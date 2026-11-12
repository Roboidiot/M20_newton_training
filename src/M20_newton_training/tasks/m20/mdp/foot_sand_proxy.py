"""Apply neural or analytical M20 foot-sand wrenches each physics tick without policy action slots."""

from __future__ import annotations

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from M20_newton_training.sand_proxy.foot_model import FootSandModel
from M20_newton_training.sand_proxy.neural.foot_model import DEFAULT_MODEL_PATH as DEFAULT_NEURAL_MODEL_PATH
from M20_newton_training.sand_proxy.neural.foot_model import NeuralFootSandModel
from M20_newton_training.sand_proxy.neural.wheel_model_runtime import shift_torque_reference
from M20_newton_training.tasks.m20.sand_walk_settings import SAND_WALK_SURFACE_HEIGHT

from .proxy_contact_rewards import ProxyFootContactState


def rotate_xyzw(quaternion, vector):
    xyz, w = quaternion[..., :3], quaternion[..., 3:]
    return vector + 2 * torch.linalg.cross(xyz, torch.linalg.cross(xyz, vector) + w * vector)


class FootSandProxyAction(ActionTerm):
    """Physical contact forces with no policy action slots and no observation changes."""

    def __init__(self, cfg, env):
        physics = env.cfg.sim.physics
        if not isinstance(physics, NewtonCfg) or not isinstance(physics.solver_cfg, MJWarpSolverCfg):
            raise ValueError("The foot proxy task requires its plain Newton MJWarp backend; do not use MPM coupling")
        super().__init__(cfg, env)
        self._export_IO_descriptor = False
        self.body_ids, names = self._asset.find_bodies(cfg.foot_names, preserve_order=True)
        if names != list(cfg.foot_names):
            raise ValueError(f"Expected four ordered wheel bodies: {names}")
        self.model_kind = cfg.model_kind
        self.model: NeuralFootSandModel | FootSandModel
        if cfg.model_kind == "neural":
            self.model = NeuralFootSandModel.load(cfg.model_path, num_envs=env.num_envs, device=env.device)
            self.model.runtime.check_physics_dt(env.physics_dt)
            self.joint_ids, names = self._asset.find_joints(cfg.wheel_joint_names, preserve_order=True)
            if list(names) != list(cfg.wheel_joint_names):
                raise ValueError(f"Expected ordered wheel joints: {names}")
        elif cfg.model_kind == "analytical":
            self.model = FootSandModel.load(
                cfg.model_path, num_envs=env.num_envs, num_feet=4, device=env.device, keep=cfg.footprint_keep
            )
        else:
            raise ValueError(f"Unknown sand model_kind: {cfg.model_kind!r}; use neural or analytical")
        self._empty = torch.empty((env.num_envs, 0), device=env.device)
        shape = self.model.parameters["geometry"]
        if shape["type"] != "capsule":
            raise ValueError("Current M20 proxy expects its calibrated capsule wheel geometry")
        if cfg.model_kind == "neural":
            wheels = shape["wheels"]
            if [wheel["body_name"] for wheel in wheels] != list(cfg.foot_names):
                raise ValueError("Neural deployment geometry must match FL/FR/HL/HR body order")
            self.radius = torch.tensor([wheel["radius"] for wheel in wheels], device=env.device)
            self.half_length = torch.tensor([wheel["half_length"] for wheel in wheels], device=env.device)
            self.local_center = torch.tensor([wheel["local_center"] for wheel in wheels], device=env.device)
            self.local_axis = torch.tensor([wheel["local_axis"] for wheel in wheels], device=env.device)
            self.local_center = self.local_center.expand(env.num_envs, 4, 3)
            self.local_axis = self.local_axis.expand(env.num_envs, 4, 3)
        else:
            self.radius, self.half_length = shape["radius"], shape["length"] / 2
            self.local_center = torch.tensor(shape["xyz"], device=env.device).expand(env.num_envs, 4, 3)
            primitive_quat = torch.tensor(shape["quat_xyzw"], device=env.device)
            axis = rotate_xyzw(primitive_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device))
            self.local_axis = axis.expand(env.num_envs, 4, 3)
        self.surface_height = env.scene.env_origins[:, 2:3] + cfg.surface_height
        self.force_w = torch.zeros((env.num_envs, 4, 3), device=env.device)
        self.torque_w = torch.zeros_like(self.force_w)
        # Match the three-sample history used by the inherited contact-force penalty.
        # Record the actual applied force, including its stability limiter, at each physics tick.
        self.force_history_w = torch.zeros((env.num_envs, 3, 4, 3), device=env.device)
        self._contact_state = ProxyFootContactState(env.num_envs, env.device)
        self.contact_force_w = torch.zeros_like(self.force_w)
        self._contact_pending = False
        self._rigid_sensor = None
        if cfg.rigid_contact_sensor is not None:
            self._rigid_sensor = env.scene.sensors[cfg.rigid_contact_sensor]
            self._rigid_sensor_ids, names = self._rigid_sensor.find_bodies(cfg.foot_names, preserve_order=True)
            if list(names) != list(cfg.foot_names):
                raise ValueError(f"Expected ordered rigid contact sensor feet: {names}")
            if self._rigid_sensor.num_filter_objects < 1:
                raise ValueError("The rigid foot contact sensor must match the ground collision shapes")
        self._physics_tick = 0
        self.application_points_w = torch.zeros_like(self.force_w)
        self.effective_mass = self._asset.data.default_mass.torch.sum(dim=1, keepdim=True) / 4

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._empty

    @property
    def processed_actions(self):
        return self._empty

    @property
    def contact_state(self):
        self.update_contact_state()
        return self._contact_state

    @torch.no_grad()
    def update_contact_state(self):
        """Publish support once after physics, using proxy plus filtered rigid floor reactions."""
        if not self._contact_pending:
            return
        self.contact_force_w.copy_(self.force_w)
        if self._rigid_sensor is not None:
            forces = self._rigid_sensor.data.force_matrix_w
            if forces is None:
                raise RuntimeError("Rigid foot contacts require the ground-filtered force matrix")
            forces = forces if isinstance(forces, torch.Tensor) else forces.torch
            self.contact_force_w.add_(forces[:, self._rigid_sensor_ids].sum(dim=2))
        self._contact_state.update(
            self.contact_force_w,
            self._env.physics_dt * self._env.cfg.decimation,
            self.cfg.contact_force_threshold,
            metrics_force_w=self.force_w,
        )
        self._contact_pending = False

    def process_actions(self, actions):
        self._contact_state.begin_step()
        self._physics_tick = 0

    @torch.no_grad()
    def apply_actions(self):
        data = self._asset.data
        ids = self.body_ids
        q = data.body_link_quat_w.torch[:, ids]
        center = data.body_link_pos_w.torch[:, ids] + rotate_xyzw(q, self.local_center)
        axis = rotate_xyzw(q, self.local_axis)
        point = center.clone()
        point[..., 2] -= self.radius + self.half_length * axis[..., 2].abs()
        if isinstance(self.model, NeuralFootSandModel):
            angular = data.body_com_ang_vel_w.torch[:, ids]
            com = data.body_com_pos_w.torch[:, ids]
            center_velocity = data.body_com_lin_vel_w.torch[:, ids] + torch.linalg.cross(angular, center - com, dim=-1)
            self.force_w.copy_(
                self.model.step(
                    self.surface_height - point[..., 2],
                    center_velocity,
                    center[..., :2],
                    self._env.physics_dt,
                    angular_velocity_w=angular,
                    axis_w=axis,
                    joint_velocity=data.joint_vel.torch[:, self.joint_ids],
                )
            )
            reference_com = data.body_link_pos_w.torch[:, ids] + rotate_xyzw(
                q, self.model.reference_com_b.expand_as(center)
            )
            self.torque_w.copy_(shift_torque_reference(self.force_w, self.model.torque, reference_com, com))
            # The predicted torque already includes the resultant force's moment about the source COM.
            # Apply at current COM after shifting the torque, never at the bottom a second time.
            self._asset.instantaneous_wrench_composer.add_forces_and_torques_index(
                forces=self.force_w, torques=self.torque_w, positions=com, body_ids=ids, is_global=True
            )
        else:
            self._apply_analytical_force(data, ids, center, point)
        self.force_history_w.copy_(self.force_history_w.roll(1, dims=1))
        self.force_history_w[:, 0].copy_(self.force_w)
        self._physics_tick += 1
        if self._physics_tick == self._env.cfg.decimation:
            # Rigid sensor forces are current only after the last sim.step + scene.update.
            # Rewards publish the combined sample lazily, once per 20 ms control interval.
            self._contact_pending = True
            if self._rigid_sensor is None:
                self.update_contact_state()
        self.application_points_w.copy_(point)

    def _apply_analytical_force(self, data, ids, center, point):
        assert isinstance(self.model, FootSandModel)
        velocity = data.body_com_lin_vel_w.torch[:, ids] + torch.linalg.cross(
            data.body_com_ang_vel_w.torch[:, ids], point - data.body_com_pos_w.torch[:, ids]
        )
        # Rotate horizontal anisotropic drag with the chassis heading; gravity stays world-up.
        base_q = data.root_link_quat_w.torch
        x, y, z, w = base_q.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        local_velocity = torch.stack(
            [
                c * velocity[..., 0] + s * velocity[..., 1],
                -s * velocity[..., 0] + c * velocity[..., 1],
                velocity[..., 2],
            ],
            dim=-1,
        )
        force = self.model.step(
            self.surface_height - point[..., 2],
            local_velocity,
            center[..., :2],
            self._env.physics_dt,
            bearing_scale=self.cfg.bearing_scale,
            drag_scale=self.cfg.drag_scale,
            stability_mass_kg=self.effective_mass,
        )
        self.force_w.copy_(
            torch.stack(
                [c * force[..., 0] - s * force[..., 1], s * force[..., 0] + c * force[..., 1], force[..., 2]], dim=-1
            )
        )
        # Composer adds r x F about each wheel CoM. No duplicate explicit torque is supplied.
        self._asset.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=self.force_w, positions=point, body_ids=ids, is_global=True
        )

    def reset(self, env_ids=None):
        self.contact_state.log_force_metrics(self._env, env_ids)
        self.model.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.force_w[ids] = 0
        self.torque_w[ids] = 0
        self.force_history_w[ids] = 0
        self.contact_force_w[ids] = 0
        self.contact_state.reset(env_ids)


@configclass
class FootSandProxyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FootSandProxyAction
    asset_name: str = "robot"
    foot_names: tuple[str, ...] = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")
    model_kind: str = "neural"
    model_path: str = str(DEFAULT_NEURAL_MODEL_PATH)
    wheel_joint_names: tuple[str, ...] = ("fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint")
    surface_height: float = SAND_WALK_SURFACE_HEIGHT
    bearing_scale: float = 1.0
    drag_scale: float = 1.0
    footprint_keep: float = 0.7
    contact_force_threshold: float = 1.0
    rigid_contact_sensor: str | None = None


def proxy_contact_forces(env, threshold: float, action_term_name: str = "sand_proxy"):
    """Sum per-foot force excess over a threshold, using three applied-force samples.

    Unlike a rigid ContactSensor's normal-only force, this uses the complete proxy
    force vector (support and horizontal drag) in newtons. No model state is advanced
    during reward evaluation. The environment evaluates rewards after physics and
    before resetting the action's force history.
    """
    term = env.action_manager.get_term(action_term_name)
    peak_force = torch.linalg.vector_norm(term.force_history_w, dim=-1).amax(dim=1)
    return (peak_force - threshold).clamp(min=0.0).sum(dim=1)


def proxy_depth_limit(env, limit_m: float | None = None):
    """End unsupported penetrations instead of relying on unmodeled shin/body support."""
    term = env.action_manager.get_term("sand_proxy")
    if limit_m is None:
        parameters = term.model.parameters
        limit_m = parameters.get("training_depth_limit_m", parameters.get("domain", {}).get("max_depth_m"))
        if limit_m is None:
            raise ValueError("This model has no calibrated depth limit; pass limit_m explicitly")
    return (term.model.depth > limit_m).any(dim=1)
