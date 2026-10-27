"""Apply the calibrated M20 foot sand force every physics tick via a zero-dimensional action term."""

from __future__ import annotations

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH, FootSandModel

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
        self.model = FootSandModel.load(
            cfg.model_path, num_envs=env.num_envs, num_feet=4, device=env.device, keep=cfg.footprint_keep
        )
        self._empty = torch.empty((env.num_envs, 0), device=env.device)
        shape = self.model.parameters["geometry"]
        if shape["type"] != "capsule":
            raise ValueError("Current M20 proxy expects its calibrated capsule wheel geometry")
        self.radius, self.half_length = shape["radius"], shape["length"] / 2
        self.local_center = torch.tensor(shape["xyz"], device=env.device).expand(env.num_envs, 4, 3)
        primitive_quat = torch.tensor(shape["quat_xyzw"], device=env.device)
        axis = rotate_xyzw(primitive_quat, torch.tensor([0.0, 0.0, 1.0], device=env.device))
        self.local_axis = axis.expand(env.num_envs, 4, 3)
        self.surface_height = env.scene.env_origins[:, 2:3] + cfg.surface_height
        self.force_w = torch.zeros((env.num_envs, 4, 3), device=env.device)
        # Match the three-sample history used by the inherited contact-force penalty.
        # Record the actual applied force, including its stability limiter, at each physics tick.
        self.force_history_w = torch.zeros((env.num_envs, 3, 4, 3), device=env.device)
        self.contact_state = ProxyFootContactState(env.num_envs, env.device)
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

    def process_actions(self, actions):
        self.contact_state.begin_step()
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
        self.force_history_w.copy_(self.force_history_w.roll(1, dims=1))
        self.force_history_w[:, 0].copy_(self.force_w)
        self._physics_tick += 1
        if self._physics_tick == self._env.cfg.decimation:
            # Match newton-terrain: sample the last applied force once per 20 ms
            # control interval. Physical sand forces still update at every tick.
            self.contact_state.update(
                self.force_w, self._env.physics_dt * self._env.cfg.decimation, self.cfg.contact_force_threshold
            )
        self.application_points_w.copy_(point)
        # Composer adds r x F about each wheel CoM. No duplicate explicit torque is supplied.
        self._asset.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=self.force_w, positions=self.application_points_w, body_ids=ids, is_global=True
        )

    def reset(self, env_ids=None):
        self.model.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.force_w[ids] = 0
        self.force_history_w[ids] = 0
        self.contact_state.reset(env_ids)


@configclass
class FootSandProxyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FootSandProxyAction
    asset_name: str = "robot"
    foot_names: tuple[str, ...] = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")
    model_path: str = str(DEFAULT_MODEL_PATH)
    surface_height: float = 0.30
    bearing_scale: float = 1.0
    drag_scale: float = 1.0
    footprint_keep: float = 0.7
    contact_force_threshold: float = 1.0


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
        limit_m = term.model.parameters.get("training_depth_limit_m", term.model.parameters["domain"]["max_depth_m"])
    return (term.model.depth > limit_m).any(dim=1)
