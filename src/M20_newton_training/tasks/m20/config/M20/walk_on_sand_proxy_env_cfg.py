"""Walk-compatible training on calibrated foot-only analytic sand instead of MPM particles."""

from typing import cast

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass, replace
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg

from isaaclab_tasks.utils import PresetCfg

from M20_newton_training.tasks.m20.mdp.foot_sand_proxy import FootSandProxyActionCfg
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import (
    ProxyDiagonalLiftConsistency,
    ProxyRotationGaitStatus,
    ProxyRotationGaitSymmetry,
    proxy_feet_air_time_command_scaled,
    proxy_feet_slide_ang_z_cmd,
    proxy_hipx_symmetry_l2,
)
from M20_newton_training.tasks.m20.velocity_env_cfg import MySceneCfg

from .rough_env_cfg import DeeproboticsM20ActionsCfg
from .walk_env_cfg import DeeproboticsM20WalkEnvCfg
from .walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandRewardsCfg


@configclass
class M20FootProxyPhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=100, nconmax=64, use_mujoco_contacts=True, integrator="implicitfast"),
        num_substeps=2,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    default = newton_mjwarp


@configclass
class FootProxySceneCfg(MySceneCfg):
    sand_surface_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SandSurfaceVisual",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.298)),
        spawn=sim_utils.CuboidCfg(
            size=(2.4, 2.4, 0.003),
            collision_props=None,
            rigid_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.53, 0.31), opacity=0.35),
        ),
    )


@configclass
class FootProxyActionsCfg(DeeproboticsM20ActionsCfg):
    sand_proxy = FootSandProxyActionCfg(contact_force_threshold=5.0)


@configclass
class FootProxyRewardsCfg(DeeproboticsM20WalkOnSandRewardsCfg):
    diagonal_lift_consistency = RewardTermCfg(
        func=ProxyDiagonalLiftConsistency,
        weight=-300.0,
        params={
            "group_a_body_names": ["fl_wheel", "hr_wheel"],
            "group_b_body_names": ["fr_wheel", "hl_wheel"],
            "height_scale": 0.25,
            "min_air_time": 0.1,
            "max_pair_age_s": 2.0,
            "action_term_name": "sand_proxy",
        },
    )
    hipx_symmetry = RewardTermCfg(
        func=proxy_hipx_symmetry_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["fl_hipx_joint", "fr_hipx_joint", "hl_hipx_joint", "hr_hipx_joint"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "lateral_cmd_threshold": 0.05,
            "yaw_cmd_threshold": 0.1,
            "maneuver_scale": 0.25,
        },
    )


@configclass
class DeeproboticsM20WalkOnSandProxyEnvCfg(DeeproboticsM20WalkEnvCfg):
    scene: FootProxySceneCfg = FootProxySceneCfg(num_envs=2048, env_spacing=3.5)
    actions: FootProxyActionsCfg = FootProxyActionsCfg()
    rewards: FootProxyRewardsCfg = FootProxyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = cast(PhysicsCfg, M20FootProxyPhysicsCfg())
        # Explicit whole-robot contact damping needs 2 kHz force updates; policy remains 50 Hz.
        self.sim.dt = 0.0005
        self.decimation = 40
        self.sim.render_interval = self.decimation
        robot = self.scene.robot
        self.scene.robot = replace(robot, init_state=replace(robot.init_state, pos=(0.0, 0.0, 0.30 + 0.58)))
        self.scene.contact_forces = None
        # Tracking kernels divide squared velocity error by std**2 = 0.25.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        # Reuse Walk's reward slots with proxy contact state and diagonal gait rewards.
        for name, func in (
            ("feet_air_time", proxy_feet_air_time_command_scaled),
            ("feet_slide_ang_z_cmd", proxy_feet_slide_ang_z_cmd),
            ("rotation_gait_status", ProxyRotationGaitStatus),
            ("rotation_gait_symmetry", ProxyRotationGaitSymmetry),
        ):
            term = getattr(self.rewards, name)
            term.func = func
            term.params.pop("sensor_cfg")
            term.params["action_term_name"] = "sand_proxy"
        # Air times below 0.1 s receive a penalty only when commanded to move.
        self.rewards.feet_air_time.params.update(threshold=0.1, max_air_time=1.0)
        # Replace the old turn-only gate with a full velocity-command norm gate.
        for name in ("rotation_gait_status", "rotation_gait_symmetry"):
            for param in ("lin_vel_threshold", "ang_vel_threshold"):
                getattr(self.rewards, name).params.pop(param)
        for name in ("feet_air_time", "rotation_gait_status", "rotation_gait_symmetry", "diagonal_lift_consistency"):
            getattr(self.rewards, name).params.update(command_name="base_velocity", min_command_norm=0.1)
        self.rewards.rotation_gait_status.params.pop("asset_cfg")
        self.rewards.rotation_gait_status.params["target_height"] = 0.25
        # Favor actual lift progress over contact-pattern occupancy and prolonged unloading.
        self.rewards.rotation_gait_status.weight = 100.0
        symmetry = self.rewards.rotation_gait_symmetry
        symmetry.weight = 6.0
        symmetry.params.pop("target_duty")
        symmetry.params.pop("std")
        symmetry.params["window_s"] = 1.0
        self.rewards.flat_orientation_l2.weight = -7.5
        # Keep lateral posture control while allowing sagittal stepping freedom.
        self.rewards.hipx_joint_pos_penalty.weight = -1.5
        self.rewards.feet_air_time.weight = 5.0
        for name in (
            "contact_forces",
            "hipy_joint_pos_penalty",
            "knee_joint_pos_penalty",
            "joint_acc_wheel_l2",
            "action_smooth_l2",
            "feet_slide_ang_z_cmd",
            "joint_mirror",
            "stand_still",
            "undesired_contacts",
            "sand_diagonal_steps",
            "sand_step_completion",
        ):
            getattr(self.rewards, name).weight = 0.0
        self.events.randomize_reset_base.params["pose_range"].update(
            x=(0.0, 0.0), y=(0.0, 0.0), roll=(-0.05, 0.05), pitch=(-0.05, 0.05)
        )
        # Begin within the measured translational speed range; strength saturates outside it.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.disable_zero_weight_rewards()
