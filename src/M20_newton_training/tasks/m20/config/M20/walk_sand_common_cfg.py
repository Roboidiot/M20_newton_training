"""Shared WalkOnSand/WalkOnSandProxy rewards, commands and robot reset configuration."""

from collections.abc import Callable
from typing import cast

from torch import Tensor

from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass, replace

from M20_newton_training.tasks.m20.mdp.crawl_rewards import BalancedFootClearanceReward
from M20_newton_training.tasks.m20.mdp.position_tracking_rewards import CommandPositionTrackingReward
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import (
    ProxyFootForwardStepReward,
    ProxyFootLiftConsistency,
    ProxyRotationGaitStatus,
    ProxyRotationGaitSymmetry,
    com_forward_support_l2,
    forward_pitch_orientation_l2,
    front_wheels_height_reward,
    no_turn_ang_vel_z_l2,
    proxy_feet_air_time_command_scaled,
    proxy_feet_slide_ang_z_cmd,
    proxy_hipx_symmetry_l2,
    sand_base_height_l2,
)
from M20_newton_training.tasks.m20.mdp.rewards import track_ang_vel_z_exp, track_lin_vel_xy_exp
from M20_newton_training.tasks.m20.mdp.sand_walk_terminations import sand_reset_penalty
from M20_newton_training.tasks.m20.mdp.single_axis_velocity import SingleAxisVelocityCommandCfg
from M20_newton_training.tasks.m20.mdp.walk_on_sand_rewards import (
    WalkOnSandDiagonalStepReward,
    WalkOnSandStepCompletionReward,
)
from M20_newton_training.tasks.m20.sand_walk_settings import SAND_WALK_SURFACE_HEIGHT
from M20_newton_training.tasks.m20.velocity_env_cfg import TerminationsCfg

from .rough_env_cfg import DeeproboticsM20RewardsCfg


@configclass
class DeeproboticsM20WalkOnSandRewardsCfg(DeeproboticsM20RewardsCfg):
    sand_diagonal_steps = RewardTermCfg(
        func=cast(Callable[..., Tensor], WalkOnSandDiagonalStepReward),
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "command_name": "base_velocity",
            "target_lift": 0.07,
            "height_std": 0.035,
            "sync_std": 0.025,
            "command_threshold": 0.1,
            "side_multiplier": 2.0,
            "turn_multiplier": 3.0,
        },
    )
    sand_step_completion = RewardTermCfg(
        func=cast(Callable[..., Tensor], WalkOnSandStepCompletionReward),
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "command_name": "base_velocity",
            "lift_threshold": 0.035,
            "lower_threshold": 0.01,
            "target_lift": 0.07,
            "min_stride": 0.06,
            "min_swing_time": 0.0,
            "max_swing_time": 0.8,
            "command_threshold": 0.1,
            "side_multiplier": 2.0,
            "turn_multiplier": 3.0,
        },
    )


@configclass
class SandWalkRewardsCfg(DeeproboticsM20WalkOnSandRewardsCfg):
    track_position_xy = RewardTermCfg(
        func=cast(Callable[..., Tensor], CommandPositionTrackingReward),
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "window_s": 2.0,
            "min_tracking_time": 0.2,
            "min_target_distance": 0.03,
            "min_command_norm": 0.1,
        },
    )
    base_height_reset_penalty = RewardTermCfg(
        func=sand_reset_penalty,
        weight=0.0,
        params={"termination_name": "base_height_low"},
    )
    front_wheels_reset_penalty = RewardTermCfg(
        func=sand_reset_penalty,
        weight=0.0,
        params={"termination_name": "front_wheels_low"},
    )
    front_wheels_height = RewardTermCfg(
        func=front_wheels_height_reward,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
        },
    )
    com_forward_l2 = RewardTermCfg(
        func=com_forward_support_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "action_term_name": "sand_proxy",
            "allowed_forward_offset": 0.01,
            "distance_scale": 0.10,
            "min_command_norm": 0.1,
            "min_support_feet": 2,
        },
    )
    ang_vel_z_no_turn_l2 = RewardTermCfg(
        func=no_turn_ang_vel_z_l2,
        weight=0.0,
        params={"command_name": "base_velocity", "zero_command_threshold": 1e-6},
    )
    feet_sand_clearance = RewardTermCfg(
        # IsaacLab instantiates ManagerTermBase classes, but func's annotation only describes the resulting callable.
        func=cast(Callable[..., Tensor], BalancedFootClearanceReward),
        weight=0.0,
        params={
            "target_clearance": 0.04,
            "min_air_time": 0.0,
            "max_swing_time": 1.0,
            "surface_height": SAND_WALK_SURFACE_HEIGHT,
            "min_base_height": 0.35,
            "max_tilt": 0.5,
            "action_term_name": "sand_proxy",
            "command_name": "base_velocity",
            "min_command_norm": 0.1,
            "max_progress_lead": 0.5,
        },
    )
    feet_forward_step = RewardTermCfg(
        func=cast(Callable[..., Tensor], ProxyFootForwardStepReward),
        weight=50.0,
        params={
            "target_stride": 0.12,
            "repeat_decay": 1.0,
            "min_air_time": 0.0,
            "action_term_name": "sand_proxy",
            "command_name": "base_velocity",
            "min_command_norm": 0.1,
        },
    )
    feet_lift_consistency = RewardTermCfg(
        func=cast(Callable[..., Tensor], ProxyFootLiftConsistency),
        weight=0.0,
        params={
            "height_scale": 0.25,
            "min_air_time": 0.0,
            "max_sample_age_s": 4.0,
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
class SandWalkTerminationsCfg(TerminationsCfg):
    base_height_low: TerminationTermCfg | None = None
    front_wheels_low: TerminationTermCfg | None = None


def configure_sand_walk_task(cfg, contact_term_name):
    """Bind shared rewards to a force provider without changing the policy interfaces."""
    robot = cfg.scene.robot
    cfg.scene.robot = replace(
        robot, init_state=replace(robot.init_state, pos=(0.0, 0.0, SAND_WALK_SURFACE_HEIGHT + 0.53))
    )
    if contact_term_name == "sand_proxy":
        cfg.actions.sand_proxy.surface_height = SAND_WALK_SURFACE_HEIGHT
    cfg.scene.contact_forces = None
    # Restore body-frame exponential tracking with denominator std**2 = 0.25.
    cfg.rewards.track_lin_vel_xy_exp.weight = 5.0
    cfg.rewards.track_ang_vel_z_exp.weight = 2.0
    cfg.rewards.track_lin_vel_xy_exp.func = track_lin_vel_xy_exp
    cfg.rewards.track_ang_vel_z_exp.func = track_ang_vel_z_exp
    cfg.rewards.track_lin_vel_xy_exp.params = {"command_name": "base_velocity", "std": 0.5}
    cfg.rewards.track_ang_vel_z_exp.params = {"command_name": "base_velocity", "std": 0.5}
    # Reuse Walk's contact slots with the selected force provider.
    for name, func in (
        ("feet_air_time", proxy_feet_air_time_command_scaled),
        ("feet_slide_ang_z_cmd", proxy_feet_slide_ang_z_cmd),
    ):
        term = getattr(cfg.rewards, name)
        term.func = func
        term.params.pop("sensor_cfg")
        term.params["action_term_name"] = contact_term_name
    cfg.rewards.feet_lift_consistency.params["action_term_name"] = contact_term_name
    cfg.rewards.feet_sand_clearance.params["action_term_name"] = contact_term_name
    cfg.rewards.feet_forward_step.params["action_term_name"] = contact_term_name
    cfg.rewards.com_forward_l2.params["action_term_name"] = contact_term_name
    # Match Walk's capped air time and reduced short-air penalty with the selected force provider.
    cfg.rewards.feet_air_time.params.update(threshold=0.3, max_air_time=1.0, short_air_penalty_scale=0.2)
    for name in ("feet_air_time", "feet_lift_consistency"):
        getattr(cfg.rewards, name).params.update(command_name="base_velocity", min_command_norm=0.1)
    # Match Walk's sustained diagonal lift and phase scoring with sand contact forces.
    # Lift compares mean swing/support world-Z heights without an air-time minimum.
    cfg.rewards.rotation_gait_status = RewardTermCfg(
        func=cast(Callable[..., Tensor], ProxyRotationGaitStatus),
        weight=3.0,
        params={
            "target_height": 0.10,
            "group_a_body_names": ["fl_wheel", "hr_wheel"],
            "group_b_body_names": ["fr_wheel", "hl_wheel"],
            "min_air_time": 0.0,
            "reward_mode": "rough",
            "action_term_name": contact_term_name,
            "command_name": "base_velocity",
            "min_command_norm": 0.1,
        },
    )
    cfg.rewards.rotation_gait_symmetry = RewardTermCfg(
        func=cast(Callable[..., Tensor], ProxyRotationGaitSymmetry),
        weight=15.0,
        params={
            "window_s": 4.0,
            "group_a_body_names": ["fl_wheel", "hr_wheel"],
            "group_b_body_names": ["fr_wheel", "hl_wheel"],
            "action_term_name": contact_term_name,
            "command_name": "base_velocity",
            "min_command_norm": 0.1,
        },
    )
    cfg.rewards.base_height_l2.func = sand_base_height_l2
    cfg.rewards.base_height_l2.params = {"target_height": 0.38, "surface_height": SAND_WALK_SURFACE_HEIGHT}
    cfg.rewards.base_height_l2.weight = -10.0
    cfg.rewards.flat_orientation_l2.weight = -20.0
    cfg.rewards.flat_orientation_l2.func = forward_pitch_orientation_l2
    cfg.rewards.flat_orientation_l2.params = {
        "command_name": "base_velocity",
        "forward_pitch_deg": -15.0,
        "forward_threshold": 0.1,
    }
    # Keep lateral posture control while allowing sagittal stepping freedom.
    cfg.rewards.hipx_joint_pos_penalty.weight = -0.5
    cfg.rewards.joint_mirror.weight = -0.03
    cfg.rewards.lin_vel_z_l2.weight = -2.0
    cfg.rewards.joint_vel_wheel_l2.weight = -10.0
    cfg.rewards.feet_air_time.weight = 5.0
    cfg.rewards.joint_acc_l2.weight = 0.0
    cfg.rewards.joint_torques_l2.weight = 0.0
    cfg.rewards.action_rate_l2.weight = 0.0
    cfg.rewards.ang_vel_xy_l2.weight = 0.0
    for name in (
        "contact_forces",
        "hipy_joint_pos_penalty",
        "knee_joint_pos_penalty",
        "joint_acc_wheel_l2",
        "action_smooth_l2",
        "feet_slide_ang_z_cmd",
        "stand_still",
        "undesired_contacts",
        "sand_diagonal_steps",
        "sand_step_completion",
        "feet_lift_consistency",
        "joint_acc_l2",
        "joint_torques_l2",
    ):
        getattr(cfg.rewards, name).weight = 0.0
    cfg.events.randomize_reset_base.params["pose_range"].update(
        x=(0.0, 0.0), y=(0.0, 0.0), roll=(-0.02, 0.02), pitch=(-0.02, 0.02),yaw=(0.0, 0.0)
    )
    # Choose among nonzero-range axes uniformly; (0, 0) disables an axis.
    command = cfg.commands.base_velocity
    cfg.commands.base_velocity = SingleAxisVelocityCommandCfg(
        asset_name=command.asset_name,
        resampling_time_range=command.resampling_time_range,
        debug_vis=command.debug_vis,
        min_abs_command=0.15,
        ranges=replace(
            command.ranges,
            lin_vel_x=(0.15, 0.25),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-0.0, 0.0),
            heading=None,
        ),
    )
