# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Upright rough-road locomotion environment for the M20 wheeled quadruped.

The robot is initialized in a user-provided standing posture with its ``base_link`` +X axis oriented along the
world +Z (up) axis, and follows body Y/Z linear and body X angular velocity commands while maintaining
``0.8028 m`` clearance above the local terrain and gravity aligned with the base -X axis.

Only random rough roads are generated, with a fixed height-noise range and no terrain difficulty curriculum.
Orientation terms use base +X = up.
"""

from __future__ import annotations

import math

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, RigidObject
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from M20_newton_training.assets.deeprobotics import DEEPROBOTICS_M20_CFG
from M20_newton_training.tasks.m20.config.M20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from M20_newton_training.tasks.m20.config.M20.stand_terrains import StandRandomRoughTerrainCfg
from M20_newton_training.tasks.m20.mdp.stand_commands import StandVelocityCommandCfg
from M20_newton_training.tasks.m20.mdp.stand_keyboard import KeyboardStandVelocityCommand

##
# Standing pose
##

# Joint-space standing posture (rad): hipx = 0, front hipy = -1.64, hind hipy = -0.92,
# front knee = 1.02, hind knee = -0.99, wheels = 0. Provided by the user from their rigging tool.
STANDING_JOINT_POS = {
    ".*hipx_joint": 0.0,
    "f[l,r]_hipy_joint": -1.64,
    "h[l,r]_hipy_joint": -0.92,
    "f[l,r]_knee_joint": 1.02,
    "h[l,r]_knee_joint": -0.99,
    ".*wheel_joint": 0.0,
}

# Base-link clearance above local terrain in the Z-up world frame [m].
STAND_BASE_HEIGHT = 0.8028

# Quaternion (x, y, z, w) mapping base_link +X onto world +Z (up): a -90 deg rotation about the world Y axis.
STAND_BASE_ROT = (0.0, -0.5 * math.sqrt(2.0), 0.0, 0.5 * math.sqrt(2.0))

DEEPROBOTICS_M20_STAND_CFG = DEEPROBOTICS_M20_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, STAND_BASE_HEIGHT),
        rot=STAND_BASE_ROT,
        joint_pos=STANDING_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
)


##
# Stand-specific MDP terms.
#
# Keep balance and smoothness penalties active at every terrain level instead of
# gating them by the parent's global ``gait_level`` scalar.
##


def stand_track_lin_vel_yz_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Track command vx/vy along the upright robot's body Y/Z axes."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b.torch[:, 1:3]), dim=1)
    return torch.exp(-error / std**2)


def stand_track_ang_vel_x_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Track command wz about the upright robot's body X axis."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_ang_vel_b.torch[:, 0])
    return torch.exp(-error / std**2)


def stand_lin_vel_x_l2(env, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Penalize vertical motion along the upright robot's body X axis."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b.torch[:, 0])


def stand_ang_vel_yz_l2(env, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Penalize tipping rates about the upright robot's body Y/Z axes."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, 1:3]), dim=1)


def stand_base_height_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg | None = None,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Maintain terrain-relative height, ignoring invalid rays per environment."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    ground_height = 0.0
    if sensor_cfg is not None:
        ray_heights = env.scene[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
        valid = torch.isfinite(ray_heights)
        count = valid.sum(dim=1)
        mean_height = torch.where(valid, ray_heights, 0.0).sum(dim=1) / count.clamp_min(1)
        ground_height = torch.where(count > 0, mean_height, env.scene.env_origins[:, 2])
    return torch.square(asset.data.root_pos_w.torch[:, 2] - ground_height - target_height)


def stand_still_joint_deviation_l1(
    env, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float = 0.1
) -> torch.Tensor:
    """Hold the upright joint pose only when translation AND turning commands are small."""
    command = env.command_manager.get_command(command_name)
    return base_mdp.joint_deviation_l1(env, asset_cfg) * (torch.linalg.norm(command, dim=1) < command_threshold)


def stand_front_joint_mirror_l2(
    env,
    asset_cfg: SceneEntityCfg,
    mirror_asset_cfg: SceneEntityCfg,
    joint_signs: tuple[float, float, float] = (-1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Penalize front-leg mirror error without command, gravity, or gait-level gates.

    In hipx/hipy/knee order, hipx mirrors with opposite joint angles while hipy
    and knee mirror with equal angles, matching the M20's left/right joint axes.
    """
    asset = env.scene[asset_cfg.name]
    left = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    right = asset.data.joint_pos.torch[:, mirror_asset_cfg.joint_ids]
    error = left - right * left.new_tensor(joint_signs)
    return torch.sum(torch.square(error), dim=1)


def stand_action_rate_l2(env) -> torch.Tensor:
    """Penalize first-order action change."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def stand_action_smooth_l2(env) -> torch.Tensor:
    """Penalize second-order action change using an env-local action history."""
    cache_name = "_stand_action_smooth_prev_prev"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, torch.zeros_like(env.action_manager.action))
    prev_prev_action = getattr(env, cache_name)
    diff = torch.square(env.action_manager.action + prev_prev_action - 2.0 * env.action_manager.prev_action)
    diff = diff * (env.action_manager.prev_action != 0)
    diff = diff * (prev_prev_action != 0)
    reward = torch.sum(diff, dim=1)
    setattr(env, cache_name, env.action_manager.prev_action.clone())
    return reward


def stand_joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied joint torques."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def stand_orientation_x_up_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize deviation of gravity from the base -X axis (i.e. keep base +X pointing up)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    desired = torch.tensor([-1.0, 0.0, 0.0], device=env.device)
    error = asset.data.projected_gravity_b - desired
    return torch.sum(torch.square(error), dim=1)


def stand_bad_orientation_x_up(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the base +X axis tips too far from world +Z."""
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    flipped = gravity[:, 0] > 0.0
    tilted = (gravity[:, 1:].abs() > 0.7).any(dim=-1)
    return flipped | tilted


def stand_bad_orientation_x_up_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty (0/1) matching the :func:`stand_bad_orientation_x_up` termination condition."""
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b
    flipped = gravity[:, 0] > 0.0
    tilted = (gravity[:, 1:].abs() > 0.7).any(dim=-1)
    return (flipped | tilted).float()


##
# Environment configuration
##


@configclass
class DeeproboticsM20StandEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Track reduced velocity commands on rough roads with the base +X axis up."""

    def __post_init__(self):
        # Inherit Rough's sensors and randomization, then disable terrain curricula.
        super().__post_init__()

        # --- Robot: standing pose, base +X -> world +Z, base height 0.8028 m ---
        self.scene.robot = DEEPROBOTICS_M20_STAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Every tile uses the same roughness range; no progression between levels.
        self.scene.terrain.terrain_generator = self.scene.terrain.terrain_generator.replace(
            curriculum=False, slope_threshold=None, sub_terrains={"random_rough": StandRandomRoughTerrainCfg()}
        )
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.curriculum.gait_level = None
        # World-aligned scans avoid extracting yaw from an upright base frame.
        self.scene.height_scanner.ray_alignment = "world"
        self.scene.height_scanner_base.ray_alignment = "world"
        self.scene.height_scanner_base.update_period = self.decimation * self.sim.dt

        # --- Commands: half of Rough's ranges, interpreted in body Y/Z/X ---
        self.commands.base_velocity = StandVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.commands.base_velocity.resampling_time_range,
            ranges=StandVelocityCommandCfg.Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)),
        )

        # --- Zero out locomotion / gait / contact rewards inherited from the parent configs ---
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = 0.0
        self.rewards.joint_acc_wheel_l2.weight = 0.0
        self.rewards.hipx_joint_pos_penalty.weight = 0.0
        self.rewards.hipy_joint_pos_penalty.weight = 0.0
        self.rewards.knee_joint_pos_penalty.weight = 0.0
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.contact_forces.weight = 0.0
        self.rewards.feet_air_time_ang_z_M20.weight = 0.0
        self.rewards.feet_slide_ang_z_cmd.weight = 0.0
        self.rewards.rotation_gait_status.weight = 0.0
        self.rewards.rotation_gait_symmetry.weight = 0.0
        self.rewards.bad_orientation_penalty.weight = 0.0

        # --- Stand rewards ---
        # Hold the standing leg posture at rest; allow movement and pure turning.
        self.rewards.stand_still.func = stand_still_joint_deviation_l1
        self.rewards.stand_still.weight = -0.001
        # Terrain-relative height replaces the old fixed world-height target.
        self.rewards.base_height_l2.func = stand_base_height_l2
        self.rewards.base_height_l2.params = {
            "target_height": STAND_BASE_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
        }
        self.rewards.base_height_l2.weight = -10.0
        # Retain reward keys/weights, but map command slots to body Y/Z/X.
        self.rewards.track_lin_vel_xy_exp.func = stand_track_lin_vel_yz_exp
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.func = stand_track_ang_vel_x_exp
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        # Body X is vertical; tipping occurs about body Y/Z.
        self.rewards.lin_vel_z_l2.func = stand_lin_vel_x_l2
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.func = stand_ang_vel_yz_l2
        self.rewards.ang_vel_xy_l2.weight = -0.5
        # action smoothness (custom, gait-free)
        self.rewards.action_rate_l2.func = stand_action_rate_l2
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.action_smooth_l2.func = stand_action_smooth_l2
        self.rewards.action_smooth_l2.weight = -0.05
        # effort penalty (custom, gait-free)
        self.rewards.joint_torques_l2.func = stand_joint_torques_l2
        self.rewards.joint_torques_l2.weight = -2.5e-5

        # Rough-road safety and regularization, active even on the easiest terrain.
        self.rewards.undesired_contacts.func = base_mdp.undesired_contacts
        self.rewards.undesired_contacts.weight = -0.0
        self.rewards.contact_forces.func = base_mdp.contact_forces
        self.rewards.contact_forces.weight = -1.5e-5
        self.rewards.joint_acc_l2.weight = -2e-7
        self.rewards.joint_acc_wheel_l2.weight = -1e-7
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_deviation_l1.weight = -0.01

        # Mirror the upper/front legs; the rear legs remain free to balance and steer.
        # Rough's diagonal pairing and Z-up/gait gates do not apply to upright Stand.
        self.rewards.joint_mirror = RewTerm(
            func=stand_front_joint_mirror_l2,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint"], preserve_order=True
                ),
                "mirror_asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint"], preserve_order=True
                ),
                "joint_signs": (-1.0, 1.0, 1.0),
            },
        )

        # custom X-up orientation terms
        setattr(self.rewards, "orientation_x_up_l2", RewTerm(func=stand_orientation_x_up_l2, weight=-30.0))
        setattr(
            self.rewards,
            "bad_orientation_x_up_penalty",
            RewTerm(func=stand_bad_orientation_x_up_penalty, weight=-50.0),
        )

        # --- Terminations ---
        # replace the Z-up bad-orientation check with the base +X = up convention
        self.terminations.bad_orientation_2 = DoneTerm(func=stand_bad_orientation_x_up)

        # --- Events: mild reset randomization around the standing pose ---
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                # Reset rotations are applied in the local frame: body X is yaw.
                "roll": (-math.pi, math.pi),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        }
        # Keep the startup/reset randomization but omit abrupt kicks during upright locomotion.
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        # drop zero-weight rewards
        self.disable_zero_weight_rewards()

    def play_mode(self):
        """Use keyboard commands for playback; training keeps the random sampler."""
        super().play_mode()
        self.scene.num_envs = 1
        self.commands.base_velocity.class_type = KeyboardStandVelocityCommand
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
