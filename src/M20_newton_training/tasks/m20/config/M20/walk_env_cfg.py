# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-ground stepping with command-dependent air-time rewards and wheel-speed penalties."""

from collections.abc import Callable
from typing import cast

from torch import Tensor

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass, replace

from M20_newton_training.tasks.m20.mdp.keyboard_velocity import KeyboardVelocityCommandCfg
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import (
    ProxyRotationGaitStatus,
    ProxyRotationGaitSymmetry,
    proxy_feet_air_time_command_scaled,
)
from M20_newton_training.tasks.m20.mdp.single_axis_velocity import SingleAxisVelocityCommandCfg
from M20_newton_training.tasks.m20.mdp.walk_rewards import (
    feet_air_time_command_scaled,
    wheel_velocity_command_scaled_l2,
)

from .flat_env_cfg import DeeproboticsM20FlatEnvCfg


@configclass
class DeeproboticsM20WalkEnvCfg(DeeproboticsM20FlatEnvCfg):
    """Keep Flat's interfaces with selected penalties and gait shaping without terrain levels."""

    reward_gait_level: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.gait_level = None

        # Replace the turn-only air-time term, rather than rewarding the same touchdown twice.
        self.rewards.feet_air_time_ang_z_M20.weight = 0.0
        self.rewards.feet_air_time.func = feet_air_time_command_scaled
        self.rewards.feet_air_time.weight = 5.0
        self.rewards.feet_air_time.params = {
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
            "threshold": 0.3,
            "max_air_time": 1.0,
            "command_threshold": 0.1,
            "side_multiplier": 2.0,
            "turn_multiplier": 3.0,
        }

        self.rewards.joint_vel_wheel_l2.func = wheel_velocity_command_scaled_l2
        self.rewards.joint_vel_wheel_l2.weight = -0.2
        self.rewards.joint_vel_wheel_l2.params = {
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names, preserve_order=True),
            "turn_threshold": 0.1,
            "turn_multiplier": 5.0,
        }

        if self.__class__.__name__ == "DeeproboticsM20WalkEnvCfg":
            # Bind only the rigid Walk task; its sand subclasses keep their own force providers/configs.
            self.rewards.hipx_joint_pos_penalty.weight = -3.0
            self.rewards.lin_vel_z_l2.weight = -2.0
            self.rewards.joint_mirror.weight = -0.03
            self.rewards.flat_orientation_l2.weight = -50.0
            self.rewards.base_height_l2.weight = -10.0
            self.rewards.base_height_l2.params.update(target_height=0.58, sensor_cfg=None)
            feet = ["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"]
            sensor = SceneEntityCfg("contact_forces", body_names=feet, preserve_order=True)
            self.rewards.feet_air_time.func = proxy_feet_air_time_command_scaled
            self.rewards.feet_air_time.params.update(
                sensor_cfg=sensor, min_command_norm=0.1, max_air_time=1.0, short_air_penalty_scale=0.2
            )
            self.rewards.rotation_gait_status = RewardTermCfg(
                func=cast(Callable[..., Tensor], ProxyRotationGaitStatus),
                weight=3.0,
                params={
                    "sensor_cfg": sensor,
                    "command_name": "base_velocity",
                    "target_height": 0.10,
                    "min_air_time": 0.0,
                    "reward_mode": "rough",
                    "min_command_norm": 0.1,
                    "group_a_body_names": ["fl_wheel", "hr_wheel"],
                    "group_b_body_names": ["fr_wheel", "hl_wheel"],
                },
            )
            self.rewards.rotation_gait_symmetry = RewardTermCfg(
                func=cast(Callable[..., Tensor], ProxyRotationGaitSymmetry),
                weight=15.0,
                params={
                    "sensor_cfg": sensor,
                    "command_name": "base_velocity",
                    "window_s": 4.0,
                    "min_command_norm": 0.1,
                    "group_a_body_names": ["fl_wheel", "hr_wheel"],
                    "group_b_body_names": ["fr_wheel", "hl_wheel"],
                },
            )
            command = self.commands.base_velocity
            self.commands.base_velocity = SingleAxisVelocityCommandCfg(
                asset_name=command.asset_name,
                resampling_time_range=command.resampling_time_range,
                debug_vis=command.debug_vis,
                min_abs_command=0.25,
                ranges=replace(
                    command.ranges, lin_vel_x=(0.25, 0.4), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=None
                ),
            )
            for name in (
                "ang_vel_xy_l2",
                "joint_torques_l2",
                "hipy_joint_pos_penalty",
                "knee_joint_pos_penalty",
                "action_rate_l2",
                "contact_forces",
                "undesired_contacts"
            ):
                getattr(self.rewards, name).weight = 0.0
            self.disable_zero_weight_rewards()

    def play_mode(self):
        """Use body-frame keyboard velocities for playback; training retains random commands."""
        super().play_mode()
        self.scene.num_envs = 1
        command = self.commands.base_velocity
        self.commands.base_velocity = KeyboardVelocityCommandCfg(
            asset_name=command.asset_name,
            ranges=replace(command.ranges),
            debug_vis=command.debug_vis,
            keyboard_speeds=(0.4, 0.2, 0.4),
        )
