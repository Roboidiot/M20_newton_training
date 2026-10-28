# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-ground stepping with command-dependent air-time rewards and wheel-speed penalties."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass,replace
from isaaclab.physics import PhysicsCfg

from M20_newton_training.tasks.m20.mdp.walk_rewards import (
    feet_air_time_command_scaled,
    wheel_velocity_command_scaled_l2,
)

from .flat_env_cfg import DeeproboticsM20FlatEnvCfg
from M20_newton_training.tasks.m20.mdp.keyboard_velocity import KeyboardVelocityCommandCfg


@configclass
class DeeproboticsM20WalkEnvCfg(DeeproboticsM20FlatEnvCfg):
    """Keep Flat's interfaces and other reward weights; enable gait shaping without terrain levels."""

    reward_gait_level: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.gait_level = None

        # Replace the turn-only air-time term, rather than rewarding the same touchdown twice.
        self.rewards.feet_air_time_ang_z_M20.weight = 0.0
        self.rewards.feet_air_time.func = feet_air_time_command_scaled
        self.rewards.feet_air_time.weight = 10.0
        self.rewards.feet_air_time.params = {
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
            "threshold": 0.5,
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
            keyboard_speeds=(0.4, 0.0, 0.0),
        )