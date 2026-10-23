# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Command-scaled kinematic stepping scores; no contact-force or sand-clearance estimate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .sand_rewards import SandDiagonalStepReward, SandStepCompletionReward

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command_scale(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    side_multiplier: float,
    turn_multiplier: float,
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    side = command[:, 1].abs() > command_threshold
    turn = command[:, 2].abs() > command_threshold
    return 1.0 + (side_multiplier - 1.0) * side + (turn_multiplier - 1.0) * turn


class WalkOnSandDiagonalStepReward(SandDiagonalStepReward):
    """Sand's relative diagonal lift score, amplified for side and turning commands."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        target_lift: float = 0.07,
        height_std: float = 0.035,
        sync_std: float = 0.025,
        command_threshold: float = 0.1,
        side_multiplier: float = 2.0,
        turn_multiplier: float = 3.0,
    ) -> torch.Tensor:
        reward = super().__call__(env, asset_cfg, command_name, target_lift, height_std, sync_std)
        return reward * _command_scale(env, command_name, command_threshold, side_multiplier, turn_multiplier)


class WalkOnSandStepCompletionReward(SandStepCompletionReward):
    """Sand's alternating completed-stroke credit with Walk's command multipliers."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        lift_threshold: float = 0.035,
        lower_threshold: float = 0.01,
        target_lift: float = 0.07,
        min_stride: float = 0.06,
        min_swing_time: float = 0.08,
        max_swing_time: float = 0.8,
        command_threshold: float = 0.1,
        side_multiplier: float = 2.0,
        turn_multiplier: float = 3.0,
    ) -> torch.Tensor:
        reward = super().__call__(
            env,
            asset_cfg,
            command_name,
            lift_threshold,
            lower_threshold,
            target_lift,
            min_stride,
            min_swing_time,
            max_swing_time,
        )
        return reward * _command_scale(env, command_name, command_threshold, side_multiplier, turn_multiplier)
