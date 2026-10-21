# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Command-dependent stepping rewards for flat-ground M20 walking."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .rewards import _contact_tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time_command_scaled(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.2,
    max_air_time: float = 0.5,
    command_threshold: float = 0.1,
    side_multiplier: float = 2.0,
    turn_multiplier: float = 3.0,
) -> torch.Tensor:
    """Score air time on touchdown; amplify side/turn commands, including mixed commands.

    Signed credit discourages brief contact chatter. Capping air time bounds the reward for holding a leg up.
    Multipliers add their increments: side plus turn gives ``side_multiplier + turn_multiplier - 1``.
    """
    sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
    first_contact = sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    air_time = _contact_tensor(sensor.data.last_air_time)[:, sensor_cfg.body_ids]
    credit = ((air_time.clamp(max=max_air_time) - threshold) * first_contact).sum(dim=1)
    command = env.command_manager.get_command(command_name)
    side = command[:, 1].abs() > command_threshold
    turn = command[:, 2].abs() > command_threshold
    moving = (torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold) | turn
    scale = 1.0 + (side_multiplier - 1.0) * side + (turn_multiplier - 1.0) * turn
    return credit * scale * moving


def wheel_velocity_command_scaled_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    turn_threshold: float = 0.1,
    turn_multiplier: float = 3.0,
) -> torch.Tensor:
    """Penalize actual wheel speeds in both directions, more strongly for any turning command."""
    asset: Articulation = env.scene[asset_cfg.name]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    penalty = velocity.square().mean(dim=1)
    turning = env.command_manager.get_command(command_name)[:, 2].abs() > turn_threshold
    return penalty * (1.0 + (turn_multiplier - 1.0) * turning)
