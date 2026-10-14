# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity commands for the M20 with its base +X axis pointing up.

The three command slots retain the interface ``[vx, vy, wz]`` but represent
body-frame ``[linear Y, linear Z, angular X]`` in this task.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class StandVelocityCommand(UniformVelocityCommand):
    """Track Y/Z translation and X-axis rotation, using episode-mean errors."""

    def __init__(self, cfg: StandVelocityCommandCfg, env):
        if cfg.heading_command:
            raise ValueError("Stand velocity commands require heading_command=False for the X-up frame.")
        proportions = (cfg.rel_zero_vel_envs, cfg.rel_only_lin_y_envs, cfg.rel_only_lin_x_envs, cfg.rel_only_ang_z_envs)
        if any(value < 0.0 or value > 1.0 for value in proportions) or sum(proportions) > 1.0:
            raise ValueError("Stand special-command proportions must be in [0, 1] and sum to at most 1.")
        super().__init__(cfg, env)
        self.metrics["base_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["knee_pos"] = torch.zeros(self.num_envs, device=self.device)
        self._base_z_sum = torch.zeros(self.num_envs, device=self.device)
        self._knee_pos_sum = torch.zeros(self.num_envs, device=self.device)
        self._knee_joint_ids = self.robot.find_joints(".*[Kk]nee.*")[0]

    def _resample_command(self, env_ids: Sequence[int]):
        """Use Rough's sampling proportions, with direct X-axis turn commands."""
        super()._resample_command(env_ids)
        command = self.vel_command_b[env_ids].clone()
        sampled_linear = command[:, :2].clone()
        command[:, :2] *= (torch.linalg.norm(command[:, :2], dim=1) > 0.2).unsqueeze(1)

        slot = torch.rand(len(env_ids), device=self.device)
        zero_end = self.cfg.rel_zero_vel_envs
        y_end = zero_end + self.cfg.rel_only_lin_y_envs
        x_end = y_end + self.cfg.rel_only_lin_x_envs
        turn_end = x_end + self.cfg.rel_only_ang_z_envs
        zero = slot < zero_end
        only_y = (slot >= zero_end) & (slot < y_end)
        only_x = (slot >= y_end) & (slot < x_end)
        only_turn = (slot >= x_end) & (slot < turn_end)

        # Single-axis samples use the full configured range, including small
        # commands that would otherwise be suppressed by the mixed-sample threshold.
        command[zero] = 0.0
        command[only_y, 0] = 0.0
        command[only_y, 1] = sampled_linear[only_y, 1]
        command[only_y, 2] = 0.0
        command[only_x, 0] = sampled_linear[only_x, 0]
        command[only_x, 1:] = 0.0
        command[only_turn, :2] = 0.0
        self.vel_command_b[env_ids] = command
        self.is_standing_env[env_ids] = zero
        self.is_heading_env[env_ids] = False

    def _update_metrics(self):
        data = self.robot.data
        planar_velocity = data.root_lin_vel_b.torch[:, 1:3]
        # Keep the existing log keys: "xy" denotes the two command slots and
        # "yaw" denotes turning about the task's vertical (body X) axis.
        self._error_xy_sum += torch.linalg.norm(self.command[:, :2] - planar_velocity, dim=1)
        self._error_yaw_sum += torch.abs(self.command[:, 2] - data.root_ang_vel_b.torch[:, 0])
        self._step_count += 1.0
        self._base_z_sum += data.root_pos_w.torch[:, 2]

        knee_deviation = torch.linalg.norm(
            data.joint_pos.torch[:, self._knee_joint_ids] - data.default_joint_pos.torch[:, self._knee_joint_ids],
            dim=1,
        )
        moving = (torch.linalg.norm(self.command, dim=1) > 0.1) | (torch.linalg.norm(planar_velocity, dim=1) > 0.5)
        self._knee_pos_sum += torch.where(moving, knee_deviation, 5.0 * knee_deviation)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # Use the upstream reset to finalize tracking errors and success_rate,
        # log episode means, and clear only the selected environments' buffers.
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        count = self._step_count[ids].clamp_min(1.0)
        self.metrics["base_z"][ids] = self._base_z_sum[ids] / count
        self.metrics["knee_pos"][ids] = self._knee_pos_sum[ids] / count
        extras = super().reset(ids)
        self._base_z_sum[ids] = 0.0
        self._knee_pos_sum[ids] = 0.0
        return extras


@configclass
class StandVelocityCommandCfg(UniformVelocityCommandCfg):
    """The range fields lin_vel_x/lin_vel_y/ang_vel_z map to body Y/Z/X.

    Disable the inherited XY arrow visualization, whose axes do not match this
    task. The actor still receives three command slots in the original order.
    """

    class_type: type = StandVelocityCommand
    heading_command: bool = False
    debug_vis: bool = False
    element_names: list[str] = ["lin_vel_y", "lin_vel_z", "ang_vel_x"]
    rel_zero_vel_envs: float = 0.2
    rel_only_lin_y_envs: float = 0.02
    rel_only_lin_x_envs: float = 0.02
    rel_only_ang_z_envs: float = 0.2
