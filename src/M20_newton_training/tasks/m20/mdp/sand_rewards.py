# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-free stepping rewards for Sand, without a terrain curriculum or policy clock.

Wheel-center lift relative to the other legs is a kinematic proxy, not a measurement of the deforming sand
surface or contact force. Pair coordination and completed lift/advance/lower motions encourage stepping
without adding phase observations to the pretrained actor.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _moving(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return (torch.linalg.vector_norm(command[:, :2], dim=1) > 0.1) | (command[:, 2].abs() > 0.1)


class _SandFootReward(ManagerTermBase):
    """Keep an environment-local neutral foot reference, refreshed after each reset."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cast(SceneEntityCfg, cfg.params["asset_cfg"])
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        if self.asset.data.body_pos_w.torch[:, self.asset_cfg.body_ids].shape[1] != 4:
            raise ValueError("Sand stepping rewards require four wheel links ordered FL, FR, HL, HR.")
        self._reference = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._reference_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._reference_valid[slice(None) if env_ids is None else env_ids] = False

    def _foot_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        position_w = self.asset.data.body_pos_w.torch[:, self.asset_cfg.body_ids]
        relative_w = position_w - self.asset.data.root_pos_w.torch[:, None, :]
        rotation = self.asset.data.root_quat_w.torch[:, None, :].expand(-1, 4, -1)
        position_b = quat_apply_inverse(rotation, relative_w)
        self._reference.copy_(torch.where(self._reference_valid[:, None, None], self._reference, position_b))
        self._reference_valid.fill_(True)
        displacement_z = position_b[:, :, 2] - self._reference[:, :, 2]
        # Common vertical motion cannot earn lift credit (e.g. body bobbing or lifting all four legs).
        lift = displacement_z - displacement_z.amin(dim=1, keepdim=True)
        return position_b, lift

    def _upright(self) -> torch.Tensor:
        return (-self.asset.data.projected_gravity_b.torch[:, 2]).clamp(0.0, 1.0)


class SandDiagonalStepReward(_SandFootReward):
    """Reward equal lift within diagonal pairs and separation between the pairs; no hidden clock."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str = "base_velocity",
        target_lift: float = 0.07,
        height_std: float = 0.035,
        sync_std: float = 0.025,
    ) -> torch.Tensor:
        _, lift = self._foot_state()
        pair_error = 0.5 * ((lift[:, 0] - lift[:, 3]).square() + (lift[:, 1] - lift[:, 2]).square())
        separation = (0.5 * (lift[:, 0] + lift[:, 3] - lift[:, 1] - lift[:, 2])).abs()
        height_score = (separation / target_lift).clamp(0.0, 1.0)
        height_score *= torch.exp(-((separation - target_lift) / height_std).square())
        return height_score * torch.exp(-pair_error / sync_std**2) * _moving(env, command_name) * self._upright()


class SandStepCompletionReward(_SandFootReward):
    """Credit a leg only after it lifts, advances in the body plane, and lowers again.

    Returns completed-step credit per second; RewardManager's dt factor integrates each event once.
    Height hysteresis and swing-duration limits reject tiny vibrations and indefinitely held legs.
    Completed diagonal pairs must alternate to receive repeated credit.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._swing = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._start_xy = torch.zeros(self.num_envs, 4, 2, device=self.device)
        self._peak_lift = torch.zeros(self.num_envs, 4, device=self.device)
        self._travel = torch.zeros_like(self._peak_lift)
        self._duration = torch.zeros_like(self._peak_lift)
        self._last_pair = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._swing[ids] = False
        self._start_xy[ids] = 0.0
        self._peak_lift[ids] = 0.0
        self._travel[ids] = 0.0
        self._duration[ids] = 0.0
        self._last_pair[ids] = -1

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
    ) -> torch.Tensor:
        position_b, lift = self._foot_state()
        moving = _moving(env, command_name)[:, None]
        entered = (~self._swing) & (lift > lift_threshold) & moving
        self._start_xy.copy_(torch.where(entered[:, :, None], position_b[:, :, :2], self._start_xy))
        active = (self._swing | entered) & moving
        self._duration.copy_(torch.where(active, self._duration + env.step_dt, 0.0))
        self._peak_lift.copy_(torch.where(active, torch.maximum(self._peak_lift, lift), 0.0))
        distance = torch.linalg.vector_norm(position_b[:, :, :2] - self._start_xy, dim=-1)
        self._travel.copy_(torch.where(active, torch.maximum(self._travel, distance), 0.0))

        lowered = self._swing & (lift < lower_threshold) & moving
        valid_time = (self._duration >= min_swing_time) & (self._duration <= max_swing_time)
        credit = (self._peak_lift / target_lift).clamp(0.0, 1.0) * (self._travel / min_stride).clamp(0.0, 1.0)
        completed = credit * lowered * valid_time
        pair_credit = torch.stack(
            (0.5 * (completed[:, 0] + completed[:, 3]), 0.5 * (completed[:, 1] + completed[:, 2])), dim=1
        )
        eligible = self._last_pair[:, None] != torch.arange(2, device=self.device)[None, :]
        eligible_credit = pair_credit * eligible
        best_credit, best_pair = eligible_credit.max(dim=1)
        self._last_pair.copy_(torch.where(best_credit > 0.0, best_pair, self._last_pair))
        self._last_pair.copy_(torch.where(moving[:, 0], self._last_pair, -1))
        reward = 0.5 * best_credit / env.step_dt
        self._swing.copy_(active & ~lowered & (self._duration <= max_swing_time))
        # Completed/expired strokes must not leak their credit into the next swing.
        self._peak_lift *= self._swing
        self._travel *= self._swing
        self._duration *= self._swing
        return reward * self._upright()


class SandSupportMotionPenalty(_SandFootReward):
    """Penalize horizontal wheel-center motion for kinematically low legs, without contact sensing."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        support_threshold: float = 0.01,
        velocity_reference: float = 0.5,
    ) -> torch.Tensor:
        _, lift = self._foot_state()
        velocity = self.asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
        error = (velocity / velocity_reference).square().sum(dim=-1).clamp(max=25.0)
        return (error * (lift < support_threshold)).mean(dim=1)


def sand_wheel_velocity_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, velocity_reference: float = 5.0
) -> torch.Tensor:
    """Discourage pure rolling with normalized actual wheel speeds, while retaining all 16 actions."""
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        (asset.data.joint_vel.torch[:, asset_cfg.joint_ids] / velocity_reference).square().clamp(max=25.0).mean(dim=1)
    )


def sand_stand_still(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Hold the default pose at rest; pure turning is movement too."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = (asset.data.joint_pos.torch - asset.data.default_joint_pos.torch)[:, asset_cfg.joint_ids].abs().sum(dim=1)
    return error * ~_moving(env, command_name)


class SandActionSmoothness(ManagerTermBase):
    """Second-order action penalty with per-environment reset and no curriculum scaling."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_previous = torch.zeros_like(env.action_manager.action)
        self._history_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._previous_previous[ids] = 0.0
        self._history_steps[ids] = 0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        manager = env.action_manager
        error = (manager.action - 2.0 * manager.prev_action + self._previous_previous).square().sum(dim=1)
        reward = error * (self._history_steps >= 2)
        self._previous_previous.copy_(manager.prev_action)
        self._history_steps += 1
        return reward
