"""Balanced crawl rewards shared by analytic and MPM sand, sampled at control rate.

Legacy helpers remain in proxy_contact_rewards.py. These replacements use the
same force provider interface, policy observations and action dimensions.
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .proxy_contact_rewards import (
    ProxyFootClearanceReward,
    _advance_participation_balance,
    _foot_participation_scale,
    _gait_command_gate,
    _log_foot_participation,
    _ordered_hipx_body_ids,
    _single_swing_patterns,
)


def _positive(cfg, name, default):
    value = cfg.params.get(name, default)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class CrawlForwardStepReward(ManagerTermBase):
    """Pay a supported forward swing only when foot placement and chassis advance.

    HipX-relative stroke is measured from LAST SUPPORT BEFORE LIFTOFF, not the
    previous touchdown. World placement compares consecutive landings. Chassis
    progress integrates signed root-position increments along the command's
    yaw-aligned world direction. Each foot keeps a nondecreasing progress mark,
    so moving back and returning to an old position cannot repeatedly earn credit.
    Raw credit is min(stroke, placement, new chassis progress)/target_stride.
    No sand-surface extraction is required. Repeat limits are foot-specific.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        for name, default in (
            ("target_stride", 0.12),
            ("min_air_time", 0.1),
            ("max_progress_lead", 0.5),
            ("min_command_norm", 0.1),
        ):
            _positive(cfg, name, default)
        self.hip_body_ids = _ordered_hipx_body_ids(env)
        shape = (env.num_envs, 4)
        self.last_support_relative = torch.zeros((*shape, 3), device=env.device)
        self.last_touchdown_world = torch.zeros_like(self.last_support_relative)
        self.support_seen = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.swing_qualified = torch.zeros_like(self.support_seen)
        self.previous_direction = torch.zeros((env.num_envs, 2), device=env.device)
        self.previous_world_direction = torch.zeros_like(self.previous_direction)
        self.previous_root = torch.zeros((env.num_envs, 3), device=env.device)
        self.position_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.body_progress = torch.zeros(env.num_envs, device=env.device)
        self.body_marks = torch.zeros(shape, device=env.device)
        self.completed_steps = torch.zeros(shape, dtype=torch.long, device=env.device)
        self.participation_progress = torch.zeros(shape, device=env.device)
        self.balance_progress = torch.zeros(shape, device=env.device)

    def reset(self, env_ids=None):
        _log_foot_participation(
            self._env,
            "forward_steps",
            self.completed_steps,
            self.participation_progress,
            env_ids,
        )
        self._reset_state(env_ids)

    def _reset_state(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in (
            "last_support_relative",
            "last_touchdown_world",
            "support_seen",
            "swing_qualified",
            "previous_direction",
            "previous_world_direction",
            "previous_root",
            "position_seen",
            "body_progress",
            "body_marks",
            "completed_steps",
            "participation_progress",
            "balance_progress",
        ):
            getattr(self, name)[ids] = 0

    def __call__(
        self,
        env,
        target_stride=0.12,
        min_air_time=0.1,
        max_progress_lead=0.5,
        action_term_name="sand_proxy",
        command_name="base_velocity",
        min_command_norm=0.1,
    ):
        command = env.command_manager.get_command(command_name)[:, :2]
        norm = command.norm(dim=1)
        direction = command / norm.clamp_min(1e-6)[:, None]
        gate = norm >= min_command_norm
        changed = (direction * self.previous_direction).sum(dim=1) < 0.95
        self._reset_state(~gate | changed)
        term = env.action_manager.get_term(action_term_name)
        state = term.contact_state  # Read before positions: MPM publishes its control window here.
        contact = state.vertical_contact
        point = term.application_points_w
        data = env.scene["robot"].data
        root = data.root_link_pos_w.torch
        rotation = yaw_quat(data.root_quat_w.torch)
        world_direction = quat_apply(rotation, torch.cat((direction, torch.zeros_like(norm[:, None])), dim=1))[:, :2]
        delta = ((root - self.previous_root)[:, :2] * self.previous_world_direction).sum(dim=1)
        self.body_progress.add_(torch.where(self.position_seen & gate, delta, 0.0))
        self.previous_root.copy_(root)
        self.previous_world_direction.copy_(world_direction)
        self.previous_direction.copy_(direction)
        self.position_seen.copy_(gate)
        relative = quat_apply_inverse(
            rotation[:, None].expand(-1, 4, -1), point - data.body_link_pos_w.torch[:, self.hip_body_ids]
        )
        mature = state.current_air_time + 1e-6 >= min_air_time
        self.swing_qualified |= _single_swing_patterns(contact) & mature & self.support_seen & gate[:, None]
        landed = state.touchdown[:, 0] & contact
        duration = (state.touchdown_air_time[:, 0] - env.step_dt).clamp_min(0.0)
        valid = landed & self.swing_qualified & (duration + 1e-6 >= min_air_time) & gate[:, None]
        stroke = ((relative - self.last_support_relative)[..., :2] * direction[:, None]).sum(dim=-1)
        placement = ((point - self.last_touchdown_world)[..., :2] * world_direction[:, None]).sum(dim=-1)
        body_advance = self.body_progress[:, None] - self.body_marks
        advance = torch.minimum(torch.minimum(stroke, placement), body_advance)
        score = torch.where(valid, (advance / target_stride).clamp(0.0, 1.0), 0.0)
        scale = _foot_participation_scale(self.balance_progress, 0.0, max_progress_lead)
        rewarded = score * scale
        result = rewarded.sum(dim=1)
        self.completed_steps.add_((score > 0).long())
        self.participation_progress.add_(score)
        _advance_participation_balance(self.balance_progress, score, max_progress_lead)
        update = contact & (~self.support_seen | landed) & gate[:, None]
        self.last_touchdown_world.copy_(torch.where(update[..., None], point, self.last_touchdown_world))
        self.body_marks.copy_(
            torch.where(update, torch.maximum(self.body_marks, self.body_progress[:, None]), self.body_marks)
        )
        supported = contact & gate[:, None]
        self.last_support_relative.copy_(torch.where(supported[..., None], relative, self.last_support_relative))
        self.support_seen |= supported
        self.swing_qualified.masked_fill_(contact, False)
        return result


class ProgressiveCrawlPhaseReward(ManagerTermBase):
    """Reward pairwise balance of REAL completed swings in a sliding window.

    p is each foot's share of valid completed swings in the last window_s.
    2/3 * sum_{i<j} sqrt(p_i*p_j) yields 0, 1/3, 2/3, 1 for equal participation
    of 1, 2, 3, 4 feet. A held unloaded pose cannot maintain the reward.
    A swing needs >=min_air_time unloading, three supports, and >=min_lift rise
    BOTH in world Z and relative to its own HipX; buried swings remain eligible.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        for name, default in (("window_s", 2.0), ("min_lift", 0.03), ("min_air_time", 0.1), ("min_command_norm", 0.1)):
            _positive(cfg, name, default)
        self.buffer_size = max(1, round(cfg.params.get("window_s", 2.0) / env.step_dt))
        self.pattern_buffer = torch.zeros((env.num_envs, 4, self.buffer_size), device=env.device)
        self.buffer_idx = 0
        self.hip_body_ids = _ordered_hipx_body_ids(env)
        shape = (env.num_envs, 4)
        self.support_relative = torch.zeros(shape, device=env.device)
        self.support_world = torch.zeros_like(self.support_relative)
        self.support_seen = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.peak_lift = torch.zeros_like(self.support_relative)
        self.previous_command = torch.zeros((env.num_envs, 3), device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in (
            "pattern_buffer",
            "support_relative",
            "support_world",
            "support_seen",
            "peak_lift",
            "previous_command",
        ):
            getattr(self, name)[ids] = 0

    def __call__(
        self,
        env,
        window_s=2.0,
        min_lift=0.03,
        min_air_time=0.1,
        action_term_name="sand_proxy",
        command_name="base_velocity",
        min_command_norm=0.1,
    ):
        command = env.command_manager.get_command(command_name)
        norm = command.norm(dim=1)
        direction = command / norm.clamp_min(1e-6)[:, None]
        gate = norm >= min_command_norm
        changed = (direction * self.previous_command).sum(dim=1) < 0.95
        self.reset(~gate | changed)
        self.previous_command.copy_(direction)
        term = env.action_manager.get_term(action_term_name)
        state = term.contact_state
        contact = state.vertical_contact
        world = term.application_points_w[..., 2]
        relative = world - env.scene["robot"].data.body_link_pos_w.torch[:, self.hip_body_ids, 2]
        valid = (
            _single_swing_patterns(contact)
            & self.support_seen
            & gate[:, None]
            & (state.current_air_time + 1e-6 >= min_air_time)
        )
        lift = torch.minimum(world - self.support_world, relative - self.support_relative).clamp_min(0.0)
        self.peak_lift.copy_(torch.where(valid, torch.maximum(self.peak_lift, lift), self.peak_lift))
        duration = (state.touchdown_air_time[:, 0] - env.step_dt).clamp_min(0.0)
        completed = (
            state.touchdown[:, 0]
            & contact
            & (self.peak_lift >= min_lift)
            & (duration + 1e-6 >= min_air_time)
            & gate[:, None]
        )
        self.pattern_buffer[:, :, self.buffer_idx] = completed
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.peak_lift.masked_fill_(contact, 0.0)
        supported = contact & gate[:, None]
        self.support_world.copy_(torch.where(supported, world, self.support_world))
        self.support_relative.copy_(torch.where(supported, relative, self.support_relative))
        self.support_seen |= supported
        counts = self.pattern_buffer.sum(dim=2)
        p = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        return sum(torch.sqrt(p[:, i] * p[:, j]) for i in range(4) for j in range(i + 1, 4)) * (2.0 / 3.0)


class BalancedFeetAirTimeReward(ManagerTermBase):
    """Retain the air-time formula and short-air costs, limit repeated single-foot bonuses."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        _positive(cfg, "max_progress_lead", 0.5)
        _positive(cfg, "participation_time", 0.3)
        self.progress = torch.zeros((env.num_envs, 4), device=env.device)

    def reset(self, env_ids=None):
        self.progress[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env,
        command_name,
        threshold=0.1,
        max_air_time=1.0,
        command_threshold=0.1,
        side_multiplier=2.0,
        turn_multiplier=3.0,
        action_term_name="sand_proxy",
        min_command_norm=0.1,
        max_progress_lead=0.5,
        participation_time=0.3,
    ):
        gate = _gait_command_gate(env, command_name, min_command_norm)
        self.reset(~gate)
        state = env.action_manager.get_term(action_term_name).contact_state
        credit = (state.touchdown_air_time[:, 0].clamp(max=max_air_time) - threshold) * state.touchdown[:, 0]
        positive = credit.clamp_min(0.0)
        scale = _foot_participation_scale(self.progress, 0.0, max_progress_lead)
        _advance_participation_balance(self.progress, positive / participation_time * gate[:, None], max_progress_lead)
        command = env.command_manager.get_command(command_name)
        command_scale = (
            1.0
            + (side_multiplier - 1.0) * (command[:, 1].abs() > command_threshold)
            + (turn_multiplier - 1.0) * (command[:, 2].abs() > command_threshold)
        )
        return (positive * scale + credit.clamp_max(0.0)).sum(dim=1) * command_scale * gate


class BalancedFootClearanceReward(ProxyFootClearanceReward):
    """Keep the extraction bonus, without an unlimited single-foot alternative."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.max_progress_lead = _positive(cfg, "max_progress_lead", 0.5)
        self.progress = torch.zeros((env.num_envs, 4), device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.progress[slice(None) if env_ids is None else env_ids] = 0.0

    def _score(self, clearance, forward, enough_stride, eligible, completed):
        score = super()._score(clearance, forward, enough_stride, eligible, completed)
        scale = _foot_participation_scale(self.progress, 0.0, self.max_progress_lead)
        _advance_participation_balance(self.progress, score, self.max_progress_lead)
        return score * scale

    def __call__(
        self,
        env,
        target_clearance=0.04,
        target_stride=0.08,
        min_stride=0.03,
        min_air_time=0.1,
        max_swing_time=1.0,
        surface_height=0.30,
        min_base_height=0.35,
        max_tilt=0.5,
        action_term_name="sand_proxy",
        command_name="base_velocity",
        min_command_norm=0.1,
        max_progress_lead=0.5,
    ):
        self.max_progress_lead = max_progress_lead
        return super().__call__(
            env,
            target_clearance=target_clearance,
            target_stride=target_stride,
            min_stride=min_stride,
            min_air_time=min_air_time,
            max_swing_time=max_swing_time,
            surface_height=surface_height,
            min_base_height=min_base_height,
            max_tilt=max_tilt,
            action_term_name=action_term_name,
            command_name=command_name,
            min_command_norm=min_command_norm,
        )
