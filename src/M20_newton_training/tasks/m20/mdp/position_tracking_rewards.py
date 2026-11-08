"""Control-rate position tracking from integrated body-frame velocity commands."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, yaw_quat


class CommandPositionTrackingReward(ManagerTermBase):
    """Compare actual world-XY displacement with the integrated command over a window.

    At each control step, rotate the current planar command into world XY using
    chassis yaw, then integrate it into a target position. For window anchor p0,
    target p*, and actual position p, return 1 - ||p-p*|| / ||p*-p0||. Exact
    tracking scores 1, stationary feet/body with a straight command score 0,
    overshoot/lateral error reduce credit, and negative scores are not clamped.

    The first sample establishes the anchor. Minimum tracking time and target
    displacement suppress tiny-denominator transients. Score the last sample of
    each window before re-anchoring at the actual position, so old errors do not
    accumulate across the full episode. Idle/pure-yaw commands and changes of
    planar command direction clear the window. Magnitude changes integrate
    normally. Repeated evaluation in one common_step_counter cannot advance it.

    This term only changes the reward: the target/error is not a new observation.
    RewardManager applies the configured weight and env.step_dt as usual.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        for name, default in (("window_s", 2.0), ("min_target_distance", 0.03), ("min_command_norm", 0.1)):
            value = cfg.params.get(name, default)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        warmup = cfg.params.get("min_tracking_time", 0.2)
        if not math.isfinite(warmup) or not 0 <= warmup <= cfg.params.get("window_s", 2.0):
            raise ValueError("min_tracking_time must be finite and between zero and window_s")
        self.anchor_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.command_displacement_xy = torch.zeros_like(self.anchor_xy)
        self.previous_direction = torch.zeros_like(self.anchor_xy)
        self.elapsed_samples = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.cached_reward = torch.zeros(env.num_envs, device=env.device)

    @property
    def elapsed(self):
        """Derive time from integer samples, avoiding drift at window boundaries."""
        return self.elapsed_samples * self._env.step_dt

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for buffer in (
            self.anchor_xy, self.command_displacement_xy, self.previous_direction,
            self.elapsed_samples, self.initialized, self.cached_reward,
        ):
            buffer[ids] = 0
        self.last_step[ids] = -1

    def __call__(
        self,
        env,
        command_name="base_velocity",
        window_s=2.0,
        min_tracking_time=0.2,
        min_target_distance=0.03,
        min_command_norm=0.1,
        asset_cfg: SceneEntityCfg | None = None,
    ):
        data = env.scene["robot" if asset_cfg is None else asset_cfg.name].data
        position = data.root_link_pos_w.torch[:, :2]
        command = env.command_manager.get_command(command_name)[:, :2]
        magnitude = command.norm(dim=1)
        direction = command / magnitude.clamp_min(1e-6)[:, None]
        active = magnitude >= min_command_norm
        changed = (direction * self.previous_direction).sum(dim=1) < 0.95
        new_sample = self.last_step != env.common_step_counter
        begin = new_sample & (~self.initialized | changed | ~active)
        advance = new_sample & self.initialized & ~changed & active
        command_w = quat_apply(
            yaw_quat(data.root_quat_w.torch), torch.cat((command, torch.zeros_like(command[:, :1])), dim=1)
        )[:, :2]
        desired_displacement = self.command_displacement_xy + command_w * env.step_dt
        samples = self.elapsed_samples + 1
        elapsed = samples * env.step_dt
        distance = desired_displacement.norm(dim=1)
        error = (position - self.anchor_xy - desired_displacement).norm(dim=1)
        qualified = advance & (elapsed + 1e-6 >= min_tracking_time) & (distance + 1e-6 >= min_target_distance)
        reward = torch.where(qualified, 1.0 - error / distance.clamp_min(min_target_distance), 0.0)
        self.cached_reward.copy_(torch.where(new_sample, reward, self.cached_reward))
        # Finish this window's score before starting the next window at the actual position.
        rollover = advance & (elapsed + 1e-6 >= window_s)
        anchor = begin | rollover
        self.anchor_xy.copy_(torch.where(anchor[:, None], position, self.anchor_xy))
        self.command_displacement_xy.copy_(
            torch.where(
                anchor[:, None], 0.0,
                torch.where(advance[:, None], desired_displacement, self.command_displacement_xy),
            )
        )
        self.elapsed_samples.copy_(torch.where(anchor, 0, torch.where(advance, samples, self.elapsed_samples)))
        self.previous_direction.copy_(torch.where(new_sample[:, None], direction, self.previous_direction))
        self.initialized.copy_(torch.where(new_sample, active, self.initialized))
        self.last_step.fill_(env.common_step_counter)
        return self.cached_reward
