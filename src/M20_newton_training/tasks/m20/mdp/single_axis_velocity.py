"""Sample one enabled translation/turn axis without near-zero movement commands."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from .commands import UniformThresholdVelocityCommand, UniformThresholdVelocityCommandCfg


class SingleAxisVelocityCommand(UniformThresholdVelocityCommand):
    """Choose uniformly among enabled axes; (0, 0) disables an axis.

    All-zero ranges produce stationary commands. Enabled axes retain the common
    minimum magnitude and the existing command observations/metrics.
    """

    cfg: SingleAxisVelocityCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.min_abs_command) or cfg.min_abs_command <= 0:
            raise ValueError("min_abs_command must be finite and positive")
        if cfg.heading_command or cfg.rel_standing_envs != 0:
            raise ValueError("Single-axis sampling requires heading_command=False and rel_standing_envs=0")
        self._axis_intervals = []
        for bounds in (cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z):
            lo, hi = bounds
            if not all(math.isfinite(v) for v in bounds) or lo > hi:
                raise ValueError(f"Invalid velocity range: {bounds}")
            if lo == hi == 0.0:
                self._axis_intervals.append([])
                continue
            intervals = []
            if lo <= -cfg.min_abs_command:
                intervals.append((lo, min(hi, -cfg.min_abs_command)))
            if hi >= cfg.min_abs_command:
                intervals.append((max(lo, cfg.min_abs_command), hi))
            if not intervals:
                raise ValueError(f"Range {bounds} contains no command with magnitude >= {cfg.min_abs_command}")
            self._axis_intervals.append(intervals)
        self._enabled_axes = torch.tensor(
            [axis for axis, intervals in enumerate(self._axis_intervals) if intervals],
            dtype=torch.long,
            device=self.device,
        )

    def _sample_values(self, axis, count):
        intervals = self._axis_intervals[axis]
        lo, hi = intervals[0]
        sample = torch.rand(count, device=self.device)
        if len(intervals) == 1:
            return lo + sample * (hi - lo)
        other_lo, other_hi = intervals[1]
        widths = (hi - lo, other_hi - other_lo)
        total = sum(widths)
        probability = widths[0] / total if total > 0 else 0.5
        first = torch.rand(count, device=self.device) < probability
        return torch.where(first, lo + sample * widths[0], other_lo + sample * widths[1])

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # Do not run the parent's mixed sampling or its <=0.2 m/s zeroing rule.
        self.vel_command_b[env_ids] = 0.0
        self.is_standing_env[env_ids] = self._enabled_axes.numel() == 0
        self.is_heading_env[env_ids] = False
        if self._enabled_axes.numel() == 0:
            return
        axes = self._enabled_axes[torch.randint(self._enabled_axes.numel(), (len(env_ids),), device=self.device)]
        for axis in range(3):
            if not self._axis_intervals[axis]:
                continue
            selected = env_ids[axes == axis]
            self.vel_command_b[selected, axis] = self._sample_values(axis, len(selected))


@configclass
class SingleAxisVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    class_type: type = SingleAxisVelocityCommand
    min_abs_command: float = 0.16
    heading_command: bool = False
    rel_heading_envs: float = 0.0
    rel_standing_envs: float = 0.0
    rel_zero_vel_envs: float = 0.0
    rel_only_lin_x_envs: float = 0.0
    rel_only_lin_y_envs: float = 0.0
    rel_only_ang_z_envs: float = 0.0
