# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-window keyboard velocity commands in the robot's body X/Y/Z convention."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass

from .commands import UniformThresholdVelocityCommand, UniformThresholdVelocityCommandCfg


def read_keyboard_command(viewer, ranges, speeds: tuple[float, float, float]) -> tuple[float, float, float]:
    """Read held keys as body-frame [linear X, linear Y, angular Z] commands."""
    gui = getattr(viewer, "gui", None)
    if (gui is not None and gui.is_keyboard_capturing()) or viewer.is_key_down("backspace"):
        return (0.0, 0.0, 0.0)
    pairs = (("i", "k"), ("j", "l"), ("u", "o"))
    limits = (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
    command = []
    for (positive, negative), (low, high), speed in zip(pairs, limits, speeds, strict=True):
        direction = int(viewer.is_key_down(positive)) - int(viewer.is_key_down(negative))
        command.append(min(max(direction * speed, low), high))
    return (command[0], command[1], command[2])


class KeyboardVelocityCommand(UniformThresholdVelocityCommand):
    """Drive planar locomotion from Newton keys without random or heading commands."""

    cfg: KeyboardVelocityCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._keyboard_viewer = None
        self._last_keyboard_command = None
        print("[Keyboard] Click the Newton window. I/K: vx, J/L: vy, U/O: wz; release to stop; Backspace: zero.")

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice):
        # Reset/timer callbacks must not invoke the parent's random command sampler.
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False

    def _update_command(self):
        if self._keyboard_viewer is None:
            for visualizer in self._env.sim.visualizers:
                viewer = getattr(visualizer, "_viewer", None)
                if viewer is not None and callable(getattr(viewer, "is_key_down", None)):
                    self._keyboard_viewer = viewer
                    break
            if self._keyboard_viewer is None:
                raise RuntimeError("Sand keyboard playback requires --visualizer newton_gl.")

        if not self._keyboard_viewer.is_running():
            raise KeyboardInterrupt
        command = read_keyboard_command(self._keyboard_viewer, self.cfg.ranges, self.cfg.keyboard_speeds)
        self.vel_command_b[:] = torch.tensor(command, dtype=self.vel_command_b.dtype, device=self.device)
        if command != self._last_keyboard_command:
            print(f"[Keyboard] vx={command[0]:+.2f} m/s, vy={command[1]:+.2f} m/s, wz={command[2]:+.2f} rad/s")
            self._last_keyboard_command = command


@configclass
class KeyboardVelocityCommandCfg(UniformThresholdVelocityCommandCfg):
    """Manual body X/Y/Z commands for playback, with configurable held-key speeds."""

    class_type: type = KeyboardVelocityCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    heading_command: bool = False
    rel_heading_envs: float = 0.0
    rel_standing_envs: float = 0.0
    keyboard_speeds: tuple[float, float, float] = (0.5, 0.3, 0.5)
    """Positive key speeds for [vx (m/s), vy (m/s), wz (rad/s)], clipped to command ranges."""
