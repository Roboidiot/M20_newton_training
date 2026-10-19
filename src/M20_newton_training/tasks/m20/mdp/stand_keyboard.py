# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-window keyboard commands for upright M20 policy playback."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .stand_commands import StandVelocityCommand


def read_keyboard_command(viewer, ranges) -> tuple[float, float, float]:
    """Read held keys as command slots [body linear Y, linear Z, angular X]."""
    gui = getattr(viewer, "gui", None)
    if (gui is not None and gui.is_keyboard_capturing()) or viewer.is_key_down("backspace"):
        return (0.0, 0.0, 0.0)
    pairs = (("j", "l"), ("i", "k"), ("u", "o"))
    limits = (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
    command = []
    for (positive, negative), (low, high) in zip(pairs, limits, strict=True):
        direction = int(viewer.is_key_down(positive)) - int(viewer.is_key_down(negative))
        speed = 0.75 * max(abs(low), abs(high))
        command.append(min(max(direction * speed, low), high))
    return tuple(command)


class KeyboardStandVelocityCommand(StandVelocityCommand):
    """Poll Newton keys each control step, without randomly sampling commands.

    I/K: +/-vx (body Y); J/L: +/-vy (body Z); U/O: +/-wz (body X).
    Releasing keys stops that axis. Backspace overrides all held keys to zero.
    WASD and arrows remain available to the viewer's camera controls.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._keyboard_viewer = None
        self._last_keyboard_command = None
        print("[Keyboard] Click the Newton window. I/K: vx, J/L: vy, U/O: wz; release to stop; Backspace: zero.")

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice):
        # Called on episode resets and by the command timer. Preserve manual
        # commands instead of invoking the parent's random sampler.
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False

    def _update_command(self):
        if self._keyboard_viewer is None:
            # SimulationContext exposes active visualizers. Newton keeps its
            # public key-polling API on the backend viewer in _viewer.
            for visualizer in self._env.sim.visualizers:
                viewer = getattr(visualizer, "_viewer", None)
                if viewer is not None and callable(getattr(viewer, "is_key_down", None)):
                    self._keyboard_viewer = viewer
                    break
            if self._keyboard_viewer is None:
                raise RuntimeError("Stand keyboard playback requires --visualizer newton.")

        if not self._keyboard_viewer.is_running():
            # The upstream playback loop catches this and closes the environment.
            raise KeyboardInterrupt
        command = read_keyboard_command(self._keyboard_viewer, self.cfg.ranges)
        self.vel_command_b[:] = torch.tensor(command, dtype=self.vel_command_b.dtype, device=self.device)
        if command != self._last_keyboard_command:
            print(f"[Keyboard] vx={command[0]:+.2f} m/s, vy={command[1]:+.2f} m/s, wz={command[2]:+.2f} rad/s")
            self._last_keyboard_command = command
