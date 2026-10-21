"""CPU regressions for Sand's play-only body-frame keyboard commands."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import to_dict

from M20_newton_training.tasks.m20.config.M20.sand_env_cfg import DeeproboticsM20SandEnvCfg
from M20_newton_training.tasks.m20.mdp.commands import UniformThresholdVelocityCommand
from M20_newton_training.tasks.m20.mdp.keyboard_velocity import (
    KeyboardVelocityCommand,
    KeyboardVelocityCommandCfg,
    read_keyboard_command,
)


def _viewer(keys=(), *, capturing=False, running=True):
    return SimpleNamespace(
        is_key_down=lambda key: key in keys,
        is_running=lambda: running,
        gui=SimpleNamespace(is_keyboard_capturing=lambda: capturing),
    )


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (("i",), (0.5, 0.0, 0.0)),
        (("k",), (-0.5, 0.0, 0.0)),
        (("j",), (0.0, 0.3, 0.0)),
        (("l",), (0.0, -0.3, 0.0)),
        (("u",), (0.0, 0.0, 0.5)),
        (("o",), (0.0, 0.0, -0.5)),
        (("i", "k", "j", "l", "u", "o"), (0.0, 0.0, 0.0)),
        (("i", "j", "backspace"), (0.0, 0.0, 0.0)),
        ((), (0.0, 0.0, 0.0)),
    ],
)
def test_body_axes_and_stop_controls(keys, expected):
    cfg = KeyboardVelocityCommandCfg()
    ranges = DeeproboticsM20SandEnvCfg().commands.base_velocity.ranges
    assert read_keyboard_command(_viewer(keys), ranges, cfg.keyboard_speeds) == expected
    assert read_keyboard_command(_viewer(keys, capturing=True), ranges, cfg.keyboard_speeds) == (0.0, 0.0, 0.0)


def test_play_mode_preserves_training_observations_and_sensor_configuration():
    cfg = DeeproboticsM20SandEnvCfg()
    observations = to_dict(cfg.observations)
    scanner = to_dict(cfg.scene.height_scanner)
    scanner_base = to_dict(cfg.scene.height_scanner_base)
    ranges = to_dict(cfg.commands.base_velocity.ranges)
    cfg.play_mode()
    command = cfg.commands.base_velocity
    assert isinstance(command, KeyboardVelocityCommandCfg)
    assert command.class_type is KeyboardVelocityCommand
    assert not command.heading_command
    assert command.resampling_time_range == (1.0e9, 1.0e9)
    assert cfg.scene.num_envs == 1
    assert to_dict(command.ranges) == ranges
    for group in ("policy", "critic"):
        observations[group]["enable_corruption"] = False
    assert to_dict(cfg.observations) == observations
    assert to_dict(cfg.scene.height_scanner) == scanner
    assert to_dict(cfg.scene.height_scanner_base) == scanner_base
    training = DeeproboticsM20SandEnvCfg().commands.base_velocity
    assert training.class_type is UniformThresholdVelocityCommand
    assert training.heading_command
    assert training.resampling_time_range == (10.0, 10.0)


def _term(viewer):
    term = KeyboardVelocityCommand.__new__(KeyboardVelocityCommand)
    cfg = DeeproboticsM20SandEnvCfg()
    cfg.play_mode()
    assert isinstance(cfg.commands.base_velocity, KeyboardVelocityCommandCfg)
    term.cfg = cfg.commands.base_velocity
    term._env = cast(
        ManagerBasedEnv,
        SimpleNamespace(device="cpu", num_envs=2, sim=SimpleNamespace(visualizers=[SimpleNamespace(_viewer=viewer)])),
    )
    term.vel_command_b = torch.zeros(2, 3)
    term.is_standing_env = torch.ones(2, dtype=torch.bool)
    term.is_heading_env = torch.ones(2, dtype=torch.bool)
    term._keyboard_viewer = None
    term._last_keyboard_command = None
    return term


def test_reset_does_not_resample_or_replace_manual_turn_with_heading_control(monkeypatch):
    def random_sampler_must_not_run(*args):
        raise AssertionError("Keyboard playback must not sample random commands.")

    monkeypatch.setattr(UniformThresholdVelocityCommand, "_resample_command", random_sampler_must_not_run)
    term = _term(_viewer(("i", "o")))
    term._update_command()
    expected = torch.tensor([[0.5, 0.0, -0.5], [0.5, 0.0, -0.5]])
    term._resample_command(torch.tensor([0, 1]))
    term._update_command()
    torch.testing.assert_close(term.command, expected)
    assert not term.is_standing_env.any()
    assert not term.is_heading_env.any()
    term._keyboard_viewer = _viewer()
    term._update_command()
    torch.testing.assert_close(term.command, torch.zeros(2, 3))


def test_keyboard_requires_an_open_newton_window():
    term = _term(None)
    with pytest.raises(RuntimeError, match="--visualizer newton_gl"):
        term._update_command()
    term = _term(_viewer(running=False))
    with pytest.raises(KeyboardInterrupt):
        term._update_command()
