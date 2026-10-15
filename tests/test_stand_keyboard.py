"""CPU checks for manual commands and the play-only configuration switch."""

from types import SimpleNamespace

import pytest
import torch

from M20_newton_training.tasks.m20.config.M20.stand_env_cfg import DeeproboticsM20StandEnvCfg
from M20_newton_training.tasks.m20.mdp.stand_commands import StandVelocityCommand
from M20_newton_training.tasks.m20.mdp.stand_keyboard import KeyboardStandVelocityCommand, read_keyboard_command


def _viewer(keys=(), *, capturing=False, running=True):
    return SimpleNamespace(
        is_key_down=lambda key: key in keys,
        is_running=lambda: running,
        gui=SimpleNamespace(is_keyboard_capturing=lambda: capturing),
    )


def test_keyboard_uses_stand_command_slots_and_key_release_stops():
    ranges = DeeproboticsM20StandEnvCfg().commands.base_velocity.ranges
    assert read_keyboard_command(_viewer(["i", "j", "u"]), ranges) == (0.5, 0.25, 0.5)
    assert read_keyboard_command(_viewer(["k", "l", "o"]), ranges) == (-0.5, -0.25, -0.5)
    assert read_keyboard_command(_viewer(["i", "k", "j", "l", "u", "o"]), ranges) == (0.0, 0.0, 0.0)
    assert read_keyboard_command(_viewer(), ranges) == (0.0, 0.0, 0.0)
    assert read_keyboard_command(_viewer(["i", "backspace"]), ranges) == (0.0, 0.0, 0.0)
    assert read_keyboard_command(_viewer(["i"], capturing=True), ranges) == (0.0, 0.0, 0.0)


def test_play_switch_preserves_training_configuration_and_observation_shape():
    cfg = DeeproboticsM20StandEnvCfg()
    before = cfg.observations.to_dict()
    assert cfg.commands.base_velocity.class_type is StandVelocityCommand
    cfg.play_mode()
    assert cfg.commands.base_velocity.class_type is KeyboardStandVelocityCommand
    assert cfg.commands.base_velocity.resampling_time_range == (1.0e9, 1.0e9)
    assert cfg.scene.num_envs == 1
    # Only the inherited noise settings change; all observation terms remain.
    after = cfg.observations.to_dict()
    for group in ("policy", "critic"):
        before[group]["enable_corruption"] = False
    assert before == after
    assert DeeproboticsM20StandEnvCfg().commands.base_velocity.class_type is StandVelocityCommand


def _term(viewer):
    term = KeyboardStandVelocityCommand.__new__(KeyboardStandVelocityCommand)
    term.cfg = DeeproboticsM20StandEnvCfg().commands.base_velocity
    term._env = SimpleNamespace(
        device="cpu", num_envs=2, sim=SimpleNamespace(visualizers=[SimpleNamespace(_viewer=viewer)])
    )
    term.vel_command_b = torch.zeros(2, 3)
    term.is_standing_env = torch.ones(2, dtype=torch.bool)
    term.is_heading_env = torch.ones(2, dtype=torch.bool)
    term._keyboard_viewer = None
    term._last_keyboard_command = None
    return term


def test_manual_command_survives_resampling_and_broadcasts_to_play_envs(monkeypatch):
    term = _term(_viewer(["i", "u"]))
    term._update_command()
    expected = torch.tensor([[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]])
    torch.testing.assert_close(term.command, expected)

    def random_sampler_must_not_run(*args):
        raise AssertionError("Keyboard playback must not generate random commands.")

    monkeypatch.setattr(StandVelocityCommand, "_resample_command", random_sampler_must_not_run)
    term._resample_command(torch.tensor([0]))
    torch.testing.assert_close(term.command, expected)
    assert not term.is_standing_env[0]
    assert not term.is_heading_env[0]
    term._keyboard_viewer = _viewer()
    term._update_command()
    torch.testing.assert_close(term.command, torch.zeros(2, 3))


def test_keyboard_requires_a_viewer_and_window_close_exits_playback():
    term = _term(None)
    with pytest.raises(RuntimeError, match="--visualizer newton"):
        term._update_command()
    term = _term(_viewer(running=False))
    with pytest.raises(KeyboardInterrupt):
        term._update_command()
