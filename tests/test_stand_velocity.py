"""CPU regressions for the upright robot's velocity axes and episode metrics."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv

from M20_newton_training.tasks.m20 import mdp
from M20_newton_training.tasks.m20.config.M20.flat_env_cfg import DeeproboticsM20FlatEnvCfg
from M20_newton_training.tasks.m20.config.M20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from M20_newton_training.tasks.m20.config.M20.stand_env_cfg import (
    DeeproboticsM20StandEnvCfg,
    stand_ang_vel_yz_l2,
    stand_lin_vel_x_l2,
    stand_track_ang_vel_x_exp,
    stand_track_lin_vel_yz_exp,
)
from M20_newton_training.tasks.m20.mdp.stand_commands import StandVelocityCommand, StandVelocityCommandCfg


def _env(linear_velocity, angular_velocity, commands):
    """Provide Isaac Lab's explicit torch views without starting a simulator."""
    linear_velocity = torch.tensor(linear_velocity, dtype=torch.float32)
    num_envs = len(linear_velocity)
    data = SimpleNamespace(
        root_lin_vel_b=SimpleNamespace(torch=linear_velocity),
        root_ang_vel_b=SimpleNamespace(torch=torch.tensor(angular_velocity, dtype=torch.float32)),
        root_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.8028]]).repeat(num_envs, 1)),
        joint_pos=SimpleNamespace(torch=torch.ones(num_envs, 1)),
        default_joint_pos=SimpleNamespace(torch=torch.zeros(num_envs, 1)),
    )
    asset = SimpleNamespace(data=data, find_joints=lambda pattern: ([0], ["fl_knee_joint"]))
    commands = torch.tensor(commands, dtype=torch.float32)
    return cast(
        ManagerBasedRLEnv,
        SimpleNamespace(
            num_envs=num_envs,
            device="cpu",
            scene={"robot": asset},
            command_manager=SimpleNamespace(get_command=lambda name: commands),
            sim=SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None)),
            extras={"log": {}},
        ),
    )


def test_nonzero_commands_track_yz_translation_and_x_rotation():
    # Vertical speed and tipping rates must not change planar/turn tracking.
    env = _env([[7.0, 0.4, -0.2]], [[0.3, 8.0, 9.0]], [[0.4, -0.2, 0.3]])
    torch.testing.assert_close(stand_track_lin_vel_yz_exp(env, 1.0, "base_velocity"), torch.ones(1))
    torch.testing.assert_close(stand_track_ang_vel_x_exp(env, 1.0, "base_velocity"), torch.ones(1))


def test_zero_command_tracks_horizontal_drift_and_turning():
    env = _env([[0.0, 3.0, 4.0]], [[2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    assert stand_track_lin_vel_yz_exp(env, 5.0, "base_velocity").item() == pytest.approx(0.36787944)
    assert stand_track_ang_vel_x_exp(env, 2.0, "base_velocity").item() == pytest.approx(0.36787944)


def test_vertical_and_tipping_penalties_use_complementary_axes():
    env = _env([[2.0, 3.0, 4.0]], [[7.0, 3.0, 4.0]], [[0.0, 0.0, 0.0]])
    assert stand_lin_vel_x_l2(env).item() == pytest.approx(4.0)
    assert stand_ang_vel_yz_l2(env).item() == pytest.approx(25.0)
    env = _env([[0.0, 3.0, 4.0]], [[7.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    assert stand_lin_vel_x_l2(env).item() == 0.0
    assert stand_ang_vel_yz_l2(env).item() == 0.0


def test_metrics_are_episode_means_and_partial_reset_preserves_other_envs():
    env = _env([[7.0, 0.4, -0.2], [0.0, 3.0, 4.0]], [[0.3, 8.0, 9.0], [0.2, 0.0, 0.0]], [[0.0] * 3] * 2)
    cfg = DeeproboticsM20StandEnvCfg().commands.base_velocity
    assert isinstance(cfg, StandVelocityCommandCfg)
    term = StandVelocityCommand(cfg, env)
    term.vel_command_b[:] = torch.tensor([[0.4, -0.2, 0.3], [0.0, 0.0, 0.0]])
    for _ in range(2):
        term._update_metrics()

    extras = term.reset(torch.tensor([0]))
    assert extras["error_vel_xy"] == 0.0
    assert extras["error_vel_yaw"] == 0.0
    assert extras["base_z"] == pytest.approx(0.8028)
    assert extras["knee_pos"] == pytest.approx(1.0)
    assert env.extras["log"]["Metrics/success_rate"] == 1.0
    torch.testing.assert_close(term._step_count, torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(term._error_xy_sum, torch.tensor([0.0, 10.0]))

    extras = term.reset(torch.tensor([1]))
    assert extras["error_vel_xy"] == pytest.approx(5.0)
    assert extras["error_vel_yaw"] == pytest.approx(0.2)
    assert extras["knee_pos"] == pytest.approx(1.0)
    assert env.extras["log"]["Metrics/success_rate"] == 0.0
    for buffer in (term._step_count, term._error_xy_sum, term._error_yaw_sum, term._base_z_sum, term._knee_pos_sum):
        torch.testing.assert_close(buffer, torch.zeros(2))


def test_reset_all_and_next_episode_do_not_reuse_error_history():
    env = _env([[0.0, 3.0, 4.0]], [[0.2, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
    cfg = DeeproboticsM20StandEnvCfg().commands.base_velocity
    assert isinstance(cfg, StandVelocityCommandCfg)
    term = StandVelocityCommand(cfg, env)
    term._update_metrics()
    assert term.reset()["error_vel_xy"] == pytest.approx(5.0)
    env.scene["robot"].data.root_lin_vel_b.torch[:] = 0.0
    env.scene["robot"].data.root_ang_vel_b.torch[:] = 0.0
    term.vel_command_b[:] = 0.0
    term._update_metrics()
    extras = term.reset()
    assert extras["error_vel_xy"] == 0.0
    assert extras["error_vel_yaw"] == 0.0
    assert extras["knee_pos"] == pytest.approx(5.0)
    assert env.extras["log"]["Metrics/success_rate"] == 1.0


def test_stand_configuration_selects_correct_functions_and_reduced_commands():
    cfg = DeeproboticsM20StandEnvCfg()
    assert isinstance(cfg.commands.base_velocity, StandVelocityCommandCfg)
    assert cfg.commands.base_velocity.class_type is StandVelocityCommand
    assert cfg.commands.base_velocity.element_names == ["lin_vel_y", "lin_vel_z", "ang_vel_x"]
    assert cfg.commands.base_velocity.ranges.lin_vel_x == (-1.0, 1.0)
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (-0.5, 0.5)
    assert cfg.commands.base_velocity.ranges.ang_vel_z == (-1.0, 1.0)
    assert not cfg.commands.base_velocity.heading_command
    assert cfg.rewards.track_lin_vel_xy_exp.func is stand_track_lin_vel_yz_exp
    assert cfg.rewards.track_ang_vel_z_exp.func is stand_track_ang_vel_x_exp
    assert cfg.rewards.lin_vel_z_l2.func is stand_lin_vel_x_l2
    assert cfg.rewards.ang_vel_xy_l2.func is stand_ang_vel_yz_l2


@pytest.mark.parametrize("env_ids", [slice(None), list(range(5)), torch.arange(5)])
def test_special_samples_include_stopping_translation_and_direct_turning(monkeypatch, env_ids):
    env = _env([[0.0] * 3] * 5, [[0.0] * 3] * 5, [[0.0] * 3] * 5)
    cfg = DeeproboticsM20StandEnvCfg().commands.base_velocity
    assert isinstance(cfg, StandVelocityCommandCfg)
    cfg.ranges.lin_vel_x = (0.05, 0.05)
    cfg.ranges.lin_vel_y = (0.04, 0.04)
    cfg.ranges.ang_vel_z = (0.3, 0.3)
    term = StandVelocityCommand(cfg, env)
    # One sample in each disjoint interval. Small single-axis commands must
    # survive the mixed-command threshold, and pure turning must not use heading.
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.1, 0.21, 0.23, 0.3, 0.9]))
    term._resample_command(env_ids)
    term._update_command()
    expected = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.04, 0.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.3]])
    torch.testing.assert_close(term.command, expected)
    assert term.is_standing_env.tolist() == [True, False, False, False, False]
    assert not term.is_heading_env.any()


@pytest.mark.parametrize("config_type", [DeeproboticsM20FlatEnvCfg, DeeproboticsM20RoughEnvCfg])
def test_locomotion_tasks_keep_original_command_and_reward_axes(config_type):
    cfg = config_type()
    assert cfg.commands.base_velocity.class_type is mdp.UniformThresholdVelocityCommand
    assert cfg.rewards.track_lin_vel_xy_exp.func is mdp.track_lin_vel_xy_exp
    assert cfg.rewards.track_ang_vel_z_exp.func is mdp.track_ang_vel_z_exp
    assert cfg.rewards.lin_vel_z_l2.func is mdp.lin_vel_z_l2
    assert cfg.rewards.ang_vel_xy_l2.func is mdp.ang_vel_xy_l2


@pytest.mark.parametrize("command_type", [StandVelocityCommand, mdp.UniformThresholdVelocityCommand])
@pytest.mark.parametrize("env_ids", [None, slice(None), slice(1, 4, 2), [1, 3], torch.tensor([1, 3])])
def test_command_reset_accepts_full_and_partial_environment_selectors(command_type, env_ids):
    env = _env([[0.0, 3.0, 4.0]] * 4, [[0.2, 0.0, 0.0]] * 4, [[0.0] * 3] * 4)
    cfg_type = DeeproboticsM20StandEnvCfg if command_type is StandVelocityCommand else DeeproboticsM20RoughEnvCfg
    cfg = cfg_type().commands.base_velocity
    cfg.debug_vis = False
    if command_type is StandVelocityCommand:
        assert isinstance(cfg, StandVelocityCommandCfg)
        term = StandVelocityCommand(cfg, env)
    else:
        assert isinstance(cfg, mdp.UniformThresholdVelocityCommandCfg)
        term = mdp.UniformThresholdVelocityCommand(cfg, env)
    term._update_metrics()
    term._update_metrics()
    error_before = term._error_xy_sum.clone()

    extras = term.reset(env_ids)

    selected = slice(None) if env_ids is None else env_ids
    expected_steps = torch.full((4,), 2.0)
    expected_steps[selected] = 0.0
    expected_error = error_before.clone()
    expected_error[selected] = 0.0
    torch.testing.assert_close(term._step_count, expected_steps)
    torch.testing.assert_close(term._error_xy_sum, expected_error)
    assert extras["error_vel_xy"] == pytest.approx(error_before[selected].mean().item() / 2.0)
    assert extras["base_z"] == pytest.approx(0.8028)
    torch.testing.assert_close(term.command_counter, (expected_steps == 0).long())
    assert "Metrics/success_rate" in env.extras["log"]
    # A second reset before any new steps must not reuse the previous episode.
    assert term.reset(env_ids)["error_vel_xy"] == 0.0


@pytest.mark.parametrize("env_ids", [slice(None), slice(1, 4, 2), slice(0, 0), [1, 3], torch.tensor([1, 3])])
def test_discrete_command_sampling_respects_environment_selector(env_ids):
    term = mdp.DiscreteCommandController.__new__(mdp.DiscreteCommandController)
    term._env = cast(ManagerBasedRLEnv, SimpleNamespace(num_envs=4, device="cpu"))
    term.available_commands = [10, 20]
    term.command_buffer = torch.full((4,), -1, dtype=torch.int32)
    term._resample_command(env_ids)
    expected_selected = torch.zeros(4, dtype=torch.bool)
    expected_selected[env_ids] = True
    torch.testing.assert_close(term.command_buffer != -1, expected_selected)
    assert torch.isin(term.command_buffer[env_ids], torch.tensor([10, 20])).all()


@pytest.mark.parametrize("env_ids", [None, slice(None), slice(1, 4, 2), slice(0, 0), torch.tensor([1, 3])])
def test_rotation_reward_reset_respects_environment_selector(env_ids):
    term = mdp.RotationGaitSymmetry.__new__(mdp.RotationGaitSymmetry)
    term.contact_buffer = torch.ones(4, 2, 3)
    term.reset(env_ids)
    expected = torch.ones(4, 2, 3)
    expected[slice(None) if env_ids is None else env_ids] = 0.0
    torch.testing.assert_close(term.contact_buffer, expected)
