"""Body-frame v/cmd rewards, zero-command safety and isolated task bindings."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils import to_dict

from M20_newton_training.tasks.m20.config.M20.walk_env_cfg import DeeproboticsM20WalkEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.rewards import (
    command_direction_ang_vel_z,
    command_direction_lin_vel_xy,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
)


def _env(commands, velocities, turns=None):
    commands = torch.tensor(commands, dtype=torch.float32)
    n = len(commands)

    def wrap(value):
        return SimpleNamespace(torch=torch.as_tensor(value, dtype=torch.float32))

    angular = torch.zeros((n, 3))
    if turns is not None:
        angular[:, 2] = torch.tensor(turns)
    data = SimpleNamespace(
        root_lin_vel_b=wrap(velocities),
        root_ang_vel_b=wrap(angular),
    )
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: commands),
    )


def test_translation_uses_signed_velocity_to_command_ratios():
    env = _env(
        [[0.3, 0, 0], [-0.3, 0, 0], [0, 0.3, 0], [0, -0.3, 0], [0.5, 0, 0], [0, 0, 0.3], [0.3, 0, 0]],
        [[0.2, 0, 0], [-0.2, 0, 0], [0, -0.2, 0], [0, -0.2, 0], [0.2, 0, 0], [2, 3, 4], [0, 0, 0]],
    )
    torch.testing.assert_close(
        command_direction_lin_vel_xy(env, "base_velocity"), torch.tensor([2 / 3, 2 / 3, -2 / 3, 2 / 3, 0.4, 0.0, 0.0])
    )


def test_translation_averages_active_axes_and_ignores_zero_command_components():
    env = _env([[0.3, 0.4, 0], [0.3, 0.4, 0], [0.3, 0, 0]], [[0.3, 0.4, 0], [0.15, 0.2, 0], [0.3, 10.0, 0]])
    torch.testing.assert_close(command_direction_lin_vel_xy(env, "base_velocity"), torch.tensor([1.0, 0.5, 1.0]))


def test_translation_reads_body_xy_directly_without_pose_or_world_velocity():
    env = _env(
        [[0.3, 0, 0], [0, 0.3, 0], [0.3, 0, 0]],
        [[0.2, 0, -2], [0, -0.2, 2], [0, 0, -2]],
    )
    torch.testing.assert_close(command_direction_lin_vel_xy(env, "base_velocity"), torch.tensor([2 / 3, -2 / 3, 0.0]))


def test_turn_progress_has_both_signs_and_no_credit_for_zero_turn_command():
    env = _env(
        [[0, 0, 0.3], [0, 0, -0.3], [0, 0, 0.3], [0.3, 0, 0], [0, 0, 0.3]],
        [[0, 0, 0]] * 5,
        turns=[0.2, -0.2, -0.2, 2.0, 0.0],
    )
    torch.testing.assert_close(
        command_direction_ang_vel_z(env, "base_velocity"), torch.tensor([2 / 3, 2 / 3, -2 / 3, 0.0, 0.0])
    )


def test_unclipped_ratio_reward_allows_speed_fluctuation_for_a_fixed_command():
    steady = _env([[0.3, 0, 0]] * 4, [[0.2, 0, 0]] * 4)
    varying = _env([[0.3, 0, 0]] * 4, [[0.4, 0, 0], [0, 0, 0], [0.4, 0, 0], [0, 0, 0]])
    a = (5 * 0.02 * command_direction_lin_vel_xy(steady, "base_velocity")).sum()
    b = (5 * 0.02 * command_direction_lin_vel_xy(varying, "base_velocity")).sum()
    torch.testing.assert_close(a, b)
    assert a.item() == pytest.approx(0.08 / 0.3)
    cancellation = _env([[0.3, 0, 0]] * 3, [[0.2, 0, 0], [0.2, 0, 0], [-0.4, 0, 0]])
    assert command_direction_lin_vel_xy(cancellation, "base_velocity").sum().item() == pytest.approx(0.0)
    unbounded = _env([[0.3, 0, 0]], [[-2.0, 0, 0]])
    assert command_direction_lin_vel_xy(unbounded, "base_velocity").item() == pytest.approx(-2.0 / 0.3)


def test_exact_matching_positive_and_negative_commands_score_one():
    commands = [[0.25, 0, 0], [-0.4, 0, 0], [0, 0.3, 0], [0, -0.25, 0], [0, 0, 0.3], [0, 0, -0.4]]
    env = _env(commands, [row[:2] + [0.0] for row in commands], turns=[row[2] for row in commands])
    torch.testing.assert_close(
        command_direction_lin_vel_xy(env, "base_velocity"), torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(
        command_direction_ang_vel_z(env, "base_velocity"), torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    )


def test_zero_commands_never_divide_by_zero_even_with_nonzero_body_motion():
    env = _env([[0.0, 0.0, 0.0], [1e-8, -1e-8, 1e-8]], [[2.0, -3.0, 1.0], [2.0, -3.0, 1.0]], turns=[2.0, -2.0])
    torch.testing.assert_close(command_direction_lin_vel_xy(env, "base_velocity"), torch.zeros(2))
    torch.testing.assert_close(command_direction_ang_vel_z(env, "base_velocity"), torch.zeros(2))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("command", [0.3, -0.3])
def test_active_linear_reward_decreases_after_peak_and_preserves_negative_tail(axis, command):
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg().rewards.track_lin_vel_xy_exp
    ratios = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, -10.0]
    commands = [[0.0, 0.0, 0.0] for _ in range(len(ratios) + 1)]
    velocities = [[0.0, 0.0, 0.0] for _ in commands]
    for row, ratio in enumerate(ratios):
        commands[row][axis] = command
        velocities[row][axis] = command * ratio
    velocities[-1][axis] = 2.0  # Zero command remains zero reward despite motion.
    env = _env(commands, velocities)
    values = cfg.func(env, **cfg.params)
    torch.testing.assert_close(values, torch.tensor([0.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -10.0, 0.0]))
    assert (values[1:7].diff() < 0).all()
    assert (values * cfg.weight * 0.02).max().item() == pytest.approx(0.10)


@pytest.mark.parametrize("command", [0.3, -0.3])
def test_active_turn_reward_matches_linear_shaping_and_preserves_negative_tail(command):
    rewards = DeeproboticsM20WalkOnSandProxyEnvCfg().rewards
    cfg = rewards.track_ang_vel_z_exp
    ratios = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, -10.0]
    commands = [[command, 0.0, command] for _ in ratios] + [[0.0, 0.0, 0.0]]
    turns = [command * ratio for ratio in ratios] + [2.0]
    env = _env(commands, [[velocity, 0.0, 0.0] for velocity in turns], turns=turns)
    values = cfg.func(env, **cfg.params)
    torch.testing.assert_close(values, torch.tensor([0.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -10.0, 0.0]))
    linear = rewards.track_lin_vel_xy_exp
    torch.testing.assert_close(values, linear.func(env, **linear.params))
    assert (values[1:7].diff() < 0).all()
    assert (values * cfg.weight * 0.02).max().item() == pytest.approx(0.04)


def test_only_proxy_bindings_change_and_original_exponential_helpers_remain_usable():
    before = to_dict(DeeproboticsM20WalkEnvCfg())
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg()
    assert cfg.rewards.track_lin_vel_xy_exp.func is command_direction_lin_vel_xy
    assert cfg.rewards.track_ang_vel_z_exp.func is command_direction_ang_vel_z
    assert cfg.rewards.track_lin_vel_xy_exp.params == {"command_name": "base_velocity", "max_reward": 1.0}
    assert cfg.rewards.track_ang_vel_z_exp.params == {"command_name": "base_velocity", "max_reward": 1.0}
    assert to_dict(DeeproboticsM20WalkEnvCfg()) == before
    env = _env([[0.3, 0, 0.2]], [[0.3, 0, 0]], turns=[0.2])
    torch.testing.assert_close(track_lin_vel_xy_exp(env, 0.3, "base_velocity"), torch.ones(1))
    torch.testing.assert_close(track_ang_vel_z_exp(env, 0.3, "base_velocity"), torch.ones(1))
