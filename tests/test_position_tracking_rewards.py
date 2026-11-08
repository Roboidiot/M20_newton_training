"""Integrated position tracking: frames, timing, resets, errors and native manager binding."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardManager, RewardTermCfg
from isaaclab.utils.math import quat_apply, yaw_quat

from M20_newton_training.tasks.m20.mdp.position_tracking_rewards import CommandPositionTrackingReward


def _env(command=(0.2, 0.0, 0.0), **params):
    n = 3
    position = torch.tensor([[0.0, 0.0, 0.8], [3.5, -3.5, 0.8], [-7.0, 7.0, 0.8]])
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1)
    command = torch.tensor(command).repeat(n, 1)
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        step_dt=0.02,
        common_step_counter=0,
        sim=SimpleNamespace(is_playing=lambda: True),
        position=position,
        quaternion=quaternion,
        command=command,
        command_manager=SimpleNamespace(get_command=lambda _: command),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=SimpleNamespace(torch=position),
            root_quat_w=SimpleNamespace(torch=quaternion),
        ))},
    )
    env.cfg = RewardTermCfg(func=CommandPositionTrackingReward, weight=5.0, params=params)
    env.reward = CommandPositionTrackingReward(env.cfg, env)
    return env


def _sample(env, world_velocity=None):
    if world_velocity is not None:
        env.position[:, :2] += torch.as_tensor(world_velocity) * env.step_dt
    env.common_step_counter += 1
    return env.reward(env, **env.cfg.params).clone()


@pytest.mark.parametrize("velocity,expected", [
    ((0.2, 0.0), 1.0), ((0.0, 0.0), 0.0), ((-0.2, 0.0), -1.0),
    ((0.4, 0.0), 0.0), ((0.6, 0.0), -1.0), ((0.2, 0.2), 0.0), ((-0.6, 0.0), -3.0),
])
def test_signed_position_error_distinguishes_tracking_standing_reverse_overshoot_and_lateral_error(velocity, expected):
    env = _env()
    assert not _sample(env).any()
    for _ in range(30):
        result = _sample(env, velocity)
    torch.testing.assert_close(result, torch.full((3,), expected), atol=1e-4, rtol=1e-4)


def test_current_body_commands_rotate_to_world_xy_without_using_height_roll_or_pitch():
    env = _env()
    angles = torch.tensor([0.0, math.pi / 4, math.pi / 2])
    # yaw composed with a 60-degree pitch: only yaw is used for the planar reference.
    env.quaternion[:, 0] = -torch.sin(angles / 2) * math.sin(math.pi / 6)
    env.quaternion[:, 1] = torch.cos(angles / 2) * math.sin(math.pi / 6)
    env.quaternion[:, 2] = torch.sin(angles / 2) * math.cos(math.pi / 6)
    env.quaternion[:, 3] = torch.cos(angles / 2) * math.cos(math.pi / 6)
    env.command[:] = torch.tensor([[0.2, 0.0, 0.0], [0.0, -0.2, 0.0], [-0.2, 0.0, 0.0]])
    world = quat_apply(yaw_quat(env.quaternion), env.command.clone())[:, :2]
    _sample(env)
    for _ in range(30):
        env.position[:, 2] += 0.1
        result = _sample(env, world)
    torch.testing.assert_close(result, torch.ones(3), atol=1e-4, rtol=1e-4)


def test_speed_fluctuations_can_finish_at_the_same_target_position_with_full_credit():
    env = _env()
    _sample(env)
    for i in range(100):
        result = _sample(env, (0.4 if i % 2 == 0 else 0.0, 0.0))
    # Last instantaneous speed is zero, but the integrated distance is exactly .4 m.
    torch.testing.assert_close(result, torch.ones(3), atol=1e-4, rtol=1e-4)


def test_warmup_distance_gate_and_repeated_calls_never_advance_time_twice():
    env = _env()
    _sample(env)
    for _ in range(9):
        assert not _sample(env, (0.2, 0.0)).any()
    result = _sample(env, (0.2, 0.0))
    torch.testing.assert_close(result, torch.ones(3), atol=1e-4, rtol=1e-4)
    elapsed = env.reward.elapsed.clone()
    displacement = env.reward.command_displacement_xy.clone()
    for _ in range(5):
        torch.testing.assert_close(env.reward(env, **env.cfg.params), result)
    torch.testing.assert_close(env.reward.elapsed, elapsed)
    torch.testing.assert_close(env.reward.command_displacement_xy, displacement)
    distant = _env(min_tracking_time=0.0, min_target_distance=0.1)
    _sample(distant)
    for _ in range(24):
        assert not _sample(distant, (0.2, 0.0)).any()
    torch.testing.assert_close(_sample(distant, (0.2, 0.0)), torch.ones(3), atol=1e-4, rtol=1e-4)


def test_window_scores_its_endpoint_before_reanchoring_and_does_not_keep_old_error():
    env = _env()
    _sample(env)
    for _ in range(100):
        result = _sample(env, (0.0, 0.0))
    torch.testing.assert_close(result, torch.zeros(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(env.reward.elapsed, torch.zeros(3))
    torch.testing.assert_close(env.reward.anchor_xy, env.position[:, :2])
    torch.testing.assert_close(env.reward.command_displacement_xy, torch.zeros(3, 2))
    for _ in range(10):
        result = _sample(env, (0.2, 0.0))
    torch.testing.assert_close(result, torch.ones(3), atol=1e-4, rtol=1e-4)


def test_partial_reset_does_not_teleport_the_reference_in_other_worlds():
    env = _env()
    _sample(env)
    for _ in range(20):
        _sample(env, (0.2, 0.0))
    env.position[1, :2] = torch.tensor([30.0, -30.0])
    env.reward.reset([1])
    result = _sample(env, (0.2, 0.0))
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 1.0]), atol=1e-4, rtol=1e-4)
    for _ in range(10):
        result = _sample(env, (0.2, 0.0))
    torch.testing.assert_close(result, torch.ones(3), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("command", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.3), (0.05, 0.0, 0.0), (-0.2, 0.0, 0.0)])
def test_idle_pure_yaw_small_or_reversed_commands_clear_previous_reference(command):
    env = _env()
    _sample(env)
    for _ in range(20):
        _sample(env, (0.2, 0.0))
    env.command[:] = torch.tensor(command)
    assert not _sample(env).any()
    assert not env.reward.elapsed.any()
    env.command[:] = torch.tensor([0.2, 0.0, 0.0])
    assert not _sample(env).any()
    for _ in range(10):
        result = _sample(env, (0.2, 0.0))
    torch.testing.assert_close(result, torch.ones(3), atol=1e-4, rtol=1e-4)


def test_command_magnitude_changes_integrate_into_the_existing_window():
    env = _env()
    _sample(env)
    for _ in range(20):
        _sample(env, (0.2, 0.0))
    env.command[:, 0] = 0.4
    for _ in range(20):
        result = _sample(env, (0.4, 0.0))
    torch.testing.assert_close(env.reward.elapsed, torch.full((3,), 0.8))
    torch.testing.assert_close(result, torch.ones(3), atol=1e-4, rtol=1e-4)


def test_native_reward_manager_applies_weight_and_control_dt_once():
    env = _env()
    manager = RewardManager({"track_position_xy": env.cfg}, env)
    manager.compute(env.step_dt)
    for _ in range(10):
        env.position[:, 0] += 0.2 * env.step_dt
        env.common_step_counter += 1
        result = manager.compute(env.step_dt)
    torch.testing.assert_close(result, torch.full((3,), 0.1), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", ["window_s", "min_target_distance", "min_command_norm"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_positive_parameters_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        _env(**{name: value})


@pytest.mark.parametrize("value", [-1.0, 3.0, float("nan"), float("inf")])
def test_invalid_warmup_is_rejected(value):
    with pytest.raises(ValueError, match="min_tracking_time"):
        _env(min_tracking_time=value)
