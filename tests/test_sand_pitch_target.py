"""Forward-command pitch target sign, yaw invariance and task isolation."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz

from M20_newton_training.tasks.m20.config.M20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_env_cfg import DeeproboticsM20WalkEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import forward_pitch_orientation_l2, no_turn_ang_vel_z_l2
from M20_newton_training.tasks.m20.mdp.rewards import flat_orientation_l2


def _env(commands, pitches, rolls=None, yaws=None):
    commands = torch.tensor(commands, dtype=torch.float32)
    n = len(commands)
    pitch = torch.deg2rad(torch.tensor(pitches, dtype=torch.float32))
    roll = torch.deg2rad(torch.tensor([0.] * n if rolls is None else rolls))
    yaw = torch.deg2rad(torch.tensor([0.] * n if yaws is None else yaws))
    q = quat_from_euler_xyz(roll, pitch, yaw)
    gravity = quat_apply_inverse(q, torch.tensor([0., 0., -1.]).repeat(n, 1))
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=SimpleNamespace(torch=gravity)))},
        command_manager=SimpleNamespace(get_command=lambda name: commands),
        rotation=q,
    )


def test_negative_pitch_raises_body_forward_axis_and_has_zero_target_error():
    env = _env([[.2, 0., 0.]] * 3, [-5., 0., 5.])
    forward_w = quat_apply(env.rotation, torch.tensor([1., 0., 0.]).repeat(3, 1))
    assert forward_w[0, 2] > 0 and forward_w[2, 2] < 0
    reward = forward_pitch_orientation_l2(env, "base_velocity")
    torch.testing.assert_close(reward[0], torch.tensor(0.), atol=1e-12, rtol=0)
    assert reward[2] > reward[1] > reward[0]


@pytest.mark.parametrize("yaw", [0., 45., 90., -150.])
def test_target_pitch_is_independent_of_reset_heading_and_keeps_roll_level(yaw):
    env = _env([[.2, 0., 0.]] * 2, [-5., -5.], rolls=[0., 10.], yaws=[yaw, yaw])
    reward = forward_pitch_orientation_l2(env, "base_velocity")
    assert reward[0].item() == pytest.approx(0., abs=1e-12)
    assert reward[1] > 0


@pytest.mark.parametrize("command", [[0., 0., 0.], [-.2, 0., 0.], [0., .2, 0.], [0., 0., .3], [.1, 0., 0.]])
def test_stop_backward_side_and_turn_commands_keep_the_level_target(command):
    env = _env([command, command], [0., -5.])
    reward = forward_pitch_orientation_l2(env, "base_velocity")
    torch.testing.assert_close(reward, flat_orientation_l2(env))
    assert reward[0] == 0 and reward[1] > 0


def test_pitch_target_parameter_can_be_adjusted_or_disabled():
    env = _env([[.2, 0., 0.]] * 2, [-8., 0.])
    reward = forward_pitch_orientation_l2(env, "base_velocity", forward_pitch_deg=-8.)
    assert reward[0].item() == pytest.approx(0., abs=1e-12)
    torch.testing.assert_close(
        forward_pitch_orientation_l2(env, "base_velocity", forward_pitch_deg=0.), flat_orientation_l2(env)
    )


def test_shared_sand_binding_keeps_other_tasks_level_and_reward_weight_unchanged():
    for cls in (DeeproboticsM20WalkOnSandProxyEnvCfg, DeeproboticsM20WalkOnSandEnvCfg):
        term = cls().rewards.flat_orientation_l2
        assert term.func is forward_pitch_orientation_l2 and term.weight == -20.
        assert term.params["command_name"] == "base_velocity"
        assert term.params["forward_pitch_deg"] < 0  # User-tunable nose-up target.
        assert term.params["forward_threshold"] == .1
    for cls in (DeeproboticsM20WalkEnvCfg, DeeproboticsM20RoughEnvCfg):
        assert cls().rewards.flat_orientation_l2.func is flat_orientation_l2
    assert math.sin(math.radians(-5.)) < 0  # Nose-up target has negative gravity X.


def test_no_turn_yaw_penalty_applies_to_both_rotation_signs_and_idle_commands():
    commands = [[.2, 0., 0.], [-.2, 0., 0.], [0., .2, 0.], [0., 0., 0.], [0., 0., .2], [0., 0., -.2]]
    env = _env(commands, [0.] * 6)
    env.scene["robot"].data.root_ang_vel_b = SimpleNamespace(
        torch=torch.tensor([[10., -10., 2.], [10., 10., -2.], [0., 0., 2.], [0., 0., -2.], [0., 0., 2.], [0., 0., -2.]])
    )
    torch.testing.assert_close(no_turn_ang_vel_z_l2(env, "base_velocity"), torch.tensor([4., 4., 4., 4., 0., 0.]))
    env.scene["robot"].data.root_ang_vel_b.torch.zero_()
    assert not no_turn_ang_vel_z_l2(env, "base_velocity").any()


def test_no_turn_yaw_penalty_is_shared_by_sand_tasks_without_mutating_walk_or_rough():
    for cls in (DeeproboticsM20WalkOnSandProxyEnvCfg, DeeproboticsM20WalkOnSandEnvCfg):
        term = cls().rewards.ang_vel_z_no_turn_l2
        assert term.func is no_turn_ang_vel_z_l2 and term.weight == -1.
        assert term.params == {"command_name": "base_velocity", "zero_command_threshold": 1e-6}
    for cls in (DeeproboticsM20WalkEnvCfg, DeeproboticsM20RoughEnvCfg):
        assert not hasattr(cls().rewards, "ang_vel_z_no_turn_l2")
