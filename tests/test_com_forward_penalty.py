"""Whole-robot CoM bias, body-command frames and contact-provider isolation."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from M20_newton_training.tasks.m20.config.M20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_env_cfg import DeeproboticsM20WalkEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import ProxyFootContactState, com_forward_support_l2


def _env(commands, coms, contacts=None):
    n = len(commands)
    state = ProxyFootContactState(n, "cpu")
    state.vertical_contact[:] = True if contacts is None else torch.tensor(contacts, dtype=torch.bool)
    points = torch.tensor([[[0.4, 0.2, 0.0], [0.4, -0.2, 0.0], [-0.4, 0.2, 0.0], [-0.4, -0.2, 0.0]]]).repeat(n, 1, 1)
    data = SimpleNamespace(
        body_mass=SimpleNamespace(torch=torch.ones((n, 1))),
        body_com_pos_w=SimpleNamespace(torch=torch.tensor(coms, dtype=torch.float32)[:, None]),
        root_link_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1)),
        # Deliberately wrong root CoM: this is not the whole-robot CoM and must be ignored.
        root_com_pos_w=SimpleNamespace(torch=torch.full((n, 3), 100.0)),
    )
    term = SimpleNamespace(contact_state=state, application_points_w=points)
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor(commands)),
        action_manager=SimpleNamespace(get_term=lambda name: term),
        term=term,
    )


def test_deadband_and_one_sided_squared_excess():
    env = _env([[0.2, 0.0, 0.0]] * 5, [[x, 0.0, 0.6] for x in (-0.2, 0.0, 0.02, 0.03, 0.13)])
    torch.testing.assert_close(com_forward_support_l2(env), torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    # RewardManager applies the configured negative weight and 50 Hz integration.
    assert (com_forward_support_l2(env)[-1] * -1.0 * 0.02).item() == pytest.approx(-0.02)


def test_mass_weighted_whole_robot_com_uses_current_per_environment_masses():
    env = _env([[0.2, 0.0, 0.0]] * 2, [[0.0, 0.0, 0.6]] * 2)
    data = env.scene["robot"].data
    data.body_mass.torch = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
    data.body_com_pos_w.torch = torch.tensor([[[-0.1, 0.0, 0.6], [0.3, 0.0, 0.6]]]).repeat(2, 1, 1)
    # Identical link positions, different masses: CoM x=.2 or x=0, not the unweighted mean .1.
    torch.testing.assert_close(com_forward_support_l2(env), torch.tensor([2.89, 0.0]))
    data.body_mass.torch.copy_(torch.tensor([[3.0, 1.0], [1.0, 3.0]]))
    torch.testing.assert_close(com_forward_support_l2(env), torch.tensor([0.0, 2.89]))


@pytest.mark.parametrize(
    "command,position",
    [
        ([0.2, 0.0, 0.0], [0.13, 0.0, 0.6]),
        ([-0.2, 0.0, 0.0], [-0.13, 0.0, 0.6]),
        ([0.0, 0.2, 0.0], [0.0, 0.13, 0.6]),
        ([0.0, -0.2, 0.0], [0.0, -0.13, 0.6]),
        ([0.2, 0.2, 0.5], [0.13 / math.sqrt(2), 0.13 / math.sqrt(2), 0.6]),
    ],
)
def test_forward_backward_lateral_and_diagonal_commands_use_command_direction(command, position):
    env = _env([command], [position])
    torch.testing.assert_close(com_forward_support_l2(env), torch.ones(1))
    env.scene["robot"].data.body_com_pos_w.torch[..., :2].neg_()
    assert com_forward_support_l2(env).item() == 0.0


def test_zero_pure_yaw_and_small_planar_commands_are_disabled():
    env = _env([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.05, 0.0, 0.5], [0.1, 0.0, 0.0]], [[0.13, 0.0, 0.6]] * 4)
    torch.testing.assert_close(com_forward_support_l2(env), torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_yaw_rotation_and_world_translation_do_not_change_the_penalty():
    env = _env([[0.2, 0.0, 0.0]] * 3, [[0.13, 0.0, 0.6]] * 3)
    data = env.scene["robot"].data
    yaw = torch.tensor([math.pi / 2, -math.pi / 2, math.pi])
    q = quat_from_euler_xyz(torch.zeros(3), torch.zeros(3), yaw)
    origin = torch.tensor([[20.0, 30.0, 10.0], [-5.0, 7.0, -0.5], [8.0, -10.0, 3.0]])
    data.body_com_pos_w.torch.copy_(quat_apply(q[:, None], data.body_com_pos_w.torch) + origin[:, None])
    env.term.application_points_w.copy_(quat_apply(q[:, None], env.term.application_points_w) + origin[:, None])
    data.root_link_quat_w.torch.copy_(q)
    torch.testing.assert_close(com_forward_support_l2(env), torch.ones(3), atol=2e-5, rtol=2e-5)


def test_pitch_and_roll_do_not_rotate_the_planar_command_out_of_the_horizontal_plane():
    env = _env([[0.2, 0.0, 0.0]], [[0.13, 0.0, 0.6]])
    data = env.scene["robot"].data
    data.root_link_quat_w.torch.copy_(quat_from_euler_xyz(torch.tensor([0.3]), torch.tensor([-0.4]), torch.zeros(1)))
    torch.testing.assert_close(com_forward_support_l2(env), torch.ones(1))


def test_only_supporting_feet_define_reference_and_no_support_is_finite_zero():
    env = _env(
        [[0.2, 0.0, 0.0]] * 4,
        [[0.13, 0.0, 0.6]] * 4,
        [
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, False],
            [False, False, False, False],
        ],
    )
    env.term.application_points_w[1, 1:3, 0] = 100.0  # Two swing feet must not shift the reference.
    torch.testing.assert_close(com_forward_support_l2(env), torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_three_support_reference_is_the_geometric_mean_not_force_weighted():
    env = _env([[0.2, 0.0, 0.0]], [[0.13, 0.0, 0.6]])
    force = torch.tensor([[[0.0, 0.0, 10.0], [100.0, 0.0, 5.0], [0.0, 0.0, 50.0], [0.0, 0.0, 100.0]]])
    env.term.contact_state.update(force, 0.02, 5.0)
    # Fz==5 is not support even with a large horizontal force. Other support-force magnitudes do not weight the mean.
    support_x = (0.4 - 0.4 - 0.4) / 3
    expected = ((0.13 - support_x - 0.03) / 0.1) ** 2
    assert com_forward_support_l2(env).item() == pytest.approx(expected)


@pytest.mark.parametrize(
    "params",
    [
        {"allowed_forward_offset": -0.01},
        {"allowed_forward_offset": float("nan")},
        {"distance_scale": 0.0},
        {"distance_scale": float("inf")},
        {"min_command_norm": 0.0},
        {"min_support_feet": 0},
        {"min_support_feet": 5},
        {"min_support_feet": True},
    ],
)
def test_invalid_parameters_fail_with_an_explicit_message(params):
    with pytest.raises(ValueError):
        com_forward_support_l2(_env([[0.2, 0.0, 0.0]], [[0.13, 0.0, 0.6]]), **params)


def test_zero_deadband_penalizes_any_forward_offset():
    env = _env([[0.2, 0.0, 0.0]], [[0.01, 0.0, 0.6]])
    assert com_forward_support_l2(env, allowed_forward_offset=0.0).item() == pytest.approx(0.01)


def test_shared_task_binding_uses_each_force_provider_and_does_not_change_walk_or_rough():
    for cls, provider in (
        (DeeproboticsM20WalkOnSandProxyEnvCfg, "sand_proxy"),
        (DeeproboticsM20WalkOnSandEnvCfg, "mpm_contact"),
    ):
        term = cls().rewards.com_forward_l2
        assert term.func is com_forward_support_l2
        assert term.weight == -1.0
        assert term.params == {
            "command_name": "base_velocity",
            "action_term_name": provider,
            "allowed_forward_offset": 0.03,
            "distance_scale": 0.10,
            "min_command_norm": 0.1,
            "min_support_feet": 2,
        }
    for cls in (DeeproboticsM20WalkEnvCfg, DeeproboticsM20RoughEnvCfg):
        assert not hasattr(cls().rewards, "com_forward_l2")
