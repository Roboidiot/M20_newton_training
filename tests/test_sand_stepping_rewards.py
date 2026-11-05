"""Extraction, forward touchdown, reference frames and invalid-swing rejection."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import (
    DeeproboticsM20WalkOnSandProxyEnvCfg,
    FootProxyRewardsCfg,
)
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import (
    ProxyExtractedFootForwardStepReward,
    ProxyFootContactState,
    sand_base_height_l2,
)


class Scene(dict):
    pass


def _env(command=(0.2, 0.0, 0.0), yaw=0.0):
    n = 3
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg()
    cfg.rewards.feet_sand_clearance = FootProxyRewardsCfg().feet_sand_clearance
    # Retain regression coverage for the old extraction-only stride helper.
    # The active touchdown-to-touchdown reward is covered in test_foot_touchdown_progress.py.
    cfg.rewards.feet_forward_step = RewardTermCfg(
        func=ProxyExtractedFootForwardStepReward, weight=150.0,
        params={
            "target_clearance": .04, "target_stride": .08, "min_stride": .03,
            "min_air_time": .1, "max_swing_time": 1., "surface_height": .30,
            "min_base_height": .35, "max_tilt": .5,
            "action_term_name": "sand_proxy", "command_name": "base_velocity", "min_command_norm": .1,
        },
    )
    origins = torch.tensor([[0., 0., 0.], [3., -4., 1.], [-2., 5., -0.2]])
    root = origins + torch.tensor([0., 0., 0.78])
    q = torch.tensor([0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]).repeat(n, 1)
    feet_b = torch.tensor([[.3, .2, -.54], [.3, -.2, -.54], [-.3, .2, -.54], [-.3, -.2, -.54]])
    points = root[:, None] + quat_apply(q[:, None].expand(-1, 4, -1), feet_b[None].expand(n, -1, -1))

    def wrap(value):
        return SimpleNamespace(torch=value)

    data = SimpleNamespace(
        root_pos_w=wrap(root), root_quat_w=wrap(q),
        projected_gravity_b=wrap(torch.tensor([0., 0., -1.]).repeat(n, 1)),
    )
    scene = Scene(robot=SimpleNamespace(data=data))
    scene.env_origins = origins
    term = SimpleNamespace(application_points_w=points, contact_state=ProxyFootContactState(n, "cpu"))
    env = SimpleNamespace(
        cfg=cfg, scene=scene, num_envs=n, device="cpu", step_dt=.02, term=term,
        command=torch.tensor(command).repeat(n, 1),
        action_manager=SimpleNamespace(get_term=lambda name: term),
    )
    env.command_manager = SimpleNamespace(get_command=lambda name: env.command)
    env.rewards = []
    for name in ("feet_sand_clearance", "feet_forward_step"):
        term_cfg = getattr(cfg.rewards, name)
        env.rewards.append((term_cfg.func(term_cfg, env), term_cfg.params))
    return env


def _step(env, swing=(), dz=None, displacement=None):
    force = torch.zeros(env.num_envs, 4, 3)
    force[..., 2] = 50.
    force[:, list(swing), 2] = -20.  # Pull-out is unloaded even with nonzero force.
    env.term.contact_state.begin_step()
    env.term.contact_state.update(force, env.step_dt, 5.)
    if dz is not None:
        env.term.application_points_w[:, 0, 2] = env.scene.env_origins[:, 2] + .30 + dz
    if displacement is not None:
        env.term.application_points_w[:, 0, :2] = displacement
    return tuple(reward(env, **params) for reward, params in env.rewards)


def _extract(env, swing=(0,)):
    _step(env)
    for _ in range(5):
        result = _step(env, swing, dz=.05)
    return result


def _forward(env, distance=.08):
    command = env.command[:, :2]
    direction = command / command.norm(dim=1, keepdim=True).clamp_min(1e-6)
    direction3 = torch.cat((direction, torch.zeros(env.num_envs, 1)), dim=1)
    return quat_apply(env.scene["robot"].data.root_quat_w.torch, direction3)[:, :2] * distance


def test_height_uses_sand_reference_and_each_environment_origin():
    env = _env()
    torch.testing.assert_close(sand_base_height_l2(env, .48), torch.zeros(env.num_envs), atol=1e-12, rtol=0)
    env.scene["robot"].data.root_pos_w.torch[:, 2] -= .2
    torch.testing.assert_close(sand_base_height_l2(env, .48), torch.full((env.num_envs,), .04))


def test_clearance_requires_leaving_sand_and_only_credits_new_capped_progress():
    env = _env()
    _step(env)
    for _ in range(5):
        clearance, forward = _step(env, (0,), dz=-.01)
        assert not clearance.any() and not forward.any()
    clearance, _ = _step(env, (0,), dz=.02)
    torch.testing.assert_close(clearance, torch.full((env.num_envs,), .5), atol=1e-5, rtol=1e-5)
    assert not _step(env, (0,), dz=.02)[0].any()
    assert not _step(env, (0,), dz=.01)[0].any()
    assert not _step(env, (0,), dz=.02)[0].any()
    clearance, _ = _step(env, (0,), dz=.05)
    torch.testing.assert_close(clearance, torch.full((env.num_envs,), .5), atol=1e-5, rtol=1e-5)
    assert not _step(env, (0,), dz=.2)[0].any()
    assert not _step(env, dz=-.02)[0].any()


@pytest.mark.parametrize("command", [(0.2, 0., 0.), (-0.2, 0., 0.), (0., -.2, 0.)])
@pytest.mark.parametrize("yaw", [0., math.pi / 2])
def test_forward_stroke_is_directional_and_rewards_only_one_confirmed_landing(command, yaw):
    env = _env(command, yaw)
    clearance, forward = _extract(env)
    torch.testing.assert_close(clearance, torch.ones(env.num_envs))
    assert not forward.any()
    original = env.term.application_points_w[:, 0, :2].clone()
    _, forward = _step(env, (0,), dz=.05, displacement=original + _forward(env))
    assert not forward.any()  # Hovering in front never earns the touchdown reward.
    _, forward = _step(env, dz=-.02)
    torch.testing.assert_close(forward, torch.ones(env.num_envs), atol=1e-5, rtol=1e-5)
    assert not _step(env)[1].any()


@pytest.mark.parametrize("distance, expected", [(.02, 0.), (.04, .5), (.16, 1.), (-.08, 0.)])
def test_forward_stride_has_minimum_and_upper_bound(distance, expected):
    env = _env()
    _extract(env)
    original = env.term.application_points_w[:, 0, :2].clone()
    _step(env, (0,), dz=.05, displacement=original + _forward(env, distance))
    _, result = _step(env, dz=-.02)
    torch.testing.assert_close(result, torch.full((env.num_envs,), expected), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", [
    "in_place", "buried", "pre_extraction", "retracted", "chassis_forward", "chassis_backward",
])
def test_failed_strokes_cannot_collect_forward_touchdown_credit(mode):
    env = _env()
    original = env.term.application_points_w[:, 0, :2].clone()
    if mode in ("buried", "pre_extraction"):
        _step(env)
        for _ in range(5):
            _step(env, (0,), dz=-.01, displacement=original + _forward(env))
        if mode == "pre_extraction":
            _step(env, (0,), dz=.05)  # Forward movement before extraction does not count.
    else:
        _extract(env)
    if mode == "retracted":
        _step(env, (0,), dz=.05, displacement=original + _forward(env))
        _step(env, (0,), dz=.05, displacement=original)
    if mode == "chassis_forward":
        shift = _forward(env)
        env.scene["robot"].data.root_pos_w.torch[:, :2] += shift
        env.term.application_points_w[..., :2] += shift[:, None]
        _step(env, (0,), dz=.05)
    if mode == "chassis_backward":
        env.scene["robot"].data.root_pos_w.torch[:, :2] -= _forward(env)
        _step(env, (0,), dz=.05)
    assert not _step(env, dz=-.02)[1].any()


def test_chassis_yaw_with_rigidly_held_legs_is_not_a_forward_stroke():
    env = _env()
    _extract(env)
    data = env.scene["robot"].data
    relative = quat_apply_inverse(
        data.root_quat_w.torch[:, None].expand(-1, 4, -1),
        env.term.application_points_w - data.root_pos_w.torch[:, None],
    )
    data.root_quat_w.torch[:] = torch.tensor([0., 0., 2**-.5, 2**-.5])
    env.term.application_points_w.copy_(
        data.root_pos_w.torch[:, None] + quat_apply(data.root_quat_w.torch[:, None].expand(-1, 4, -1), relative)
    )
    _step(env, (0,), dz=.05)
    assert not _step(env, dz=-.02)[1].any()


@pytest.mark.parametrize("invalid", ["flight", "one_support", "low_body", "tilt", "short", "long", "reverse", "zero"])
def test_invalid_swings_are_not_rescued_by_later_extraction_and_landing(invalid):
    env = _env()
    _step(env)
    if invalid == "low_body":
        env.scene["robot"].data.root_pos_w.torch[:, 2] -= .2
    if invalid == "tilt":
        env.scene["robot"].data.projected_gravity_b.torch[:] = torch.tensor([.8, 0., -.6])
    count = 1 if invalid == "short" else 5
    original = env.term.application_points_w[:, 0, :2].clone()
    for i in range(count):
        swing = (0, 1, 2, 3) if invalid == "flight" and i == 1 else (0,)
        if invalid == "one_support" and i == 1:
            swing = (0, 1, 2)
        _step(env, swing, dz=.05)
    _step(env, (0,), dz=.05, displacement=original + _forward(env))
    if invalid == "long":
        for _ in range(51):
            _step(env, (0,), dz=.05)
    if invalid == "reverse":
        env.command *= -1
        _step(env, (0,), dz=.05)
    if invalid == "zero":
        env.command.zero_()
        _step(env, (0,), dz=.05)
    assert not _step(env, dz=-.02)[1].any()


def test_initial_airborne_feet_and_partial_resets_need_fresh_support():
    env = _env()
    for _ in range(5):
        assert not _step(env, (0,), dz=.05)[0].any()
    assert not _step(env, dz=-.02)[1].any()
    _extract(env)
    original = env.term.application_points_w[:, 0, :2].clone()
    _step(env, (0,), dz=.05, displacement=original + _forward(env))
    for reward, _ in env.rewards:
        reward.reset([1])
    _, result = _step(env, dz=-.02)
    torch.testing.assert_close(result, torch.tensor([1., 0., 1.]), atol=1e-5, rtol=1e-5)
    # The reset world can earn credit again after establishing support.
    _extract(env)
    original = env.term.application_points_w[:, 0, :2].clone()
    _step(env, (0,), dz=.05, displacement=original + _forward(env))
    torch.testing.assert_close(_step(env, dz=-.02)[1], torch.ones(env.num_envs), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("feet", [(0,), (1,), (2,), (3,), (0, 3), (0, 1)])
def test_clearance_allows_independent_feet_without_imposing_diagonal_pattern(feet):
    env = _env()
    _step(env)
    for _ in range(4):
        _step(env, feet)
    env.term.application_points_w[:, list(feet), 2] = env.scene.env_origins[:, 2:3] + .35
    result, _ = _step(env, feet)
    torch.testing.assert_close(result, torch.full((env.num_envs,), float(len(feet))))


@pytest.mark.parametrize("command", [(0., 0., 0.), (.05, 0., 0.), (0., 0., .2)])
def test_new_stepping_terms_require_a_planar_movement_command(command):
    env = _env(command)
    assert not _extract(env)[0].any()
    assert not _step(env, dz=-.02)[1].any()


@pytest.mark.parametrize("params", [
    {"target_clearance": 0.}, {"target_stride": float("nan")},
    {"min_stride": .2}, {"min_air_time": 2.}, {"max_tilt": 2.},
])
def test_invalid_swing_configuration_is_rejected(params):
    env = _env()
    reward, arguments = env.rewards[0]
    cfg = env.cfg.rewards.feet_sand_clearance
    cfg.params = {**arguments, **params}
    with pytest.raises(ValueError):
        type(reward)(cfg, env)
