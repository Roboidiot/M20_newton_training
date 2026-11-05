"""Opt-in regression trajectories for retained experimental balanced crawl helpers."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardManager
from isaaclab.utils.math import quat_apply

from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.crawl_rewards import (
    BalancedFeetAirTimeReward,
    BalancedFootClearanceReward,
    CrawlForwardStepReward,
    ProgressiveCrawlPhaseReward,
)
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import ProxyFootContactState

NAMES = ("rotation_gait_status", "feet_forward_step", "rotation_gait_symmetry", "feet_air_time", "feet_sand_clearance")
STORAGE = ("hr_hipx", "fl_hipx", "hl_hipx", "fr_hipx")


class Scene(dict):
    env_origins = torch.zeros(3, 3)


def _env(cfg_cls=DeeproboticsM20WalkOnSandProxyEnvCfg, command=(0.2, 0.0, 0.0), yaw=0.0, dt=0.02):
    cfg = cfg_cls()
    # These experimental replacements are retained but no longer bound by the active task.
    lift = cfg.rewards.rotation_gait_status
    lift.params.pop("repeat_decay", None)
    lift.params["max_progress_lead"] = 0.5
    forward = cfg.rewards.feet_forward_step
    forward.func = CrawlForwardStepReward
    forward.params.pop("repeat_decay", None)
    forward.params["max_progress_lead"] = 0.5
    phase = cfg.rewards.rotation_gait_symmetry
    phase.func = ProgressiveCrawlPhaseReward
    phase.params.update(min_lift=0.03, min_air_time=0.1)
    air = cfg.rewards.feet_air_time
    air.func = BalancedFeetAirTimeReward
    air.params.update(max_progress_lead=0.5, participation_time=0.3)
    root = torch.tensor([[0.0, 0.0, 0.75]]).repeat(3, 1)
    rotation = torch.tensor([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]]).repeat(3, 1)
    point = torch.tensor([[[0.3, 0.2, 0.1], [0.3, -0.2, 0.1], [-0.3, 0.2, 0.1], [-0.3, -0.2, 0.1]]]).repeat(3, 1, 1)
    point = quat_apply(rotation[:, None].expand(-1, 4, -1), point)
    data = SimpleNamespace(
        root_link_pos_w=SimpleNamespace(torch=root),
        root_pos_w=SimpleNamespace(torch=root),
        root_quat_w=SimpleNamespace(torch=rotation),
        body_link_pos_w=SimpleNamespace(torch=torch.zeros(3, 4, 3)),
        projected_gravity_b=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1)),
    )
    term = SimpleNamespace(contact_state=ProxyFootContactState(3, "cpu"), application_points_w=point)
    commands = torch.tensor([command]).repeat(3, 1)
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=dt,
        cfg=cfg,
        data=data,
        term=term,
        command=commands,
        scene=Scene(
            robot=SimpleNamespace(
                data=data,
                find_bodies=lambda names, preserve_order: ([STORAGE.index(name) for name in names], list(names)),
            )
        ),
        command_manager=SimpleNamespace(get_command=lambda name: commands),
        action_manager=SimpleNamespace(get_term=lambda name: term),
        extras={},
    )
    _hips(env)
    env.params = {name: getattr(cfg.rewards, name).params for name in NAMES}
    env.rewards = {name: getattr(cfg.rewards, name).func(getattr(cfg.rewards, name), env) for name in NAMES}
    return env


def _hips(env):
    offsets = torch.tensor(
        [[[-0.3141, -0.0685, 0.0], [0.3141, 0.0685, 0.0], [-0.3141, 0.0685, 0.0], [0.3141, -0.0685, 0.0]]]
    )
    env.data.body_link_pos_w.torch = env.data.root_link_pos_w.torch[:, None] + quat_apply(
        env.data.root_quat_w.torch[:, None].expand(-1, 4, -1), offsets.expand(3, -1, -1)
    )


def _sample(env, unloaded=()):
    _hips(env)
    force = torch.zeros(3, 4, 3)
    force[..., 2] = 50.0
    force[:, list(unloaded), 2] = 0.0
    state = env.term.contact_state
    state.begin_step()
    state.update(force, env.step_dt, 5.0)
    scores = {name: reward(env, **env.params[name]) for name, reward in env.rewards.items()}
    assert all(torch.isfinite(score).all() for score in scores.values())
    return scores


def _swing(env, foot=0, lift=0.15, body=0.0, stroke=0.0, steps=5):
    point = env.term.application_points_w
    z = point[:, foot, 2].clone()
    direction = env.command[:, :2]
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
    world = quat_apply(env.data.root_quat_w.torch, torch.cat((direction, torch.zeros(3, 1)), dim=1))[:, :2]
    point[:, foot, 2] += lift
    scores = []
    for sample in range(steps):
        if sample == steps - 1:
            env.data.root_link_pos_w.torch[:, :2] += body * world
            point[:, foot, :2] += (body + stroke) * world
        scores.append(_sample(env, (foot,)))
    point[:, foot, 2] = z
    landing = _sample(env)
    return scores, landing


@pytest.mark.parametrize("cfg_cls", [DeeproboticsM20WalkOnSandProxyEnvCfg, DeeproboticsM20WalkOnSandEnvCfg])
def test_opt_in_helpers_pass_native_manager_parameter_checks(cfg_cls):
    env = _env(cfg_cls)
    expected = (
        CrawlForwardStepReward,
        ProgressiveCrawlPhaseReward,
        BalancedFeetAirTimeReward,
        BalancedFootClearanceReward,
    )
    for name, cls in zip(NAMES[1:], expected, strict=True):
        assert getattr(env.cfg.rewards, name).func is cls
    manager = object.__new__(RewardManager)
    manager._resolve_terms_handle = None
    manager._env = SimpleNamespace(sim=SimpleNamespace(is_playing=lambda: False))
    for name in NAMES:
        manager._resolve_common_term_cfg(name, getattr(env.cfg.rewards, name))


def test_repeated_single_leg_lifts_stop_paying_and_other_feet_reenable_the_leg():
    env = _env()
    _sample(env)
    first, _ = _swing(env)
    torch.testing.assert_close(sum(s["rotation_gait_status"] for s in first), torch.full((3,), 0.5))
    for _ in range(12):
        scores, _ = _swing(env)
        assert not sum(s["rotation_gait_status"] for s in scores).any()
    torch.testing.assert_close(env.rewards["rotation_gait_status"].balance_progress[:, 0], torch.full((3,), 0.5))
    for foot in (1, 2, 3):
        scores, _ = _swing(env, foot)
        assert sum(s["rotation_gait_status"] for s in scores).min() > 0
    scores, _ = _swing(env)
    torch.testing.assert_close(sum(s["rotation_gait_status"] for s in scores), torch.full((3,), 0.5))


def test_phase_gives_progressive_credit_for_two_three_four_real_swing_participants():
    env = _env()
    _sample(env)
    for foot, expected in enumerate((0.0, 1 / 3, 2 / 3, 1.0)):
        _, result = _swing(env, foot, lift=0.05)
        torch.testing.assert_close(result["rotation_gait_symmetry"], torch.full((3,), expected))
    for _ in range(101):
        result = _sample(env, (0,))
    assert not result["rotation_gait_symmetry"].any()  # Holding cannot extend the event window.


@pytest.mark.parametrize("lift,steps", [(0.02, 5), (0.15, 4)])
def test_phase_rejects_tiny_or_short_unloading(lift, steps):
    env = _env()
    _sample(env)
    for foot in range(4):
        _, scores = _swing(env, foot, lift=lift, steps=steps)
        assert not scores["rotation_gait_symmetry"].any()


@pytest.mark.parametrize(
    "command,yaw", [((0.2, 0.0, 0.0), 0.0), ((-0.2, 0.0, 0.0), 0.0), ((0.0, 0.2, 0.0), 0.8), ((0.2, 0.0, 0.0), 1.2)]
)
def test_forward_requires_real_body_advance_and_a_forward_swing_in_command_coordinates(command, yaw):
    env = _env(command=command, yaw=yaw)
    _sample(env)
    _, result = _swing(env, body=0.06, stroke=0.12)
    torch.testing.assert_close(result["feet_forward_step"], torch.full((3,), 0.5), atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize("body,stroke", [(0.0, 0.12), (-0.12, 0.24), (0.12, 0.0)])
def test_forward_rejects_stationary_swings_backward_body_or_rigidly_carried_legs(body, stroke):
    env = _env()
    _sample(env)
    _, result = _swing(env, body=body, stroke=stroke)
    assert result["feet_forward_step"].abs().max() < 1e-6


def test_same_relative_touchdown_location_still_scores_after_stance_then_forward_swing():
    env = _env()
    _sample(env)
    # Stance carries the chassis forward over a fixed foot, retracting that leg relative to HipX.
    env.data.root_link_pos_w.torch[:, 0] += 0.12
    _sample(env)
    # Swing advances foot .24 and chassis .12. Final relative landing equals the first landing.
    _, result = _swing(env, body=0.12, stroke=0.12)
    torch.testing.assert_close(result["feet_forward_step"], torch.ones(3), atol=2e-6, rtol=1e-6)


def test_body_backward_forward_oscillation_cannot_repeatedly_buy_forward_credit():
    env = _env()
    _sample(env)
    _swing(env, body=0.06, stroke=0.12)
    reward = env.rewards["feet_forward_step"]
    # Temporarily remove only the participation cap to isolate the chassis high-water condition.
    reward.balance_progress.zero_()
    env.data.root_link_pos_w.torch[:, 0] -= 0.06
    _sample(env)
    _, result = _swing(env, body=0.06, stroke=0.12)
    assert result["feet_forward_step"].abs().max() < 1e-6


def test_forward_single_leg_limit_and_recovery_do_not_accumulate_unpaid_debt():
    env = _env()
    _sample(env)
    _, result = _swing(env, body=0.12, stroke=0.12)
    torch.testing.assert_close(result["feet_forward_step"], torch.ones(3), atol=2e-6, rtol=1e-6)
    for _ in range(8):
        _, result = _swing(env, body=0.12, stroke=0.12)
        assert not result["feet_forward_step"].any()
    for foot in (1, 2, 3, 0):
        _, result = _swing(env, foot, body=0.12, stroke=0.12)
        assert result["feet_forward_step"].min() > 0.99


def test_airtime_positive_bonus_is_limited_but_short_air_penalty_survives():
    env = _env()
    _sample(env)
    _, result = _swing(env, steps=15)
    assert result["feet_air_time"].min() > 0
    _, result = _swing(env, steps=15)
    assert not result["feet_air_time"].any()
    _, result = _swing(env, steps=1)
    torch.testing.assert_close(result["feet_air_time"], torch.full((3,), -0.06))


@pytest.mark.parametrize("dt,steps", [(0.01, 10), (0.02, 5), (0.025, 4)])
@pytest.mark.parametrize("cfg_cls", [DeeproboticsM20WalkOnSandProxyEnvCfg, DeeproboticsM20WalkOnSandEnvCfg])
def test_opt_in_forward_duration_gate_excludes_the_landing_interval(cfg_cls, dt, steps):
    env = _env(cfg_cls, dt=dt)
    _sample(env)
    _, result = _swing(env, body=0.06, stroke=0.12, steps=steps - 1)
    assert not result["feet_forward_step"].any()
    _, result = _swing(env, body=0.06, stroke=0.12, steps=steps)
    torch.testing.assert_close(result["feet_forward_step"], torch.full((3,), 0.5), atol=2e-6, rtol=1e-6)


def test_reset_logs_only_counts_and_progress_and_preserves_other_worlds():
    env = _env()
    _sample(env)
    for _ in range(5):
        _swing(env, body=0.12, stroke=0.12)
    for name in ("rotation_gait_status", "feet_forward_step"):
        env.rewards[name].reset([1])
    log = env.extras["log"]
    torch.testing.assert_close(log["Metrics/forward_steps/fl_progress"], torch.tensor(5.0), atol=2e-6, rtol=1e-6)
    assert len(log) == 16
    assert all(key.endswith(("_count", "_progress")) for key in log)
    assert not any(key.endswith("_rewarded_progress") for key in log)
    assert not env.rewards["feet_forward_step"].participation_progress[1].any()
    assert env.rewards["feet_forward_step"].participation_progress[0, 0] > 0


def test_clearance_cannot_be_used_as_an_unlimited_single_leg_alternative():
    env = _env()
    _sample(env)
    first, _ = _swing(env, lift=0.26)
    assert sum(s["feet_sand_clearance"] for s in first).min() > 0
    repeated, _ = _swing(env, lift=0.26)
    assert not sum(s["feet_sand_clearance"] for s in repeated).any()


def test_partial_reset_and_zero_commands_clear_new_reward_state():
    env = _env()
    _sample(env)
    for foot in range(4):
        _swing(env, foot, body=0.06, stroke=0.12)
    for reward in env.rewards.values():
        reward.reset([1])
    result = _sample(env)
    assert result["rotation_gait_symmetry"][1] == 0
    assert result["rotation_gait_symmetry"][0] > 0
    env.command.zero_()
    for _ in range(7):
        assert all(not score.any() for score in _sample(env, (0,)).values())
    assert not env.rewards["feet_forward_step"].body_progress.any()
    assert not env.rewards["rotation_gait_status"].balance_progress.any()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_participation_limits_are_rejected(name, value):
    env = _env()
    key = "min_lift" if name == "rotation_gait_symmetry" else "max_progress_lead"
    cfg = getattr(env.cfg.rewards, name)
    cfg.params[key] = value
    with pytest.raises(ValueError):
        cfg.func(cfg, env)
