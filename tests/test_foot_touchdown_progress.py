"""Retained legacy touchdown reward regressions; active crawl trajectories have separate coverage."""

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import ProxyFootContactState, ProxyFootForwardStepReward

HIP_STORAGE = ("hr_hipx", "fl_hipx", "hl_hipx", "fr_hipx")


def _env(
    command=(0.2, 0.0, 0.0),
    yaw=0.0,
    cfg_cls=DeeproboticsM20WalkOnSandProxyEnvCfg,
    target_stride=0.08,
    repeat_decay=0.0,
    min_air_time=0.0,
):
    n = 3
    cfg = cfg_cls().rewards.feet_forward_step
    # Explicit legacy coverage; new active helper is tested in test_balanced_crawl_rewards.py.
    cfg.func = ProxyFootForwardStepReward
    cfg.params.pop("max_progress_lead", None)
    cfg.params.setdefault("repeat_decay", 1.0)
    # Keep generic function cases independent of the task's tunable stride scale.
    if target_stride is not None:
        cfg.params["target_stride"] = target_stride
    if repeat_decay is not None:
        cfg.params["repeat_decay"] = repeat_decay
    # Isolate the existing geometric/participation tests from the task's duration gate.
    if min_air_time is not None:
        cfg.params["min_air_time"] = min_air_time
    state = ProxyFootContactState(n, "cpu")
    points = torch.zeros(n, 4, 3)
    points[..., 2] = -0.1  # All feet remain buried throughout these experiments.
    points[1, :, 0] = 3.0
    points[2, :, 1] = -4.0
    root = points[:, 0].clone()
    root[:, 2] = 0.6
    rotation = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]).repeat(n, 1)
    term = SimpleNamespace(contact_state=state, application_points_w=points)
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        step_dt=0.02,
        command=torch.tensor(command).repeat(n, 1),
        term=term,
        params=cfg.params,
        action_manager=SimpleNamespace(get_term=lambda name: term),
        # Body-relative progress needs the root position, but no gravity or height gate.
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_quat_w=SimpleNamespace(torch=rotation),
                    root_link_pos_w=SimpleNamespace(torch=root),
                    body_link_pos_w=SimpleNamespace(torch=torch.zeros(n, 4, 3)),
                ),
                find_bodies=lambda names, preserve_order: ([HIP_STORAGE.index(name) for name in names], list(names)),
            )
        },
    )
    env.command_manager = SimpleNamespace(get_command=lambda name: env.command)
    env.manual_hips = False
    _sync_hips(env)
    env.reward = cfg.func(cfg, env)
    return env


def _sync_hips(env):
    if env.manual_hips:
        return
    data = env.scene["robot"].data
    # Public articulation storage is deliberately different from the reward foot order.
    offsets = torch.tensor(
        [[-0.3141, -0.0685, 0.0], [0.3141, 0.0685, 0.0], [-0.3141, 0.0685, 0.0], [0.3141, -0.0685, 0.0]]
    )
    data.body_link_pos_w.torch = data.root_link_pos_w.torch[:, None] + quat_apply(
        data.root_quat_w.torch[:, None].expand(-1, 4, -1),
        offsets[None].expand(env.num_envs, -1, -1),
    )


def _shift(env, distance, feet=(0,)):
    direction = env.command[:, :2]
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
    direction = torch.cat((direction, torch.zeros(env.num_envs, 1)), dim=1)
    world = quat_apply(env.scene["robot"].data.root_quat_w.torch, direction)
    env.term.application_points_w[:, list(feet), :2] += distance * world[:, None, :2]


def _step(env, unloaded=()):
    _sync_hips(env)
    force = torch.zeros(env.num_envs, 4, 3)
    force[..., 2] = 50.0
    force[:, list(unloaded), 0] = 100.0
    force[:, list(unloaded), 2] = -20.0  # Large drag/pull still counts as unloaded.
    env.term.contact_state.begin_step()
    env.term.contact_state.update(force, env.step_dt, 5.0)
    result = env.reward(env, **env.params)
    assert torch.isfinite(result).all()
    return result


@pytest.mark.parametrize("command", [(0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, -0.2, 0.0)])
@pytest.mark.parametrize("yaw", [0.0, math.pi / 2])
def test_buried_foot_progress_uses_chassis_relative_displacement_and_command_heading(command, yaw):
    env = _env(command, yaw)
    assert not _step(env).any()  # First support only establishes the landing position.
    assert not _step(env, (0,)).any()
    _shift(env, 0.02)
    torch.testing.assert_close(_step(env), torch.full((env.num_envs,), 0.25), atol=1e-5, rtol=1e-5)
    assert not _step(env).any()
    assert (env.term.application_points_w[..., 2] < 0.3).all()


@pytest.mark.parametrize("distance, expected", [(0.001, 0.0125), (0.04, 0.5), (0.16, 1.0), (0.0, 0.0), (-0.08, 0.0)])
def test_any_positive_advance_scores_and_target_stride_only_sets_normalization_and_cap(distance, expected):
    env = _env()
    _step(env)
    _step(env, (0,))
    _shift(env, distance)
    torch.testing.assert_close(_step(env), torch.full((env.num_envs,), expected), atol=1e-5, rtol=1e-5)


def test_anchor_is_previous_touchdown_not_latest_stance_or_liftoff_position():
    env = _env()
    _step(env)
    _shift(env, 0.04)
    assert not _step(env).any()  # A foot sliding while supported is not another touchdown.
    _step(env, (0,))
    _shift(env, 0.04)
    torch.testing.assert_close(_step(env), torch.ones(env.num_envs), atol=1e-5, rtol=1e-5)
    _step(env, (0,))
    assert not _step(env).any()  # No displacement between this pair of touchdown positions.


def test_backward_landing_updates_anchor_and_returning_airborne_motion_is_not_progress():
    env = _env()
    _step(env)
    _step(env, (0,))
    _shift(env, -0.04)
    assert not _step(env).any()
    _step(env, (0,))
    _shift(env, 0.04)
    torch.testing.assert_close(_step(env), torch.full((env.num_envs,), 0.5), atol=1e-5, rtol=1e-5)
    _step(env, (0,))
    _shift(env, 0.08)
    _shift(env, -0.08)
    assert not _step(env).any()


@pytest.mark.parametrize("air_steps", [1, 60])
def test_zero_duration_override_requires_no_air_duration_or_number_of_other_supporting_feet(air_steps):
    env = _env()
    _step(env)
    for _ in range(air_steps):
        assert not _step(env, (0, 1, 2, 3)).any()
    _shift(env, 0.08)
    torch.testing.assert_close(_step(env), torch.ones(env.num_envs), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("feet", [(0,), (1,), (2,), (3,), (0, 3)])
def test_each_foot_keeps_its_own_touchdown_anchor(feet):
    env = _env()
    _step(env)
    _step(env, feet)
    _shift(env, 0.08, feet)
    torch.testing.assert_close(_step(env), torch.full((env.num_envs,), float(len(feet))), atol=1e-5, rtol=1e-5)


def test_initial_fall_and_partial_reset_do_not_create_fake_progress():
    env = _env()
    _step(env, (0, 1, 2, 3))
    _shift(env, 0.16)
    assert not _step(env).any()
    _step(env, (0,))
    _shift(env, 0.08)
    env.reward.reset([1])
    torch.testing.assert_close(_step(env), torch.tensor([1.0, 0.0, 1.0]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("command", [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.2), (-0.2, 0.0, 0.0)])
def test_inactive_or_reversed_command_clears_pending_touchdown_progress(command):
    env = _env()
    _step(env)
    _step(env, (0,))
    _shift(env, 0.08)
    env.command[:] = torch.tensor(command)
    assert not _step(env).any()
    assert not _step(env).any()


@pytest.mark.parametrize("command", [(0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, -0.2, 0.0)])
@pytest.mark.parametrize("yaw", [0.0, math.pi / 2])
def test_common_foot_and_chassis_translation_is_not_a_forward_step(command, yaw):
    env = _env(command, yaw, min_air_time=0.1)
    _step(env)
    for _ in range(5):
        _step(env, (0,))
    before = env.term.application_points_w[:, 0].clone()
    _shift(env, 0.08, (0, 1, 2, 3))
    delta = env.term.application_points_w[:, 0] - before
    env.scene["robot"].data.root_link_pos_w.torch.add_(delta)
    assert not _step(env).any()
    assert not env.reward.completed_steps.any()
    assert not env.reward.participation_progress.any()


@pytest.mark.parametrize(
    "foot_advance,body_advance,expected",
    [
        (0.12, 0.04, 1.0),
        (0.08, 0.04, 0.5),
        (0.04, 0.08, 0.0),
        (0.08, 0.08, 0.0),
        (0.0, -0.08, 1.0),
    ],
)
def test_stride_uses_difference_of_foot_minus_chassis_at_the_two_touchdowns(foot_advance, body_advance, expected):
    env = _env(min_air_time=0.1)
    _step(env)
    for _ in range(5):
        _step(env, (0,))
    _shift(env, foot_advance)
    env.scene["robot"].data.root_link_pos_w.torch[:, 0] += body_advance
    torch.testing.assert_close(_step(env), torch.full((3,), expected), atol=1e-5, rtol=1e-5)


def test_each_environment_subtracts_its_own_chassis_motion():
    env = _env(min_air_time=0.1)
    _step(env)
    for _ in range(5):
        _step(env, (0,))
    _shift(env, 0.08)
    env.scene["robot"].data.root_link_pos_w.torch[:, 0] += torch.tensor([0.0, 0.04, 0.08])
    torch.testing.assert_close(_step(env), torch.tensor([1.0, 0.5, 0.0]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("yaw", [math.pi / 2, math.pi, -0.7])
@pytest.mark.parametrize("relative_advance", [0.0, 0.08])
def test_touchdown_offsets_use_each_instant_heading_so_rigid_turns_do_not_count_as_leg_swing(yaw, relative_advance):
    env = _env(min_air_time=0.1)
    data = env.scene["robot"].data
    env.term.application_points_w[..., :2] += torch.tensor([[0.4, 0.2], [0.4, -0.2], [-0.4, 0.2], [-0.4, -0.2]])
    _step(env)
    for _ in range(5):
        _step(env, (0,))
    relative = quat_apply_inverse(
        data.root_quat_w.torch[:, None].expand(-1, 4, -1),
        env.term.application_points_w - data.root_link_pos_w.torch[:, None],
    )
    relative[:, 0, 0] += relative_advance
    data.root_quat_w.torch[:] = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    data.root_link_pos_w.torch[:, :2] += torch.tensor([0.3, -0.2])
    env.term.application_points_w.copy_(
        data.root_link_pos_w.torch[:, None]
        + quat_apply(
            data.root_quat_w.torch[:, None].expand(-1, 4, -1),
            relative,
        )
    )
    torch.testing.assert_close(_step(env), torch.full((3,), relative_advance / 0.08), atol=1e-5, rtol=1e-5)


def test_chassis_relative_reference_remains_previous_touchdown_not_liftoff():
    env = _env(min_air_time=0.1)
    _step(env)
    env.scene["robot"].data.root_link_pos_w.torch[:, 0] += 0.04
    assert not _step(env).any()
    for _ in range(5):
        _step(env, (0,))
    env.scene["robot"].data.root_link_pos_w.torch[:, 0] += 0.04
    _shift(env, 0.12)
    # .12 foot progress - .08 chassis progress = .04 relative progress since touchdown.
    torch.testing.assert_close(_step(env), torch.full((3,), 0.5), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "cfg_cls, provider",
    [
        (DeeproboticsM20WalkOnSandProxyEnvCfg, "sand_proxy"),
        (DeeproboticsM20WalkOnSandEnvCfg, "mpm_contact"),
    ],
)
def test_shared_task_binding_has_no_old_extraction_or_minimum_stride_parameters(cfg_cls, provider):
    cfg = cfg_cls()
    term = cfg.rewards.feet_forward_step
    assert term.func is ProxyFootForwardStepReward and term.weight > 0.0
    assert term.params == {
        "target_stride": term.params["target_stride"],
        "action_term_name": provider,
        "command_name": "base_velocity",
        "min_command_norm": 0.1,
        "repeat_decay": 1.0,
        "min_air_time": 0.1,
    }
    env = _env(cfg_cls=cfg_cls, target_stride=None, repeat_decay=None, min_air_time=None)
    _step(env)
    for _ in range(5):
        _step(env, (0,))
    _shift(env, term.params["target_stride"])
    result = _step(env)
    torch.testing.assert_close(result, torch.ones(env.num_envs), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        result * term.weight * env.step_dt, torch.full((env.num_envs,), term.weight * env.step_dt)
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_invalid_stride_scale_is_rejected(value):
    env = _env()
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg().rewards.feet_forward_step
    cfg.params["target_stride"] = value
    with pytest.raises(ValueError):
        ProxyFootForwardStepReward(cfg, env)


def _completed_step(env, foot, distance=0.08):
    _step(env, (foot,))
    _shift(env, distance, (foot,))
    return _step(env)


def test_same_foot_repeat_discount_is_progressive_and_other_feet_start_at_full_credit():
    env = _env(repeat_decay=1.0)
    _step(env)
    for i in range(4):
        torch.testing.assert_close(
            _completed_step(env, 3), torch.full((env.num_envs,), 1.0 / (1.0 + i)), atol=1e-5, rtol=1e-5
        )
    for foot in (0, 1, 2):
        torch.testing.assert_close(_completed_step(env, foot), torch.ones(env.num_envs), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(env.reward.participation_progress, torch.tensor([[1.0, 1.0, 1.0, 4.0]]).repeat(3, 1))


def test_balanced_foot_progress_restores_full_gain_without_prescribing_a_gait_order():
    env = _env(repeat_decay=1.0)
    _step(env)
    _completed_step(env, 3)
    _completed_step(env, 3)
    for foot in (2, 0, 1):
        _completed_step(env, foot)
        _completed_step(env, foot)
    torch.testing.assert_close(env.reward.participation_progress, torch.full((3, 4), 2.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(_completed_step(env, 3), torch.ones(3), atol=1e-5, rtol=1e-5)


def test_partial_stride_contributes_only_its_actual_progress_to_participation():
    env = _env(repeat_decay=1.0)
    _step(env)
    torch.testing.assert_close(_completed_step(env, 0, 0.04), torch.full((3,), 0.5), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(_completed_step(env, 0, 0.04), torch.full((3,), 1.0 / 3.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(env.reward.participation_progress[:, 0], torch.ones(3), atol=1e-5, rtol=1e-5)
    assert (env.reward.completed_steps[:, 0] == 2).all()
    torch.testing.assert_close(_completed_step(env, 1, 0.001), torch.full((3,), 0.0125), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(env.reward.participation_progress[:, 1], torch.full((3,), 0.0125), atol=1e-5, rtol=1e-5)


def test_backward_and_stationary_contacts_cannot_reset_the_repeat_discount():
    env = _env(repeat_decay=1.0)
    _step(env)
    _completed_step(env, 3)
    assert not _completed_step(env, 3, -0.04).any()
    assert not _completed_step(env, 3, 0.0).any()
    assert (env.reward.completed_steps[:, 3] == 1).all()
    torch.testing.assert_close(_completed_step(env, 3), torch.full((3,), 0.5), atol=1e-5, rtol=1e-5)


def test_forward_participation_reset_logs_per_foot_counts_and_clears_only_selected_worlds():
    env = _env(repeat_decay=1.0)
    env.extras = {"log": {}}
    _step(env)
    _completed_step(env, 3)
    env.reward.reset([1])
    assert env.extras["log"]["Metrics/forward_steps/hr_count"] == 1
    assert env.extras["log"]["Metrics/forward_steps/fl_count"] == 0
    assert env.extras["log"]["Metrics/forward_steps/hr_progress"] == pytest.approx(1.0, abs=1e-5)
    assert not env.reward.completed_steps[1].any() and not env.reward.participation_progress[1].any()
    _step(env)  # Establish the reset world's new touchdown anchor.
    torch.testing.assert_close(_completed_step(env, 3), torch.tensor([0.5, 1.0, 0.5]), atol=1e-5, rtol=1e-5)
    env.command[0] = 0
    _step(env)
    assert not env.reward.completed_steps[0].any() and not env.reward.participation_progress[0].any()


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_repeat_discount_is_rejected(value):
    env = _env()
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg().rewards.feet_forward_step
    cfg.params["repeat_decay"] = value
    with pytest.raises(ValueError, match="repeat_decay"):
        ProxyFootForwardStepReward(cfg, env)
