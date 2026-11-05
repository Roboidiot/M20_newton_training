"""Continuous unloading gates lift/stride credit and cannot be farmed by short contacts."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.math import quat_apply

from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import (
    ProxyCrawlGaitStatus,
    ProxyFootContactState,
    ProxyFootForwardStepReward,
)

HIP_STORAGE = ("hr_hipx", "fl_hipx", "hl_hipx", "fr_hipx")


def _env(cfg_cls=DeeproboticsM20WalkOnSandProxyEnvCfg, min_air_time=0.1, dt=0.02):
    cfg = cfg_cls()
    terms = [cfg.rewards.rotation_gait_status, cfg.rewards.feet_forward_step]
    # Preserve standalone legacy geometry/timer regressions; active replacements have trajectory tests.
    terms[1].func = ProxyFootForwardStepReward
    for term_cfg in terms:
        term_cfg.params.pop("max_progress_lead", None)
        term_cfg.params["repeat_decay"] = 1.0
    state = ProxyFootContactState(3, "cpu")
    points = torch.zeros(3, 4, 3)
    points[..., 2] = -0.2  # Buried swings remain valid once their duration is sufficient.
    term = SimpleNamespace(contact_state=state, application_points_w=points)
    command = torch.tensor([[0.2, 0.0, 0.0]]).repeat(3, 1)
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=dt,
        term=term,
        command=command,
        command_manager=SimpleNamespace(get_command=lambda name: command),
        action_manager=SimpleNamespace(get_term=lambda name: term),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(3, 1)),
                    root_link_pos_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.6]]).repeat(3, 1)),
                    body_link_pos_w=SimpleNamespace(torch=torch.zeros(3, 4, 3)),
                ),
                find_bodies=lambda names, preserve_order: ([HIP_STORAGE.index(name) for name in names], list(names)),
            )
        },
    )
    env.manual_hips = False
    _sync_hips(env)
    for reward_cfg in terms:
        reward_cfg.params["min_air_time"] = min_air_time
    env.cfgs = terms
    env.rewards = [reward_cfg.func(reward_cfg, env) for reward_cfg in terms]
    return env


def _sync_hips(env):
    if env.manual_hips:
        return
    data = env.scene["robot"].data
    offsets = torch.tensor(
        [[-0.3141, -0.0685, 0.0], [0.3141, 0.0685, 0.0], [-0.3141, 0.0685, 0.0], [0.3141, -0.0685, 0.0]]
    )
    data.body_link_pos_w.torch = data.root_link_pos_w.torch[:, None] + quat_apply(
        data.root_quat_w.torch[:, None].expand(-1, 4, -1),
        offsets[None].expand(env.num_envs, -1, -1),
    )


def _evaluate(env):
    _sync_hips(env)
    scores = [reward(env, **cfg.params) for reward, cfg in zip(env.rewards, env.cfgs, strict=True)]
    assert all(torch.isfinite(score).all() for score in scores)
    return scores


def _sample(env, unloaded=(), z=None, dx=0.0, feet=(0,)):
    if z is not None:
        env.term.application_points_w[:, list(feet), 2] = z
    env.term.application_points_w[:, list(feet), 0] += dx
    force = torch.zeros(3, 4, 3)
    force[..., 2] = 50.0
    force[:, list(unloaded), 2] = 0.0
    env.term.contact_state.begin_step()
    env.term.contact_state.update(force, env.step_dt, 5.0)
    return _evaluate(env)


@pytest.mark.parametrize("cfg_cls", [DeeproboticsM20WalkOnSandProxyEnvCfg, DeeproboticsM20WalkOnSandEnvCfg])
@pytest.mark.parametrize("air_steps", [1, 2, 3, 4])
def test_short_lift_and_forward_landings_cannot_score_or_increase_participation(cfg_cls, air_steps):
    env = _env(cfg_cls)
    _sample(env)
    for _ in range(air_steps):
        assert all(not score.any() for score in _sample(env, (0,), z=-0.05))
    scores = _sample(env, z=-0.2, dx=env.cfgs[1].params["target_stride"])
    assert all(not score.any() for score in scores)
    for reward in env.rewards:
        assert not reward.completed_steps.any()
        assert not reward.participation_progress.any()
        if isinstance(reward, ProxyCrawlGaitStatus):
            assert not reward.peak_progress.any()


@pytest.mark.parametrize("dt,minimum_samples", [(0.01, 10), (0.02, 5), (0.025, 4)])
def test_exact_threshold_qualifies_at_control_rate_and_landing_does_not_complete_the_timer(dt, minimum_samples):
    env = _env(dt=dt)
    _sample(env)
    for _ in range(minimum_samples - 1):
        assert all(not score.any() for score in _sample(env, (0,), z=-0.05))
    # A landing interval is not itself unloading. This first stride is rejected.
    assert all(not score.any() for score in _sample(env, z=-0.2, dx=env.cfgs[1].params["target_stride"]))
    for _ in range(minimum_samples - 1):
        assert all(not score.any() for score in _sample(env, (0,), z=-0.05))
    lift, stride = _sample(env, (0,), z=-0.05)
    torch.testing.assert_close(lift, torch.full((3,), 0.5))
    assert not stride.any()
    # Re-evaluating rewards cannot advance time or award the same height twice.
    assert all(not score.any() for score in _evaluate(env))
    lift, stride = _sample(env, z=-0.2, dx=env.cfgs[1].params["target_stride"])
    assert not lift.any()
    torch.testing.assert_close(stride, torch.ones(3), atol=1e-6, rtol=1e-6)
    assert (env.rewards[0].completed_steps[:, 0] == 1).all()
    torch.testing.assert_close(env.rewards[0].participation_progress[:, 0], torch.full((3,), 0.5))
    assert (env.rewards[1].completed_steps[:, 0] == 1).all()


def test_many_short_bursts_cannot_bank_height_displacement_or_repeat_discount_progress():
    env = _env()
    _sample(env)
    for _ in range(20):
        assert all(not score.any() for score in _sample(env, (0,), z=-0.05))
        assert all(not score.any() for score in _sample(env, z=-0.2, dx=0.03))
    for reward in env.rewards:
        assert not reward.participation_progress.any()
        assert not reward.completed_steps.any()
    for i in range(5):
        lift, stride = _sample(env, (0,), z=-0.05)
        torch.testing.assert_close(lift, torch.full((3,), 0.5 if i == 4 else 0.0))
        assert not stride.any()
    # Only displacement from the last short landing to this valid landing counts.
    _, stride = _sample(env, z=-0.2, dx=env.cfgs[1].params["target_stride"] * 0.25)
    torch.testing.assert_close(stride, torch.full((3,), 0.25), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        env.rewards[1].participation_progress[:, 0], torch.full((3,), 0.25), atol=1e-6, rtol=1e-6
    )


def test_early_height_peak_that_has_disappeared_before_minimum_duration_is_not_paid_later():
    env = _env()
    _sample(env)
    for _ in range(4):
        assert all(not score.any() for score in _sample(env, (0,), z=0.1))
    assert all(not score.any() for score in _sample(env, (0,), z=-0.2))
    assert all(not score.any() for score in _sample(env, z=-0.2))
    assert not env.rewards[0].completed_steps.any()
    assert not env.rewards[0].participation_progress.any()


def test_repeated_evaluations_of_one_unloaded_sample_do_not_advance_timing():
    env = _env()
    _sample(env)
    _sample(env, (0,), z=-0.05)
    for _ in range(20):
        assert all(not score.any() for score in _evaluate(env))
    torch.testing.assert_close(env.term.contact_state.current_air_time[:, 0], torch.full((3,), 0.02))
    assert all(not score.any() for score in _sample(env, z=-0.2, dx=0.2))


def test_recontact_restarts_the_continuous_unloading_requirement():
    env = _env()
    _sample(env)
    for samples in (3, 2, 4):
        for _ in range(samples):
            assert all(not score.any() for score in _sample(env, (0,), z=-0.05))
        assert all(not score.any() for score in _sample(env, z=-0.2, dx=0.05))
    assert not env.rewards[0].completed_steps.any()
    assert not env.rewards[1].completed_steps.any()


def test_partial_reward_reset_during_unloading_cannot_inherit_a_prior_swing():
    env = _env()
    _sample(env)
    for _ in range(4):
        _sample(env, (0,), z=-0.05)
    for reward in env.rewards:
        reward.reset([1])
    lift, _ = _sample(env, (0,), z=-0.05)
    torch.testing.assert_close(lift, torch.tensor([0.5, 0.0, 0.5]))
    _, stride = _sample(env, z=-0.2, dx=env.cfgs[1].params["target_stride"])
    torch.testing.assert_close(stride, torch.tensor([1.0, 0.0, 1.0]), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("cfg_cls", [DeeproboticsM20WalkOnSandProxyEnvCfg, DeeproboticsM20WalkOnSandEnvCfg])
@pytest.mark.parametrize("translation", [-0.2, 0.2])
def test_whole_robot_vertical_translation_is_not_lift(cfg_cls, translation):
    env = _env(cfg_cls)
    _sample(env)
    env.scene["robot"].data.root_link_pos_w.torch[:, 2] += translation
    # All four feet move with the chassis; foot-minus-body Z remains constant.
    env.term.application_points_w[..., 2] += translation
    for _ in range(5):
        assert all(not score.any() for score in _sample(env, (0,)))
    _sample(env)
    assert not env.rewards[0].completed_steps.any()
    assert not env.rewards[0].participation_progress.any()


@pytest.mark.parametrize(
    "foot_delta,body_delta,expected",
    [
        (0.15, 0.0, 0.5),
        (0.15, 0.06, 0.3),
        (0.06, -0.06, 0.4),
        (0.0, -0.15, 0.5),
        (0.15, 0.30, 0.0),
        (0.0, 0.15, 0.0),
    ],
)
def test_lift_is_change_in_world_z_foot_minus_body_difference(foot_delta, body_delta, expected):
    env = _env()
    _sample(env)
    env.scene["robot"].data.root_link_pos_w.torch[:, 2] += body_delta
    for sample in range(5):
        lift, _ = _sample(env, (0,), z=-0.2 + foot_delta)
        torch.testing.assert_close(lift, torch.full((3,), expected if sample == 4 else 0.0), atol=1e-6, rtol=1e-6)
    assert not _evaluate(env)[0].any()  # Holding the relative height adds no repeated credit.


def test_reference_tracks_body_and_foot_z_at_the_latest_support_sample():
    env = _env()
    _sample(env)
    body = env.scene["robot"].data.root_link_pos_w.torch
    body[:, 2] = 0.7
    _sample(env, z=-0.18)
    # Latest support difference is -.18 - .70 = -.88, independently per foot.
    torch.testing.assert_close(env.rewards[0].support_height[:, 0], torch.full((3,), -0.88))
    body[:, 2] = 0.8
    for sample in range(5):
        lift, _ = _sample(env, (0,), z=-0.03)
        # Current difference = -.83; relative lift = .05, not world-foot rise .15.
        torch.testing.assert_close(lift, torch.full((3,), (1.0 / 6.0) if sample == 4 else 0.0), atol=1e-6, rtol=1e-6)


def test_each_environment_uses_its_own_chassis_z_and_relative_reference():
    env = _env()
    body = env.scene["robot"].data.root_link_pos_w.torch
    body[:, 2] = torch.tensor([0.6, 1.6, -0.4])
    _sample(env)
    body[:, 2] += torch.tensor([0.0, 0.15, -0.15])
    for sample in range(5):
        lift, _ = _sample(env, (0,), z=-0.05)
        torch.testing.assert_close(
            lift, torch.tensor([0.5, 0.0, 1.0]) if sample == 4 else torch.zeros(3), atol=1e-6, rtol=1e-6
        )


@pytest.mark.parametrize("cfg_cls", [DeeproboticsM20WalkOnSandProxyEnvCfg, DeeproboticsM20WalkOnSandEnvCfg])
@pytest.mark.parametrize("foot", range(4))
def test_each_foot_uses_its_own_hipx_origin_with_scrambled_articulation_storage(cfg_cls, foot):
    env = _env(cfg_cls)
    _sample(env)
    env.manual_hips = True
    lift_reward, forward_reward = env.rewards
    expected_ids = [HIP_STORAGE.index(name) for name in ("fl_hipx", "fr_hipx", "hl_hipx", "hr_hipx")]
    assert lift_reward.hip_body_ids == forward_reward.hip_body_ids == expected_ids
    hip = env.scene["robot"].data.body_link_pos_w.torch[:, expected_ids[foot]]
    hip[:, 2] -= 0.15
    hip[:, 0] -= env.cfgs[1].params["target_stride"]
    for sample in range(5):
        lift, stride = _sample(env, (foot,))  # Foot and root stay fixed; only this HipX origin moved.
        torch.testing.assert_close(lift, torch.full((3,), 0.5 if sample == 4 else 0.0), atol=1e-6, rtol=1e-6)
        assert not stride.any()
    _, stride = _sample(env)
    torch.testing.assert_close(stride, torch.ones(3), atol=1e-6, rtol=1e-6)
    for reward in env.rewards:
        assert (reward.completed_steps[:, foot] == 1).all()
        assert not reward.completed_steps[:, [i for i in range(4) if i != foot]].any()


def test_chassis_origin_translation_alone_does_not_replace_the_measured_hipx_reference():
    env = _env()
    _sample(env)
    env.manual_hips = True
    # Hip poses are supplied independently, so neither reward may substitute root position.
    env.scene["robot"].data.root_link_pos_w.torch += torch.tensor([-0.2, 0.1, -0.2])
    for _ in range(5):
        assert all(not score.any() for score in _sample(env, (0,)))
    assert all(not score.any() for score in _sample(env))


def test_a_different_legs_hip_motion_cannot_create_this_legs_progress():
    env = _env()
    _sample(env)
    env.manual_hips = True
    hip_id = HIP_STORAGE.index("hr_hipx")
    env.scene["robot"].data.body_link_pos_w.torch[:, hip_id] += torch.tensor([-0.2, 0.0, -0.2])
    for _ in range(5):
        assert all(not score.any() for score in _sample(env, (0,)))
    assert all(not score.any() for score in _sample(env))


def test_common_foot_and_own_hip_translation_cancels_with_a_fixed_chassis_origin():
    env = _env()
    _sample(env)
    env.manual_hips = True
    hip_id = HIP_STORAGE.index("fl_hipx")
    env.scene["robot"].data.body_link_pos_w.torch[:, hip_id] += torch.tensor([0.1, 0.0, 0.15])
    for sample in range(5):
        scores = _sample(env, (0,), z=-0.05, dx=0.1 if sample == 0 else 0.0)
        for score in scores:
            torch.testing.assert_close(score, torch.zeros(3), atol=1e-5, rtol=0.0)
    for score in _sample(env, z=-0.05):
        torch.testing.assert_close(score, torch.zeros(3), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("cls", [ProxyCrawlGaitStatus, ProxyFootForwardStepReward])
@pytest.mark.parametrize(
    "ids,names",
    [
        ([0, 1, 2, 3], ["fr_hipx", "fl_hipx", "hl_hipx", "hr_hipx"]),
        ([0, 0, 2, 3], ["fl_hipx", "fr_hipx", "hl_hipx", "hr_hipx"]),
        ([0, 1, 2], ["fl_hipx", "fr_hipx", "hl_hipx"]),
    ],
)
def test_missing_or_misordered_hipx_mapping_fails_during_reward_construction(cls, ids, names):
    env = _env()
    env.scene["robot"].find_bodies = lambda requested, preserve_order: (ids, names)
    cfg = env.cfgs[0 if cls is ProxyCrawlGaitStatus else 1]
    with pytest.raises(ValueError, match="ordered HipX"):
        cls(cfg, env)


def test_deployed_m20_usd_hipx_child_origins_are_the_joint_pivots():
    from pathlib import Path

    from pxr import Usd, UsdPhysics

    path = Path(__file__).parents[1] / "deep_robotics_model/M20/M20_usd/M20.usd"
    stage = Usd.Stage.Open(str(path))
    for name in ("fl_hipx", "fr_hipx", "hl_hipx", "hr_hipx"):
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"/M20/joints/{name}_joint"))
        assert joint
        assert tuple(joint.GetLocalPos1Attr().Get()) == (0.0, 0.0, 0.0)
        assert str(joint.GetBody1Rel().GetTargets()[0]) == f"/M20/{name}"


def test_duration_filter_can_be_disabled_explicitly_for_legacy_experiments():
    env = _env(min_air_time=0.0)
    _sample(env)
    lift, _ = _sample(env, (0,), z=-0.05)
    torch.testing.assert_close(lift, torch.full((3,), 0.5))
    _, stride = _sample(env, z=-0.2, dx=env.cfgs[1].params["target_stride"])
    torch.testing.assert_close(stride, torch.ones(3), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("cls", [ProxyCrawlGaitStatus, ProxyFootForwardStepReward])
def test_invalid_duration_is_rejected_at_reward_construction(cls, value):
    env = _env()
    cfg = env.cfgs[0 if cls is ProxyCrawlGaitStatus else 1]
    cfg.params["min_air_time"] = value
    with pytest.raises(ValueError, match="min_air_time"):
        cls(cfg, env)
