"""Control-rate contact events, all-command diagonal gait rewards and partial resets."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import to_dict

from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import (
    DeeproboticsM20WalkOnSandProxyEnvCfg,
    FootProxyRewardsCfg,
)
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import (
    ProxyCrawlGaitStatus,
    ProxyCrawlGaitSymmetry,
    ProxyFootContactState,
    ProxyFootLiftConsistency,
    ProxyRotationGaitStatus,
    ProxyRotationGaitSymmetry,
    proxy_feet_air_time_command_scaled,
    proxy_feet_slide_ang_z_cmd,
    proxy_hipx_symmetry_l2,
)


def _env():
    commands = torch.tensor([[0.2, 0, 0], [0, -0.15, 0], [0, 0, -0.3], [0, 0.15, 0.3], [-0.2, 0, 0]])
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg()
    # These tests exercise the retained stateless air-time formula independently.
    cfg.rewards.feet_air_time.params.pop("max_progress_lead", None)
    cfg.rewards.feet_air_time.params.pop("participation_time", None)
    # Height consistency is disabled in the task; exercise its reusable implementation independently.
    cfg.rewards.feet_lift_consistency = FootProxyRewardsCfg().feet_lift_consistency
    cfg.rewards.feet_lift_consistency.params.update(command_name="base_velocity", min_command_norm=0.1)
    # The task now uses crawl helpers; retain independent regression coverage of the diagonal helpers.
    cfg.rewards.rotation_gait_status = RewardTermCfg(
        func=ProxyRotationGaitStatus, weight=100.0,
        params={
            "command_name": "base_velocity", "min_command_norm": 0.1,
            "action_term_name": "sand_proxy", "target_height": 0.30,
            "group_a_body_names": ["fl_wheel", "hr_wheel"],
            "group_b_body_names": ["fr_wheel", "hl_wheel"],
        },
    )
    cfg.rewards.rotation_gait_symmetry = RewardTermCfg(
        func=ProxyRotationGaitSymmetry, weight=0.5,
        params={
            "command_name": "base_velocity", "min_command_norm": 0.1,
            "action_term_name": "sand_proxy", "window_s": 4.0,
            "group_a_body_names": ["fl_wheel", "hr_wheel"],
            "group_b_body_names": ["fr_wheel", "hl_wheel"],
        },
    )
    n = len(commands)
    state = ProxyFootContactState(n, "cpu")
    # Intentionally permute articulation body slots relative to ordered proxy feet.
    term = SimpleNamespace(
        cfg=cfg.actions.sand_proxy,
        body_ids=[2, 0, 3, 1],
        contact_state=state,
        force_history_w=torch.zeros((n, 3, 4, 3)),
        application_points_w=torch.zeros((n, 4, 3)),
    )

    def wrap(value):
        return SimpleNamespace(torch=value)

    data = SimpleNamespace(
        body_pos_w=wrap(torch.zeros((n, 4, 3))),
        body_lin_vel_w=wrap(torch.zeros((n, 4, 3))),
        root_lin_vel_w=wrap(torch.zeros((n, 3))),
        root_quat_w=wrap(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1)),
        root_link_pos_w=wrap(torch.tensor([[0.0, 0.0, 0.6]]).repeat(n, 1)),
        body_link_pos_w=wrap(torch.tensor([
            [-.3141, -.0685, .6], [.3141, .0685, .6], [-.3141, .0685, .6], [.3141, -.0685, .6],
        ]).repeat(n, 1, 1)),
        joint_pos=wrap(torch.zeros((n, 16))),
    )
    return SimpleNamespace(
        cfg=cfg,
        device="cpu",
        num_envs=n,
        step_dt=0.02,
        action_manager=SimpleNamespace(get_term=lambda name: term),
        command_manager=SimpleNamespace(get_command=lambda name: commands),
        scene={"robot": SimpleNamespace(
            data=data,
            find_bodies=lambda names, preserve_order: (
                [("hr_hipx", "fl_hipx", "hl_hipx", "fr_hipx").index(name) for name in names], list(names),
            ),
        )},
        commands=commands,
        term=term,
    )


def _force(n, value):
    force = torch.zeros((n, 4, 3))
    force[..., 2] = value
    return force


def test_airtime_records_control_events_scales_commands_and_ignores_initial_drop():
    env = _env()
    state = env.term.contact_state
    params = env.cfg.rewards.feet_air_time.params
    force = _force(env.num_envs, 50.0)
    for _ in range(25):
        state.update(force * 0, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    assert not proxy_feet_air_time_command_scaled(env, **params).any()
    state.begin_step()
    # One foot completes 0.30 s of sampled air time, including the landing interval.
    airborne = force.clone()
    airborne[:, 0] = 0
    for _ in range(14):
        state.update(airborne, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    expected = torch.tensor([0.2, 0.4, 0.6, 0.8, 0.2])
    for _ in range(2):  # reward evaluation must not advance timers or consume events
        torch.testing.assert_close(proxy_feet_air_time_command_scaled(env, **params), expected)
    assert state.vertical_contact[:, 0].all()
    state.begin_step()
    state.update(airborne, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    assert not proxy_feet_air_time_command_scaled(env, **params).any()
    assert (state.current_air_time[:, 0] > 0).all()


def test_short_repeated_touches_are_penalized_and_partial_reset_clears_only_one_env():
    env = _env()
    state = env.term.contact_state
    force = _force(env.num_envs, 50.0)
    state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    state.begin_step()
    airborne = force.clone()
    airborne[:, 0] = 0
    params = env.cfg.rewards.feet_air_time.params
    for _ in range(2):
        state.begin_step()
        state.update(airborne, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
        state.begin_step()
        state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
        torch.testing.assert_close(
            proxy_feet_air_time_command_scaled(env, **params), torch.tensor([-0.06, -0.12, -0.18, -0.24, -0.06])
        )
    state.reset([0])
    assert not state.contact[0].any() and not state.has_contact[0].any()
    assert not state.vertical_contact[0].any()
    assert not state.touchdown[0].any()
    assert state.touchdown[1].sum() == 1
    assert proxy_feet_air_time_command_scaled(env, **params)[0] == 0


@pytest.mark.parametrize("airborne_steps", [3, 4, 5])
def test_airtime_disables_positive_and_negative_credit_for_zero_and_small_commands(airborne_steps):
    env = _env()
    env.commands[:] = torch.tensor([[0.05, 0, 0], [0, -0.05, 0], [0, 0, -0.05], [0.03, 0.03, 0.05], [0, 0, 0]])
    state = env.term.contact_state
    force = _force(env.num_envs, 50.0)
    state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    airborne = force.clone()
    airborne[:, 0] = 0
    for _ in range(airborne_steps):
        state.begin_step()
        state.update(airborne, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
        assert not proxy_feet_air_time_command_scaled(env, **env.cfg.rewards.feet_air_time.params).any()
    state.begin_step()
    state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    torch.testing.assert_close(
        proxy_feet_air_time_command_scaled(env, **env.cfg.rewards.feet_air_time.params),
        torch.zeros(env.num_envs),
    )
    state.begin_step()
    state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    assert not proxy_feet_air_time_command_scaled(env, **env.cfg.rewards.feet_air_time.params).any()


def test_airtime_uses_signed_vertical_support_despite_large_drag_and_pullout_forces():
    env = _env()
    state = env.term.contact_state
    force = _force(env.num_envs, 50.0)
    state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    state.begin_step()
    unloaded = force.clone()
    unloaded[:, 0] = torch.tensor([100.0, -80.0, -50.0])  # downward pull-out force
    unloaded[:, 1] = torch.tensor([100.0, 0.0, 0.0])  # horizontal drag only
    threshold = env.term.cfg.contact_force_threshold
    unloaded[:, 2] = torch.tensor([100.0, 0.0, threshold])  # exactly the support threshold
    unloaded[:, 3] = torch.tensor([0.0, 0.0, threshold + 0.001])  # still supported
    for _ in range(14):
        state.update(unloaded, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    assert not state.vertical_contact[:, :3].any()
    assert state.vertical_contact[:, 3].all()
    torch.testing.assert_close(state.current_air_time[:, :3], torch.full((env.num_envs, 3), 0.28))
    assert not state.current_air_time[:, 3].any()
    assert not state.touchdown.any()
    assert state.contact.all()  # full-force diagnostic remains separate from vertical support
    state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    assert state.vertical_contact.all()
    assert state.touchdown.sum() == env.num_envs * 3
    torch.testing.assert_close(
        proxy_feet_air_time_command_scaled(env, **env.cfg.rewards.feet_air_time.params),
        torch.tensor([0.6, 1.2, 1.8, 2.4, 0.6]),
    )


def test_long_airtime_is_capped_and_one_new_policy_window_does_not_erase_pending_swing():
    env = _env()
    state = env.term.contact_state
    force = _force(env.num_envs, 50.0)
    state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    airborne = force.clone()
    airborne[:, 0] = 0
    for _ in range(60):
        state.begin_step()
        state.update(airborne, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    state.begin_step()
    state.update(force, dt=0.02, threshold=env.term.cfg.contact_force_threshold)
    torch.testing.assert_close(
        proxy_feet_air_time_command_scaled(env, **env.cfg.rewards.feet_air_time.params),
        torch.tensor([0.9, 1.8, 2.7, 3.6, 0.9]),
    )


def _support(env, feet):
    force = _force(env.num_envs, -50.0)
    force[..., 0] = 100.0  # full-force magnitude must not be mistaken for upward support
    force[:, feet, 2] = 50.0
    env.term.contact_state.update(force, dt=env.step_dt, threshold=env.term.cfg.contact_force_threshold)


def test_rotation_lift_rewards_both_feet_relative_to_support_and_only_new_progress():
    env = _env()
    cfg = env.cfg.rewards.rotation_gait_status
    reward = ProxyRotationGaitStatus(cfg, env)
    target = cfg.params["target_height"]
    height = env.term.application_points_w[..., 2]
    height[:] = -0.15  # swing may begin below the reference sand surface
    _support(env, [0, 1, 2, 3])
    assert not reward(env, **cfg.params).any()
    # The lower foot controls progress, even if its paired foot exceeds the target.
    height[:, 1] += 2 * target
    height[:, 2] += 0.2 * target
    _support(env, [0, 3])
    torch.testing.assert_close(reward(env, **cfg.params), torch.full((env.num_envs,), 0.2))
    assert env.term.contact_state.contact.all()
    assert not reward(env, **cfg.params).any()  # hovering is not rewarded again
    height[:, 2] -= 0.2 * target
    assert not reward(env, **cfg.params).any()
    height[:, 2] += 0.2 * target
    assert not reward(env, **cfg.params).any()  # revisiting an already credited height
    height[:, 2] += 0.4 * target
    torch.testing.assert_close(reward(env, **cfg.params), torch.full((env.num_envs,), 0.4))
    height[:, 2] += 2 * target
    torch.testing.assert_close(reward(env, **cfg.params), torch.full((env.num_envs,), 0.4))
    assert not reward(env, **cfg.params).any()  # capped at the configured target per foot
    _support(env, [0, 1, 2, 3])
    assert not reward(env, **cfg.params).any()
    # A new swing starts at the new support height, not at an old world-Z anchor.
    _support(env, [0, 3])
    assert not reward(env, **cfg.params).any()
    height[:, [1, 2]] += 0.5 * target
    torch.testing.assert_close(reward(env, **cfg.params), torch.full((env.num_envs,), 0.5))


@pytest.mark.parametrize("feet", [[0, 1, 2, 3], [0, 1], [0], []])
def test_rotation_lift_rejects_mixed_support_all_support_and_no_support(feet):
    env = _env()
    cfg = env.cfg.rewards.rotation_gait_status
    reward = ProxyRotationGaitStatus(cfg, env)
    _support(env, [0, 1, 2, 3])
    reward(env, **cfg.params)
    env.term.application_points_w[:, [1, 2], 2] += 0.10
    # The old implementation rewarded all four supported with different diagonal heights.
    env.scene["robot"].data.body_pos_w.torch[:, [0, 3], 2] = 0.10
    _support(env, feet)
    assert not reward(env, **cfg.params).any()


def test_rotation_lift_requires_support_after_reset_and_tracks_opposite_swing():
    env = _env()
    cfg = env.cfg.rewards.rotation_gait_status
    reward = ProxyRotationGaitStatus(cfg, env)
    _support(env, [0, 3])  # no prior support for the airborne diagonal after initialization
    env.term.application_points_w[:, [1, 2], 2] += 0.10
    assert not reward(env, **cfg.params).any()
    _support(env, [0, 1, 2, 3])
    reward(env, **cfg.params)
    _support(env, [1, 2])
    env.term.application_points_w[:, [0, 3], 2] += 0.02
    torch.testing.assert_close(
        reward(env, **cfg.params), torch.full((env.num_envs,), 0.02 / cfg.params["target_height"])
    )
    other_anchor = reward.support_height[3].clone()
    reward.reset([2])
    assert not reward.support_seen[2].any() and not reward.peak_progress[2].any()
    torch.testing.assert_close(reward.support_height[3], other_anchor)
    env.term.application_points_w[:, [0, 3], 2] += 0.02
    torch.testing.assert_close(
        reward(env, **cfg.params), torch.tensor([1, 1, 0, 1, 1]) * (0.02 / cfg.params["target_height"])
    )
    reward.reset()
    assert not reward.support_seen.any() and not reward.peak_progress.any()


def test_rotation_lift_preserves_support_and_credited_progress_across_command_changes():
    env = _env()
    cfg = env.cfg.rewards.rotation_gait_status
    reward = ProxyRotationGaitStatus(cfg, env)
    _support(env, [0, 1, 2, 3])
    reward(env, **cfg.params)
    _support(env, [0, 3])
    env.term.application_points_w[:, [1, 2], 2] += 0.05
    torch.testing.assert_close(
        reward(env, **cfg.params), torch.full((env.num_envs,), 0.05 / cfg.params["target_height"])
    )
    anchors = reward.support_height.clone()
    env.commands[:] = torch.tensor([[0, 0.15, 0], [-0.4, 0, 0], [0.1, 0, 0], [0, 0, -0.3], [0.4, 0.15, 0.5]])
    assert not reward(env, **cfg.params).any()  # changing direction cannot re-credit the same height
    torch.testing.assert_close(reward.support_height, anchors)
    assert reward.support_seen.all()
    env.term.application_points_w[:, [1, 2], 2] += 0.05
    torch.testing.assert_close(
        reward(env, **cfg.params), torch.full((env.num_envs,), 0.05 / cfg.params["target_height"])
    )


def test_turn_sliding_uses_force_history_and_preserves_body_frame_definition():
    env = _env()
    env.commands[:] = torch.tensor([0, 0, 0.3])
    env.commands[1] = torch.tensor([0.2, 0, 0.3])  # translation suppresses pure-turn penalty
    env.commands[2] = 0
    data = env.scene["robot"].data
    data.root_lin_vel_w.torch[:, 0] = 2.0
    data.body_lin_vel_w.torch[:, :, 0] = 2.0
    data.body_lin_vel_w.torch[:, 2, 1] = 0.5  # FL relative horizontal velocity
    env.term.force_history_w[:, 2, 0, 2] = 50.0
    assert env.cfg.rewards.feet_slide_ang_z_cmd is None
    # Exercise the reusable function directly; the proxy task keeps this term disabled.
    params = {"command_name": "base_velocity", "asset_cfg": SimpleNamespace(name="robot")}
    torch.testing.assert_close(proxy_feet_slide_ang_z_cmd(env, **params), torch.tensor([0.5, 0, 0, 0.5, 0.5]))
    env.term.force_history_w.zero_()
    assert not proxy_feet_slide_ang_z_cmd(env, **params).any()


def test_symmetry_scores_real_opposing_diagonals_and_partial_reset():
    env = _env()
    cfg = env.cfg.rewards.rotation_gait_symmetry
    reward = ProxyRotationGaitSymmetry(cfg, env)
    before = to_dict(env.cfg)
    assert reward.buffer_size == round(cfg.params["window_s"] / env.step_dt)
    for i in range(reward.buffer_size):
        _support(env, [0, 3] if i % 2 == 0 else [1, 2])
        assert env.term.contact_state.contact.all()
        result = reward(env, **cfg.params)
    torch.testing.assert_close(result, torch.ones(env.num_envs))
    torch.testing.assert_close(reward.samples, torch.full((env.num_envs,), float(reward.buffer_size)))
    other = reward.pattern_buffer[3].clone()
    reward.reset([2])
    assert reward.samples[2] == 0 and not reward.pattern_buffer[2].any()
    torch.testing.assert_close(reward.pattern_buffer[3], other)
    result = reward(env, **cfg.params)
    assert result[2] == 0 and result[3] > 0.9
    assert to_dict(env.cfg) == before
    reward.reset()
    assert not reward.samples.any() and not reward.pattern_buffer.any()


@pytest.mark.parametrize("feet", [[0, 1], [0, 3], [0, 1, 2, 3], []])
def test_symmetry_rejects_fixed_front_support_single_diagonal_and_all_or_no_support(feet):
    env = _env()
    cfg = env.cfg.rewards.rotation_gait_symmetry
    reward = ProxyRotationGaitSymmetry(cfg, env)
    for _ in range(60):
        _support(env, feet)
        assert not reward(env, **cfg.params).any()


def test_symmetry_rejects_all_feet_moving_in_phase_despite_half_contact_duty():
    env = _env()
    cfg = env.cfg.rewards.rotation_gait_symmetry
    reward = ProxyRotationGaitSymmetry(cfg, env)
    for i in range(50):
        _support(env, [0, 1, 2, 3] if i % 2 == 0 else [])
        assert not reward(env, **cfg.params).any()


def test_symmetry_occupancy_formula_warmup_and_old_samples_expire():
    env = _env()
    cfg = env.cfg.rewards.rotation_gait_symmetry
    cfg.params["window_s"] = 0.08  # four 50 Hz samples
    reward = ProxyRotationGaitSymmetry(cfg, env)
    # A and B each occupy 1/4 of the window; the rest is double support.
    for feet in ([0, 3], [1, 2], [0, 1, 2, 3], [0, 1, 2, 3]):
        _support(env, feet)
        result = reward(env, **cfg.params)
    torch.testing.assert_close(result, torch.full((env.num_envs,), 0.5))
    _support(env, [0, 1, 2, 3])
    assert not reward(env, **cfg.params).any()  # A's only sample has expired


def test_symmetry_preserves_window_for_straight_sideways_and_reversed_active_commands():
    env = _env()
    env.commands[:] = torch.tensor([[0, 0, 0.1], [0.5, 0, 0.3], [0, 0, -0.3], [0.2, 0, 0.3], [0.1, 0, 0]])
    cfg = env.cfg.rewards.rotation_gait_symmetry
    reward = ProxyRotationGaitSymmetry(cfg, env)
    for i in range(reward.buffer_size):
        _support(env, [0, 3] if i % 2 == 0 else [1, 2])
        result = reward(env, **cfg.params)
    torch.testing.assert_close(result, torch.ones(env.num_envs))
    env.commands[:] = torch.tensor([[0.4, 0, 0], [0, -0.15, 0], [0.1, 0, 0], [0, 0, -0.3], [0, 0, 0.5]])
    _support(env, [0, 3])
    torch.testing.assert_close(reward(env, **cfg.params), torch.ones(env.num_envs))
    torch.testing.assert_close(reward.samples, torch.full((env.num_envs,), float(reward.buffer_size)))
    _support(env, [1, 2])
    torch.testing.assert_close(reward(env, **cfg.params), torch.ones(env.num_envs))


def test_hipx_mirror_uses_opposite_angles_and_ordered_joint_slots_with_maneuver_relaxation():
    env = _env()
    cfg = env.cfg.rewards.hipx_symmetry
    cfg.params["asset_cfg"].joint_ids = [7, 2, 12, 5]
    positions = env.scene["robot"].data.joint_pos.torch
    positions[:, [7, 2, 12, 5]] = torch.tensor(
        [[0.2, -0.2, 0.1, -0.1], [0.2, 0.2, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]]
    )
    torch.testing.assert_close(proxy_hipx_symmetry_l2(env, **cfg.params), torch.tensor([0.0, 0.025, 0.04, 0.04, 0.16]))
    env.commands.zero_()
    torch.testing.assert_close(proxy_hipx_symmetry_l2(env, **cfg.params), torch.tensor([0.0, 0.1, 0.16, 0.16, 0.16]))


def _completed_swing(env, reward, params, swing, lifts, steps=4, flight=False):
    """Four unloaded samples and a landing make 0.1 s under the task convention."""
    height = env.term.application_points_w[..., 2]
    baseline = height[:, swing].clone()
    support = [i for i in range(4) if i not in swing]
    for i in range(steps):
        _support(env, [] if flight and i == 1 else support)
        height[:, swing] = baseline + torch.as_tensor(lifts)
        assert not reward(env, **params).any()  # never compare a stance foot to a swinging foot
    height[:, swing] = baseline
    _support(env, [0, 1, 2, 3])
    return reward(env, **params)


@pytest.mark.parametrize(
    ("lifts_a", "lifts_b", "expected"),
    [
        ([0.15, 0.15], [0.15, 0.15], 0.0),
        ([0.25, 0.25], [0.05, 0.05], 0.64),
        ([0.25, 0.05], [0.25, 0.05], 0.64),
        ([1.0, 1.0], [0.0, 0.0], 4.0),
    ],
)
def test_lift_consistency_compares_four_relative_peaks_and_preserves_old_cost_scale(lifts_a, lifts_b, expected):
    env = _env()
    cfg = env.cfg.rewards.feet_lift_consistency
    reward = ProxyFootLiftConsistency(cfg, env)
    env.term.application_points_w[..., 2] = torch.tensor([-0.25, -0.05, -0.1, -0.2])
    _support(env, [0, 1, 2, 3])
    assert not reward(env, **cfg.params).any()
    assert not _completed_swing(env, reward, cfg.params, [0, 3], lifts_a).any()
    env.commands *= -1  # pending heights survive command changes
    result = _completed_swing(env, reward, cfg.params, [1, 2], lifts_b)
    torch.testing.assert_close(result, torch.full((env.num_envs,), expected))
    assert not reward.pending.any()
    assert not reward(env, **cfg.params).any()  # holding support cannot repeat a cycle penalty


@pytest.mark.parametrize(("steps", "flight"), [(1, False), (4, True)])
def test_lift_consistency_ignores_short_touches_and_all_feet_flight(steps, flight):
    env = _env()
    cfg = env.cfg.rewards.feet_lift_consistency
    reward = ProxyFootLiftConsistency(cfg, env)
    _support(env, [0, 1, 2, 3])
    reward(env, **cfg.params)
    assert not _completed_swing(env, reward, cfg.params, [0, 3], [0.25, 0.25], steps, flight).any()
    assert not reward.pending.any()
    assert not _completed_swing(env, reward, cfg.params, [1, 2], [0.05, 0.05]).any()
    assert not reward.pending[:, [0, 3]].any() and reward.pending[:, [1, 2]].all()


def test_lift_consistency_ignores_initial_drop_and_expires_old_samples():
    env = _env()
    cfg = env.cfg.rewards.feet_lift_consistency
    reward = ProxyFootLiftConsistency(cfg, env)
    cfg.params["max_sample_age_s"] = 0.1
    # No observed support for the first feet: an initial drop must not become a swing.
    assert not _completed_swing(env, reward, cfg.params, [0, 3], [0.25, 0.25]).any()
    assert not reward.pending.any()
    assert not _completed_swing(env, reward, cfg.params, [0, 3], [0.25, 0.25]).any()
    assert reward.pending[:, [0, 3]].all()
    for _ in range(6):
        assert not reward(env, **cfg.params).any()
    assert not reward.pending.any()
    assert not _completed_swing(env, reward, cfg.params, [1, 2], [0.05, 0.05]).any()


def test_lift_consistency_partial_reset_clears_pending_and_support_only_in_selected_world():
    env = _env()
    cfg = env.cfg.rewards.feet_lift_consistency
    reward = ProxyFootLiftConsistency(cfg, env)
    _support(env, [0, 1, 2, 3])
    reward(env, **cfg.params)
    _completed_swing(env, reward, cfg.params, [0, 3], [0.25, 0.25])
    reward.reset([0])
    assert not reward.support_seen[0].any() and not reward.pending[0].any()
    assert reward.pending[1:, [0, 3]].all()
    result = _completed_swing(env, reward, cfg.params, [1, 2], [0.05, 0.05])
    torch.testing.assert_close(result, torch.tensor([0.0, 0.64, 0.64, 0.64, 0.64]))
    reward.reset()
    assert not reward.pending.any() and not reward.support_seen.any() and not reward.swing_active.any()


def _set_command_gate_cases(env):
    env.commands[:] = torch.tensor([[0, 0, 0], [0.0999, 0, 0], [0.1, 0, 0], [0, 0, 0.1], [0.06, 0.08, 0]])


def test_airtime_gate_includes_threshold_pure_yaw_and_combined_command_norm():
    env = _env()
    _set_command_gate_cases(env)
    state = env.term.contact_state
    force = _force(env.num_envs, 50.0)
    state.update(force, env.step_dt, env.term.cfg.contact_force_threshold)
    airborne = force.clone()
    airborne[:, 0] = 0
    for _ in range(14):
        state.update(airborne, env.step_dt, env.term.cfg.contact_force_threshold)
    state.update(force, env.step_dt, env.term.cfg.contact_force_threshold)
    torch.testing.assert_close(
        proxy_feet_air_time_command_scaled(env, **env.cfg.rewards.feet_air_time.params),
        torch.tensor([0, 0, 0.2, 0.2, 0.2]),
    )


def test_lift_gate_respects_command_norm_and_reactivation_requires_new_support():
    env = _env()
    _set_command_gate_cases(env)
    cfg = env.cfg.rewards.rotation_gait_status
    reward = ProxyRotationGaitStatus(cfg, env)
    _support(env, [0, 1, 2, 3])
    reward(env, **cfg.params)
    _support(env, [0, 3])
    env.term.application_points_w[:, [1, 2], 2] += 0.5 * cfg.params["target_height"]
    torch.testing.assert_close(reward(env, **cfg.params), torch.tensor([0, 0, 0.5, 0.5, 0.5]))
    env.commands[2] = 0
    assert not reward(env, **cfg.params).any()
    assert not reward.support_seen[2].any() and not reward.peak_progress[2].any()
    assert reward.support_seen[3:].all()
    env.commands[2] = torch.tensor([0.1, 0, 0])
    assert not reward(env, **cfg.params).any()  # no carry-over swing after waiting below threshold
    env.term.application_points_w[:, [1, 2], 2] += 0.1 * cfg.params["target_height"]
    torch.testing.assert_close(reward(env, **cfg.params), torch.tensor([0, 0, 0, 0.1, 0.1]))


def test_symmetry_gate_clears_only_inactive_worlds_and_warms_up_after_reactivation():
    env = _env()
    _set_command_gate_cases(env)
    cfg = env.cfg.rewards.rotation_gait_symmetry
    reward = ProxyRotationGaitSymmetry(cfg, env)
    for i in range(reward.buffer_size):
        _support(env, [0, 3] if i % 2 == 0 else [1, 2])
        result = reward(env, **cfg.params)
    torch.testing.assert_close(result, torch.tensor([0, 0, 1.0, 1.0, 1.0]))
    env.commands[2] = 0
    _support(env, [0, 3])
    result = reward(env, **cfg.params)
    assert result[2] == 0 and result[3] > 0.9
    assert not reward.pattern_buffer[2].any() and reward.samples[2] == 0
    env.commands[2] = torch.tensor([0.1, 0, 0])
    assert reward(env, **cfg.params)[2] == 0
    _support(env, [1, 2])
    torch.testing.assert_close(reward(env, **cfg.params)[2], torch.tensor(2.0 / reward.buffer_size))


def test_lift_consistency_gate_disables_low_commands_and_drops_pending_samples():
    env = _env()
    _set_command_gate_cases(env)
    cfg = env.cfg.rewards.feet_lift_consistency
    reward = ProxyFootLiftConsistency(cfg, env)
    _support(env, [0, 1, 2, 3])
    reward(env, **cfg.params)
    _completed_swing(env, reward, cfg.params, [0, 3], [0.25, 0.25])
    assert not reward.pending[:2].any() and reward.pending[2:, [0, 3]].all()
    env.commands[2] = 0
    assert not reward(env, **cfg.params).any()
    assert not reward.pending[2].any() and not reward.support_seen[2].any()
    env.commands[2] = torch.tensor([0.1, 0, 0])
    result = _completed_swing(env, reward, cfg.params, [1, 2], [0.05, 0.05])
    torch.testing.assert_close(result, torch.tensor([0, 0, 0, 0.64, 0.64]))


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [0, 3, 1, 2], [3, 2, 1, 0]])
def test_foot_lift_consistency_accepts_single_foot_crawl_in_any_order(order):
    env = _env()
    cfg = env.cfg.rewards.feet_lift_consistency
    reward = ProxyFootLiftConsistency(cfg, env)
    _support(env, [0, 1, 2, 3])
    reward(env, **cfg.params)
    lifts = [0.1, 0.15, 0.2, 0.25]
    for foot in order[:-1]:
        assert not _completed_swing(env, reward, cfg.params, [foot], [lifts[foot]]).any()
    result = _completed_swing(env, reward, cfg.params, [order[-1]], [lifts[order[-1]]])
    torch.testing.assert_close(result, torch.full((env.num_envs,), 0.2))
    assert not reward.pending.any()
    assert not reward(env, **cfg.params).any()


def test_foot_lift_consistency_tracks_staggered_diagonal_liftoffs_and_landings_independently():
    env = _env()
    cfg = env.cfg.rewards.feet_lift_consistency
    reward = ProxyFootLiftConsistency(cfg, env)
    height = env.term.application_points_w[..., 2]
    _support(env, [0, 1, 2, 3])
    reward(env, **cfg.params)
    for _ in range(2):
        _support(env, [1, 2, 3])
        height[:, 0] = 0.1
        assert not reward(env, **cfg.params).any()
    for _ in range(2):
        _support(env, [1, 2])
        height[:, 3] = 0.25
        assert not reward(env, **cfg.params).any()
    _support(env, [0, 1, 2])
    height[:, 0] = 0
    assert not reward(env, **cfg.params).any()
    assert reward.pending[:, 0].all() and not reward.pending[:, 3].any()
    assert reward.swing_active[:, 3].all()
    assert not reward(env, **cfg.params).any()
    _support(env, [0, 1, 2, 3])
    height[:, 3] = 0
    assert not reward(env, **cfg.params).any()
    assert reward.pending[:, [0, 3]].all()
    assert not _completed_swing(env, reward, cfg.params, [1], [0.15]).any()
    result = _completed_swing(env, reward, cfg.params, [2], [0.2])
    torch.testing.assert_close(result, torch.full((env.num_envs,), 0.2))


def test_foot_lift_consistency_waits_for_all_feet_and_uses_latest_completed_sample():
    env = _env()
    cfg = env.cfg.rewards.feet_lift_consistency
    reward = ProxyFootLiftConsistency(cfg, env)
    _support(env, [0, 1, 2, 3])
    reward(env, **cfg.params)
    assert not _completed_swing(env, reward, cfg.params, [0], [0.25]).any()
    assert not _completed_swing(env, reward, cfg.params, [0], [0.15]).any()
    assert not _completed_swing(env, reward, cfg.params, [1], [0.15]).any()
    assert not _completed_swing(env, reward, cfg.params, [2], [0.15]).any()
    result = _completed_swing(env, reward, cfg.params, [3], [0.15])
    assert not result.any()
    assert not reward.pending.any()


def _crawl_reward(env, cls, **params):
    # Keep geometric/discount regressions independent of the new task timing gate.
    if cls is ProxyCrawlGaitStatus:
        params.setdefault("min_air_time", 0.)
    cfg = RewardTermCfg(
        func=cls,
        weight=1.0,
        params={"action_term_name": "sand_proxy", "command_name": "base_velocity", "min_command_norm": 0.1,
                **params},
    )
    return cls(cfg, env), cfg.params


@pytest.mark.parametrize("swing", range(4))
def test_crawl_lift_uses_each_ordered_foot_support_height_and_only_new_capped_progress(swing):
    env = _env()
    reward, params = _crawl_reward(env, ProxyCrawlGaitStatus, target_height=0.30)
    height = env.term.application_points_w[..., 2]
    height[:] = -0.4  # Entire test may stay below the sand surface; no clearance condition.
    _support(env, [0, 1, 2, 3])
    assert not reward(env, **params).any()
    _support(env, [i for i in range(4) if i != swing])
    height[:, swing] += 0.06
    torch.testing.assert_close(reward(env, **params), torch.full((env.num_envs,), 0.2))
    assert env.term.contact_state.contact.all()  # Full-force magnitude is deliberately misleading.
    assert not reward(env, **params).any()
    height[:, swing] -= 0.03
    assert not reward(env, **params).any()
    height[:, swing] += 0.03
    assert not reward(env, **params).any()
    height[:, swing] += 0.12
    torch.testing.assert_close(reward(env, **params), torch.full((env.num_envs,), 0.4))
    height[:, swing] += 0.6
    torch.testing.assert_close(reward(env, **params), torch.full((env.num_envs,), 0.4))
    assert not reward(env, **params).any()
    height[:, swing] = -0.2
    _support(env, [0, 1, 2, 3])
    assert not reward(env, **params).any()
    _support(env, [i for i in range(4) if i != swing])
    height[:, swing] += 0.15
    torch.testing.assert_close(reward(env, **params), torch.full((env.num_envs,), 0.5))


@pytest.mark.parametrize("feet", [[0, 1, 2, 3], [0, 3], [0], []])
def test_crawl_lift_rejects_every_support_count_except_three(feet):
    env = _env()
    reward, params = _crawl_reward(env, ProxyCrawlGaitStatus)
    _support(env, [0, 1, 2, 3])
    reward(env, **params)
    env.term.application_points_w[..., 2] = 0.3
    _support(env, feet)
    assert not reward(env, **params).any()


def test_crawl_lift_requires_observed_support_and_clears_only_reset_or_inactive_worlds():
    env = _env()
    reward, params = _crawl_reward(env, ProxyCrawlGaitStatus)
    height = env.term.application_points_w[..., 2]
    _support(env, [1, 2, 3])
    height[:, 0] = 0.15
    assert not reward(env, **params).any()
    _support(env, [0, 1, 2, 3])
    reward(env, **params)
    _support(env, [1, 2, 3])
    height[:, 0] += 0.15
    torch.testing.assert_close(reward(env, **params), torch.full((env.num_envs,), 0.5))
    env.commands *= -1
    assert not reward(env, **params).any()  # Direction changes cannot re-credit height.
    reward.reset([2])
    assert not reward.support_seen[2].any() and not reward.peak_progress[2].any()
    assert reward.support_seen[0].all()
    env.commands[0] = 0
    height[:, 0] += 0.03
    torch.testing.assert_close(reward(env, **params), torch.tensor([0.0, 0.1, 0.0, 0.1, 0.1]))
    assert not reward.support_seen[0].any()
    reward.reset()
    assert not reward.support_seen.any() and not reward.peak_progress.any()


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 1, 0, 2]])
def test_crawl_phase_balances_all_four_swing_states_without_imposing_order(order):
    env = _env()
    reward, params = _crawl_reward(env, ProxyCrawlGaitSymmetry)
    assert reward.buffer_size == 100  # Default 2 s at 50 Hz.
    for i in range(reward.buffer_size):
        swing = order[i % 4]
        _support(env, [foot for foot in range(4) if foot != swing])
        result = reward(env, **params)
    torch.testing.assert_close(result, torch.ones(env.num_envs))
    torch.testing.assert_close(reward.samples, torch.full((env.num_envs,), 100.0))
    other = reward.pattern_buffer[0].clone()
    reward.reset([2])
    assert not reward.pattern_buffer[2].any() and reward.samples[2] == 0
    torch.testing.assert_close(reward.pattern_buffer[0], other)
    result = reward(env, **params)
    assert result[2] == 0 and result[0] > 0.99


@pytest.mark.parametrize("feet", [[0, 1, 2, 3], [1, 2, 3], [0, 3], [0], []])
def test_crawl_phase_rejects_fixed_single_leg_and_non_crawl_patterns(feet):
    env = _env()
    reward, params = _crawl_reward(env, ProxyCrawlGaitSymmetry)
    for _ in range(reward.buffer_size + 1):
        _support(env, feet)
        assert not reward(env, **params).any()


def test_crawl_phase_window_occupancy_warmup_expiry_and_command_gate():
    env = _env()
    _set_command_gate_cases(env)
    reward, params = _crawl_reward(env, ProxyCrawlGaitSymmetry, window_s=0.16)
    assert reward.buffer_size == 8
    for swing in range(4):
        _support(env, [i for i in range(4) if i != swing])
        result = reward(env, **params)
    torch.testing.assert_close(result, torch.tensor([0., 0., 0.5, 0.5, 0.5]))
    for _ in range(4):
        _support(env, [0, 1, 2, 3])
        torch.testing.assert_close(reward(env, **params), torch.tensor([0., 0., 0.5, 0.5, 0.5]))
    env.commands[2] = 0
    _support(env, [0, 1, 2, 3])
    assert not reward(env, **params).any()  # Oldest foot state expired; inactive world also clears.
    assert not reward.pattern_buffer[2].any() and reward.samples[2] == 0
    reward.reset()
    assert not reward.pattern_buffer.any() and not reward.samples.any()


@pytest.mark.parametrize(("cls", "params"), [
    (ProxyCrawlGaitStatus, {"target_height": 0}),
    (ProxyCrawlGaitSymmetry, {"window_s": -1}),
])
def test_crawl_rewards_reject_invalid_scales(cls, params):
    with pytest.raises(ValueError):
        _crawl_reward(_env(), cls, **params)


@pytest.mark.parametrize("cfg_cls, provider", [
    (DeeproboticsM20WalkOnSandProxyEnvCfg, "sand_proxy"),
    (DeeproboticsM20WalkOnSandEnvCfg, "mpm_contact"),
])
def test_active_sand_tasks_reward_diagonal_lift_and_phase_while_all_feet_stay_buried(cfg_cls, provider):
    env = _env()
    cfg = cfg_cls()
    assert cfg.rewards.feet_sand_clearance is None
    lift = cfg.rewards.rotation_gait_status
    phase = cfg.rewards.rotation_gait_symmetry
    assert lift.func is ProxyRotationGaitStatus and lift.weight > 0.
    assert phase.func is ProxyRotationGaitSymmetry and phase.weight > 0.
    assert "repeat_decay" not in lift.params
    assert "max_progress_lead" not in lift.params
    assert cfg.rewards.feet_air_time.func is proxy_feet_air_time_command_scaled
    assert "max_progress_lead" not in cfg.rewards.feet_air_time.params
    assert "participation_time" not in cfg.rewards.feet_air_time.params
    assert cfg.rewards.feet_forward_step.params["repeat_decay"] == 1.0
    assert "max_progress_lead" not in cfg.rewards.feet_forward_step.params
    assert lift.params["min_air_time"] == .5
    assert lift.params["action_term_name"] == phase.params["action_term_name"] == provider
    assert lift.params["target_height"] == 0.20
    assert phase.params["window_s"] == 4.0
    assert cfg.rewards.feet_forward_step.params["target_stride"] > 0.
    assert cfg.rewards.feet_forward_step.params["min_air_time"] == .5
    assert "target_clearance" not in cfg.rewards.feet_forward_step.params
    assert cfg.rewards.feet_forward_step.weight > 0.
    height = env.term.application_points_w[..., 2]
    height[:] = -0.30
    _support(env, [0, 1, 2, 3])
    lift_reward = lift.func(lift, env)
    phase_reward = phase.func(phase, env)
    assert not lift_reward(env, **lift.params).any()
    assert phase_reward.buffer_size == 200  # Four seconds at 50 Hz.
    steps = round(lift.params["min_air_time"] / env.step_dt)
    for swing in ([0, 3], [1, 2]):
        height[:] = -0.30
        _support(env, [0, 1, 2, 3])
        lift_reward(env, **lift.params)
        phase_reward(env, **phase.params)
        height[:, swing[0]] += .10
        height[:, swing[1]] += .20
        for sample in range(steps):
            _support(env, [other for other in range(4) if other not in swing])
            expected = .5 if sample == steps - 1 else 0.
            torch.testing.assert_close(lift_reward(env, **lift.params), torch.full((env.num_envs,), expected))
            result = phase_reward(env, **phase.params)
            assert (height < 0.30).all()  # No clearance of the nominal sand surface.
        height[:, swing] = -.30
        _support(env, [0, 1, 2, 3])
        lift_reward(env, **lift.params)
        result = phase_reward(env, **phase.params)
    torch.testing.assert_close(result, torch.full((env.num_envs,), 2 * steps / phase_reward.buffer_size))


def _crawl_cycle(env, reward, params, foot, lift=.30):
    height = env.term.application_points_w[..., 2]
    baseline = height[:, foot].clone()
    _support(env, [i for i in range(4) if i != foot])
    height[:, foot] += lift
    result = reward(env, **params)
    assert not reward(env, **params).any()  # Hovering must not add progress or completions.
    height[:, foot] = baseline
    _support(env, [0, 1, 2, 3])
    assert not reward(env, **params).any()
    return result


def test_crawl_same_foot_discount_and_other_feet_restoring_balance_below_sand():
    env = _env()
    reward, params = _crawl_reward(env, ProxyCrawlGaitStatus, repeat_decay=1.)
    env.term.application_points_w[..., 2] = -.6
    _support(env, [0, 1, 2, 3])
    reward(env, **params)
    for expected in (1., .5, 1. / 3.):
        torch.testing.assert_close(_crawl_cycle(env, reward, params, 3), torch.full((5,), expected))
    for foot in (1, 0, 2):
        # The last unused foot remains the least-participating foot while catching up.
        expected_scores = (1., 1., 1.) if foot == 2 else (1., .5, 1. / 3.)
        for expected in expected_scores:
            torch.testing.assert_close(_crawl_cycle(env, reward, params, foot), torch.full((5,), expected))
    torch.testing.assert_close(reward.participation_progress, torch.full((5, 4), 3.))
    torch.testing.assert_close(_crawl_cycle(env, reward, params, 3), torch.ones(5))
    assert (env.term.application_points_w[..., 2] < .30).all()


def test_crawl_partial_lifts_use_undiscounted_progress_and_reset_diagnostics():
    env = _env()
    env.extras = {"log": {}}
    reward, params = _crawl_reward(env, ProxyCrawlGaitStatus, repeat_decay=1.)
    _support(env, [0, 1, 2, 3])
    reward(env, **params)
    torch.testing.assert_close(_crawl_cycle(env, reward, params, 0, .15), torch.full((5,), .5))
    torch.testing.assert_close(_crawl_cycle(env, reward, params, 0, .15), torch.full((5,), 1. / 3.))
    assert (reward.completed_steps[:, 0] == 2).all()
    torch.testing.assert_close(reward.participation_progress[:, 0], torch.ones(5))
    reward.reset([2])
    assert env.extras["log"]["Metrics/lift_steps/fl_count"] == 2
    assert env.extras["log"]["Metrics/lift_steps/fl_progress"] == 1
    assert not reward.completed_steps[2].any() and not reward.participation_progress[2].any()
    assert reward.completed_steps[1, 0] == 2
    env.commands[0] = 0
    reward(env, **params)
    assert not reward.completed_steps[0].any() and not reward.participation_progress[0].any()


def test_crawl_contact_jitter_without_lift_does_not_count_as_a_completed_step():
    env = _env()
    reward, params = _crawl_reward(env, ProxyCrawlGaitStatus, repeat_decay=1.)
    _support(env, [0, 1, 2, 3])
    reward(env, **params)
    assert not _crawl_cycle(env, reward, params, 3, 0.).any()
    assert not reward.completed_steps.any() and not reward.participation_progress.any()


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_crawl_invalid_repeat_discount_is_rejected(value):
    with pytest.raises(ValueError, match="repeat_decay"):
        _crawl_reward(_env(), ProxyCrawlGaitStatus, repeat_decay=value)
