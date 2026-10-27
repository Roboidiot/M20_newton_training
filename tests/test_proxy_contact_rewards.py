"""Control-rate contact events, all-command diagonal gait rewards and partial resets."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils import to_dict

from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import (
    ProxyFootContactState,
    ProxyRotationGaitStatus,
    ProxyRotationGaitSymmetry,
    proxy_feet_air_time_command_scaled,
    proxy_feet_slide_ang_z_cmd,
)


def _env():
    commands = torch.tensor([[0.2, 0, 0], [0, -0.15, 0], [0, 0, -0.3], [0, 0.15, 0.3], [0, 0, 0]])
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg()
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
    )
    return SimpleNamespace(
        cfg=cfg,
        device="cpu",
        num_envs=n,
        step_dt=0.02,
        action_manager=SimpleNamespace(get_term=lambda name: term),
        command_manager=SimpleNamespace(get_command=lambda name: commands),
        scene={"robot": SimpleNamespace(data=data)},
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


@pytest.mark.parametrize(("airborne_steps", "expected"), [(3, -0.02), (4, 0.0), (5, 0.02)])
def test_airtime_is_enabled_for_zero_and_small_commands_without_side_turn_amplification(airborne_steps, expected):
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
        torch.full((env.num_envs,), expected),
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
    assert cfg.params["target_height"] == 0.25
    height = env.term.application_points_w[..., 2]
    height[:] = -0.15  # swing may begin below the reference sand surface
    _support(env, [0, 1, 2, 3])
    assert not reward(env, **cfg.params).any()
    # FR rises 50 cm, but HL only 5 cm: paired progress must follow the lower foot.
    height[:, 1] += 0.50
    height[:, 2] += 0.05
    _support(env, [0, 3])
    torch.testing.assert_close(reward(env, **cfg.params), torch.full((env.num_envs,), 0.2))
    assert env.term.contact_state.contact.all()
    assert not reward(env, **cfg.params).any()  # hovering is not rewarded again
    height[:, 2] -= 0.05
    assert not reward(env, **cfg.params).any()
    height[:, 2] += 0.05
    assert not reward(env, **cfg.params).any()  # revisiting an already credited height
    height[:, 2] += 0.10
    torch.testing.assert_close(reward(env, **cfg.params), torch.full((env.num_envs,), 0.4))
    height[:, 2] += 0.50
    torch.testing.assert_close(reward(env, **cfg.params), torch.full((env.num_envs,), 0.4))
    assert not reward(env, **cfg.params).any()  # capped at 25 cm per foot
    _support(env, [0, 1, 2, 3])
    assert not reward(env, **cfg.params).any()
    # A new swing starts at the new support height, not at an old world-Z anchor.
    _support(env, [0, 3])
    assert not reward(env, **cfg.params).any()
    height[:, [1, 2]] += 0.125
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
    torch.testing.assert_close(reward(env, **cfg.params), torch.full((env.num_envs,), 0.08))
    other_anchor = reward.support_height[3].clone()
    reward.reset([2])
    assert not reward.support_seen[2].any() and not reward.peak_progress[2].any()
    torch.testing.assert_close(reward.support_height[3], other_anchor)
    env.term.application_points_w[:, [0, 3], 2] += 0.02
    torch.testing.assert_close(reward(env, **cfg.params), torch.tensor([0.08, 0.08, 0, 0.08, 0.08]))
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
    torch.testing.assert_close(reward(env, **cfg.params), torch.full((env.num_envs,), 0.2))
    anchors = reward.support_height.clone()
    env.commands[:] = torch.tensor([[0, 0.15, 0], [-0.4, 0, 0], [0, 0, 0], [0, 0, -0.3], [0.4, 0.15, 0.5]])
    assert not reward(env, **cfg.params).any()  # changing direction cannot re-credit the same height
    torch.testing.assert_close(reward.support_height, anchors)
    assert reward.support_seen.all()
    env.term.application_points_w[:, [1, 2], 2] += 0.05
    torch.testing.assert_close(reward(env, **cfg.params), torch.full((env.num_envs,), 0.2))


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
    assert reward.buffer_size == 50  # 1 s at the 50 Hz reward rate
    for i in range(50):
        _support(env, [0, 3] if i % 2 == 0 else [1, 2])
        assert env.term.contact_state.contact.all()
        result = reward(env, **cfg.params)
    torch.testing.assert_close(result, torch.ones(env.num_envs))
    torch.testing.assert_close(reward.samples, torch.full((env.num_envs,), 50.0))
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


def test_symmetry_preserves_window_for_straight_sideways_zero_and_reversed_commands():
    env = _env()
    env.commands[:] = torch.tensor([[0, 0, 0.05], [0.5, 0, 0.3], [0, 0, -0.3], [0.2, 0, 0.3], [0, 0, 0]])
    cfg = env.cfg.rewards.rotation_gait_symmetry
    reward = ProxyRotationGaitSymmetry(cfg, env)
    for i in range(50):
        _support(env, [0, 3] if i % 2 == 0 else [1, 2])
        result = reward(env, **cfg.params)
    torch.testing.assert_close(result, torch.ones(env.num_envs))
    env.commands[:] = torch.tensor([[0.4, 0, 0], [0, -0.15, 0], [0, 0, 0], [0, 0, -0.3], [0, 0, 0.5]])
    _support(env, [0, 3])
    torch.testing.assert_close(reward(env, **cfg.params), torch.ones(env.num_envs))
    torch.testing.assert_close(reward.samples, torch.full((env.num_envs,), 50.0))
    _support(env, [1, 2])
    torch.testing.assert_close(reward(env, **cfg.params), torch.ones(env.num_envs))
