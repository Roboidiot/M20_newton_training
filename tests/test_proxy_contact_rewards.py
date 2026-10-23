"""Physics-rate contact events, command gates and partial resets without a simulator."""

from types import SimpleNamespace

import torch

from isaaclab.utils import to_dict

from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import (
    ProxyFootContactState,
    ProxyRotationGaitSymmetry,
    proxy_feet_air_time_command_scaled,
    proxy_feet_slide_ang_z_cmd,
    proxy_rotation_gait_status,
)


def _env():
    commands = torch.tensor([[0.2, 0, 0], [0, -0.15, 0], [0, 0, -0.3], [0, 0.15, 0.3], [0, 0, 0]])
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg()
    n = len(commands)
    state = ProxyFootContactState(n, "cpu", decimation=40)
    # Intentionally permute articulation body slots relative to ordered proxy feet.
    term = SimpleNamespace(
        cfg=cfg.actions.sand_proxy,
        body_ids=[2, 0, 3, 1],
        contact_state=state,
        force_history_w=torch.zeros((n, 3, 4, 3)),
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


def test_airtime_records_physics_events_scales_commands_and_ignores_initial_drop():
    env = _env()
    state = env.term.contact_state
    params = env.cfg.rewards.feet_air_time.params
    force = _force(env.num_envs, 10.0)
    for _ in range(25):
        state.update(force * 0, dt=0.01, threshold=1.0)
    state.update(force, dt=0.01, threshold=1.0)
    assert not proxy_feet_air_time_command_scaled(env, **params).any()
    state.begin_step()
    # One foot completes 0.25 s of air time, then lifts again before the policy reward.
    airborne = force.clone()
    airborne[:, 0] = 0
    for _ in range(24):
        state.update(airborne, dt=0.01, threshold=1.0)
    state.update(force, dt=0.01, threshold=1.0)
    state.update(airborne, dt=0.01, threshold=1.0)
    expected = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.0])
    for _ in range(2):  # reward evaluation must not advance timers or consume events
        torch.testing.assert_close(proxy_feet_air_time_command_scaled(env, **params), expected)
    assert not state.contact[:, 0].any()
    state.begin_step()
    assert not proxy_feet_air_time_command_scaled(env, **params).any()
    assert (state.current_air_time[:, 0] > 0).all()


def test_short_repeated_touches_are_penalized_and_partial_reset_clears_only_one_env():
    env = _env()
    state = env.term.contact_state
    force = _force(env.num_envs, 10.0)
    state.update(force, dt=0.01, threshold=1.0)
    state.begin_step()
    airborne = force.clone()
    airborne[:, 0] = 0
    for _ in range(2):
        state.update(airborne, dt=0.01, threshold=1.0)
        state.update(force, dt=0.01, threshold=1.0)
    params = env.cfg.rewards.feet_air_time.params
    torch.testing.assert_close(
        proxy_feet_air_time_command_scaled(env, **params), torch.tensor([-0.36, -0.72, -1.08, -1.44, 0.0])
    )
    state.reset([0])
    assert not state.contact[0].any() and not state.has_contact[0].any()
    assert not state.touchdown[0].any()
    assert state.touchdown[1].sum() == 2
    assert proxy_feet_air_time_command_scaled(env, **params)[0] == 0


def test_long_airtime_is_capped_and_one_new_policy_window_does_not_erase_pending_swing():
    env = _env()
    state = env.term.contact_state
    force = _force(env.num_envs, 10.0)
    state.update(force, dt=0.02, threshold=1.0)
    airborne = force.clone()
    airborne[:, 0] = 0
    for _ in range(60):
        state.begin_step()
        state.update(airborne, dt=0.02, threshold=1.0)
    state.begin_step()
    state.update(force, dt=0.02, threshold=1.0)
    torch.testing.assert_close(
        proxy_feet_air_time_command_scaled(env, **env.cfg.rewards.feet_air_time.params),
        torch.tensor([0.8, 1.6, 2.4, 3.2, 0.0]),
    )


def test_rotation_support_uses_ordered_proxy_contacts_and_real_body_slots():
    env = _env()
    data = env.scene["robot"].data
    # FL/HR grounded; FR/HL lifted by 6 cm.
    env.term.contact_state.contact[:, [0, 3]] = True
    data.body_pos_w.torch[:, [0, 3], 2] = 0.06  # body slots for FR/HL
    params = env.cfg.rewards.rotation_gait_status.params
    torch.testing.assert_close(proxy_rotation_gait_status(env, **params), torch.tensor([0.0, 0, 1, 1, 0]))
    env.term.contact_state.contact.zero_()
    assert not proxy_rotation_gait_status(env, **params).any()
    env.term.contact_state.contact[:, [1, 2]] = True
    data.body_pos_w.torch[:, :, 2] = 0
    data.body_pos_w.torch[:, [2, 1], 2] = 0.06
    torch.testing.assert_close(proxy_rotation_gait_status(env, **params), torch.tensor([0.0, 0, 1, 1, 0]))


def test_turn_sliding_uses_force_history_and_preserves_body_frame_definition():
    env = _env()
    env.commands[:] = torch.tensor([0, 0, 0.3])
    env.commands[1] = torch.tensor([0.2, 0, 0.3])  # translation suppresses pure-turn penalty
    env.commands[2] = 0
    data = env.scene["robot"].data
    data.root_lin_vel_w.torch[:, 0] = 2.0
    data.body_lin_vel_w.torch[:, :, 0] = 2.0
    data.body_lin_vel_w.torch[:, 2, 1] = 0.5  # FL relative horizontal velocity
    env.term.force_history_w[:, 2, 0, 2] = 10.0
    params = env.cfg.rewards.feet_slide_ang_z_cmd.params
    torch.testing.assert_close(proxy_feet_slide_ang_z_cmd(env, **params), torch.tensor([0.5, 0, 0, 0.5, 0.5]))
    env.term.force_history_w.zero_()
    assert not proxy_feet_slide_ang_z_cmd(env, **params).any()


def test_symmetry_history_warms_up_and_resets_per_env_without_affecting_others():
    env = _env()
    env.step_dt = 0.5  # ten samples fill the five-second window
    cfg = env.cfg.rewards.rotation_gait_symmetry
    reward = ProxyRotationGaitSymmetry(cfg, env)
    before = to_dict(env.cfg)
    for i in range(10):
        env.term.contact_state.contact.zero_()
        env.term.contact_state.contact[:, [0, 3] if i % 2 == 0 else [1, 2]] = True
        result = reward(env, **cfg.params)
    torch.testing.assert_close(result, torch.tensor([0.0, 0, 1.0, 1.0, 0.0]))
    # Alternating diagonal support gives both groups a 50% duty cycle.
    assert reward.samples.eq(10).all()
    other = reward.contact_buffer[3].clone()
    reward.reset([2])
    assert reward.samples[2] == 0 and not reward.contact_buffer[2].any()
    torch.testing.assert_close(reward.contact_buffer[3], other)
    result = reward(env, **cfg.params)
    assert result[2] < result[3]
    assert to_dict(env.cfg) == before
    reward.reset()
    assert not reward.samples.any() and not reward.contact_buffer.any()
