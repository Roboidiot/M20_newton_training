"""Real sensor forces drive four-foot gait rewards without a sand model or action term."""

from types import SimpleNamespace

import pytest
import torch

from M20_newton_training.tasks.m20.config.M20.walk_env_cfg import DeeproboticsM20WalkEnvCfg
from M20_newton_training.tasks.m20.mdp.rigid_foot_contact import FOOT_NAMES, rigid_foot_contacts

SENSOR_STORAGE = ("hr_wheel", "base_link", "fr_wheel", "fl_wheel", "hl_wheel")
BODY_STORAGE = ("base_link", "hr_wheel", "fl_wheel", "hl_wheel", "fr_wheel")


class Scene(dict):
    pass


def make_env():
    cfg = DeeproboticsM20WalkEnvCfg()
    forces = torch.zeros(2, 5, 3)
    forces[:, 1, 2] = 9999  # Non-foot contacts must never enter the gait predicates.
    sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=SimpleNamespace(torch=forces)),
        find_bodies=lambda names, preserve_order: ([SENSOR_STORAGE.index(n) for n in names], list(names)),
    )
    position = torch.zeros(2, 5, 3)
    position[..., 2] = 0.09
    quaternion = torch.zeros(2, 5, 4)
    quaternion[..., 3] = 1
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=SimpleNamespace(torch=position), body_link_quat_w=SimpleNamespace(torch=quaternion)
        ),
        find_bodies=lambda names, preserve_order: ([BODY_STORAGE.index(n) for n in names], list(names)),
    )
    scene = Scene(robot=robot)
    scene.sensors = {"contact_forces": sensor}
    command = torch.tensor([[0.3, 0.0, 0.0]]).repeat(2, 1)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        common_step_counter=0,
        scene=scene,
        cfg=cfg,
        extras={},
        command=command,
        command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    # Deliberately no action_manager: rigid rewards cannot accidentally read sand_proxy.
    env.lift_cfg = cfg.rewards.rotation_gait_status
    env.phase_cfg = cfg.rewards.rotation_gait_symmetry
    env.air_cfg = cfg.rewards.feet_air_time
    env.lift = env.lift_cfg.func(env.lift_cfg, env)
    env.phase = env.phase_cfg.func(env.phase_cfg, env)
    return env


def sample(env, swing=(), height=0.025):
    forces = env.scene.sensors["contact_forces"].data.net_forces_w.torch
    points = env.scene["robot"].data.body_link_pos_w.torch
    for foot in FOOT_NAMES:
        forces[:, SENSOR_STORAGE.index(foot)] = torch.tensor([0.0, 0.0, 50.0])
        points[:, BODY_STORAGE.index(foot), 2] = 0.09
    for i in swing:
        foot = FOOT_NAMES[i]
        # Large horizontal force must not count as vertical support.
        forces[:, SENSOR_STORAGE.index(foot)] = torch.tensor([100.0, 0.0, 0.0])
        points[:, BODY_STORAGE.index(foot), 2] += height
    env.common_step_counter += 1
    air = env.air_cfg.func(env, **env.air_cfg.params)
    lift = env.lift(env, **env.lift_cfg.params)
    phase = env.phase(env, **env.phase_cfg.params)
    return air, lift, phase


def test_direct_force_mapping_and_shared_contact_timers_sample_only_once_per_50hz_step():
    env = make_env()
    sample(env)
    sample(env, (0,))
    provider = rigid_foot_contacts(env)
    assert provider.sensor_ids == [3, 2, 4, 0]
    assert provider.body_ids == [2, 4, 3, 1]
    assert provider.contact_state.vertical_contact[0].tolist() == [False, True, True, True]
    assert provider.contact_state.contact.all()  # Horizontal force is present on the unloaded foot.
    for _ in range(10):
        provider.update()
    torch.testing.assert_close(provider.contact_state.current_air_time[:, 0], torch.full((2,), 0.02))
    torch.testing.assert_close(provider.contact_state.force_duration, torch.full((2,), 0.04))
    torch.testing.assert_close(provider.application_points_w[:, 0, 2], torch.full((2,), 0.025))


@pytest.mark.parametrize("air_steps", [1, 24, 25, 60])
def test_walk_lift_scores_immediately_and_repeatedly_and_air_time_has_no_upper_cap(air_steps):
    env = make_env()
    sample(env)
    total_lift = torch.zeros(2)
    for _ in range(air_steps):
        air, lift, _ = sample(env, (0, 3), height=0.25)
        total_lift += lift
        assert not air.any()
    torch.testing.assert_close(total_lift, torch.full((2,), float(air_steps)))
    air, _, _ = sample(env)
    per_foot_credit = (air_steps + 1) * env.step_dt - env.air_cfg.params["threshold"]
    expected_air = 2 * per_foot_credit * (0.5 if per_foot_credit < 0 else 1.0)
    torch.testing.assert_close(air, torch.full((2,), expected_air), atol=1e-6, rtol=1e-6)


def test_diagonal_pairs_must_alternate_while_single_front_and_fixed_pair_states_do_not_score():
    env = make_env()
    sample(env)
    for swing in ((0,), (1,), (2,), (3,), (0, 1), (0, 3)):
        _, _, phase = sample(env, swing)
        assert not phase.any()
    _, _, phase = sample(env, (1, 2))
    # Each opposing diagonal occupies 1/200 of the 4 s window: 2*sqrt(pA*pB)=0.01.
    torch.testing.assert_close(phase, torch.full((2,), 0.01))
    for i in range(200):
        _, _, phase = sample(env, (0, 3) if i % 2 == 0 else (1, 2))
    torch.testing.assert_close(phase, torch.ones(2))
    env.command.zero_()
    assert all(not score.any() for score in sample(env, (0,)))


def test_short_air_penalty_is_scaled_per_foot_before_summing_mixed_touchdowns():
    env = make_env()
    sample(env)
    state = rigid_foot_contacts(env).contact_state
    state.touchdown[:, 0, :2] = True
    state.touchdown_air_time[:, 0, :2] = torch.tensor([0.1, 0.5])
    # Short swing: (0.1 - 0.3)*0.5=-0.1; long swing remains 0.5-0.3=0.2.
    reward = env.air_cfg.func(env, **env.air_cfg.params)
    torch.testing.assert_close(reward, torch.full((2,), 0.1))
    legacy_params = {k: v for k, v in env.air_cfg.params.items() if k != "short_air_penalty_scale"}
    torch.testing.assert_close(env.air_cfg.func(env, **legacy_params), torch.zeros(2), atol=1e-6, rtol=1e-6)


def test_partial_episode_reset_clears_contact_history_only_for_reset_worlds():
    env = make_env()
    # Keep coverage of the incremental mode still used by the sand tasks.
    env.lift_cfg.params.update(reward_mode="progress", min_air_time=0.5, target_height=0.20)
    sample(env)
    for _ in range(24):
        sample(env, (0, 3), height=0.10)
    env.lift.reset([0])
    env.phase.reset([0])
    _, lift, _ = sample(env, (0, 3), height=0.10)
    torch.testing.assert_close(lift, torch.tensor([0.0, 0.5]))
    state = rigid_foot_contacts(env).contact_state
    torch.testing.assert_close(state.current_air_time[:, 0], torch.tensor([0.02, 0.5]))
    assert state.has_contact[0].tolist() == [False, True, True, False]
    assert state.has_contact[1].all()


@pytest.mark.parametrize("minimum_time", [0.1, 0.2, 0.5])
def test_both_diagonal_feet_must_reach_the_selected_air_time_before_paired_lift_scores(minimum_time):
    env = make_env()
    env.lift_cfg.params.update(reward_mode="progress", min_air_time=minimum_time, target_height=0.20)
    sample(env)
    steps = round(minimum_time / env.step_dt)
    for _ in range(steps):
        # One diagonal foot already airborne cannot score without its partner.
        _, lift, _ = sample(env, (0,), height=0.10)
        assert not lift.any()
    for i in range(steps):
        _, lift, _ = sample(env, (0, 3), height=0.10)
        torch.testing.assert_close(lift, torch.full((2,), 0.5 if i == steps - 1 else 0.0))
    _, lift, _ = sample(env, (0, 3), height=0.10)
    assert not lift.any()  # Holding the lift cannot earn repeated progress.
    sample(env)
    assert env.lift.completed_steps[0].tolist() == [1, 0, 0, 1]
    torch.testing.assert_close(env.lift.participation_progress[0], torch.tensor([0.5, 0.0, 0.0, 0.5]))


def test_walk_sustained_lift_uses_group_mean_height_difference_and_exact_diagonal_contact():
    env = make_env()
    sample(env)
    # Below the 20 cm threshold: no height status credit, even on a valid diagonal.
    _, lift, _ = sample(env, (0, 3), height=0.10)
    assert not lift.any()
    # Both mean height and contact pattern matter; standing all four feet earns zero.
    _, lift, _ = sample(env, (0, 1), height=0.25)
    assert not lift.any()
    _, lift, _ = sample(env, (0, 3), height=0.25)
    torch.testing.assert_close(lift, torch.ones(2))
    # A second reward read at the same held height still scores in sustained mode.
    torch.testing.assert_close(env.lift(env, **env.lift_cfg.params), torch.ones(2))
    # Rough-style mean reference: one foot can be below target if the pair mean exceeds it.
    points = env.scene["robot"].data.body_link_pos_w.torch
    points[:, BODY_STORAGE.index("fl_wheel"), 2] = 0.09 + 0.12
    points[:, BODY_STORAGE.index("hr_wheel"), 2] = 0.09 + 0.30
    env.common_step_counter += 1
    torch.testing.assert_close(env.lift(env, **env.lift_cfg.params), torch.ones(2))
