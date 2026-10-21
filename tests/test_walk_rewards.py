"""Check Walk's command shaping, reward isolation, and Flat-compatible policy interface on CPU."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import to_dict

from M20_newton_training.tasks.m20.config.M20.agents.rsl_rl_ppo_cfg import (
    DeeproboticsM20FlatPPORunnerCfg,
    DeeproboticsM20WalkPPORunnerCfg,
)
from M20_newton_training.tasks.m20.config.M20.flat_env_cfg import DeeproboticsM20FlatEnvCfg
from M20_newton_training.tasks.m20.config.M20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_env_cfg import DeeproboticsM20WalkEnvCfg
from M20_newton_training.tasks.m20.mdp import rewards as shared_rewards
from M20_newton_training.tasks.m20.mdp.walk_rewards import (
    feet_air_time_command_scaled,
    wheel_velocity_command_scaled_l2,
)


def _env(commands):
    command = torch.tensor(commands, dtype=torch.float32)
    n = len(command)
    first_contact = torch.zeros(n, 6, dtype=torch.bool)
    first_contact[:, 1] = True
    sensor = SimpleNamespace(
        data=SimpleNamespace(last_air_time=SimpleNamespace(torch=torch.full((n, 6), 0.4))),
        compute_first_contact=lambda dt: SimpleNamespace(torch=first_contact),
        first_contact=first_contact,
    )
    scene = SimpleNamespace(
        sensors={"contact_forces": sensor},
    )
    return SimpleNamespace(
        num_envs=n,
        device="cpu",
        step_dt=0.02,
        command_manager=SimpleNamespace(get_command=lambda name: command),
        scene=scene,
    )


@pytest.mark.parametrize(
    ("command", "multiplier"),
    [
        ([0.4, 0.0, 0.0], 1),
        ([0.0, 0.4, 0.0], 2),
        ([0.0, -0.4, 0.0], 2),
        ([0.4, 0.0, 0.4], 3),
        ([0.0, 0.0, -0.4], 3),
        ([0.4, -0.4, -0.4], 4),
        ([0.0, 0.0, 0.0], 0),
        ([0.05, 0.0, 0.05], 0),
    ],
)
def test_air_time_scales_side_and_turn_commands_including_mixed_commands(command, multiplier):
    env = _env([command])
    sensor = SceneEntityCfg("contact_forces", body_ids=[1, 3, 4, 5])
    reward = feet_air_time_command_scaled(env, "base_velocity", sensor)
    assert reward.item() == pytest.approx(0.2 * multiplier)


def test_air_time_only_scores_touchdown_and_caps_long_swings_without_rewarding_chatter():
    env = _env([[0.4, 0.0, 0.0]] * 3)
    sensor = env.scene.sensors["contact_forces"]
    sensor.first_contact[0] = False
    sensor.data.last_air_time.torch[1] = 10.0
    sensor.data.last_air_time.torch[2] = 0.1
    cfg = SceneEntityCfg("contact_forces", body_ids=[1, 3, 4, 5])
    reward = feet_air_time_command_scaled(env, "base_velocity", cfg)
    torch.testing.assert_close(reward, torch.tensor([0.0, 0.3, -0.1]))
    sensor.first_contact[:] = False
    assert not feet_air_time_command_scaled(env, "base_velocity", cfg).any()


def test_air_time_requires_contact_air_time_tracking():
    env = _env([[0.4, 0.0, 0.0]])
    env.scene.sensors["contact_forces"].data.last_air_time = None
    with pytest.raises(RuntimeError, match="track_air_time"):
        feet_air_time_command_scaled(env, "base_velocity", SceneEntityCfg("contact_forces"))


def test_actual_wheel_speed_penalty_ignores_leg_speeds_and_increases_for_turning():
    env = _env([[0.4, 0, 0], [0, 0.4, 0], [0.4, 0.4, -0.4], [0, 0, 0.4], [0, 0, 0]])
    velocity = torch.full((5, 16), 1000.0)
    wheel_ids = [3, 7, 11, 15]
    velocity[:, wheel_ids] = torch.tensor([5.0, -5.0, 5.0, -5.0])
    env.scene = {"robot": SimpleNamespace(data=SimpleNamespace(joint_vel=SimpleNamespace(torch=velocity)))}
    cfg = SceneEntityCfg("robot", joint_ids=wheel_ids)
    penalty = wheel_velocity_command_scaled_l2(env, "base_velocity", cfg)
    torch.testing.assert_close(penalty, torch.tensor([25.0, 25.0, 75.0, 75.0, 25.0]))
    velocity[:, wheel_ids] = 0.0
    assert not wheel_velocity_command_scaled_l2(env, "base_velocity", cfg).any()


def test_walk_keeps_flat_interfaces_and_other_weights_without_mutating_siblings():
    flat_before = DeeproboticsM20FlatEnvCfg()
    walk = DeeproboticsM20WalkEnvCfg()
    flat = DeeproboticsM20FlatEnvCfg()
    rough = DeeproboticsM20RoughEnvCfg()
    assert to_dict(flat_before) == to_dict(flat)
    assert to_dict(walk.scene) == to_dict(flat.scene)
    assert to_dict(walk.observations) == to_dict(flat.observations)
    assert to_dict(walk.actions) == to_dict(flat.actions)
    assert to_dict(walk.commands) == to_dict(flat.commands)
    assert to_dict(walk.events) == to_dict(flat.events)
    assert to_dict(walk.terminations) == to_dict(flat.terminations)
    assert walk.scene.contact_forces.track_air_time
    assert walk.curriculum.terrain_levels is None
    assert walk.curriculum.gait_level is None
    assert flat.curriculum.gait_level is not None
    assert rough.curriculum.terrain_levels is not None
    assert walk.rewards.feet_air_time_ang_z_M20 is None
    assert flat.rewards.feet_air_time_ang_z_M20.weight == 50.0
    assert walk.rewards.feet_air_time.func is feet_air_time_command_scaled
    assert walk.rewards.joint_vel_wheel_l2.func is wheel_velocity_command_scaled_l2
    assert walk.rewards.feet_air_time.weight == 10.0
    assert walk.rewards.joint_vel_wheel_l2.weight == -0.2
    changed_terms = {"feet_air_time", "feet_air_time_ang_z_M20", "joint_vel_wheel_l2"}
    for name, term in vars(flat.rewards).items():
        if name not in changed_terms:
            assert to_dict(getattr(walk.rewards, name)) == to_dict(term)
    runner = DeeproboticsM20WalkPPORunnerCfg()
    flat_runner = DeeproboticsM20FlatPPORunnerCfg()
    assert runner.experiment_name == "deeprobotics_m20_walk"
    assert to_dict(runner.actor) == to_dict(flat_runner.actor)
    assert to_dict(runner.critic) == to_dict(flat_runner.critic)
    assert to_dict(runner.algorithm) == to_dict(flat_runner.algorithm)


def test_walk_gait_override_is_local_and_keeps_regularization_active(monkeypatch):
    monkeypatch.setattr(shared_rewards, "gait_level", 0.0)
    env = _env([[0.4, 0, 0]])
    env.cfg = DeeproboticsM20WalkEnvCfg()
    env.action_manager = SimpleNamespace(action=torch.ones(1, 16), prev_action=torch.zeros(1, 16))
    torch.testing.assert_close(shared_rewards.get_gait_level_tensor(env), torch.ones(1))
    assert shared_rewards.action_rate_l2(env).item() == 16.0
    assert shared_rewards.gait_level == 0.0
    env.cfg = DeeproboticsM20FlatEnvCfg()
    assert shared_rewards.action_rate_l2(env).item() == 0.0
    monkeypatch.setattr(shared_rewards, "gait_level", 0.4)
    torch.testing.assert_close(shared_rewards.get_gait_level_tensor(env), torch.tensor([0.4]))
    env.cfg = DeeproboticsM20WalkEnvCfg()
    assert shared_rewards.action_rate_l2(env).item() == 16.0
