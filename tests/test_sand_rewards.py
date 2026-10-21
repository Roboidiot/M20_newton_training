"""Contact-free Sand reward checks, including asynchronous reset and checkpoint interface compatibility."""

import inspect
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import to_dict

from M20_newton_training.tasks.m20.config.M20.agents.rsl_rl_ppo_cfg import (
    DeeproboticsM20RoughPPORunnerCfg,
    DeeproboticsM20SandPPORunnerCfg,
)
from M20_newton_training.tasks.m20.config.M20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from M20_newton_training.tasks.m20.config.M20.sand_env_cfg import DeeproboticsM20SandEnvCfg
from M20_newton_training.tasks.m20.mdp import rewards as shared_rewards
from M20_newton_training.tasks.m20.mdp.sand_rewards import (
    SandActionSmoothness,
    sand_stand_still,
    sand_wheel_velocity_l2,
)


def _array(tensor):
    return SimpleNamespace(torch=tensor)


def _env() -> Any:
    root = torch.tensor([[0.0, 0.0, 0.78], [4.0, 0.0, 0.78]])
    feet = torch.tensor([[0.3, 0.2, -0.58], [0.3, -0.2, -0.58], [-0.3, 0.2, -0.58], [-0.3, -0.2, -0.58]])
    neutral = root[:, None, :] + feet[None, :, :]
    data = SimpleNamespace(
        body_pos_w=_array(neutral.clone()),
        body_lin_vel_w=_array(torch.zeros(2, 4, 3)),
        root_pos_w=_array(root),
        root_quat_w=_array(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)),
        projected_gravity_b=_array(torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1)),
        joint_pos=_array(torch.ones(2, 16)),
        default_joint_pos=_array(torch.zeros(2, 16)),
        joint_vel=_array(torch.zeros(2, 16)),
    )
    commands = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
    return cast(
        ManagerBasedRLEnv,
        SimpleNamespace(
            num_envs=2,
            device="cpu",
            step_dt=0.02,
            episode_length_buf=torch.zeros(2, dtype=torch.long),
            scene={
                "robot": SimpleNamespace(data=data, actuators=SimpleNamespace(applied_effort=_array(torch.ones(2, 16))))
            },
            command_manager=SimpleNamespace(get_command=lambda name: commands),
            action_manager=SimpleNamespace(action=torch.zeros(2, 16), prev_action=torch.zeros(2, 16)),
            neutral=neutral,
            commands=commands,
        ),
    )


def _term(env, name):
    cfg = getattr(DeeproboticsM20SandEnvCfg().rewards, name)
    if "asset_cfg" in cfg.params:
        cfg.params["asset_cfg"].body_ids = [0, 1, 2, 3]
    return cfg.func(cfg, env), cfg.params


def _pose(env, feet, *, lift=0.0, advance=0.0, env_ids=(0, 1)):
    data = env.scene["robot"].data
    for env_id in env_ids:
        for foot in feet:
            data.body_pos_w.torch[env_id, foot] = env.neutral[env_id, foot] + torch.tensor([advance, 0.0, lift])


def test_sand_preserves_actor_actions_and_scanners_and_ungates_all_active_rewards():
    sand = DeeproboticsM20SandEnvCfg()
    rough = DeeproboticsM20RoughEnvCfg()
    assert to_dict(sand.observations) == to_dict(rough.observations)
    assert to_dict(sand.actions) == to_dict(rough.actions)
    assert to_dict(sand.scene.height_scanner) == to_dict(rough.scene.height_scanner)
    assert to_dict(sand.scene.height_scanner_base) == to_dict(rough.scene.height_scanner_base)
    assert to_dict(DeeproboticsM20SandPPORunnerCfg().actor) == to_dict(DeeproboticsM20RoughPPORunnerCfg().actor)
    assert sand.curriculum.gait_level is None
    assert rough.curriculum.gait_level is not None
    assert sand.rewards.joint_torques_l2.func is base_mdp.joint_torques_l2
    assert sand.rewards.action_rate_l2.func is base_mdp.action_rate_l2
    assert sand.rewards.action_smooth_l2.func is SandActionSmoothness
    assert sand.rewards.joint_mirror is None
    for name in ("hipx_joint_pos_penalty", "hipy_joint_pos_penalty", "knee_joint_pos_penalty", "stand_still"):
        assert getattr(sand.rewards, name).weight == pytest.approx(getattr(rough.rewards, name).weight / 10)
    for term in vars(sand.rewards).values():
        if term is not None and hasattr(term, "func"):
            assert "get_gait_level_tensor" not in inspect.getsource(term.func)


def test_diagonal_coordination_rejects_hopping_and_is_independent_of_clock_and_gait_level(monkeypatch):
    env = _env()
    term, params = _term(env, "sand_diagonal_steps")
    assert not term(env, **params).any()
    _pose(env, [0, 3], lift=0.05)
    smaller_lift = term(env, **params)
    _pose(env, [0, 3], lift=0.07)
    diagonal = term(env, **params)
    assert (diagonal > smaller_lift).all()
    torch.testing.assert_close(diagonal, torch.ones(2), atol=1e-5, rtol=1e-5)
    monkeypatch.setattr(shared_rewards, "gait_level", 0.0)
    env.episode_length_buf[:] = torch.tensor([17, 931])
    torch.testing.assert_close(term(env, **params), diagonal)
    monkeypatch.setattr(shared_rewards, "gait_level", 1.0)
    torch.testing.assert_close(term(env, **params), diagonal)
    _pose(env, [0, 3])
    _pose(env, [0, 1], lift=0.07)
    assert (term(env, **params) < 0.1 * diagonal).all()
    _pose(env, [0, 1, 2, 3], lift=0.07)
    assert not term(env, **params).any()
    _pose(env, [0, 1, 2, 3])
    _pose(env, [0, 3], lift=0.07)
    env.commands[:] = 0.0
    assert not term(env, **params).any()
    env.commands[:, 2] = 0.4
    assert (term(env, **params) > 0).all()


def test_step_credit_requires_lift_travel_and_lowering_and_is_not_repeated():
    env = _env()
    term, params = _term(env, "sand_step_completion")
    term(env, **params)
    _pose(env, [0, 3], lift=0.07)
    assert not term(env, **params).any()
    _pose(env, [0, 3], lift=0.07, advance=0.06)
    for _ in range(4):
        assert not term(env, **params).any()
    _pose(env, [0, 3], advance=0.06)
    assert (term(env, **params) > 0).all()
    assert not term(env, **params).any()

    # Pure sliding along the ground cannot earn a completed step.
    term.reset()
    _pose(env, [0, 1, 2, 3])
    term(env, **params)
    _pose(env, [0, 3], advance=0.1)
    assert not term(env, **params).any()


def test_partial_reset_discards_only_that_environments_pending_stroke():
    env = _env()
    term, params = _term(env, "sand_step_completion")
    term(env, **params)
    _pose(env, [0, 3], lift=0.07)
    term(env, **params)
    _pose(env, [0, 3], lift=0.07, advance=0.06)
    for _ in range(4):
        term(env, **params)
    term.reset([0])
    _pose(env, [0, 3], advance=0.06)
    credit = term(env, **params)
    assert credit[0] == 0.0
    assert credit[1] > 0.0


def test_completed_diagonal_pairs_must_alternate():
    env = _env()
    term, params = _term(env, "sand_step_completion")
    term(env, **params)

    def stroke(feet, start, end):
        _pose(env, feet, lift=0.07, advance=start)
        term(env, **params)
        _pose(env, feet, lift=0.07, advance=end)
        for _ in range(4):
            term(env, **params)
        _pose(env, feet, advance=end)
        return term(env, **params)

    assert (stroke([0, 3], 0.0, 0.06) > 0).all()
    assert not stroke([0, 3], 0.06, 0.12).any()
    assert (stroke([1, 2], 0.0, 0.06) > 0).all()
    assert (stroke([0, 3], 0.12, 0.18) > 0).all()


def test_stationary_or_expired_swings_do_not_earn_step_credit():
    for stop in (False, True):
        env = _env()
        term, params = _term(env, "sand_step_completion")
        term(env, **params)
        _pose(env, [0, 3], lift=0.07)
        term(env, **params)
        _pose(env, [0, 3], lift=0.07, advance=0.06)
        for _ in range(4):
            term(env, **params)
        if stop:
            env.commands[:] = 0.0
        else:
            params["max_swing_time"] = 0.08
        _pose(env, [0, 3], advance=0.06)
        assert not term(env, **params).any()


def test_support_and_wheel_penalties_use_kinematics_without_contact_sensors():
    env = _env()
    term, params = _term(env, "sand_support_motion")
    assert not term(env, **params).any()
    env.scene["robot"].data.body_lin_vel_w.torch[:, [0, 3], 0] = 0.5
    moving_low = term(env, **params)
    assert (moving_low > 0).all()
    _pose(env, [0, 3], lift=0.07)
    assert not term(env, **params).any()
    wheels = SceneEntityCfg("robot", joint_ids=[3, 7, 11, 15])
    env.scene["robot"].data.joint_vel.torch[:, wheels.joint_ids] = 10.0
    torch.testing.assert_close(sand_wheel_velocity_l2(env, wheels), torch.full((2,), 4.0))


def test_regularization_survives_zero_gait_and_smoothness_history_resets(monkeypatch):
    env = _env()
    monkeypatch.setattr(shared_rewards, "gait_level", 0.0)
    legs = SceneEntityCfg("robot", joint_ids=list(range(12)))
    assert (base_mdp.joint_torques_l2(env, legs) > 0).all()
    term, params = _term(env, "action_smooth_l2")
    term(env, **params)
    term(env, **params)
    env.action_manager.action[:] = 1.0
    assert (base_mdp.action_rate_l2(env) > 0).all()
    assert (term(env, **params) > 0).all()
    term.reset([0])
    result = term(env, **params)
    assert result[0] == 0
    assert result[1] > 0
    env.commands[:] = 0.0
    assert (sand_stand_still(env, "base_velocity", legs) > 0).all()
    env.commands[:, 2] = 0.4
    assert not sand_stand_still(env, "base_velocity", legs).any()
