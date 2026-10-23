"""CPU checks for Walk checkpoint compatibility and command-scaled MPM stepping proxies."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_class
from tensordict import TensorDict

from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

from M20_newton_training.tasks.m20.config.M20.agents.rsl_rl_ppo_cfg import (
    DeeproboticsM20WalkOnSandPPORunnerCfg,
    DeeproboticsM20WalkPPORunnerCfg,
)
from M20_newton_training.tasks.m20.config.M20.sand_env_cfg import DeeproboticsM20SandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_env_cfg import DeeproboticsM20WalkEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.mdp import rewards as shared_rewards

CONTACT_TERMS = {
    "undesired_contacts",
    "contact_forces",
    "feet_air_time",
    "feet_slide_ang_z_cmd",
    "rotation_gait_status",
    "rotation_gait_symmetry",
}


def test_walk_on_sand_keeps_walk_interfaces_and_sand_physics_without_mutating_siblings():
    before_walk = to_dict(DeeproboticsM20WalkEnvCfg())
    before_sand = to_dict(DeeproboticsM20SandEnvCfg())
    cfg = DeeproboticsM20WalkOnSandEnvCfg()
    walk = DeeproboticsM20WalkEnvCfg()
    sand = DeeproboticsM20SandEnvCfg()
    assert before_walk == to_dict(walk)
    assert before_sand == to_dict(sand)
    assert to_dict(cfg.observations) == to_dict(walk.observations)
    assert to_dict(cfg.actions) == to_dict(walk.actions)
    assert to_dict(cfg.commands) == to_dict(walk.commands)
    assert to_dict(cfg.terminations) == to_dict(walk.terminations)
    assert to_dict(cfg.sim.physics) == to_dict(sand.sim.physics)
    for name in ("sand", "sand_tray_floor", "sand_tray_north", "sand_tray_south", "sand_tray_east", "sand_tray_west"):
        assert to_dict(getattr(cfg.scene, name)) == to_dict(getattr(sand.scene, name))
    assert cfg.scene.env_spacing == sand.scene.env_spacing
    assert cfg.scene.robot.init_state.pos == sand.scene.robot.init_state.pos
    assert cfg.events.randomize_reset_base.params == sand.events.randomize_reset_base.params
    assert cfg.scene.height_scanner is None
    assert to_dict(cfg.scene.height_scanner_base) == to_dict(walk.scene.height_scanner_base)
    assert cfg.scene.contact_forces is None
    assert cfg.reward_gait_level == 1.0
    assert cfg.curriculum.terrain_levels is None
    assert cfg.curriculum.gait_level is None

    for name in CONTACT_TERMS:
        assert getattr(cfg.rewards, name) is None
        assert getattr(walk.rewards, name) is not None
    for name, term in vars(walk.rewards).items():
        if name not in CONTACT_TERMS:
            assert to_dict(getattr(cfg.rewards, name)) == to_dict(term)
    for term in vars(cfg.rewards).values():
        if term is not None and hasattr(term, "params"):
            sensor = term.params.get("sensor_cfg")
            assert sensor is None or sensor.name != "contact_forces"


def test_walk_on_sand_runner_changes_only_experiment_name_and_loads_walk_networks_strictly():
    cfg = DeeproboticsM20WalkOnSandPPORunnerCfg()
    walk_cfg = DeeproboticsM20WalkPPORunnerCfg()
    actual = to_dict(cfg)
    actual["experiment_name"] = walk_cfg.experiment_name
    assert actual == to_dict(walk_cfg)
    assert cfg.experiment_name == "deeprobotics_m20_walk_on_sand"
    spec = gym.spec("M20NewtonTraining-M20-M20-WalkOnSand")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":DeeproboticsM20WalkOnSandPPORunnerCfg")

    # Match the training entrypoint's compatibility conversion before constructing RSL-RL models.
    handle_deprecated_rsl_rl_cfg(cfg, check_rsl_rl_version())
    handle_deprecated_rsl_rl_cfg(walk_cfg, check_rsl_rl_version())
    obs = TensorDict({"policy": torch.zeros(2, 57), "critic": torch.zeros(2, 60)}, batch_size=[2])
    for name, output_dim in (("actor", 16), ("critic", 1)):
        model_class, params = resolve_class(to_dict(getattr(cfg, name)))
        source_class, source_params = resolve_class(to_dict(getattr(walk_cfg, name)))
        assert model_class is source_class is MLPModel
        source = source_class(obs, walk_cfg.obs_groups, name, output_dim, **source_params)
        target = model_class(obs, cfg.obs_groups, name, output_dim, **params)
        target.load_state_dict(source.state_dict(), strict=True)
        with torch.inference_mode():
            expected = source(obs, stochastic_output=False)
            actual_output = target(obs, stochastic_output=False)
        assert actual_output.shape == (2, output_dim)
        torch.testing.assert_close(actual_output, expected)


def _env():
    command = torch.tensor([[0.4, 0, 0], [0, -0.4, 0], [0.4, 0, -0.4], [0, 0.4, 0.4], [0, 0, 0]])
    n = len(command)
    root = torch.tensor([[0.0, 0.0, 0.68]]).repeat(n, 1)
    feet = torch.tensor([[0.3, 0.2, -0.58], [0.3, -0.2, -0.58], [-0.3, 0.2, -0.58], [-0.3, -0.2, -0.58]])
    neutral = root[:, None, :] + feet[None, :, :]
    data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=root),
        root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1)),
        projected_gravity_b=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1)),
        body_pos_w=SimpleNamespace(torch=neutral.clone()),
    )
    return SimpleNamespace(
        device="cpu",
        num_envs=n,
        step_dt=0.02,
        cfg=DeeproboticsM20WalkOnSandEnvCfg(),
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: command),
        command=command,
        neutral=neutral,
    )


def _reward(env, name):
    cfg = getattr(env.cfg.rewards, name)
    cfg.params["asset_cfg"].body_ids = [0, 1, 2, 3]
    return cfg.func(cfg, env), cfg.params


def _pose(env, lift=0.0, travel=0.0):
    feet = env.scene["robot"].data.body_pos_w.torch
    feet.copy_(env.neutral)
    feet[:, [0, 3], 2] += lift
    feet[:, [0, 3], 0] += travel


def test_relative_diagonal_lift_scales_commands_without_contact_or_gait_level(monkeypatch):
    env = _env()
    reward, params = _reward(env, "sand_diagonal_steps")
    assert not reward(env, **params).any()
    _pose(env, lift=0.07)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.0])
    for gait in (0.0, 1.0):
        monkeypatch.setattr(shared_rewards, "gait_level", gait)
        torch.testing.assert_close(reward(env, **params), expected, atol=1e-5, rtol=1e-5)
    # Common motion of all legs must not earn diagonal lift credit.
    env.scene["robot"].data.body_pos_w.torch.copy_(env.neutral + torch.tensor([0, 0, 0.07]))
    assert not reward(env, **params).any()


@pytest.mark.parametrize("partial_reset", [False, True])
def test_completed_strokes_scale_commands_and_clear_pending_credit_on_partial_reset(partial_reset):
    env = _env()
    reward, params = _reward(env, "sand_step_completion")
    assert not reward(env, **params).any()
    _pose(env, lift=0.07)
    assert not reward(env, **params).any()
    _pose(env, lift=0.07, travel=0.06)
    for _ in range(4):
        assert not reward(env, **params).any()
    if partial_reset:
        reward.reset([0])
    _pose(env, travel=0.06)
    expected = torch.tensor([0.0 if partial_reset else 25.0, 50.0, 75.0, 100.0, 0.0])
    torch.testing.assert_close(reward(env, **params), expected)
    assert not reward(env, **params).any()


def test_play_preserves_walk_observation_interface_and_coupled_physics():
    cfg = DeeproboticsM20WalkOnSandEnvCfg()
    walk = DeeproboticsM20WalkEnvCfg()
    physics = to_dict(cfg.sim.physics)
    cfg.play_mode()
    walk.play_mode()
    assert to_dict(cfg.observations) == to_dict(walk.observations)
    assert to_dict(cfg.commands) == to_dict(walk.commands)
    assert to_dict(cfg.sim.physics) == physics
    assert cfg.scene.contact_forces is None
