"""Trained actor inference and CLI compatibility without a simulator or CUDA."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from sand_pinn_lab.collect.config import environment_config, parse_args
from sand_pinn_lab.collect.policy import CheckpointPolicy, set_command


@pytest.fixture
def saved_actor(tmp_path):
    groups = {"actor": ["policy"], "critic": ["critic"]}
    config = {
        "class_name": "MLPModel",
        "hidden_dims": [8],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 100.0,
            "std_type": "log",
            "std_range": (1e-6, 1e6),
        },
    }
    obs = TensorDict({"policy": torch.randn(1, 57)}, batch_size=[1])
    actor = MLPModel(obs, groups, "actor", 16, **{k: v for k, v in config.items() if k != "class_name"})
    # Non-identity saved normalization must survive; no normalization is updated during inference.
    actor.obs_normalizer._mean.fill_(0.3)
    actor.obs_normalizer._std.fill_(2.0)
    actor.eval()
    checkpoint = tmp_path / "model_12500.pt"
    torch.save({"actor_state_dict": actor.state_dict(), "iter": 12500}, checkpoint)
    params = tmp_path / "params"
    params.mkdir()
    agent_config = params / "agent.yaml"
    agent_config.write_text(yaml.dump({"actor": config, "obs_groups": groups, "clip_actions": 0.1}))
    return checkpoint, agent_config, actor, obs


def test_checkpoint_uses_saved_normalizer_deterministic_mean_and_clipping(saved_actor):
    checkpoint, agent_config, reference, obs = saved_actor
    policy = CheckpointPolicy(checkpoint, agent_config)
    with torch.inference_mode():
        expected = reference(obs, stochastic_output=False)[0].clamp(-0.1, 0.1).numpy()
    original = {k: v.clone() for k, v in policy.actor.state_dict().items()}
    for seed in (1, 50):
        torch.manual_seed(seed)
        action = policy(obs)
        np.testing.assert_allclose(action, expected, atol=1e-7)
        assert action.dtype == np.float32 and action.shape == (16,)
        assert policy.segment == -1
        action[:] = 9  # Recorder and caller cannot overwrite the stored compatibility target.
        np.testing.assert_allclose(policy.target, expected, atol=1e-7)
    assert not policy.actor.training
    assert all(not p.requires_grad for p in policy.actor.parameters())
    for name, value in policy.actor.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_checkpoint_cli_resolves_saved_config_and_keeps_physics_and_recorder_contract(saved_actor):
    checkpoint, agent_config, _, _ = saved_actor
    args = parse_args(["--checkpoint", str(checkpoint), "--device", "cpu"])
    assert args.checkpoint == checkpoint and args.agent_config == agent_config
    assert args.command == [0.2, 0.0, 0.0]
    assert args.episodes == 20 and args.episode_seconds == 10.0
    cfg = environment_config(args)
    assert cfg.scene.num_envs == 1 and cfg.sim.dt == 0.005 and cfg.decimation == 4
    assert cfg.actions.mpm_contact is None
    assert not cfg.observations.policy.enable_corruption
    assert cfg.commands.base_velocity.resampling_time_range == (1e9, 1e9)
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        assert getattr(cfg.commands.base_velocity.ranges, name) == (0.0, 0.0)


def test_command_written_before_actor_observations_and_warmup_is_stationary():
    term = SimpleNamespace(
        vel_command_b=torch.zeros(1, 3),
        is_standing_env=torch.ones(1, dtype=torch.bool),
        is_heading_env=torch.ones(1, dtype=torch.bool),
    )
    raw = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: term))
    set_command(raw, [0.2, 0.0, 0.0])
    torch.testing.assert_close(term.vel_command_b, torch.tensor([[0.2, 0.0, 0.0]]))
    assert not term.is_standing_env.any() and not term.is_heading_env.any()
    set_command(raw, [0.0, 0.0, 0.0])
    assert not term.vel_command_b.any() and term.is_standing_env.all()


def test_random_collection_stays_the_default():
    args = parse_args([])
    assert args.checkpoint is None and args.agent_config is None


@pytest.mark.parametrize(
    "argv",
    [["--checkpoint", "/missing/model.pt"], ["--agent-config", "/missing/agent.yaml"], ["--command", "nan", "0", "0"]],
)
def test_bad_checkpoint_arguments_fail_before_launch(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


@pytest.mark.parametrize("bad_observation", [torch.zeros(1, 56), torch.full((1, 57), float("nan"))])
def test_wrong_or_nonfinite_actor_inputs_are_rejected(saved_actor, bad_observation):
    policy = CheckpointPolicy(*saved_actor[:2])
    with pytest.raises(ValueError, match="finite observations"):
        policy({"policy": bad_observation})
