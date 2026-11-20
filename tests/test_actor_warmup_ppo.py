"""Real CPU PPO updates exercise actor freezing, unfreezing and checkpoint resumes."""

import io
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict

from isaaclab.utils import to_dict

from isaaclab_rl.entrypoints.backends.cli_args_rsl_rl import checkpoint_load_cfg
from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg

from M20_newton_training.tasks.m20.config.M20.agents.rsl_rl_ppo_cfg import (
    ActorWarmupPpoAlgorithmCfg,
    DeeproboticsM20RoughPPORunnerCfg,
    DeeproboticsM20WalkOnSandPPORunnerCfg,
    DeeproboticsM20WalkOnSandProxyPPORunnerCfg,
    DeeproboticsM20WalkPPORunnerCfg,
)


def observations():
    return TensorDict({"policy": torch.randn(4, 60), "critic": torch.randn(4, 70)}, batch_size=[4])


def make_algorithm(warmup=100, normalization=False):
    # Use the native configuration conversion, class resolver and PPO factory.
    cfg = DeeproboticsM20WalkOnSandProxyPPORunnerCfg()
    cfg.actor.hidden_dims = cfg.critic.hidden_dims = [16]
    cfg.actor.obs_normalization = cfg.critic.obs_normalization = normalization
    cfg.num_steps_per_env = 2
    cfg.algorithm.num_learning_epochs = cfg.algorithm.num_mini_batches = 1
    if warmup is not None:
        # The live task uses native PPO; opt in explicitly to test the retained extension.
        algorithm = to_dict(cfg.algorithm)
        algorithm.pop("class_name")
        cfg.algorithm = ActorWarmupPpoAlgorithmCfg(**algorithm, actor_warmup_iterations=warmup)
    handle_deprecated_rsl_rl_cfg(cfg, check_rsl_rl_version())
    config = to_dict(cfg)
    config["multi_gpu"] = None
    alg_class = resolve_callable(config["algorithm"]["class_name"])
    return alg_class.construct_algorithm(
        observations(), SimpleNamespace(num_envs=4, num_actions=16), config, "cpu"
    )


def update(alg):
    obs = observations()
    with torch.inference_mode():
        for _ in range(alg.storage.num_transitions_per_env):
            actions = alg.act(obs)
            reward = 1. + .2 * obs["policy"][:, 0] - .01 * actions.square().sum(dim=-1)
            obs = observations()
            alg.process_env_step(obs, reward, torch.zeros(4, dtype=torch.bool), {})
        alg.compute_returns(obs)
    return alg.update()


def snapshot(module):
    return {name: value.clone() for name, value in module.state_dict().items()}


def unchanged(module, before):
    return all(torch.equal(value, before[name]) for name, value in module.state_dict().items())


def checkpoint(alg):
    payload = alg.save()
    payload["iter"] = alg.completed_updates - 1
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def test_exactly_100_updates_freeze_actor_and_std_then_update_101_unfreezes(monkeypatch):
    torch.manual_seed(42)
    alg = make_algorithm()
    actor, critic = snapshot(alg.actor), snapshot(alg.critic)
    initial_lr = alg.learning_rate

    def forbidden_kl(*args):
        pytest.fail("KL learning-rate adaptation must be paused during actor warmup")

    with monkeypatch.context() as context:
        context.setattr(alg.actor, "get_kl_divergence", forbidden_kl)
        for _ in range(100):
            assert alg.actor_frozen
            losses = update(alg)
            assert losses["actor_frozen"] == 1.
            assert unchanged(alg.actor, actor)
            assert alg.learning_rate == initial_lr
            assert alg.schedule == "adaptive"
            assert all(p not in alg.optimizer.state for p in alg.actor.parameters())
    assert not unchanged(alg.critic, critic)
    assert alg.completed_updates == 100
    assert not alg.actor_frozen
    assert all(p.requires_grad for p in alg.actor.parameters())
    losses = update(alg)
    assert losses["actor_frozen"] == 0.
    assert not unchanged(alg.actor.mlp, {k.removeprefix("mlp."): v for k, v in actor.items() if k.startswith("mlp.")})
    distribution_before = {k: v for k, v in actor.items() if k.startswith("distribution.")}
    assert distribution_before
    assert any(not torch.equal(alg.actor.state_dict()[k], v) for k, v in distribution_before.items())


@pytest.mark.parametrize("restore_optimizer", [False, True])
def test_checkpoint_resume_preserves_remaining_warmup_and_completed_warmup(restore_optimizer):
    alg = make_algorithm(warmup=2)
    update(alg)
    saved = checkpoint(alg)
    resumed = make_algorithm(warmup=2)
    assert resumed.load(saved, checkpoint_load_cfg(restore_optimizer), strict=True)
    assert resumed.completed_updates == 1 and resumed.actor_frozen
    actor = snapshot(resumed.actor)
    assert update(resumed)["actor_frozen"] == 1.
    assert unchanged(resumed.actor, actor)
    assert not resumed.actor_frozen
    finished = make_algorithm(warmup=2)
    finished.load(checkpoint(resumed), checkpoint_load_cfg(restore_optimizer), strict=True)
    assert finished.completed_updates == 2 and not finished.actor_frozen
    actor = snapshot(finished.actor)
    assert update(finished)["actor_frozen"] == 0.
    assert not unchanged(finished.actor, actor)


def test_restored_adam_momentum_cannot_move_frozen_actor():
    alg = make_algorithm(warmup=0)
    update(alg)
    saved = checkpoint(alg)
    resumed = make_algorithm(warmup=100)
    resumed.load(saved, None, strict=True)
    assert resumed.actor_frozen
    assert all(p in resumed.optimizer.state for p in resumed.actor.parameters())
    before = snapshot(resumed.actor)
    update(resumed)
    assert unchanged(resumed.actor, before)


def test_actor_observation_statistics_freeze_while_critic_statistics_update():
    alg = make_algorithm(warmup=1, normalization=True)
    actor = snapshot(alg.actor)
    update(alg)
    assert unchanged(alg.actor, actor)
    assert alg.actor.obs_normalization
    assert alg.actor.obs_normalizer.count == 0
    assert alg.critic.obs_normalizer.count > 0
    update(alg)
    assert alg.actor.obs_normalizer.count > 0


def test_legacy_checkpoint_uses_iteration_and_weights_only_load_restarts_warmup():
    alg = make_algorithm(warmup=1)
    update(alg)
    saved = checkpoint(alg)
    del saved["actor_warmup_state"]
    resumed = make_algorithm(warmup=1)
    resumed.load(saved, None, strict=True)
    assert resumed.completed_updates == 1 and not resumed.actor_frozen
    resumed.load(saved, {"actor": True, "critic": True, "iteration": False}, strict=True)
    assert resumed.completed_updates == 0 and resumed.actor_frozen


@pytest.mark.parametrize("warmup", [-1, True, 1.5])
def test_invalid_warmup_fails_clearly(warmup):
    with pytest.raises(ValueError, match="nonnegative integer"):
        make_algorithm(warmup=warmup)


def test_proxy_defaults_to_joint_training_and_preserves_ppo_hyperparameters():
    before = to_dict(DeeproboticsM20WalkOnSandPPORunnerCfg())
    proxy = DeeproboticsM20WalkOnSandProxyPPORunnerCfg()
    alg_cfg = to_dict(proxy.algorithm)
    assert resolve_callable(alg_cfg["class_name"]) is PPO
    assert "actor_warmup_iterations" not in alg_cfg
    assert alg_cfg == before["algorithm"]
    assert to_dict(DeeproboticsM20WalkOnSandPPORunnerCfg()) == before
    for cls in (DeeproboticsM20WalkPPORunnerCfg, DeeproboticsM20RoughPPORunnerCfg):
        assert cls().algorithm.class_name == "PPO"
    alg = make_algorithm(warmup=None)
    actor, critic = snapshot(alg.actor), snapshot(alg.critic)
    assert type(alg) is PPO
    assert "actor_frozen" not in update(alg)
    assert not unchanged(alg.actor, actor)
    assert not unchanged(alg.critic, critic)
