"""Sand-walking failure boundaries, wheel ordering and relative wheel-height rewards."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardManager, SceneEntityCfg, TerminationManager, TerminationTermCfg
from isaaclab.utils import to_dict

from M20_newton_training.tasks.m20.config.M20.sand_env_cfg import DeeproboticsM20SandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_env_cfg import DeeproboticsM20WalkEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import front_wheels_height_reward
from M20_newton_training.tasks.m20.mdp.sand_walk_terminations import (
    base_height_low,
    front_wheels_low,
    sand_reset_penalty,
)

WHEELS = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")
STORAGE = ("hr_wheel", "base_link", "fl_wheel", "hl_wheel", "fr_wheel")


class Scene(dict):
    pass


def _env(heights=(0.8, 0.8, 0.8, 0.8, 0.8), gaps=(0.0, 0.0, 0.0, 0.0, 0.0)):
    n = len(heights)
    origins = torch.zeros(n, 3)
    origins[:, 2] = torch.arange(n) * 2.0
    root = origins.clone()
    root[:, 2] += torch.tensor(heights)
    wheels = torch.zeros(n, len(STORAGE), 3)
    # Use large, unrelated chassis and XY coordinates to catch wrong indices/axes.
    wheels[..., :2] = 20.0
    wheels[:, STORAGE.index("base_link"), 2] = -100.0
    for name in WHEELS:
        wheels[:, STORAGE.index(name), 2] = origins[:, 2] + 0.2
    for name in ("hl_wheel", "hr_wheel"):
        wheels[:, STORAGE.index(name), 2] += torch.tensor(gaps)
    data = SimpleNamespace(root_link_pos_w=SimpleNamespace(torch=root), body_link_pos_w=SimpleNamespace(torch=wheels))
    asset = SimpleNamespace(
        data=data,
        body_names=list(STORAGE),
        num_bodies=len(STORAGE),
        find_bodies=lambda names, preserve_order: ([STORAGE.index(name) for name in names], list(names)),
    )
    scene = Scene(robot=asset)
    scene.env_origins = origins
    return SimpleNamespace(
        scene=scene,
        num_envs=n,
        device="cpu",
        step_dt=0.02,
        max_episode_length_s=20.0,
        sim=SimpleNamespace(is_playing=lambda: True),
        timeout_mask=torch.zeros(n, dtype=torch.bool),
    )


def _timeout(env):
    return env.timeout_mask


def test_base_height_uses_environment_origin_and_includes_the_060_boundary():
    env = _env(heights=(0.599, 0.600, 0.601, 0.88, 0.55))
    torch.testing.assert_close(base_height_low(env), torch.tensor([True, True, False, False, True]))


def test_front_rear_wheel_mean_height_gap_uses_ordered_world_z_and_includes_boundary():
    env = _env(gaps=(-0.2, 0.0, 0.099, 0.100, 0.101))
    cfg = SceneEntityCfg("robot", body_ids=[STORAGE.index(name) for name in WHEELS])
    torch.testing.assert_close(front_wheels_low(env, 0.10, cfg), torch.tensor([False, False, False, True, True]))
    # One raised rear wheel by .20 m also produces a rear-minus-front MEAN gap of .10 m.
    points = env.scene["robot"].data.body_link_pos_w.torch
    points[:, STORAGE.index("hl_wheel"), 2] = env.scene.env_origins[:, 2] + 0.2
    points[:, STORAGE.index("hr_wheel"), 2] = env.scene.env_origins[:, 2] + 0.40
    assert front_wheels_low(env, 0.10, cfg).all()


def test_front_wheel_height_reward_is_positive_linear_and_invariant_to_common_height():
    env = _env(gaps=(-0.20, -0.10, 0.0, 0.05, 0.10))
    cfg = SceneEntityCfg("robot", body_ids=[STORAGE.index(name) for name in WHEELS])
    expected = torch.tensor([0.20, 0.10, 0.0, 0.0, 0.0])
    torch.testing.assert_close(front_wheels_height_reward(env, cfg), expected)
    # Different FL/FR and HL/HR heights must be averaged within each pair.
    points = env.scene["robot"].data.body_link_pos_w.torch
    points[:, STORAGE.index("fl_wheel"), 2] += 0.04
    points[:, STORAGE.index("fr_wheel"), 2] -= 0.04
    points[:, STORAGE.index("hl_wheel"), 2] += 0.02
    points[:, STORAGE.index("hr_wheel"), 2] -= 0.02
    torch.testing.assert_close(front_wheels_height_reward(env, cfg), expected)
    # Neither changing the chassis height nor shifting all wheels changes the gap.
    env.scene["robot"].data.root_link_pos_w.torch[:, 2] += 3.0
    points[..., 2] += 1.5
    torch.testing.assert_close(front_wheels_height_reward(env, cfg), expected)


@pytest.mark.parametrize("cfg_cls", [DeeproboticsM20WalkOnSandProxyEnvCfg, DeeproboticsM20WalkOnSandEnvCfg])
def test_both_sand_tasks_bind_front_height_reward_and_apply_weight_at_50hz(cfg_cls):
    cfg = cfg_cls()
    env = _env(gaps=(-0.20, -0.10, 0.0, 0.05, 0.10))
    term = cfg.rewards.front_wheels_height
    assert term.func is front_wheels_height_reward
    assert term.weight == 10.0
    assert term.params["asset_cfg"].body_names == list(WHEELS)
    assert term.params["asset_cfg"].preserve_order
    manager = RewardManager({"front_wheels_height": term}, env)
    torch.testing.assert_close(manager.compute(env.step_dt), torch.tensor([0.04, 0.02, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize("cfg_cls", [DeeproboticsM20WalkOnSandProxyEnvCfg, DeeproboticsM20WalkOnSandEnvCfg])
def test_native_managers_reset_on_each_failure_and_apply_matching_penalties(cfg_cls):
    cfg = cfg_cls()
    env = _env(heights=(0.8, 0.6, 0.8, 0.6, 0.8), gaps=(0.0, 0.0, 0.13, 0.13, 0.0))
    env.timeout_mask[4] = True
    terms = {name: getattr(cfg.terminations, name) for name in ("base_height_low", "front_wheels_low")}
    terms["time_out"] = TerminationTermCfg(func=_timeout, time_out=True)
    env.termination_manager = TerminationManager(terms, env)
    torch.testing.assert_close(env.termination_manager.compute(), torch.tensor([False, True, True, True, True]))
    torch.testing.assert_close(env.termination_manager.terminated, torch.tensor([False, True, True, True, False]))
    rewards = {name: getattr(cfg.rewards, name) for name in ("base_height_reset_penalty", "front_wheels_reset_penalty")}
    manager = RewardManager(rewards, env)
    torch.testing.assert_close(manager.compute(env.step_dt), torch.tensor([0.0, -2.0, -4.0, -6.0, 0.0]))
    # A coincident timeout must not suppress a real geometric failure's own flag.
    env.timeout_mask[1:3] = True
    env.termination_manager.compute()
    torch.testing.assert_close(manager.compute(env.step_dt), torch.tensor([0.0, -2.0, -4.0, -6.0, 0.0]))


@pytest.mark.parametrize("cfg_cls", [DeeproboticsM20WalkOnSandProxyEnvCfg, DeeproboticsM20WalkOnSandEnvCfg])
def test_both_sand_tasks_share_base_reset_and_front_penalties_without_mutating_siblings(cfg_cls):
    walk_before = to_dict(DeeproboticsM20WalkEnvCfg())
    sand_before = to_dict(DeeproboticsM20SandEnvCfg())
    cfg = cfg_cls()
    assert cfg.terminations.base_height_low.func is base_height_low
    assert cfg.terminations.base_height_low.params["minimum_height"] == 0.60
    assert not cfg.terminations.base_height_low.time_out
    assert cfg.rewards.base_height_reset_penalty.func is sand_reset_penalty
    assert cfg.rewards.base_height_reset_penalty.weight == -100.0
    assert cfg.rewards.base_height_reset_penalty.params["termination_name"] == "base_height_low"
    assert cfg.rewards.flat_orientation_l2 is None
    assert cfg.terminations.front_wheels_low.func is front_wheels_low
    assert cfg.terminations.front_wheels_low.params["height_difference"] == 0.10
    assert cfg.terminations.front_wheels_low.params["asset_cfg"].body_names == list(WHEELS)
    assert cfg.terminations.front_wheels_low.params["asset_cfg"].preserve_order
    assert not cfg.terminations.front_wheels_low.time_out
    reward = cfg.rewards.front_wheels_reset_penalty
    assert reward.func is sand_reset_penalty
    assert reward.weight == -200.0
    assert reward.params["termination_name"] == "front_wheels_low"
    # Modifying one task's thresholds must not mutate another config instance.
    cfg.terminations.front_wheels_low.params["height_difference"] = 0.4
    assert cfg_cls().terminations.front_wheels_low.params["height_difference"] == 0.10
    cfg.terminations.base_height_low.params["minimum_height"] = 0.4
    assert cfg_cls().terminations.base_height_low.params["minimum_height"] == 0.60
    assert to_dict(DeeproboticsM20WalkEnvCfg()) == walk_before
    assert to_dict(DeeproboticsM20SandEnvCfg()) == sand_before


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_thresholds_fail_clearly(value):
    env = _env()
    with pytest.raises(ValueError, match="minimum_height"):
        base_height_low(env, value)
    with pytest.raises(ValueError, match="height_difference"):
        front_wheels_low(env, value, SceneEntityCfg("robot", body_ids=[0, 2, 3, 4]))


def test_invalid_wheel_count_is_not_silently_averaged():
    with pytest.raises(ValueError, match="four ordered"):
        front_wheels_low(_env(), 0.12, SceneEntityCfg("robot", body_ids=[0, 2]))
    with pytest.raises(ValueError, match="four ordered"):
        front_wheels_height_reward(_env(), SceneEntityCfg("robot", body_ids=[0, 2]))
