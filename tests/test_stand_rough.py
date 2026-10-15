"""CPU regressions for upright locomotion on roads with fixed roughness."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGenerator

from M20_newton_training.tasks.m20.config.M20.agents.rsl_rl_ppo_cfg import DeeproboticsM20StandPPORunnerCfg
from M20_newton_training.tasks.m20.config.M20.flat_env_cfg import DeeproboticsM20FlatEnvCfg
from M20_newton_training.tasks.m20.config.M20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from M20_newton_training.tasks.m20.config.M20.stand_env_cfg import (
    DeeproboticsM20StandEnvCfg,
    stand_base_height_l2,
    stand_still_joint_deviation_l1,
)
from M20_newton_training.tasks.m20.config.M20.stand_terrains import (
    StandRandomRoughTerrainCfg,
    stand_random_rough_terrain,
)


class _Scene(dict):
    """Scene entities plus environment origins, without simulator initialization."""


def test_stand_has_only_rough_roads_and_disables_terrain_curriculum():
    cfg = DeeproboticsM20StandEnvCfg()
    terrain = cfg.scene.terrain
    generator = terrain.terrain_generator
    assert terrain.terrain_type == "generator"
    assert set(generator.sub_terrains) == {"random_rough"}
    assert isinstance(generator.sub_terrains["random_rough"], StandRandomRoughTerrainCfg)
    assert not generator.curriculum
    assert terrain.max_init_terrain_level is None
    assert cfg.curriculum.terrain_levels is None
    assert cfg.curriculum.gait_level is None
    assert cfg.curriculum.command_levels is None
    assert cfg.scene.height_scanner.ray_alignment == "world"
    assert cfg.scene.height_scanner_base.ray_alignment == "world"
    assert cfg.observations.policy.height_scan is None
    assert cfg.observations.critic.height_scan is not None
    assert cfg.rewards.base_height_l2.params["sensor_cfg"].name == "height_scanner_base"
    for name in ("undesired_contacts", "contact_forces", "joint_acc_l2", "joint_acc_wheel_l2", "joint_pos_limits"):
        assert getattr(cfg.rewards, name).weight < 0.0
    assert cfg.rewards.rotation_gait_status is None
    assert cfg.rewards.rotation_gait_symmetry is None
    assert DeeproboticsM20StandPPORunnerCfg().max_iterations == 20000

    # Creating Stand must not remove stairs or enable scans/curricula in sibling tasks.
    rough = DeeproboticsM20RoughEnvCfg()
    assert "pyramid_stairs" in rough.scene.terrain.terrain_generator.sub_terrains
    assert rough.scene.terrain.max_init_terrain_level == 5
    flat = DeeproboticsM20FlatEnvCfg()
    assert flat.scene.terrain.terrain_type == "plane"
    assert flat.curriculum.terrain_levels is None


def test_generated_road_ignores_difficulty_without_mutating_config():
    cfg = StandRandomRoughTerrainCfg(size=(4.0, 4.0), slope_threshold=None)
    rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        easy, _ = stand_random_rough_terrain(0.0, cfg)
        np.random.seed(42)
        hard, _ = stand_random_rough_terrain(1.0, cfg)
    finally:
        np.random.set_state(rng_state)
    np.testing.assert_array_equal(hard[0].vertices, easy[0].vertices)
    assert cfg.noise_range == (0.01, 0.16)
    assert cfg.size == (4.0, 4.0)


def test_custom_road_can_be_built_by_isaaclab_terrain_generator():
    cfg = DeeproboticsM20StandEnvCfg().scene.terrain.terrain_generator.replace(
        seed=42, num_rows=2, num_cols=1, border_width=0.0, color_scheme="none"
    )
    terrain = TerrainGenerator(cfg, device="cpu")
    assert terrain.terrain_origins.shape == (2, 1, 3)
    assert np.isfinite(terrain.terrain_mesh.vertices).all()
    assert np.ptp(terrain.terrain_mesh.vertices[:, 2]) > 0.01


def test_height_reward_tracks_local_ground_and_handles_invalid_rays_per_env():
    root_height = torch.tensor([[0.0, 0.0, 1.0028], [0.0, 0.0, 0.9028], [0.0, 0.0, 2.8028]])
    ray_hits = torch.zeros(3, 3, 3)
    ray_hits[..., 2] = torch.tensor([[0.2, 0.2, float("nan")], [0.1, 0.1, float("inf")], [float("nan")] * 3])
    scene = _Scene(
        robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=root_height))),
        height_scanner_base=SimpleNamespace(data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=ray_hits))),
    )
    scene.env_origins = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    env = SimpleNamespace(scene=scene)
    sensor = SceneEntityCfg("height_scanner_base")
    torch.testing.assert_close(stand_base_height_l2(env, 0.8028, sensor_cfg=sensor), torch.zeros(3), atol=1e-6, rtol=0)
    root_height[0, 2] += 0.1
    torch.testing.assert_close(
        stand_base_height_l2(env, 0.8028, sensor_cfg=sensor), torch.tensor([0.01, 0.0, 0.0]), atol=1e-6, rtol=0
    )


def test_rest_posture_penalty_releases_legs_for_translation_and_pure_turning():
    commands = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0]])
    data = SimpleNamespace(
        joint_pos=SimpleNamespace(torch=torch.ones(3, 2)),
        default_joint_pos=SimpleNamespace(torch=torch.zeros(3, 2)),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: commands),
    )
    reward = stand_still_joint_deviation_l1(env, "base_velocity", SceneEntityCfg("robot", joint_ids=[0, 1]))
    torch.testing.assert_close(reward, torch.tensor([2.0, 0.0, 0.0]))


def test_special_command_proportions_match_rough_and_reject_invalid_totals():
    cfg = DeeproboticsM20StandEnvCfg().commands.base_velocity
    rough_cfg = DeeproboticsM20RoughEnvCfg().commands.base_velocity
    for name in ("rel_zero_vel_envs", "rel_only_lin_y_envs", "rel_only_lin_x_envs", "rel_only_ang_z_envs"):
        assert getattr(cfg, name) == getattr(rough_cfg, name)
    cfg.rel_zero_vel_envs = 1.0
    with pytest.raises(ValueError, match="proportions"):
        cfg.class_type(cfg, SimpleNamespace())
