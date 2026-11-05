"""Episode-aware splitting and holdout isolation for the random sliding experiment."""

import copy

import numpy as np

from sand_proxy_lab.experiments.random_sliding import random_masks, training_group


def sample_data():
    time = np.concatenate((np.arange(1, 241) * 0.025, np.arange(1, 41) * 0.025))
    rng = np.random.default_rng(41)
    pose = np.zeros((len(time), 7))
    pose[:, -1] = 1
    return {
        "metadata": {"warmup_s": 0.5, "robot_mass_kg": 34.4},
        "episode_id": np.repeat([0, 1], [240, 40]),
        "episode_time_s": time,
        "dt_s": np.full(len(time), 0.025),
        "depth_m": rng.uniform(0.01, 0.25, (len(time), 4)),
        "foot_velocity_yaw_m_s": rng.uniform(-0.5, 0.5, (len(time), 4, 3)),
        "mpm_force_w_N": rng.normal(0, 100, (len(time), 4, 3)),
        "base_pose_w": pose,
    }


def test_random_split_respects_episode_reset_and_each_warmup():
    data = sample_data()
    masks = random_masks(data)
    assert masks["random_ep0_training"].sum() == 140
    assert masks["random_ep0_validation"].sum() == 73
    assert masks["random_ep1_partial"].sum() == 20
    assert masks["random_all_post_warmup"].sum() == 240
    assert not (masks["random_ep0_training"] & masks["random_ep0_validation"]).any()
    for name in ["random_ep0_training", "random_ep0_validation"]:
        assert np.all(data["episode_id"][masks[name]] == 0)
    assert np.all(data["episode_id"][masks["random_ep1_partial"]] == 1)
    assert np.all(data["episode_time_s"][masks["random_all_post_warmup"]] > 0.5)


def test_holdout_episode_and_warmup_targets_cannot_change_training_or_normalization():
    data = sample_data()
    train = random_masks(data)["random_ep0_training"]
    group = training_group(data, train)
    mutated = copy.deepcopy(data)
    mutated["mpm_force_w_N"][~train] = 1e12
    isolated = training_group(mutated, train)
    for key in group:
        np.testing.assert_equal(group[key], isolated[key])
    assert len(group["depth"]) == 140 * 4
