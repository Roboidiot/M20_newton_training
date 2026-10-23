"""Episode boundaries and warmup exclusion for independent random-action shadow trials."""

import json

import numpy as np
import pytest

from sand_proxy_lab.shadow.random_agent import reset_reason
from sand_proxy_lab.shadow.random_report import analyze, evaluation_mask


def test_fall_boundary_and_time_limit_are_detected_in_xyzw_world_pose():
    standing = np.array([0, 0, 0.88, 0, 0, 0, 1.0])
    assert reset_reason(standing, 0.30, 1.0, 6.0) is None
    yaw_only = standing.copy()
    yaw_only[5:] = np.sin(np.pi / 4), np.cos(np.pi / 4)
    assert reset_reason(yaw_only, 0.30, 1.0, 6.0) is None
    roll = standing.copy()
    roll[3:] = np.sin(np.pi / 5), 0, 0, np.cos(np.pi / 5)
    assert reset_reason(roll, 0.30, 1.0, 6.0) == "fall_tilt_over_60deg"
    low = standing.copy()
    low[2] = 0.49
    assert reset_reason(low, 0.30, 1.0, 6.0) == "fall_base_clearance_under_020m"
    edge = standing.copy()
    edge[0] = 0.86
    assert reset_reason(edge, 0.30, 1.0, 6.0) == "tray_boundary"
    assert reset_reason(standing, 0.30, 6.0, 6.0) == "episode_time_limit"


def test_every_episode_warmup_is_excluded_not_just_first_global_prefix():
    data = {"episode_time_s": np.array([0.1, 0.5, 0.52, 1.0, 0.1, 0.5, 0.52])}
    np.testing.assert_array_equal(
        evaluation_mask(data, {"warmup_s": 0.5}), [False, False, True, True, False, False, True]
    )


def test_report_excludes_large_reset_warmup_errors_and_keeps_episode_windows(tmp_path):
    actual = np.full((6, 4, 3), 10.0)
    predicted = actual + 2
    predicted[[0, 3]] = 1e6  # Both first and second episode warmup must be excluded.
    data = {
        "time_s": np.arange(1, 7) * 0.0025,
        "dt_s": np.full(6, 0.0025),
        "control_step": np.arange(6),
        "episode_id": np.repeat([0, 1], 3),
        "episode_time_s": np.tile([0.0025, 0.005, 0.0075], 2),
        "mpm_force_w_N": actual,
        "proxy_force_w_N": predicted,
        "mpm_other_robot_force_w_N": np.zeros((6, 3)),
        "depth_m": np.full((6, 4), 0.02),
        "out_of_domain": np.zeros((6, 4), dtype=bool),
        "base_pose_w": np.tile([0, 0, 0.88, 0, 0, 0, 1], (6, 1)),
    }
    metadata = {
        "agent": "synthetic_random_report_test",
        "warmup_s": 0.0025,
        "status": "synthetic_only",
        "robot_weight_N": 100,
        "episodes": [{"surface_height_m": 0.3, "status": "finished"}] * 2,
    }
    np.savez_compressed(tmp_path / "traces.npz", **data, metadata=json.dumps(metadata))
    result = analyze(tmp_path / "traces.npz")
    assert result["episodes"] == 2
    assert result["evaluation_duration_s"] == pytest.approx(0.01)
    assert result["pooled"]["all_post_warmup"]["foot_samples"] == 16
    for axis in "xyz":
        assert result["pooled"]["all_post_warmup"][axis]["rmse_N"] == pytest.approx(2)
    with np.load(tmp_path / "control_windows.npz") as saved:
        np.testing.assert_array_equal(saved["episode_id"], [0, 0, 0, 1, 1, 1])
