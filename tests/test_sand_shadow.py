"""Independent shadow experiment: body mapping, impulse units, passive hook and time averaging."""

import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from isaaclab.utils import to_dict

from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from sand_proxy_lab.shadow.__main__ import experiment_config, settled_surface
from sand_proxy_lab.shadow.record import ShadowRecorder, body_slots
from sand_proxy_lab.shadow.report import analyze, metrics


def labels():
    return [
        "/World/envs/env_0/SandTrayFloor",
        *[
            f"/World/envs/env_0/Robot/{name}"
            for name in ("hr_wheel", "base_link", "fl_wheel", "fl_knee", "hl_wheel", "fr_wheel")
        ],
    ]


def test_mapping_uses_entry_body_labels_excludes_tray_and_separates_nonfoot_contacts():
    feet, base, slots = body_slots(labels())
    assert feet == [3, 6, 5, 1] and base == 2
    np.testing.assert_array_equal(slots, [-1, 3, 4, 0, 4, 2, 1])
    with pytest.raises(ValueError, match="one fl_wheel"):
        body_slots(labels() + ["/World/Robot/fl_wheel"])


def test_mpm_impulse_reduction_and_shadow_never_write_body_forces(tmp_path):
    wp.init()
    pose = np.zeros((7, 7), dtype=np.float32)
    pose[:, 2] = 0.35
    pose[:, 6] = 1
    state = SimpleNamespace(
        body_q=wp.array(pose, dtype=wp.transform, device="cpu"),
        body_qd=wp.zeros(7, dtype=wp.spatial_vector, device="cpu"),
        body_f=wp.zeros(7, dtype=wp.spatial_vector, device="cpu"),
    )
    impulses = wp.array([[0, 0, 1], [0, 0, 2], [0, 0, 3], [99, 99, 99]], dtype=wp.vec3, device="cpu")
    positions = wp.array([[0.1, 0, 0.26]] * 4, dtype=wp.vec3, device="cpu")
    colliders = wp.array([0, 1, 2, -1], dtype=int, device="cpu")

    class Solver:
        model = SimpleNamespace(body_label=labels(), body_com=wp.zeros(7, dtype=wp.vec3, device="cpu"))
        collider_body_index = wp.array([3, 4, 0], dtype=int, device="cpu")
        count = 0

        def step(self, *args):
            self.count += 1

        def collect_collider_impulses(self, state):
            return impulses, positions, colliders

    solver = Solver()
    with ShadowRecorder(solver, DEFAULT_MODEL_PATH, 8.0, "cpu", 0.30) as recorder:
        recorder.enabled = True
        solver.step(state, state, None, None, 0.0025)
        solver.step(state, state, None, None, 0.0025)
        assert solver.count == len(recorder.rows) == 2
        row = recorder.rows[0]
        np.testing.assert_allclose(row["mpm_force_w_N"], [[0, 0, 400], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(row["mpm_other_robot_force_w_N"], [0, 0, 800])
        np.testing.assert_allclose(row["mpm_torque_com_w_Nm"][0], [0, -40, 0], atol=1e-5)
        np.testing.assert_allclose(row["depth_m"], 0.04, atol=1e-6)
        assert (row["proxy_force_w_N"][:, 2] > 0).all()
        recorder.runtime.slip_speed_sq.fill_(1)
        recorder.unlimited.hole_depth.fill_(0.1)
        recorder.reset_episode(1, 0.31)
        assert recorder.elapsed == pytest.approx(0.005) and len(recorder.rows) == 2
        assert recorder.runtime.slip_speed_sq.count_nonzero() == 0
        assert recorder.unlimited.hole_depth.count_nonzero() == 0
        solver.step(state, state, None, None, 0.0025)
        assert recorder.rows[-1]["episode_id"] == 1
        assert recorder.rows[-1]["episode_time_s"] == pytest.approx(0.0025)
        assert recorder.rows[-1]["time_s"] == pytest.approx(0.0075)
        np.testing.assert_array_equal(state.body_f.numpy(), 0)
        recorder.save(tmp_path / "traces.npz", {"proxy_force_applied": False})
    assert "step" not in solver.__dict__
    with np.load(tmp_path / "traces.npz", allow_pickle=False) as data:
        assert data["mpm_force_w_N"].shape == (3, 4, 3)
        assert not json.loads(data["metadata"].item())["proxy_force_applied"]


def test_experiment_configuration_keeps_original_training_task_unchanged():
    before = to_dict(DeeproboticsM20WalkOnSandEnvCfg())
    args = Namespace(
        mpm_iterations=100, physics_dt=0.0025, device="cuda:0", seed=42, duration=4.0, warmup=0.5, settle=0.3
    )
    cfg = experiment_config(args)
    assert cfg.scene.num_envs == 1 and not cfg.sim.physics.use_cuda_graph
    assert cfg.sim.dt * cfg.decimation == 0.02
    assert cfg.commands.base_velocity.rel_standing_envs == 0
    assert cfg.scene.robot.init_state.pos[2] == 1.8
    assert cfg.events.randomize_push_robot is None and cfg.events.randomize_actuator_gains is None
    assert to_dict(DeeproboticsM20WalkOnSandEnvCfg()) == before


def test_impulse_metrics_and_control_window_averaging_use_all_physics_samples(tmp_path):
    result = metrics(np.array([10.0, 10.0]), np.array([12.0, 12.0]), np.array([0.01, 0.03]))
    assert result["rmse_N"] == result["bias_N"] == 2
    assert result["mpm_impulse_Ns"] == pytest.approx(0.4)
    data = {
        "time_s": np.array([0.0025, 0.005]),
        "dt_s": np.array([0.0025, 0.0025]),
        "control_step": np.array([0, 0]),
        "mpm_force_w_N": np.full((2, 4, 3), 10.0),
        "proxy_force_w_N": np.stack([np.zeros((4, 3)), np.full((4, 3), 20.0)]),
        "proxy_unlimited_force_w_N": np.full((2, 4, 3), 30.0),
        "depth_m": np.full((2, 4), 0.02),
        "out_of_domain": np.zeros((2, 4), dtype=bool),
        "mpm_other_robot_force_w_N": np.zeros((2, 3)),
        "base_pose_w": np.zeros((2, 7)),
    }
    metadata = {
        "warmup_s": 0,
        "status": "synthetic_unit_test",
        "stop_reason": "test",
        "robot_weight_N": 100,
        "surface_height_m": 0.3,
        "checkpoint": "synthetic_only",
    }
    np.savez_compressed(tmp_path / "traces.npz", **data, metadata=json.dumps(metadata))
    summary = analyze(tmp_path / "traces.npz")
    assert not summary["shadow_forces_applied"] and summary["physics_samples"] == 2
    with np.load(tmp_path / "control_windows.npz") as averaged:
        np.testing.assert_array_equal(averaged["proxy_force_w_N"], 10.0)
    assert (tmp_path / "foot_forces.png").stat().st_size > 1000


def test_reference_surface_is_measured_from_column_tops():
    points = np.array([[0.01, 0.01, z] for z in [0.01, 0.03, 0.05]])
    assert settled_surface(points, 0.02) == pytest.approx(0.06)
