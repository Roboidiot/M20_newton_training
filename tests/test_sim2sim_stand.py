"""Regressions for policy ordering, frames, and MuJoCo torque limits."""

import json

import numpy as np
import torch

from mojuco_sim.sim2sim_stand import DEFAULT_CONFIG, CommandInput, StandSimulation, action_targets, build_observation


def test_observation_matches_training_order_scales_and_legacy_mask():
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    positions = np.asarray(cfg["default_joint_positions"]) + np.arange(16)
    obs = build_observation(
        [400.0, -400.0, 1.0], [-1.0, 0.0, 0.0], [0.3, 0.1, 0.2], positions, np.arange(16), np.ones(16), cfg
    )
    assert obs.shape == (57,)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs[:3], [25.0, -25.0, 0.25])
    np.testing.assert_allclose(obs[3:6], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(obs[6:9], [0.3, 0.1, 0.2])
    np.testing.assert_array_equal(obs[9:25][[3, 7, 11, 15]], np.zeros(4))
    np.testing.assert_array_equal(obs[9:25][12:15], [12.0, 13.0, 14.0])
    np.testing.assert_allclose(obs[25:41], np.arange(16) * 0.05)
    np.testing.assert_array_equal(obs[41:], np.ones(16))


def test_action_targets_separate_leg_positions_and_wheel_speeds():
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    raw, positions, velocities = action_targets(np.ones(16), cfg)
    np.testing.assert_array_equal(raw, np.ones(16))
    np.testing.assert_allclose(positions[:3], [0.125, -1.39, 1.27])
    np.testing.assert_array_equal(velocities, np.full(4, 5.0))
    raw, _, velocities = action_targets(np.full(16, 1000.0), cfg)
    np.testing.assert_array_equal(raw, np.full(16, 100.0))
    np.testing.assert_array_equal(velocities, np.full(4, 100.0))


def test_mujoco_joint_mapping_upright_gravity_and_torque_saturation():
    sim = StandSimulation(DEFAULT_CONFIG, DEFAULT_CONFIG, pd_only=True)
    np.testing.assert_array_equal(sim.actuator_ids, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15])
    np.testing.assert_allclose(sim.body_state()[2], [-1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(sim.data.qpos[sim.qpos_ids], sim.config["default_joint_positions"])
    sim.policy = lambda obs: torch.full((1, 16), 100.0)
    sim.step([0.0, 0.0, 0.0])
    torque = sim.data.ctrl[sim.actuator_ids]
    assert np.isfinite(sim.data.qpos).all()
    assert np.max(np.abs(torque[:12])) <= 76.4
    assert np.max(np.abs(torque[12:])) <= 21.6
    np.testing.assert_allclose(sim.data.time, 0.02)
    sim.reset()
    np.testing.assert_array_equal(sim.last_action, np.zeros(16))
    assert sim.control_steps == 0


def test_keyboard_increments_clamps_stops_and_resets():
    inputs = CommandInput([0.0, 0.0, 0.0], [1.0, 0.5, 1.0])
    for _ in range(20):
        inputs.key_callback(ord("J"))
    np.testing.assert_allclose(inputs.read()[0], [0.0, 0.5, 0.0])
    inputs.key_callback(259)
    np.testing.assert_array_equal(inputs.read()[0], np.zeros(3))
    inputs.key_callback(ord("I"))
    inputs.key_callback(ord("U"))
    np.testing.assert_allclose(inputs.read()[0], [0.1, 0.0, 0.1])
    inputs.key_callback(ord(" "))
    assert inputs.read()[2]
    inputs.key_callback(ord("R"))
    command, reset, paused = inputs.read()
    assert reset and paused
    np.testing.assert_array_equal(command, np.zeros(3))
    assert not inputs.read()[1]
