"""Shadow fit holdout isolation, runtime parity, passivity and legacy static support."""

import copy
import json

import numpy as np
import torch

from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH, FootSandModel
from sand_proxy_lab.shadow.refit import fit_coefficients, predict_numpy, speed_envelope


def parameter_pair():
    base = json.loads(DEFAULT_MODEL_PATH.read_text())
    base["domain"] = base.pop("intruder_domain", base["domain"])
    base.pop("shadow_contact", None)
    base["schema_version"] = 1
    revised = copy.deepcopy(base)
    revised.update(schema_version=2, intruder_domain=copy.deepcopy(base["domain"]))
    revised["domain"] = {
        "max_depth_m": 0.28,
        "max_speed_m_s": 2.6,
        "max_lateral_speed_m_s": 0.18,
        "max_vertical_speed_m_s": 0.6,
    }
    revised["shadow_contact"] = {
        "speed_filter_tau_s": 0.03,
        "coefficients": dict(
            cx=280,
            dx=0.006,
            satx=3,
            cy=750,
            dy=0.04,
            saty=2,
            bearing=20,
            db=0.02,
            soft=0.2,
            cz=900,
            dz=0.02,
            nz=2,
            blend=0.05,
        ),
    }
    return base, revised


def test_numpy_fit_function_matches_stateful_training_replay_and_reset():
    base, revised = parameter_pair()
    legacy, model = FootSandModel(base, 1), FootSandModel(revised, 1)
    rng = np.random.default_rng(4)
    d = rng.uniform(0.01, 0.22, (100, 4))
    d[40:45] = -0.01
    v = rng.uniform(-0.3, 0.3, (100, 4, 3))
    dt = np.full(100, 0.0025)
    positions = torch.zeros((1, 4, 2))
    mass = torch.tensor([[8.6]])
    baseline, actual = [], []
    for i in range(len(d)):
        depth = torch.tensor(d[i : i + 1], dtype=torch.float32)
        velocity = torch.tensor(v[i : i + 1], dtype=torch.float32)
        baseline.append(legacy.step(depth, velocity, positions, dt[i], stability_mass_kg=mass)[0].numpy().copy())
        actual.append(model.step(depth, velocity, positions, dt[i], stability_mass_kg=mass)[0].numpy().copy())
    envelope = speed_envelope(d, v, dt)
    p = revised["shadow_contact"]["coefficients"]
    predicted = predict_numpy(d, v, np.array(baseline), envelope, list(p.values()), 8.6, dt)
    np.testing.assert_allclose(actual, predicted, atol=2e-3, rtol=1e-4)
    assert np.all(np.sum(np.array(actual)[..., :2] * v[..., :2], axis=-1) <= 1e-4)
    assert (np.array(actual)[..., 2] < 0).any()  # Upward sliding can have downward friction.
    model.reset([0])
    assert not model.slip_speed_sq.any()


def test_original_normal_support_is_preserved_at_sustained_zero_horizontal_speed():
    base, revised = parameter_pair()
    legacy, model = FootSandModel(base, 2), FootSandModel(revised, 2)
    depth = torch.full((2, 4), 0.04)
    velocity = torch.zeros((2, 4, 3))
    xy = torch.zeros((2, 4, 2))
    for _ in range(30):
        torch.testing.assert_close(model.step(depth, velocity, xy, 0.0025), legacy.step(depth, velocity, xy, 0.0025))
    velocity[..., 0] = 0.2
    model.step(depth, velocity, xy, 0.0025)
    model.reset([0])
    assert not model.slip_speed_sq[0].any() and model.slip_speed_sq[1].any()


def test_future_validation_force_targets_do_not_affect_fit_coefficients():
    _, revised = parameter_pair()
    rng = np.random.default_rng(8)
    n = 420
    depth = rng.uniform(0.01, 0.20, (n, 4))
    velocity = rng.uniform(-0.3, 0.3, (n, 4, 3))
    baseline = np.zeros((n, 4, 3))
    dt = np.full(n, 0.0025)
    pose = np.zeros((n, 7))
    pose[:, 6] = 1
    target = predict_numpy(
        depth,
        velocity,
        baseline,
        speed_envelope(depth, velocity, dt),
        list(revised["shadow_contact"]["coefficients"].values()),
        8.6,
        dt,
    )
    data = dict(
        depth_m=depth,
        foot_velocity_yaw_m_s=velocity,
        mpm_force_w_N=target,
        base_pose_w=pose,
        time_s=np.arange(1, n + 1) * 0.0025,
        dt_s=dt,
        metadata={"robot_mass_kg": 34.4},
    )
    before, _ = fit_coefficients(data, baseline, 0.75, 0.09)
    data["mpm_force_w_N"][data["time_s"] > 0.75] = 1e12
    after, _ = fit_coefficients(data, baseline, 0.75, 0.09)
    assert before == after
    np.testing.assert_allclose(
        list(before.values()), list(revised["shadow_contact"]["coefficients"].values()), rtol=1e-4
    )
