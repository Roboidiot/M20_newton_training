"""Scientific checks for the offline standalone sliding-law experiment."""

import copy

import numpy as np
import torch

from M20_newton_training.sand_proxy.shadow_response import sliding_response
from sand_proxy_lab.experiments.sliding_only import (
    INITIAL,
    SLIDING_NAMES,
    fit,
    predict,
    shadow_training,
    static_support,
)


def test_standalone_numpy_matches_runtime_law_with_clamps_and_variable_timesteps():
    rng = np.random.default_rng(13)
    depth = rng.uniform(-0.03, 0.4, (150, 4))
    velocity = rng.uniform(-4, 4, (150, 4, 3))
    dt = rng.uniform(0.0005, 0.02, 150)
    p = dict(zip(SLIDING_NAMES, INITIAL, strict=True))
    numpy_force = predict(depth, velocity, p, mass=8.6, dt=dt)
    torch_force = (
        sliding_response(
            torch.tensor(np.clip(depth, 0, 0.28)),
            torch.tensor(np.clip(velocity, -2.6, 2.6)),
            p,
            stability_mass_kg=torch.tensor(8.6),
            dt=torch.tensor(dt[:, None]),
        ).numpy()
        * (depth > 0)[..., None]
    )
    np.testing.assert_allclose(numpy_force, torch_force, rtol=1e-7, atol=1e-5)
    assert np.all(np.sum(numpy_force[..., :2] * velocity[..., :2], axis=-1) <= 1e-10)
    assert np.all(numpy_force[depth <= 0] == 0)


def test_validation_targets_cannot_enter_fit_and_synthetic_law_is_recovered():
    rng = np.random.default_rng(23)
    depth = rng.uniform(0.005, 0.25, (160, 4))
    velocity = rng.uniform(-0.8, 0.8, (160, 4, 3))
    p = dict(zip(SLIDING_NAMES, INITIAL, strict=True))
    pose = np.zeros((160, 7))
    pose[:, -1] = 1
    data = dict(
        time_s=np.arange(1, 161) * 0.0025,
        dt_s=np.full(160, 0.0025),
        depth_m=depth,
        foot_velocity_yaw_m_s=velocity,
        mpm_force_w_N=predict(depth, velocity, p, mass=8.6),
        base_pose_w=pose,
        metadata={"robot_mass_kg": 34.4},
    )
    train = shadow_training(data, 0.3)
    modified = copy.deepcopy(data)
    modified["mpm_force_w_N"][modified["time_s"] > 0.3] = 1e12
    isolated = shadow_training(modified, 0.3)
    for key in train:
        np.testing.assert_equal(train[key], isolated[key])
    recovered, info = fit([train], radius=0.09)
    assert all(s["success"] for s in info["starts"])
    np.testing.assert_allclose(
        predict(train["depth"], train["velocity"], recovered, mass=train["mass"], dt=train["dt"]),
        train["actual"],
        rtol=1e-5,
        atol=1e-5,
    )


def test_static_scan_agrees_with_analytic_peak_and_insufficient_bearing():
    p = dict(zip(SLIDING_NAMES, INITIAL, strict=True))
    d = p["db"] * np.log1p(p["soft"] / p["db"])
    peak = p["bearing"] * (1 - np.exp(-d / p["db"])) * np.exp(-d / p["soft"])
    result = static_support(p, weight=338.36)
    np.testing.assert_allclose(result["max_per_foot_N"], peak, rtol=1e-5)
    assert result["equal_load_equilibria"] == []
    assert result["max_support_weight_ratio"] < 1
