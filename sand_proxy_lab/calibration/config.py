"""Foot-only translation experiments; schema-2 raw-data settings remain reproducible."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import usd_geometry

GEOMETRIES = ("foot",)
MOTIONS = ("press", "pull", "forward", "lateral")
COLUMNS = (
    "time",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "angle_y",
    "omega_y",
    "depth",
    "measure",
    "fx",
    "fy",
    "fz",
    "tx",
    "ty",
    "tz",
    "progress",
)
COL = {name: i for i, name in enumerate(COLUMNS)}


@dataclass(frozen=True)
class Case:
    geometry: str
    motion: str
    split: str
    depth: float
    speed: float
    seed: int = 0

    @property
    def name(self) -> str:
        return (
            f"{self.geometry}_{self.motion}_d{round(self.depth * 1000):03d}"
            f"_v{round(self.speed * 1000):03d}_{self.split}_s{self.seed}"
        )


def training_snapshot() -> dict:
    from M20_newton_training.assets.deeprobotics import DEEPROBOTICS_M20_CFG
    from M20_newton_training.tasks.m20.config.M20.sand_env_cfg import (
        SAND_COLLIDER_MARGIN,
        SAND_DEPTH,
        SAND_JITTER_FRACTION,
        SAND_MATERIAL,
        SAND_VOXEL_SIZE,
    )

    geometry = usd_geometry(Path(DEEPROBOTICS_M20_CFG.spawn.usd_path), Path(__file__).resolve().parents[2])
    wheel = geometry["wheel_shape"]
    return {
        "schema_version": 2,
        "source": "M20 actual USD foot, fixed orientation, translational MPM reference",
        **geometry,
        "wheel_radius": wheel["radius"],
        "wheel_width": wheel["length"],
        "wheel_total_width": wheel["length"] + (2 * wheel["radius"] if wheel["type"] == "capsule" else 0),
        "material": dict(vars(SAND_MATERIAL)),
        "bed_size": [0.70, 0.60, max(0.40, SAND_DEPTH)],
        "voxel_size": min(0.01, SAND_VOXEL_SIZE),
        "training_voxel_size": SAND_VOXEL_SIZE,
        "particles_per_axis": 1.0,
        "jitter_fraction": SAND_JITTER_FRACTION,
        "collider_margin": SAND_COLLIDER_MARGIN,
        "collider_friction": 1.0,
        "dt": 0.005,
        "settle_seconds": 0.25,
        "insertion_speed": 0.10,
        "hold_seconds": 0.10,
        "clearance": 0.02,
        "travel": 0.08,
        "max_iterations": 100,
        "tolerance": 1e-4,
        "collider_basis": "pic27",
        "velocity_basis": "Q1",
        "strain_basis": "P0",
        "gravity": [0.0, 0.0, -9.81],
        "sampling": {
            "train_depths": [0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
            "train_speeds": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "train_seeds": [0],
            "validation_depths": [0.035, 0.075, 0.125, 0.175],
            "validation_speeds": [0.075, 0.125, 0.175, 0.225, 0.275],
            "validation_seeds": [1],
        },
        "fitting": {
            "absolute_rmse_tolerance_N": 5.0,
            "relative_rmse_tolerance": 0.25,
        },
    }


def cases(config: dict) -> list[Case]:
    if config.get("schema_version") != 2:
        raise ValueError("V2 collection needs v2 settings; old experiments are archived")
    result = []
    maximum = max(config["sampling"]["train_depths"])
    for geometry in GEOMETRIES:
        for split in ("train", "validation"):
            sampling = config["sampling"]
            for seed in sampling[f"{split}_seeds"]:
                for motion in MOTIONS:
                    # One press sweep covers every depth without repeating identical path prefixes.
                    depths = [maximum] if motion == "press" else sampling[f"{split}_depths"]
                    for depth in depths:
                        for speed in sampling[f"{split}_speeds"]:
                            result.append(Case(geometry, motion, split, depth, speed, seed))
    return result


def probe_bottom(config: dict, geometry: str) -> float:
    if geometry == "foot":
        return float(config["wheel_shape"]["xyz"][2] - config["wheel_radius"])
    raise ValueError("The active calibration pipeline supports foot geometry only")


def phase_start(config: dict, case: Case) -> float:
    if case.motion == "press":
        return 0.0
    return (config["clearance"] + case.depth) / config["insertion_speed"] + config["hold_seconds"]


def movement_duration(config: dict, case: Case) -> float:
    distance = config["clearance"] + case.depth if case.motion in ("press", "pull") else config["travel"]
    return distance / case.speed


def duration(config: dict, case: Case) -> float:
    return phase_start(config, case) + movement_duration(config, case)


def sample_times(config: dict, case: Case) -> np.ndarray:
    """Physical trajectory sample times, with exactly one sample at the endpoint."""
    total = duration(config, case)
    times = np.arange(math.floor(total / config["dt"]) + 1, dtype=float) * config["dt"]
    if abs(times[-1] - total) <= 1e-9:
        times[-1] = total
        return times
    return np.append(times, total)


def progress(config: dict, case: Case, t: float) -> float:
    return float(np.clip((t - phase_start(config, case)) / movement_duration(config, case), 0, 1))


def trajectory(config: dict, case: Case, surface: float, t: float):
    if case.geometry not in GEOMETRIES or case.motion not in MOTIONS:
        raise ValueError("Only foot/shin press, pull, forward and lateral motions are supported")
    bottom = probe_bottom(config, case.geometry)
    p = np.array([0.0, 0.0, surface - bottom + config["clearance"]])
    v = np.zeros(3)
    t = float(np.clip(t, 0, duration(config, case)))
    if case.motion == "press":
        p[2] -= case.speed * t
        v[2] = -case.speed
        return p, v, 0.0, 0.0, True
    if case.motion in ("forward", "lateral"):
        p[0 if case.motion == "forward" else 1] = -config["travel"] / 2
    insertion = (config["clearance"] + case.depth) / config["insertion_speed"]
    if t < insertion:
        p[2] -= config["insertion_speed"] * t
        v[2] = -config["insertion_speed"]
        return p, v, 0.0, 0.0, False
    p[2] = surface - bottom - case.depth
    measuring = t >= phase_start(config, case)
    elapsed = max(0.0, t - phase_start(config, case))
    if case.motion == "pull":
        p[2] += case.speed * elapsed
        v[2] = case.speed if measuring else 0.0
    else:
        axis = 0 if case.motion == "forward" else 1
        p[axis] += case.speed * elapsed
        v[axis] = case.speed if measuring else 0.0
    return p, v, 0.0, 0.0, measuring


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 2:
        raise ValueError("Expected v2 settings; v1 data cannot fit the new force model")
    for key in (
        "dt",
        "voxel_size",
        "particles_per_axis",
        "insertion_speed",
        "travel",
        "settle_seconds",
        "wheel_radius",
        "wheel_width",
        "tolerance",
        "clearance",
        "collider_margin",
    ):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    sample = config["sampling"]
    for split in ("train", "validation"):
        for quantity, low, high in (("depths", 0.01, 0.20), ("speeds", 0.05, 0.30)):
            values = sample[f"{split}_{quantity}"]
            if not values or any(not math.isfinite(v) or v < low - 1e-9 or v > high + 1e-9 for v in values):
                raise ValueError(f"{split}_{quantity} must lie in [{low}, {high}]")
            if values != sorted(set(values)):
                raise ValueError(f"{split}_{quantity} must be sorted and unique")
        seeds = sample[f"{split}_seeds"]
        if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("Each split needs distinct particle seeds")
    if set(sample["train_seeds"]) & set(sample["validation_seeds"]):
        raise ValueError("Training and validation particle seeds must be disjoint")
    for key in ("depths", "speeds"):
        lo, hi = min(sample[f"train_{key}"]), max(sample[f"train_{key}"])
        if any(v < lo or v > hi for v in sample[f"validation_{key}"]):
            raise ValueError("Validation grid must be inside the fitted grid")
    if min(config["bed_size"][:2]) < 0.60 or config["bed_size"][2] < max(sample["train_depths"]) + 0.15:
        raise ValueError("Bed needs >= 0.6 m horizontal extent and >= 15 cm bottom clearance")
    if config["hold_seconds"] < 0 or not 0 <= config["jitter_fraction"] <= 1:
        raise ValueError("Invalid preparation hold or jitter")
    if not isinstance(config["max_iterations"], int) or config["max_iterations"] <= 0:
        raise ValueError("max_iterations must be a positive integer")
    fitting = config["fitting"]
    for key in ("absolute_rmse_tolerance_N", "relative_rmse_tolerance"):
        if fitting[key] <= 0 or not math.isfinite(fitting[key]):
            raise ValueError("Force error tolerances must be positive")
    suite = cases(config)
    if len({c.name for c in suite}) != len(suite):
        raise ValueError("Case names collide; use depth/speed nodes separated by at least 1 mm or 0.001 m/s")


def case_dict(case: Case) -> dict:
    return {"name": case.name, **asdict(case)}
