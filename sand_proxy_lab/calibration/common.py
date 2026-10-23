"""Read-only collision geometry and particle-grid utilities shared by v2 calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def usd_geometry(path: Path, root: Path) -> dict:
    """Read composed USD, including instance proxies; do not substitute visual meshes or URDF cylinders."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError("Expected a readable M20 USD stage in metres")
    cache = UsdGeom.XformCache()

    def extract(body_name):
        body = stage.GetPrimAtPath(f"/M20/{body_name}")
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        shapes = []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            transform = Gf.Transform(cache.GetLocalToWorldTransform(prim) * inverse)
            scale = np.abs(np.asarray(transform.GetScale()))
            quat = transform.GetRotation().GetQuat()
            shape = {
                "type": prim.GetTypeName().lower(),
                "prim_path": str(prim.GetPath()),
                "xyz": list(transform.GetTranslation()),
                "quat_xyzw": [*quat.GetImaginary(), quat.GetReal()],
            }
            if prim.IsA(UsdGeom.Cube):
                shape["type"] = "box"
                shape["size"] = (prim.GetAttribute("size").Get() * scale).tolist()
            elif prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cylinder):
                if prim.GetAttribute("axis").Get() != "Z" or not np.isclose(scale[0], scale[1]):
                    raise ValueError(f"Unsupported USD primitive axis/scale at {prim.GetPath()}")
                shape["radius"] = float(prim.GetAttribute("radius").Get() * scale[0])
                shape["length"] = float(prim.GetAttribute("height").Get() * scale[2])
            else:
                raise ValueError(f"Unsupported USD collision primitive at {prim.GetPath()}")
            shapes.append(shape)
        return shapes

    wheels = extract("fl_wheel")
    if len(wheels) != 1 or wheels[0]["type"] not in ("capsule", "cylinder"):
        raise ValueError("Unexpected M20 USD foot geometry")
    return {
        "usd_path": str(path.relative_to(root)),
        "wheel_shape": wheels[0],
        "usd_layers_sha256": {
            str(Path(layer.realPath).relative_to(root)): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
            for layer in stage.GetUsedLayers()
            if layer.realPath
        },
    }


def particle_grid(config: dict, seed: int) -> tuple[np.ndarray, float, float]:
    """Cell-centered equal-volume particles, matching IsaacLab's placement convention."""
    size = np.asarray(config["bed_size"], dtype=float)
    spacing = config["voxel_size"] / config["particles_per_axis"]
    counts = np.floor(size / spacing + 1e-7).astype(int)
    if (counts < 1).any():
        raise ValueError("Bed must contain at least one cell along each axis")
    cell = size / counts
    axes = [(np.arange(n) + 0.5) * h for n, h in zip(counts, cell, strict=True)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points[:, :2] -= size[:2] / 2
    points += np.random.default_rng(seed).uniform(-0.5, 0.5, points.shape) * config["jitter_fraction"] * spacing
    volume = float(np.prod(cell))
    return points.astype(np.float32), volume * config["material"]["density"], 0.5 * np.cbrt(volume)


def surface_height(points: np.ndarray, config: dict) -> float:
    """Median column-top estimate near the centre, with half-cell support; not a fixed top offset."""
    spacing = config["voxel_size"] / config["particles_per_axis"]
    mask = (np.abs(points[:, 0]) < 0.18) & (np.abs(points[:, 1]) < 0.15)
    p = points[mask]
    if not len(p) or not np.isfinite(p).all():
        raise ValueError("No finite particles in central surface-estimation region")
    xy = np.floor(p[:, :2] / (2 * spacing)).astype(int)
    _, inverse = np.unique(xy, axis=0, return_inverse=True)
    tops = np.full(inverse.max() + 1, -np.inf)
    np.maximum.at(tops, inverse, p[:, 2])
    return float(np.median(tops) + spacing / 2)


def identity(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
