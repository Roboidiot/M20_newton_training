"""Coordinate conventions, model indices and actual USD collision capsules."""

import hashlib

import numpy as np

FEET = tuple(f"{leg}_wheel" for leg in ("fl", "fr", "hl", "hr"))
SHINS = tuple(f"{leg}_knee" for leg in ("fl", "fr", "hl", "hr"))


def body_ids(labels, names):
    result = []
    for name in names:
        matches = [i for i, label in enumerate(labels) if label.rstrip("/").endswith(f"/env_0/Robot/{name}")]
        if len(matches) != 1:
            raise ValueError(f"Expected one Robot/{name}, found {matches}")
        result.append(matches[0])
    return result


def joint_indices(model, names):
    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    q, qd = [], []
    for name in names:
        matches = [i for i, label in enumerate(model.joint_label) if label.rstrip("/").split("/")[-1] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one scalar joint {name}, got {matches}")
        j = matches[0]
        if q_start[j + 1] - q_start[j] != 1 or qd_start[j + 1] - qd_start[j] != 1:
            raise ValueError(f"Joint {name} must have one coordinate and one velocity")
        q.append(int(q_start[j]))
        qd.append(int(qd_start[j]))
    return q, qd


def rotate_xyzw(q, v):
    import torch

    xyz, w = q[..., :3], q[..., 3:]
    return v + 2 * torch.linalg.cross(xyz, torch.linalg.cross(xyz, v) + w * v)


def capsule_kinematics(pose, com_twist, com, local_center, local_axis, radius, half_length):
    """True lowest capsule point, including the horizontal shift of a tilted capsule."""
    import torch

    center = pose[:, :3] + rotate_xyzw(pose[:, 3:], local_center)
    axis = rotate_xyzw(pose[:, 3:], local_axis)
    point = center - torch.sign(axis[:, 2:3]) * half_length[:, None] * axis
    point[:, 2] -= radius
    center_v = com_twist[:, :3] + torch.linalg.cross(com_twist[:, 3:], center - com)
    point_v = com_twist[:, :3] + torch.linalg.cross(com_twist[:, 3:], point - com)
    return center, center_v, point, point_v, axis


def read_wheel_geometry(usd_path):
    from pathlib import Path

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError("Expected a readable metre-unit M20 USD")
    root = stage.GetDefaultPrim().GetPath()
    cache = UsdGeom.XformCache()
    wheels = []
    for name in FEET:
        body = stage.GetPrimAtPath(root.AppendChild(name))
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        collisions = [
            prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()) if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(collisions) != 1 or not collisions[0].IsA(UsdGeom.Capsule):
            raise ValueError(f"Expected one collision capsule for {name}")
        prim = collisions[0]
        transform = Gf.Transform(cache.GetLocalToWorldTransform(prim) * inverse)
        scale = np.abs(np.asarray(transform.GetScale()))
        axis_name = prim.GetAttribute("axis").Get()
        if axis_name != "Z" or not np.isclose(scale[0], scale[1]):
            raise ValueError(f"Unsupported capsule axis/scale: {prim.GetPath()}")
        axis = transform.GetRotation().TransformDir(Gf.Vec3d(0, 0, 1))
        wheels.append(
            {
                "body_name": name,
                "prim_path": str(prim.GetPath()),
                "local_center": list(transform.GetTranslation()),
                "local_axis": list(axis),
                "radius": float(prim.GetAttribute("radius").Get() * scale[0]),
                "half_length": float(prim.GetAttribute("height").Get() * scale[2] / 2),
            }
        )
    hashes = {
        layer.realPath: hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
        for layer in stage.GetUsedLayers()
        if layer.realPath
    }
    return {"wheels": wheels, "usd_layers_sha256": hashes}
