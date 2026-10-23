"""Standalone Newton MPM intruder tests using the IsaacLab task's material snapshot."""

from __future__ import annotations

import json
import time
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

from .common import identity, particle_grid, surface_height
from .config import (
    COLUMNS,
    Case,
    case_dict,
    probe_bottom,
    progress,
    sample_times,
    trajectory,
)


@wp.kernel
def _wrench(
    impulse: wp.array[wp.vec3],
    point: wp.array[wp.vec3],
    collider: wp.array[int],
    body_index: wp.array[int],
    origin: wp.vec3,
    inverse_dt: float,
    output: wp.array[wp.vec3],
):
    i = wp.tid()
    c = collider[i]
    if c >= 0:  # noqa: SIM102 -- Warp may evaluate both sides; never index with an invalid collider ID.
        if body_index[c] == 0:
            force = impulse[i] * inverse_dt
            wp.atomic_add(output, 0, force)
            wp.atomic_add(output, 1, wp.cross(point[i] - origin, force))


def build(config: dict, case: Case, device: str):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverImplicitMPM.register_custom_attributes(builder)
    body = builder.add_body(is_kinematic=True, label="M20_calibration_probe")
    shape = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=True,
        mu=config["collider_friction"],
        margin=config["collider_margin"],
    )
    if case.geometry != "foot":
        raise ValueError("The active calibration pipeline collects foot forces only")
    geometries = [config["wheel_shape"]]
    for geometry in geometries:
        transform = wp.transform(wp.vec3(*geometry["xyz"]), wp.quat(*geometry["quat_xyzw"]))
        if geometry["type"] == "box":
            sx, sy, sz = geometry["size"]
            builder.add_shape_box(body, xform=transform, hx=sx / 2, hy=sy / 2, hz=sz / 2, cfg=shape)
        elif geometry["type"] in ("cylinder", "capsule"):
            add_shape = builder.add_shape_capsule if geometry["type"] == "capsule" else builder.add_shape_cylinder
            add_shape(body, xform=transform, radius=geometry["radius"], half_height=geometry["length"] / 2, cfg=shape)
        else:
            raise ValueError(f"Unsupported probe primitive: {geometry['type']}")
    wall = newton.ModelBuilder.ShapeConfig(
        has_shape_collision=False, has_particle_collision=True, mu=0.8, margin=config["collider_margin"]
    )
    hx, hy, depth = np.asarray(config["bed_size"]) * np.array([0.5, 0.5, 1.0])
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0, 0, -0.03)), hx=hx + 0.03, hy=hy + 0.03, hz=0.03, cfg=wall)
    for x, y, sx, sy in (
        (hx + 0.02, 0, 0.02, hy),
        (-hx - 0.02, 0, 0.02, hy),
        (0, hy + 0.02, hx + 0.04, 0.02),
        (0, -hy - 0.02, hx + 0.04, 0.02),
    ):
        builder.add_shape_box(-1, xform=wp.transform(wp.vec3(x, y, depth / 2)), hx=sx, hy=sy, hz=depth, cfg=wall)
    points, mass, radius = particle_grid(config, case.seed)
    # Explicit cell-centred positions avoid Newton's boundary-grid convention and hidden jitter seed.
    builder.add_particles(
        pos=points.tolist(),
        vel=np.zeros_like(points).tolist(),
        mass=[mass] * len(points),
        radius=[float(radius)] * len(points),
    )
    model = builder.finalize(device=device)
    model.set_gravity(config["gravity"])
    for key, value in config["material"].items():
        if key != "density":
            getattr(model.mpm, key).fill_(value)
    solver_config = SolverImplicitMPM.Config()
    for key in ("voxel_size", "max_iterations", "tolerance", "collider_basis", "velocity_basis", "strain_basis"):
        setattr(solver_config, key, config[key])
    solver_config.grid_type = "sparse"
    solver_config.grid_padding = 1
    solver_config.transfer_scheme = "apic"
    solver_config.collider_velocity_mode = "forward"
    solver = SolverImplicitMPM(model, solver_config)
    return model, solver


def run_case(config: dict, case: Case, directory: Path, device: str) -> Path:
    path = directory / f"{case.name}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            metadata = json.loads(str(previous["metadata"]))
        if metadata["config_id"] != identity(config) or metadata["case"] != case_dict(case):
            raise ValueError(f"Existing {path} belongs to different settings; choose a new output directory")
        print(f"REUSE {case.name}", flush=True)
        return path
    start = time.monotonic()
    model, solver = build(config, case, device)
    state = model.state()
    output = wp.zeros(2, dtype=wp.vec3, device=device)
    dt = config["dt"]

    def set_pose(p, v, angle=0.0, omega=0.0):
        q = [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]
        state.body_q.assign(np.asarray([[*p, *q]], dtype=np.float32))
        state.body_qd.assign(np.asarray([[*v, 0.0, omega, 0.0]], dtype=np.float32))

    p = np.array([0, 0, config["bed_size"][2] - probe_bottom(config, case.geometry) + 0.04])
    set_pose(p, [0, 0, 0])
    for _ in range(round(config["settle_seconds"] / dt)):
        solver.step(state, state, None, None, dt)
    wp.synchronize()
    points = state.particle_q.numpy()
    surface = surface_height(points, config)
    settling_rms = float(np.sqrt(np.mean(state.particle_qd.numpy() ** 2)))
    records = []
    times = sample_times(config, case)
    steps = len(times)
    for i, t in enumerate(times):
        p, v, angle, omega, measuring = trajectory(config, case, surface, t)
        set_pose(p, v, angle, omega)
        solver.step(state, state, None, None, dt)
        impulse, position, collider = solver.collect_collider_impulses(state)
        output.zero_()
        wp.launch(
            _wrench,
            dim=len(impulse),
            inputs=[impulse, position, collider, solver.collider_body_index, wp.vec3(*p), 1.0 / dt, output],
            device=device,
        )
        wrench = output.numpy().reshape(6)
        if not np.isfinite(wrench).all():
            raise FloatingPointError(f"Non-finite wrench in {case.name}, t={t}")
        records.append(
            [
                t,
                *p,
                *v,
                angle,
                omega,
                surface - (p[2] + probe_bottom(config, case.geometry)),
                float(measuring),
                *wrench,
                progress(config, case, t),
            ]
        )
        if i and i % 100 == 0:
            print(f"  {case.name}: {i}/{steps} steps, F={wrench[:3].round(2)} N", flush=True)
    if not np.isfinite(state.particle_q.numpy()).all():
        raise FloatingPointError(f"Non-finite final particles in {case.name}")
    metadata = {
        "schema_version": 2,
        "data_origin": "newton_mpm",
        "config_id": identity(config),
        "config": config,
        "case": case_dict(case),
        "surface_height": surface,
        "settling_velocity_rms": settling_rms,
        "particle_count": model.particle_count,
        "newton_version": newton.__version__,
        "warp_version": wp.__version__,
        "device": str(model.device),
        "wall_seconds": time.monotonic() - start,
        "columns": list(COLUMNS),
        "orientation": "fixed identity body rotation; zero angular velocity throughout",
        "torque_origin": "probe body origin",
        "force_frame": "world XYZ",
        "force_units": "N",
        "torque_units": "N m",
    }
    data = np.asarray(records)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, data=data, metadata=json.dumps(metadata, sort_keys=True))
    temporary.replace(path)
    print(f"SAVED {path}: {len(data)} samples, {metadata['wall_seconds']:.1f}s", flush=True)
    return path
