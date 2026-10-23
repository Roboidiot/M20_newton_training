"""Refit a fixed foot-contact law on a shadow trace; preserve blocked time validation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH, FootSandModel
from sand_proxy_lab.paths import add_terrain_argument, terrain_directory

NAMES = ("cx", "dx", "satx", "cy", "dy", "saty", "bearing", "db", "soft", "cz", "dz", "nz", "blend")
FILTER_TAU = 0.03


def rotate_forces(force, pose, *, inverse=False):
    x, y, z, w = pose[:, 3:].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    if inverse:
        s = -s
    result = force.copy()
    result[..., 0] = c * force[..., 0] - s * force[..., 1]
    result[..., 1] = s * force[..., 0] + c * force[..., 1]
    return result


def speed_envelope(depth, velocity, dt, tau=FILTER_TAU):
    squared = np.sum(velocity[..., :2] ** 2, axis=-1)
    result = np.zeros_like(depth, dtype=float)
    previous = np.zeros(depth.shape[1])
    for i in range(len(depth)):
        alpha = np.exp(-dt[i] / tau)
        previous = np.where(depth[i] > 0, alpha * previous + (1 - alpha) * squared[i], 0.0)
        result[i] = previous
    return result


def predict_numpy(depth, velocity, baseline, envelope, coefficients, mass=8.6, dt=0.0025):
    p = dict(zip(NAMES, coefficients, strict=True))
    d = np.maximum(depth, 0)
    x, y, z = np.moveaxis(velocity, -1, 0)
    cx = p["cx"] * (-np.expm1(-d / p["dx"])) / np.sqrt(1 + (x / p["satx"]) ** 2)
    cy = p["cy"] * (-np.expm1(-d / p["dy"])) / np.sqrt(1 + (y / p["saty"]) ** 2)
    power = d ** p["nz"]
    cz = p["cz"] * power / (power + p["dz"] ** p["nz"])
    limit = (0.5 * mass / np.asarray(dt))[:, None] if np.ndim(dt) else 0.5 * mass / dt
    cx, cy, cz = (np.minimum(value, limit) for value in (cx, cy, cz))
    bearing = p["bearing"] * (-np.expm1(-d / p["db"])) * np.exp(-d / p["soft"])
    dynamic = np.stack([-cx * x, -cy * y, bearing - cz * z], axis=-1)
    gate = -np.expm1(-envelope / p["blend"] ** 2)
    return ((1 - gate[..., None]) * baseline + gate[..., None] * dynamic) * (depth > 0)[..., None]


def fit_coefficients(data, baseline, train_end, radius):
    depth, velocity = data["depth_m"], data["foot_velocity_yaw_m_s"]
    actual = rotate_forces(data["mpm_force_w_N"], data["base_pose_w"], inverse=True)
    envelope = speed_envelope(depth, velocity, data["dt_s"])
    train = (data["time_s"] <= train_end)[:, None] & (depth > 0)
    if train.sum() < 200:
        raise ValueError("Insufficient training contact samples")
    # Only training targets determine normalization and constitutive coefficients.
    scale = np.maximum(5, np.sqrt(np.mean(actual[train] ** 2, axis=0)))
    initial = [280, 0.006, 3, 750, 0.04, 2, 20, 0.02, 0.2, 900, 0.02, 2, 0.05]
    # Preserve static support near zero slip. This threshold is a stability prior,
    # not identifiable from the predominantly rolling forward trajectory.
    lower = [1, 0.0001, 0.1, 1, 0.0001, 0.1, 0, 0.001, 0.03, 1, 0.001, 0.5, 0.05]
    upper = [1500, radius, 20, 3000, 0.2, 20, 150, 0.15, 3, 1700, 0.15, 4, 0.20]
    mass = data["metadata"]["robot_mass_kg"] / 4

    def residual(values):
        predicted = predict_numpy(depth, velocity, baseline, envelope, values, mass, data["dt_s"])
        return ((predicted - actual)[train] / scale).ravel()

    optimum = least_squares(residual, initial, bounds=(lower, upper), max_nfev=400, x_scale="jac")
    if not optimum.success:
        raise RuntimeError(optimum.message)
    return dict(zip(NAMES, optimum.x.tolist(), strict=True)), {
        "train_contact_foot_samples": int(train.sum()),
        "axis_loss_scale_N": scale.tolist(),
        "optimizer_nfev": optimum.nfev,
        "at_parameter_bounds": [name for name, flag in zip(NAMES, optimum.active_mask, strict=True) if flag],
    }


def replay(parameters, data):
    model = FootSandModel(parameters, 1)
    force = np.empty_like(data["mpm_force_w_N"])
    depth = torch.tensor(data["depth_m"], dtype=torch.float32)
    velocity = torch.tensor(data["foot_velocity_yaw_m_s"], dtype=torch.float32)
    xy = torch.tensor(data["foot_point_w_m"][..., :2], dtype=torch.float32)
    mass = torch.tensor([[data["metadata"]["robot_mass_kg"] / 4]])
    with torch.inference_mode():
        for i, dt in enumerate(data["dt_s"]):
            force[i] = model.step(
                depth[i : i + 1], velocity[i : i + 1], xy[i : i + 1], float(dt), stability_mass_kg=mass
            )[0].numpy()
    return rotate_forces(force, data["base_pose_w"])


def comparison(data, before, after, mask):
    actual = data["mpm_force_w_N"][mask]
    result = {"physics_frames": int(mask.sum()), "foot_samples": int(mask.sum() * 4)}
    for name, value in (("before", before[mask]), ("after", after[mask])):
        rmse = np.sqrt(np.mean((value - actual) ** 2, axis=(0, 1)))
        result[name] = {
            "rmse_xyz_N": rmse.tolist(),
            "nrmse_xyz": (rmse / np.maximum(1, np.sqrt(np.mean(actual**2, axis=(0, 1))))).tolist(),
            "mean_total_Fz_N": float(value[..., 2].sum(-1).mean()),
        }
    result["mean_total_mpm_Fz_N"] = float(actual[..., 2].sum(-1).mean())
    return result


def deploy_model(candidate: Path, destination: Path = DEFAULT_MODEL_PATH, *, terrain: str | None = None):
    parameters = json.loads(candidate.read_text())
    if parameters.get("schema_version") != 2 or not parameters["shadow_calibration"]["improves_blocked_validation"]:
        raise ValueError("Candidate must improve all three blocked-validation component RMSEs")
    previous = destination.read_bytes()
    old_sha = hashlib.sha256(previous).hexdigest()
    backups = terrain_directory(terrain, candidate) / "deployment_backups"
    backups.mkdir(parents=True, exist_ok=True)
    backup = backups / f"m20_foot_sand_{old_sha[:12]}.json"
    if not backup.exists():
        backup.write_bytes(previous)
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_bytes(candidate.read_bytes())
    temporary.replace(destination)
    print(f"Deployed: {destination}; previous model: {backup}", flush=True)


def fit_trace(trace: Path, output: Path, train_end=3.0, validation_start=3.2, deploy=False, terrain=None):
    with np.load(trace, allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files if k != "metadata"}
        data["metadata"] = json.loads(saved["metadata"].item())
    if data["metadata"]["proxy_force_applied"] or data["metadata"]["num_envs"] != 1:
        raise ValueError("Need a single-environment MPM-only shadow reference")
    if "episode_id" in data and len(np.unique(data["episode_id"])) != 1:
        raise ValueError("Refit requires one continuous episode; extract an episode before fitting")
    if not 0 < train_end < validation_start < data["time_s"][-1]:
        raise ValueError("Need training prefix, temporal gap, and nonempty validation suffix")
    if any(not np.isfinite(a).all() for k, a in data.items() if k != "metadata"):
        raise ValueError("Nonfinite source data")
    baseline_path = trace.parent / "model_snapshot.json"
    baseline = json.loads(baseline_path.read_text())
    # Always refit on the original intruder backbone; never learn a correction of a correction.
    baseline["domain"] = baseline.pop("intruder_domain", baseline["domain"])
    baseline.pop("shadow_contact", None)
    baseline.pop("shadow_calibration", None)
    baseline["schema_version"] = 1
    before = replay(baseline, data)
    coefficients, fit_info = fit_coefficients(
        data, rotate_forces(before, data["base_pose_w"], inverse=True), train_end, baseline["geometry"]["radius"]
    )
    parameters = copy.deepcopy(baseline)
    parameters["calibration"]["scope"] = (
        "intruder backbone benchmark only; active-model results are in shadow_calibration"
    )
    parameters.update(
        schema_version=2,
        recommended_physics_dt_s=0.0005,
        intruder_domain=copy.deepcopy(baseline["domain"]),
        training_depth_limit_m=0.20,
        domain={
            "max_depth_m": 0.28,
            "max_speed_m_s": 2.6,
            "max_vertical_speed_m_s": 0.60,
            "max_lateral_speed_m_s": 0.18,
        },
        shadow_contact={
            "function": "saturating_viscous_contact",
            "speed_filter_tau_s": FILTER_TAU,
            "coefficients": coefficients,
        },
    )
    after = replay(parameters, data)
    if not np.isfinite(after).all():
        raise ValueError("Nonfinite refitted replay")
    report = {
        "training": comparison(data, before, after, data["time_s"] <= train_end),
        "blocked_validation": comparison(data, before, after, data["time_s"] >= validation_start),
        "full_trace": comparison(data, before, after, np.ones(len(before), dtype=bool)),
    }
    old_error = np.array(report["blocked_validation"]["before"]["rmse_xyz_N"])
    new_error = np.array(report["blocked_validation"]["after"]["rmse_xyz_N"])
    parameters["shadow_calibration"] = {
        "source_trace": str(trace.resolve()),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "source_snapshot_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "training_end_s": train_end,
        "validation_start_s": validation_start,
        "temporal_gap_s": validation_start - train_end,
        "target": "four wheel MPM forces only; other robot forces excluded",
        "fit_info": fit_info,
        "improves_blocked_validation": bool(np.all(new_error < old_error)),
        "status": "single_trajectory_refinement_not_general_validation",
        "validation": report["blocked_validation"],
        "limitations": [
            "One forward-policy trajectory only; blocked suffix is not an independent rollout",
            "Speed, penetration and time co-vary; not all parameters are identifiable",
            "Experimental contact formula; no shin, thigh or belly support",
            "Expanded bounds describe this trace, not a fully sampled calibration grid",
            "Original intruder law is retained at sustained zero horizontal speed",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline_model.json").write_text(json.dumps(baseline, indent=2) + "\n")
    (output / "proxy_model.json").write_text(json.dumps(parameters, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output / "replay.npz",
        time_s=data["time_s"],
        before_force_w_N=before,
        after_force_w_N=after,
        mpm_force_w_N=data["mpm_force_w_N"],
        depth_m=data["depth_m"],
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
    # Window means are displayed; metrics above use every physical sample.
    groups = np.unique(data["control_step"])
    times = np.array([data["time_s"][data["control_step"] == group][-1] for group in groups])
    for force, label in ((data["mpm_force_w_N"], "MPM"), (before, "before"), (after, "refitted")):
        window = np.stack([force[data["control_step"] == g].mean(0) for g in groups])
        for foot, name in enumerate(("FL", "FR", "HL", "HR")):
            for component, axis in enumerate("XYZ"):
                ax = axes[foot, component]
                ax.plot(times, window[:, foot, component], label=label, linewidth=0.9)
                ax.set(title=f"{name}: F{axis}", ylabel="N")
                ax.axvspan(train_end, validation_start, color="gray", alpha=0.1)
                ax.axvline(validation_start, color="black", linestyle="--", alpha=0.4)
                ax.grid(alpha=0.2)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Time (s); dashed line: blocked validation begins")
    fig.tight_layout()
    fig.savefig(output / "before_after.png", dpi=150)
    plt.close(fig)
    lines = [
        "# Shadow-data foot contact refit",
        "",
        "Only training-prefix force targets enter the fit. No time or foot-ID feature is used at runtime.",
        f"Train <= {train_end}s; excluded gap; blocked validation >= {validation_start}s.",
        "This is a single-trajectory development check, not an independent validation rollout.",
        "",
        "| Split | Axis | Old RMSE (N) | New RMSE (N) | New NRMSE |",
        "|---|---|---:|---:|---:|",
    ]
    for split, item in report.items():
        for axis, old, new, relative in zip(
            "XYZ", item["before"]["rmse_xyz_N"], item["after"]["rmse_xyz_N"], item["after"]["nrmse_xyz"], strict=True
        ):
            lines.append(f"| {split} | {axis} | {old:.2f} | {new:.2f} | {relative:.1%} |")
    lines += ["", "![Replay comparison](before_after.png)", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps(report["blocked_validation"], indent=2), flush=True)
    print(f"Candidate: {output / 'proxy_model.json'}", flush=True)
    if deploy:
        deploy_model(output / "proxy_model.json", terrain=terrain)
    return parameters, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_terrain_argument(parser)
    parser.add_argument(
        "--trace", type=Path, help="Continuous shadow trace; defaults to terrain/shadow/walk_forward_001"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-end", type=float, default=3.0)
    parser.add_argument("--validation-start", type=float, default=3.2)
    parser.add_argument(
        "--deploy", action="store_true", help="Back up and replace the deployed model if errors improve"
    )
    args = parser.parse_args()
    terrain = terrain_directory(args.terrain, args.trace)
    trace = args.trace or terrain / "shadow/walk_forward_001/traces.npz"
    output = args.output_dir or terrain / "fit/contact" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    fit_trace(trace, output, args.train_end, args.validation_start, args.deploy, terrain.name)


if __name__ == "__main__":
    main()
