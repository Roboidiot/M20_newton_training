"""CPU-only force comparison; no simulator or checkpoint loading."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FEET = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")


def metrics(actual, predicted, dt):
    error = predicted - actual
    rms = float(np.sqrt(np.mean(actual**2)))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "samples": len(actual),
        "rmse_N": rmse,
        "nrmse": rmse / max(1.0, rms),
        "bias_N": float(error.mean()),
        "mae_N": float(np.abs(error).mean()),
        "peak_mpm_abs_N": float(np.abs(actual).max()),
        "peak_proxy_abs_N": float(np.abs(predicted).max()),
        "mpm_impulse_Ns": float(np.sum(actual * dt)),
        "proxy_impulse_Ns": float(np.sum(predicted * dt)),
        "passed": bool(rmse <= max(5, 0.25 * rms)),
    }


def analyze(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files if k != "metadata"}
        metadata = json.loads(saved["metadata"].item())
    output = path.parent
    time, dt = data["time_s"], data["dt_s"]
    valid = time > metadata["warmup_s"] + 1e-8
    if not valid.any():
        valid = np.ones(len(time), dtype=bool)
        fallback = "No post-warmup samples: metrics include warmup and are diagnostic only."
    else:
        fallback = None
    actual, proxy = data["mpm_force_w_N"], data["proxy_force_w_N"]
    rows = []
    for foot, name in enumerate(FEET):
        contact = np.linalg.norm(actual[:, foot], axis=-1) > 1.0
        for subset, mask in (
            ("all", valid),
            ("mpm_contact", valid & contact),
            ("in_domain_contact", valid & contact & ~data["out_of_domain"][:, foot]),
        ):
            if not mask.any():
                continue
            for component, axis in enumerate("XYZ"):
                rows.append(
                    {
                        "foot": name,
                        "axis": axis,
                        "subset": subset,
                        **metrics(actual[mask, foot, component], proxy[mask, foot, component], dt[mask]),
                    }
                )
    summary = {
        "status": metadata["status"],
        "stop_reason": metadata["stop_reason"],
        "warning": fallback,
        "shadow_forces_applied": False,
        "physics_samples": len(time),
        "post_warmup_samples": int((time > metadata["warmup_s"] + 1e-8).sum()),
        "recorded_duration_s": float(dt.sum()),
        "out_of_domain_fraction_per_foot": dict(zip(FEET, data["out_of_domain"][valid].mean(0).tolist(), strict=True)),
        "depth_over_20cm_fraction_per_foot": (data["depth_m"][valid] > 0.2).mean(0).tolist(),
        "horizontal_speed_over_030_fraction_per_foot": (
            np.linalg.norm(data["foot_velocity_yaw_m_s"][valid, :, :2], axis=-1) > 0.3
        )
        .mean(0)
        .tolist()
        if "foot_velocity_yaw_m_s" in data
        else None,
        "mean_total_mpm_foot_Fz_N": float(actual[valid, :, 2].sum(-1).mean()),
        "mean_total_proxy_foot_Fz_N": float(proxy[valid, :, 2].sum(-1).mean()),
        "mean_other_robot_mpm_Fz_N": float(data["mpm_other_robot_force_w_N"][valid, 2].mean()),
        "runtime_vs_unlimited_rms_difference_N": np.sqrt(
            np.mean((proxy[valid] - data["proxy_unlimited_force_w_N"][valid]) ** 2, axis=0)
        ).tolist(),
        "robot_weight_N": metadata["robot_weight_N"],
        "base_clearance_start_end_m": (data["base_pose_w"][[0, -1], 2] - metadata["surface_height_m"]).tolist(),
        "metrics": rows,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Preserve every physical sample in NPZ and export convenient long-format foot curves.
    with (output / "forces.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "foot",
                "depth_m",
                "out_of_domain",
                "mpm_Fx_N",
                "mpm_Fy_N",
                "mpm_Fz_N",
                "proxy_Fx_N",
                "proxy_Fy_N",
                "proxy_Fz_N",
            ]
        )
        for i, t in enumerate(time):
            for foot, name in enumerate(FEET):
                writer.writerow(
                    [
                        t,
                        name,
                        data["depth_m"][i, foot],
                        data["out_of_domain"][i, foot],
                        *actual[i, foot],
                        *proxy[i, foot],
                    ]
                )
    # Same control-window impulse / duration, with no last-substep sampling bias.
    groups = np.unique(data["control_step"])
    averaged = {"control_step": groups}
    for key in ("mpm_force_w_N", "proxy_force_w_N", "proxy_unlimited_force_w_N"):
        averaged[key] = np.stack(
            [
                np.sum(data[key][data["control_step"] == group] * dt[data["control_step"] == group, None, None], axis=0)
                / dt[data["control_step"] == group].sum()
                for group in groups
            ]
        )
    averaged["time_s"] = np.array([time[data["control_step"] == group][-1] for group in groups])
    np.savez_compressed(output / "control_windows.npz", **averaged)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for filename, curves in (("foot_forces.png", data), ("foot_forces_20ms.png", averaged)):
        fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
        t = curves["time_s"]
        for foot, name in enumerate(FEET):
            for component, axis in enumerate("XYZ"):
                ax = axes[foot, component]
                for key, label, style in (
                    ("mpm_force_w_N", "MPM", "-"),
                    ("proxy_force_w_N", "proxy (runtime)", "-"),
                    ("proxy_unlimited_force_w_N", "proxy (unlimited)", "--"),
                ):
                    ax.plot(
                        t, curves[key][:, foot, component], label=label, linewidth=0.85, linestyle=style, alpha=0.85
                    )
                ax.axvspan(0, metadata["warmup_s"], color="gray", alpha=0.1)
                ax.set(title=f"{name}: F{axis}", ylabel="Force (N)")
                ax.grid(alpha=0.25)
                if foot == 3:
                    ax.set_xlabel("Time after robot release (s)")
        axes[0, 0].legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time, actual[:, :, 2].sum(-1), label="MPM feet")
    axes[0].plot(time, proxy[:, :, 2].sum(-1), label="proxy feet")
    axes[0].plot(time, data["mpm_other_robot_force_w_N"][:, 2], label="MPM other robot bodies", alpha=0.6)
    axes[0].axhline(metadata["robot_weight_N"], label="weight", color="black", linestyle="--")
    axes[0].set_ylabel("Total Fz (N)")
    for foot, name in enumerate(FEET):
        axes[1].plot(time, data["depth_m"][:, foot], label=name)
    axes[1].axhline(0.2, color="black", linestyle="--")
    axes[1].set_ylabel("Bottom depth (m)")
    axes[2].plot(time, data["base_pose_w"][:, 2] - metadata["surface_height_m"], label="base above reference sand")
    axes[2].set(ylabel="Height (m)", xlabel="Time after robot release (s)")
    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "support_and_depth.png", dpi=150)
    plt.close(fig)
    lines = [
        "# M20 single-environment MPM / shadow proxy",
        "",
        f"Checkpoint: `{metadata['checkpoint']}`",
        f"Status: {metadata['status']}; {metadata['stop_reason']}",
        f"Duration: {dt.sum():.3f} s; warmup excluded: {metadata['warmup_s']} s.",
        f"Command: {metadata.get('command', 'not recorded')} (vx, vy, wz); "
        f"reference surface: {metadata['surface_height_m']:.4f} m.",
        "Only MPM forces act on the robot. Proxy parameters are fixed; no fitting or PPO update.",
        "Inputs are sampled at the MPM solve boundary, after staggered coupling prepares colliders.",
        "The unrestricted prediction is also saved to distinguish explicit stability regularization.",
        f"Mean post-warmup total foot Fz: MPM {summary['mean_total_mpm_foot_Fz_N']:.2f} N / "
        f"proxy {summary['mean_total_proxy_foot_Fz_N']:.2f} N; robot weight {metadata['robot_weight_N']:.2f} N.",
        "",
        "| Foot | Axis | Contact RMSE (N) | NRMSE | Bias (N) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        if row["subset"] == "mpm_contact":
            lines.append(
                f"| {row['foot']} | {row['axis']} | {row['rmse_N']:.2f} | {row['nrmse']:.1%} | {row['bias_N']:.2f} |"
            )
    lines += [
        "",
        "Out-of-domain fractions (current model bounds; see model_snapshot.json):",
        json.dumps(summary["out_of_domain_fraction_per_foot"]),
        "This run includes rolling/tilt and may include non-foot support, "
        "which the foot-only calibration does not model.",
        "Large out-of-domain fractions and non-foot support prevent interpreting the run as proxy validation passed.",
        "Errors in the table are computed on raw physical samples, not smoothed curves; "
        "in-domain contact subsets are listed separately in metrics.csv.",
        "",
        "![Control-window mean forces](foot_forces_20ms.png)",
        "![Force traces](foot_forces.png)",
        "![Support and penetration](support_and_depth.png)",
        "",
    ]
    if fallback:
        lines.insert(2, fallback)
    (output / "report.md").write_text("\n".join(lines))
    return summary
