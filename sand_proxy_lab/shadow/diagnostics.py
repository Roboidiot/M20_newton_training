"""CPU-only diagnostics of recorded episode 0. No simulation, fitting, or model writes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyze(directory):
    OUT = Path(directory)
    with np.load(OUT / "traces.npz", allow_pickle=False) as saved:
        d = {k: saved[k].copy() for k in saved.files if k != "metadata"}
        meta = json.loads(saved["metadata"].item())
    valid = d["episode_time_s"] > meta["warmup_s"] + 1e-8
    if not valid.any():
        raise ValueError("No post-warmup samples are available for episode diagnostics")
    a, f = d["mpm_force_w_N"], d["proxy_force_w_N"]
    e = f - a
    mask = np.broadcast_to(valid[:, None], a.shape[:2])
    vel = d["foot_velocity_yaw_m_s"]
    depth = d["depth_m"]

    def stats(m):
        aa, ff = a[m], f[m]
        ee = ff - aa
        rmse = np.sqrt(np.mean(ee**2, axis=0))
        return {
            "foot_samples": int(m.sum()),
            "rmse_xyz_N": rmse.tolist(),
            "nrmse_xyz": (rmse / np.maximum(1, np.sqrt(np.mean(aa**2, axis=0)))).tolist(),
            "bias_xyz_N": ee.mean(0).tolist(),
            "correlation_xyz": [float(np.corrcoef(aa[:, i], ff[:, i])[0, 1]) for i in range(3)],
            "mean_Fz_mpm_N": float(aa[:, 2].mean()),
            "mean_Fz_proxy_N": float(ff[:, 2].mean()),
        }

    groups = {
        "down_vz_under_minus005": mask & (vel[..., 2] < -0.05),
        "up_vz_over005": mask & (vel[..., 2] > 0.05),
        "slow_z": mask & (np.abs(vel[..., 2]) <= 0.05),
    }
    details = {"all": stats(mask), "vertical_velocity_groups": {key: stats(m) for key, m in groups.items()}}
    for key, m in groups.items():
        details["vertical_velocity_groups"][key]["Fz_squared_error_fraction"] = float(
            np.sum(e[..., 2][m] ** 2) / np.sum(e[..., 2][mask] ** 2)
        )
    end = float(d["episode_time_s"][-1])
    edges = [meta["warmup_s"], end / 3, 2 * end / 3, end]
    if edges[1] <= edges[0]:
        edges = np.linspace(meta["warmup_s"], end, 4).tolist()
    windows = tuple(zip(edges[:-1], edges[1:], strict=True))
    details["time_windows"] = []
    for lo, hi in windows:
        m = mask & (d["episode_time_s"][:, None] > lo + 1e-8) & (d["episode_time_s"][:, None] <= hi + 1e-8)
        details["time_windows"].append({"time_range_s": [lo, hi], "mean_depth_m": float(depth[m].mean()), **stats(m)})
    details["coverage_failure_fractions"] = {
        "depth_over028": float((depth[valid] > 0.28).mean()),
        "horizontal_speed_over26": float((np.linalg.norm(vel[valid, ..., :2], axis=-1) > 2.6).mean()),
        "vertical_speed_over060": float((np.abs(vel[valid, ..., 2]) > 0.6).mean()),
        "lateral_speed_over018": float((np.abs(vel[valid, ..., 1]) > 0.18).mean()),
    }
    details["runtime_vs_unlimited_rmse_xyz_N"] = np.sqrt(
        np.mean((f[valid] - d["proxy_unlimited_force_w_N"][valid]) ** 2, axis=(0, 1))
    ).tolist()
    details["base_clearance_start_end_m"] = (
        d["base_pose_w"][[0, -1], 2] - meta["episodes"][0]["surface_height_m"]
    ).tolist()
    details["maximum_base_tilt_deg"] = float(
        np.degrees(np.arccos(np.clip(1 - 2 * np.sum(d["base_pose_w"][:, 3:5] ** 2, axis=-1), -1, 1))).max()
    )
    (OUT / "diagnostics.json").write_text(json.dumps(details, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    colors = ("#2374aa", "#c18a38", "#427b65")
    for axis, letter in enumerate("xyz"):
        ax = axes[0, axis]
        aa, ff = a[..., axis][mask], f[..., axis][mask]
        ax.scatter(aa, ff, s=4, alpha=0.15, color=colors[axis], rasterized=True)
        lo, hi = min(aa.min(), ff.min()), max(aa.max(), ff.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#666666", linewidth=1)
        ax.set(
            xlabel="MPM 实测 / N",
            ylabel="代理预测 / N",
            title=f"F{letter}：逐物理步对照，RMSE={details['all']['rmse_xyz_N'][axis]:.1f} N",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    bins = np.arange(0.04, 0.281, 0.02)
    centers, rmses = [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = mask & (depth >= lo) & (depth < hi)
        if m.sum() >= 30:
            centers.append(100 * (lo + hi) / 2)
            rmses.append(np.sqrt(np.mean(e[m] ** 2, axis=0)))
    for axis, letter in enumerate("xyz"):
        axes[1, 0].plot(centers, np.array(rmses)[:, axis], "o-", color=colors[axis], label=f"F{letter}")
    axes[1, 0].set(xlabel="足底压入深度 / cm", ylabel="RMSE / N", title="按深度分组：非受控深度试验")
    axes[1, 0].legend()
    keys = tuple(groups)
    for axis, letter in enumerate("xyz"):
        axes[1, 1].bar(
            np.arange(3) + (axis - 1) * 0.24,
            [details["vertical_velocity_groups"][k]["rmse_xyz_N"][axis] for k in keys],
            width=0.24,
            color=colors[axis],
            label=f"F{letter}",
        )
    axes[1, 1].set(
        xticks=range(3),
        xticklabels=["向下 vz<−0.05", "向上 vz>0.05", "低竖直速度"],
        ylabel="RMSE / N",
        title="按瞬时竖直速度分组（m/s）",
    )
    axes[1, 1].legend()
    for axis, letter in enumerate("xyz"):
        axes[1, 2].bar(
            np.arange(3) + (axis - 1) * 0.24,
            [entry["rmse_xyz_N"][axis] for entry in details["time_windows"]],
            width=0.24,
            color=colors[axis],
            label=f"F{letter}",
        )
    axes[1, 2].set(
        xticks=range(3),
        xticklabels=[f"{lo:g}–{hi:g} s" for lo, hi in windows],
        ylabel="RMSE / N",
        title="回合前、中、后段误差",
    )
    axes[1, 2].legend()
    for ax in axes[1]:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"RandomAgent 单回合误差诊断\n分析 {meta['warmup_s']:g}–{end:g} s；{int(mask.sum())} 个足端样本；"
        f"物理步长 {np.mean(d['dt_s']) * 1000:g} ms，无平滑或时间平移",
        fontsize=14,
    )
    fig.savefig(OUT / "error_diagnostics.png", dpi=170)
    fig.savefig(OUT / "error_diagnostics.svg")
    plt.close(fig)
    print(json.dumps(details, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.directory)


if __name__ == "__main__":
    main()
