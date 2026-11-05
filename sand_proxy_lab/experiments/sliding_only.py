"""Compare a standalone sliding law against the deployed mixed law on existing data.

Run from the project root with ``python -m sand_proxy_lab.experiments.sliding_only``.
No MPM simulation, backbone calibration, deployment or policy training is performed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH
from sand_proxy_lab.calibration.config import COL
from sand_proxy_lab.calibration.foot_contact_fit import load_foot_data
from sand_proxy_lab.calibration.foot_contact_fit import replay as replay_intruder
from sand_proxy_lab.paths import add_terrain_argument, terrain_directory
from sand_proxy_lab.shadow.refit import NAMES, replay, rotate_forces, speed_envelope

SLIDING_NAMES = NAMES[:-1]  # No blend parameter, filter, plastic state or footprint memory.
INITIAL = np.array([280, 0.006, 3, 750, 0.04, 2, 20, 0.02, 0.2, 900, 0.02, 2], dtype=float)
LOWER = np.array([1, 0.0001, 0.1, 1, 0.0001, 0.1, 0, 0.001, 0.03, 1, 0.001, 0.5])
MODELS = ("current_mixed", "sliding_unchanged", "sliding_shadow_fit", "sliding_joint_fit")
LABELS = ("Current mixed", "Slide: unchanged", "Slide: shadow fit", "Slide: joint fit")


def predict(depth, velocity, coefficients, *, mass=None, dt=0.0025, domain=None):
    """Standalone F_slide, including the same runtime clamps and explicit damping limit."""
    domain = domain or {"max_depth_m": 0.28, "max_speed_m_s": 2.6}
    active = np.asarray(depth) > 0
    d = np.clip(depth, 0, domain["max_depth_m"])
    x, y, z = np.moveaxis(np.clip(velocity, -domain["max_speed_m_s"], domain["max_speed_m_s"]), -1, 0)
    p = coefficients
    cx = p["cx"] * (-np.expm1(-d / p["dx"])) / np.sqrt(1 + (x / p["satx"]) ** 2)
    cy = p["cy"] * (-np.expm1(-d / p["dy"])) / np.sqrt(1 + (y / p["saty"]) ** 2)
    power = d ** p["nz"]
    cz = p["cz"] * power / (power + p["dz"] ** p["nz"])
    if mass is not None:
        limit = 0.5 * np.asarray(mass) / np.asarray(dt)
        while np.ndim(limit) < np.ndim(d):
            limit = np.expand_dims(limit, -1)
        cx, cy, cz = (np.minimum(c, limit) for c in (cx, cy, cz))
    bearing = p["bearing"] * (-np.expm1(-d / p["db"])) * np.exp(-d / p["soft"])
    return np.stack([-cx * x, -cy * y, bearing - cz * z], axis=-1) * active[..., None]


def upper_bounds(radius, *, joint=False):
    # Shadow-only fit keeps the original 12 bounds exactly. The joint variant
    # admits probe-scale bearing (kN); it is a separate, explicitly labelled fit.
    return np.array([1500, radius, 20, 3000, 0.2, 20, 10000 if joint else 150, 0.15, 3, 1700, 0.15, 4])


def shadow_training(data, train_end):
    mask = (data["time_s"] <= train_end)[:, None] & (data["depth_m"] > 0)
    target = rotate_forces(data["mpm_force_w_N"], data["base_pose_w"], inverse=True)[mask]
    if len(target) < 200:
        raise ValueError("Insufficient training contacts")
    return {
        "name": "shadow_training_prefix",
        "depth": data["depth_m"][mask],
        "velocity": data["foot_velocity_yaw_m_s"][mask],
        "actual": target,
        "dt": np.broadcast_to(data["dt_s"][:, None], mask.shape)[mask],
        "mass": data["metadata"]["robot_mass_kg"] / 4,
        "scale": np.maximum(5, np.sqrt(np.mean(target**2, axis=0))),
    }


def intruder_mask(experiment):
    a = experiment["data"]
    mask = a[:, COL["measure"]] > 0.5
    if experiment["case"].motion == "press":
        mask &= (a[:, COL["depth"]] >= 0.01 - 1e-8) & (a[:, COL["depth"]] <= 0.2 + 1e-8)
    return mask


def probe_training(experiments, dt):
    result = []
    for e in experiments:
        if e["case"].split != "train":
            continue
        a = e["data"][intruder_mask(e)]
        target = a[:, COL["fx"] : COL["fz"] + 1]
        result.append(
            {
                "name": e["case"].name,
                "depth": a[:, COL["depth"]],
                "velocity": a[:, COL["vx"] : COL["vz"] + 1],
                "actual": target,
                "dt": dt,
                "mass": None,  # Prescribed kinematic probe: no effective robot mass.
                "scale": np.maximum(5, np.sqrt(np.mean(target**2, axis=0))),
            }
        )
    return result


def fit(groups, radius, *, joint=False):
    """Three predeclared starts; select by training loss only, never validation."""
    upper = upper_bounds(radius, joint=joint)
    starts = [INITIAL.copy(), INITIAL.copy(), INITIAL.copy()]
    starts[1][[1, 2, 4, 5, 7, 8, 10, 11]] = [0.06, 0.4, 0.1, 5, 0.08, 1, 0.08, 1]
    starts[2][[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] = [900, 0.03, 1, 1400, 0.01, 0.5, 100, 0.04, 0.8, 1400, 0.04, 3]

    def residual(values):
        p = dict(zip(SLIDING_NAMES, values, strict=True))
        parts = []
        for i, group in enumerate(groups):
            error = (
                predict(group["depth"], group["velocity"], p, mass=group["mass"], dt=group["dt"]) - group["actual"]
            ) / group["scale"]
            if joint:
                # Equal total source weight: whole-robot train vs probe train.
                # Within probes each of the 114 experiments has equal weight.
                weight = 0.5 if i == 0 else 0.5 / (len(groups) - 1)
                error = error * np.sqrt(weight / len(error))
            parts.append(error.ravel())
        return np.concatenate(parts)

    solutions = []
    for i, start in enumerate(starts):
        optimum = least_squares(residual, start, bounds=(LOWER, upper), max_nfev=400, x_scale="jac")
        print(
            f"{'joint' if joint else 'shadow'} start {i}: cost={optimum.cost:.6g}, "
            f"nfev={optimum.nfev}, success={optimum.success}",
            flush=True,
        )
        solutions.append(optimum)
    successful = [(i, s) for i, s in enumerate(solutions) if s.success and np.isfinite(s.fun).all()]
    if not successful:
        raise RuntimeError("All optimizer starts failed")
    selected, optimum = min(successful, key=lambda item: item[1].cost)
    return dict(zip(SLIDING_NAMES, optimum.x.tolist(), strict=True)), {
        "selected_start": selected,
        "selection": "lowest successful training loss; validation targets never inspected",
        "starts": [
            {"cost": float(s.cost), "nfev": s.nfev, "success": bool(s.success), "message": s.message} for s in solutions
        ],
        "lower_bounds": dict(zip(SLIDING_NAMES, LOWER.tolist(), strict=True)),
        "upper_bounds": dict(zip(SLIDING_NAMES, upper.tolist(), strict=True)),
        "at_bounds": [k for k, flag in zip(SLIDING_NAMES, optimum.active_mask, strict=True) if flag],
        "training_groups": [
            {"name": g["name"], "samples": len(g["depth"]), "axis_scale_N": g["scale"].tolist()} for g in groups
        ],
    }


def scores(actual, predicted):
    a, p = np.asarray(actual).reshape(-1, 3), np.asarray(predicted).reshape(-1, 3)
    error = p - a
    rmse = np.sqrt(np.mean(error**2, axis=0))
    return {
        "foot_samples": len(a),
        "rmse_xyz_N": rmse.tolist(),
        "nrmse_xyz": (rmse / np.maximum(1, np.sqrt(np.mean(a**2, axis=0)))).tolist(),
        "bias_xyz_N": error.mean(0).tolist(),
        "mae_xyz_N": np.abs(error).mean(0).tolist(),
    }


def load_trace(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files if k != "metadata"}
        data["metadata"] = json.loads(z["metadata"].item())
    if data["metadata"]["proxy_force_applied"] or data["metadata"]["num_envs"] != 1:
        raise ValueError("Expected a single-environment MPM-only shadow trace")
    if "episode_id" in data and len(np.unique(data["episode_id"])) != 1:
        raise ValueError("Use one continuous episode")
    if any(not np.isfinite(v).all() for k, v in data.items() if k != "metadata"):
        raise ValueError("Nonfinite source trace")
    return data


def static_support(coefficients, weight):
    depth = np.linspace(0, 0.28, 2801)
    force = predict(depth, np.zeros((len(depth), 3)), coefficients)[:, 2]
    peak = int(np.argmax(force))
    roots = np.flatnonzero((force[:-1] - weight / 4) * (force[1:] - weight / 4) < 0)
    return {
        "max_per_foot_N": float(force[peak]),
        "max_total_N": float(4 * force[peak]),
        "peak_depth_m": float(depth[peak]),
        "max_support_weight_ratio": float(4 * force[peak] / weight),
        "equal_load_equilibria": [
            {"depth_m": float(depth[i]), "restoring_slope": bool(force[i + 1] > force[i])} for i in roots
        ],
    }


def trace_plot(output, name, data, predictions, train_end, validation_start):
    import matplotlib.pyplot as plt

    groups = np.unique(data["control_step"])
    times = np.array([data["time_s"][data["control_step"] == g][-1] for g in groups])

    def average(values):
        return np.stack(
            [
                np.average(values[data["control_step"] == g], axis=0, weights=data["dt_s"][data["control_step"] == g])
                for g in groups
            ]
        )

    fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True, layout="constrained")
    colors = ["#2878b5", "#a0a0a0", "#dc702b", "#549e47"]
    for values, label, color in [(data["mpm_force_w_N"], "MPM", "#152437")] + [
        (predictions[key], label, color) for key, label, color in zip(MODELS, LABELS, colors, strict=True)
    ]:
        mean = average(values)
        for foot, foot_name in enumerate(("FL", "FR", "HL", "HR")):
            for axis, letter in enumerate("XYZ"):
                ax = axes[foot, axis]
                ax.plot(times, mean[:, foot, axis], label=label, color=color, linewidth=0.8, alpha=0.85)
                ax.set(title=f"{foot_name} F{letter}", ylabel="N")
                ax.grid(alpha=0.2)
                if name == "fit_source":
                    ax.axvspan(train_end, validation_start, color="gray", alpha=0.1)
                    ax.axvline(validation_start, color="gray", linestyle=":")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.suptitle(f"{name}: force comparison (20 ms impulse-weighted display; metrics use raw samples)")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.savefig(output / f"{name}_forces.png", dpi=150)
    plt.close(fig)


def run(terrain, output, train_end=3.0, validation_start=3.2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deployed_bytes = DEFAULT_MODEL_PATH.read_bytes()
    current = json.loads(deployed_bytes)
    sources = {
        "fit_source": terrain / "shadow/walk_forward_001/traces.npz",
        "independent_forward": terrain / "shadow/refit_validation_offset065/traces.npz",
        "independent_random": terrain / "shadow/random_30s_001/episode_000_analysis/traces.npz",
    }
    traces = {name: load_trace(path) for name, path in sources.items()}
    if not 0 < train_end < validation_start < traces["fit_source"]["time_s"][-1]:
        raise ValueError("Need training prefix, gap and nonempty held-out suffix")
    config, experiments = load_foot_data(terrain / "intruder")
    output.mkdir(parents=True, exist_ok=False)
    (output / "current_model_snapshot.json").write_bytes(deployed_bytes)
    training = shadow_training(traces["fit_source"], train_end)
    fitted, info = fit([training], current["geometry"]["radius"])
    joint, joint_info = fit(
        [training, *probe_training(experiments, config["dt"])], current["geometry"]["radius"], joint=True
    )
    coefficients = {
        "sliding_unchanged": {k: current["shadow_contact"]["coefficients"][k] for k in SLIDING_NAMES},
        "sliding_shadow_fit": fitted,
        "sliding_joint_fit": joint,
    }
    manifest = {
        "deployed_model_sha256": hashlib.sha256(deployed_bytes).hexdigest(),
        "traces": {
            k: {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for k, p in sources.items()
        },
        "intruder_files": {e["case"].name: e["sha256"] for e in experiments},
        "train_end_s": train_end,
        "validation_start_s": validation_start,
        "fit_method": "bounded least_squares, 3 fixed starts, training-only selection",
        "shadow_objective": "same positive-depth train mask, per-axis RMS scales and bounds as original; no gate",
        "joint_objective": "50% shadow prefix, 50% probe train; equal probe-case weight; per-group axis RMS >=5 N",
        "joint_bounds_difference": "bearing upper bound raised from 150 to 10000 N for kN-scale probe loads",
        "no_backbone_parameters_used_in_pure_models": True,
        "independent_validation_used_for_fitting": False,
        "force_metrics": "world XYZ; every original physical sample; no time shift or smoothing",
        "closed_loop_simulation": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, values in coefficients.items():
        candidate = {
            "schema_version": 1,
            "model_type": "m20_foot_sliding_only_experiment",
            "geometry": copy.deepcopy(current["geometry"]),
            "material": copy.deepcopy(current["material"]),
            "domain": copy.deepcopy(current["domain"]),
            "coefficients": values,
            "formula": "F_slide directly; d<=0 => zero; no blend, filter, plastic or footprint state",
            "fit_info": info if name == "sliding_shadow_fit" else joint_info if name == "sliding_joint_fit" else None,
            "deployment_status": "offline_candidate_not_accepted_by_FootSandModel",
        }
        (output / f"{name}.json").write_text(json.dumps(candidate, indent=2) + "\n")

    result = {"shadow": {}, "intruder_validation": {}, "static_support": {}, "current_gate": {}}
    rows = []
    for name, data in traces.items():
        print(f"Replaying {name}", flush=True)
        predictions = {"current_mixed": replay(current, data)}
        for key, values in coefficients.items():
            local = predict(
                data["depth_m"],
                data["foot_velocity_yaw_m_s"],
                values,
                mass=data["metadata"]["robot_mass_kg"] / 4,
                dt=data["dt_s"],
                domain=current["domain"],
            )
            predictions[key] = rotate_forces(local, data["base_pose_w"])
        time = data["time_s"]
        post = time > data["metadata"]["warmup_s"] + 1e-8
        correction = current["shadow_contact"]
        envelope = speed_envelope(
            data["depth_m"], data["foot_velocity_yaw_m_s"], data["dt_s"], correction["speed_filter_tau_s"]
        )
        gate = -np.expm1(-envelope / correction["coefficients"]["blend"] ** 2)
        contact = post[:, None] & (data["depth_m"] > 0)
        result["current_gate"][name] = {
            "post_warmup_contact_samples": int(contact.sum()),
            "mean": float(gate[contact].mean()),
            "fraction_above_099": float((gate[contact] > 0.99).mean()),
            "fraction_above_0999": float((gate[contact] > 0.999).mean()),
        }
        masks = {
            "post_warmup": np.broadcast_to(post[:, None], data["depth_m"].shape),
            "warmup": np.broadcast_to((~post)[:, None], data["depth_m"].shape),
            "low_horizontal_speed_contact": post[:, None]
            & (data["depth_m"] > 0)
            & (np.linalg.norm(data["foot_velocity_yaw_m_s"][..., :2], axis=-1) < 0.05),
            "upward_contact": post[:, None] & (data["depth_m"] > 0) & (data["foot_velocity_yaw_m_s"][..., 2] > 0.05),
            "downward_contact": post[:, None] & (data["depth_m"] > 0) & (data["foot_velocity_yaw_m_s"][..., 2] < -0.05),
        }
        if name == "fit_source":
            masks.update(
                training=np.broadcast_to((time <= train_end)[:, None], data["depth_m"].shape),
                blocked_validation=np.broadcast_to((time >= validation_start)[:, None], data["depth_m"].shape),
            )
        result["shadow"][name] = {}
        for subset, mask in masks.items():
            if not mask.any():
                continue
            metrics = {key: scores(data["mpm_force_w_N"][mask], value[mask]) for key, value in predictions.items()}
            result["shadow"][name][subset] = metrics
            for key, value in metrics.items():
                rows.append({"dataset": name, "subset": subset, "model": key, **value})
        np.savez_compressed(
            output / f"{name}_replay.npz",
            time_s=time,
            dt_s=data["dt_s"],
            depth_m=data["depth_m"],
            mpm_force_w_N=data["mpm_force_w_N"],
            **predictions,
        )
        trace_plot(output, name, data, predictions, train_end, validation_start)

    print("Replaying held-out probe experiments", flush=True)
    validation = [e for e in experiments if e["case"].split == "validation"]
    _, mixed_probe = replay_intruder(current, validation, config["dt"])
    probe_scores = []
    for index, e in enumerate(validation):
        mask = intruder_mask(e)
        a = e["data"][mask]
        predictions = {"current_mixed": mixed_probe[index, : len(e["data"])][mask]}
        predictions.update(
            {
                key: predict(
                    a[:, COL["depth"]],
                    a[:, COL["vx"] : COL["vz"] + 1],
                    values,
                    dt=config["dt"],
                    domain=current["domain"],
                )
                for key, values in coefficients.items()
            }
        )
        actual = a[:, COL["fx"] : COL["fz"] + 1]
        axis = {"press": 2, "pull": 2, "forward": 0, "lateral": 1}[e["case"].motion]
        for key, value in predictions.items():
            metric = scores(actual, value)
            rms = float(np.sqrt(np.mean(actual[:, axis] ** 2)))
            probe_scores.append(
                {
                    "case": e["case"].name,
                    "motion": e["case"].motion,
                    "model": key,
                    "primary_axis": "XYZ"[axis],
                    "primary_rmse_N": metric["rmse_xyz_N"][axis],
                    "passed": metric["rmse_xyz_N"][axis] <= max(5, 0.25 * rms),
                    **metric,
                }
            )
    for motion in ("press", "pull", "forward", "lateral"):
        result["intruder_validation"][motion] = {}
        for key in MODELS:
            group = [r for r in probe_scores if r["model"] == key and r["motion"] == motion]
            result["intruder_validation"][motion][key] = {
                "mean_case_primary_rmse_N": float(np.mean([r["primary_rmse_N"] for r in group])),
                "passed": sum(r["passed"] for r in group),
                "cases": len(group),
            }
    (output / "probe_case_metrics.json").write_text(json.dumps(probe_scores, indent=2) + "\n")
    weight = traces["fit_source"]["metadata"]["robot_weight_N"]
    for key, values in coefficients.items():
        result["static_support"][key] = static_support(values, weight)
    result["robot_weight_N"] = weight
    result["fit_info"] = {"shadow": info, "joint": joint_info}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "shadow_metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, (name, subset) in zip(
        axes,
        [
            ("fit_source", "blocked_validation"),
            ("independent_forward", "post_warmup"),
            ("independent_random", "post_warmup"),
        ],
        strict=True,
    ):
        for i, (key, label) in enumerate(zip(MODELS, LABELS, strict=True)):
            ax.bar(
                np.arange(3) + (i - 1.5) * 0.19,
                result["shadow"][name][subset][key]["rmse_xyz_N"],
                width=0.19,
                label=label,
            )
        ax.set(title=name, ylabel="raw-sample RMSE (N)", xticks=np.arange(3), xticklabels=list("XYZ"))
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.savefig(output / "rmse_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    depth = np.linspace(0, 0.28, 1000)
    for key, label in zip(MODELS[1:], LABELS[1:], strict=True):
        ax.plot(depth * 100, 4 * predict(depth, np.zeros((len(depth), 3)), coefficients[key])[:, 2], label=label)
    ax.axhline(weight, color="black", linestyle="--", label=f"robot weight: {weight:.2f} N")
    ax.set(
        xlabel="equal penetration of four feet (cm)",
        ylabel="static total support (N)",
        title="Standalone sliding law at zero velocity; no backbone or history",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output / "static_support.png", dpi=160)
    plt.close(fig)

    lines = [
        "# 纯滑移响应离线对照实验",
        "",
        "使用现有数据，未运行新 MPM、训练或发布。",
        "纯滑移模型只保留 F_slide 的 12 个参数，不包含基底、混合权重、速度历史或足印历史。",
        "",
        "## 实验设计",
        "",
        "- current_mixed：当前部署模型，同轨迹重放。",
        "- sliding_unchanged：直接删除基底与混合，保留当前滑移系数。",
        "- sliding_shadow_fit：仅用原整机训练段重拟合，沿用原来的 12 参数边界和三轴归一化。",
        "- sliding_joint_fit：直接联合拟合整机训练段与 114 组探针训练数据，无基底标定；"
        "两来源各占一半损失，探针各工况等权。承载参数上界单独放宽至 10000 N，其余边界相同。",
        "- 每次拟合使用三个预先固定的初值，仅按训练损失选取成功解。",
        f"- 整机拟合 t≤{train_end:g} s，留出 t≥{validation_start:g} s，中间不计。"
        "独立轨迹及全部 65 组探针验证数据不参与拟合。",
        "- 误差使用世界 XYZ 的全部原始物理样本；独立验证排除前 0.5 s 预热。曲线只为显示采用 20 ms 冲量加权平均。",
        "- 探针回放使用其原始运动历史，混合模型推进完整准备/保持/测量；纯滑移模型无历史。"
        "探针为规定运动，不启用有效质量限制，整机则启用同样的限制。",
        "",
        "## 整机误差",
        "",
        "| 数据 | 模型 | Fx RMSE / N | Fy RMSE / N | Fz RMSE / N |",
        "|---|---|---:|---:|---:|",
    ]
    for name, subset in [
        ("fit_source", "training"),
        ("fit_source", "blocked_validation"),
        ("independent_forward", "post_warmup"),
        ("independent_random", "post_warmup"),
    ]:
        for key in MODELS:
            x, y, z = result["shadow"][name][subset][key]["rmse_xyz_N"]
            lines.append(f"| {name}/{subset} | {key} | {x:.2f} | {y:.2f} | {z:.2f} |")
    lines += [
        "",
        "## 探针验证",
        "",
        "表中为各验证工况主受力轴 RMSE 的算术平均；通过标准为 RMSE≤max(5 N, 25% MPM RMS)。",
        "",
        "| 运动 | 模型 | 主轴平均 RMSE / N | 通过工况 |",
        "|---|---|---:|---:|",
    ]
    for motion, models in result["intruder_validation"].items():
        for key, value in models.items():
            lines.append(
                f"| {motion} | {key} | {value['mean_case_primary_rmse_N']:.2f} | {value['passed']}/{value['cases']} |"
            )
    lines += [
        "",
        "## 静止承载检查",
        "",
        f"整机重量 {weight:.2f} N，均匀四足分担为 {weight / 4:.2f} N/足。",
        "在 0–28 cm 深度扫描纯滑移模型的 Fz(d,v=0)。这是解析承载检查，不是整机闭环支撑仿真。",
        "",
        "| 模型 | 四足最大静态承载 / N | 与体重比 | 等载平衡深度 / cm |",
        "|---|---:|---:|---|",
    ]
    for key, value in result["static_support"].items():
        roots = (
            ", ".join(
                f"{r['depth_m'] * 100:.2f} ({'恢复斜率' if r['restoring_slope'] else '负刚度'})"
                for r in value["equal_load_equilibria"]
            )
            or "无"
        )
        lines.append(f"| {key} | {value['max_total_N']:.2f} | {value['max_support_weight_ratio']:.3f} | {roots} |")
    lines += [
        "",
        "## 为何整机运动误差几乎不变",
        "",
        "下表统计当前混合模型在各轨迹预热后的正深度接触样本中，滑移分支权重 g 的分布。"
        "接近 1 表示当时已几乎完全使用滑移分支，删除基底自然难以改变这些样本的预测。",
        "",
        "| 轨迹 | 平均 g | g>0.99 比例 | g>0.999 比例 |",
        "|---|---:|---:|---:|",
    ]
    for name, value in result["current_gate"].items():
        lines.append(
            f"| {name} | {value['mean']:.6f} | {value['fraction_above_099']:.2%} | {value['fraction_above_0999']:.2%} |"
        )
    lines += [
        "",
        "## 本次结果的判断",
        "",
        "仅用原整机训练段重拟合时，纯滑移模型的运动回放误差与当前模型几乎相同，"
        "随机动作误差没有实质改善；原有向上运动阻尼偏大的问题仍存在。",
        "直接去掉基底或重新拟合纯滑移参数，均失去了足够的静止承载。"
        "联合探针训练数据后承载提高，但本次候选在标记深度范围内仍没有四足均载静止平衡，且整机竖直误差增加。",
        "探针压入误差明显恶化，说明单一滑移响应式不能同时替代当前基底承担的压入/静止承载功能。"
        "因此，本次数据与拟合设置下不适合直接用纯滑移模型替换当前训练模型。",
        "若要完全取消弹塑性状态，仍需另外设计和辨识能承载体重的静态深度项，"
        "并用独立数据与闭环支撑实验验证；这已经超出原样使用现有滑移响应式。",
    ]
    lines += [
        "",
        "## 解释边界",
        "",
        "这些是同一批 MPM 状态上的受力回放，不能由此声称策略训练或闭环运动效果改善。",
        "删除基底会同时删除弹塑性卸载、剪切松动和足印历史。纯滑移式在 vx=vy=vz=0 时仅剩 B(d)，"
        "零水平速度但有竖直运动时仍保留竖直阻尼。",
        "联合拟合既改变训练来源、权重，也放宽承载上界；其结果作为充分使用既有训练数据的补充，不作为仅删除基底的受控对照。",
        "参数触边、三个初值的收敛情况及全部归一化尺度见候选 JSON 和 metrics.json；不使用验证误差挑选模型或初值。",
        "候选是独立实验格式，不包含基底参数，不能被当前 FootSandModel/deploy 入口直接装载。",
        "",
        "## 重现",
        "",
        "```bash",
        ".venv/bin/python -m sand_proxy_lab.experiments.sliding_only --terrain " + terrain.name,
        "```",
        "",
        "![三轴误差](rmse_comparison.png)",
        "",
        "![零速度承载](static_support.png)",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    if DEFAULT_MODEL_PATH.read_bytes() != deployed_bytes:
        raise RuntimeError("Deployed model changed during experiment")
    print(
        json.dumps(
            {"shadow_fit_static_support": result["static_support"]["sliding_shadow_fit"], "output": str(output)},
            indent=2,
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_terrain_argument(parser)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-end", type=float, default=3.0)
    parser.add_argument("--validation-start", type=float, default=3.2)
    args = parser.parse_args()
    terrain = terrain_directory(args.terrain)
    output = args.output_dir or terrain / "fit/sliding_only" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run(terrain, output, args.train_end, args.validation_start)


if __name__ == "__main__":
    main()
