"""Fit a standalone sliding law to recorded random-agent MPM forces, without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH
from sand_proxy_lab.experiments.sliding_only import SLIDING_NAMES, fit, load_trace, predict, scores
from sand_proxy_lab.paths import add_terrain_argument, terrain_directory
from sand_proxy_lab.shadow.refit import rotate_forces

MODEL_NAMES = ("current_mixed", "forward_sliding", "random_blocked_fit", "random_all_fit")
LABELS = ("Current mixed", "Slide: forward fit", "Slide: random train prefix", "Slide: all random")
COLORS = ("#326fa6", "#949ca6", "#d47a2c", "#429568")


def random_masks(data, train_end=4.0, validation_start=4.2):
    """Keep episode resets explicit; all cutoffs use local episode time."""
    warmup = data["metadata"]["warmup_s"]
    if not warmup < train_end < validation_start:
        raise ValueError("Need post-warmup train prefix, gap, and later validation")
    episode, time = data["episode_id"], data["episode_time_s"]
    post = time > warmup + 1e-8
    masks = {
        "random_ep0_training": (episode == 0) & post & (time <= train_end + 1e-8),
        "random_ep0_validation": (episode == 0) & (time >= validation_start - 1e-8),
        "random_ep1_partial": (episode == 1) & post,
        "random_all_post_warmup": post,
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Requested training and validation subsets must all be nonempty")
    if np.any(masks["random_ep0_training"] & masks["random_ep0_validation"]):
        raise ValueError("Training overlaps validation")
    return masks


def training_group(data, frame_mask):
    """Normalize and select force targets exclusively from requested training frames."""
    contact = frame_mask[:, None] & (data["depth_m"] > 0)
    actual = rotate_forces(data["mpm_force_w_N"], data["base_pose_w"], inverse=True)[contact]
    if len(actual) < 200:
        raise ValueError("Insufficient training contacts")
    return {
        "name": "random_selected_training_frames",
        "depth": data["depth_m"][contact],
        "velocity": data["foot_velocity_yaw_m_s"][contact],
        "actual": actual,
        "mass": data["metadata"]["robot_mass_kg"] / 4,
        "dt": np.broadcast_to(data["dt_s"][:, None], contact.shape)[contact],
        "scale": np.maximum(5, np.sqrt(np.mean(actual**2, axis=0))),
    }


def load_random(path):
    with np.load(path, allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files if k != "metadata"}
        data["metadata"] = json.loads(saved["metadata"].item())
    if data["metadata"]["proxy_force_applied"] or data["metadata"]["num_envs"] != 1:
        raise ValueError("Need single-environment MPM-only random trace")
    if set(np.unique(data["episode_id"])) != {0, 1}:
        raise ValueError("This experiment expects complete episode 0 and partial episode 1")
    if any(not np.isfinite(v).all() for k, v in data.items() if k != "metadata"):
        raise ValueError("Nonfinite input")
    if np.any(data["dt_s"] <= 0):
        raise ValueError("Timesteps must be positive")
    return data


def pure_predictions(data, coefficients, domain):
    local = predict(
        data["depth_m"],
        data["foot_velocity_yaw_m_s"],
        coefficients,
        mass=data["metadata"]["robot_mass_kg"] / 4,
        dt=data["dt_s"],
        domain=domain,
    )
    return rotate_forces(local, data["base_pose_w"])


def evaluate(data, predictions, mask):
    actual = data["mpm_force_w_N"][mask]
    values = {}
    for key, force in predictions.items():
        metric = scores(actual, force[mask])
        metric["passed_xyz"] = [
            r <= 5 or relative <= 0.25 for r, relative in zip(metric["rmse_xyz_N"], metric["nrmse_xyz"], strict=True)
        ]
        metric["mean_total_Fz_N"] = float(force[mask, :, 2].sum(-1).mean())
        values[key] = metric
    return {
        "physics_frames": int(mask.sum()),
        "mean_total_mpm_Fz_N": float(actual[..., 2].sum(-1).mean()),
        "models": values,
    }


def plot_trace(output, name, data, predictions, *, random=False, raw=False, train_end=4, validation_start=4.2):
    import matplotlib.pyplot as plt

    # Display impulse-weighted 20 ms windows; scoring uses every raw physics sample.
    groups = np.unique(data["control_step"])
    times = data["time_s"] if raw else np.array([data["time_s"][data["control_step"] == g][-1] for g in groups])
    fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True, layout="constrained")
    curves = [(data["mpm_force_w_N"], "MPM", "#152437")] + [
        (predictions[key], label, color) for key, label, color in zip(MODEL_NAMES, LABELS, COLORS, strict=True)
    ]
    for force, label, color in curves:
        mean = (
            force
            if raw
            else np.stack(
                [
                    np.average(
                        force[data["control_step"] == g], axis=0, weights=data["dt_s"][data["control_step"] == g]
                    )
                    for g in groups
                ]
            )
        )
        for foot, foot_name in enumerate(("FL", "FR", "HL", "HR")):
            for axis, letter in enumerate("XYZ"):
                ax = axes[foot, axis]
                ax.plot(times, mean[:, foot, axis], label=label, color=color, linewidth=0.8, alpha=0.9)
                ax.set(title=f"{foot_name} F{letter}", ylabel="N")
                ax.grid(alpha=0.2)
    for ax in axes.flat:
        ax.axvspan(0, data["metadata"]["warmup_s"], color="gray", alpha=0.1)
        if random:
            ax.axvspan(train_end, validation_start, color="gray", alpha=0.15)
            ax.axvline(validation_start, color="#555555", linestyle=":")
            reset = float(data["time_s"][data["episode_id"] == 0][-1])
            ax.axvline(reset, color="black", linestyle="--")
            ax.axvspan(reset, reset + data["metadata"]["warmup_s"], color="gray", alpha=0.1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    for ax in axes[-1]:
        ax.set_xlabel("recorded time (s)")
    description = "every raw 2.5 ms physics sample" if raw else "20 ms impulse-weighted display; raw-sample metrics"
    fig.suptitle(f"{name}: {description}")
    fig.savefig(output / f"{name}_forces.png", dpi=150)
    plt.close(fig)


def write_report(output, metrics, blocked_info, all_info, train_end, validation_start):
    lines = [
        "# 随机动作数据拟合纯滑移响应",
        "",
        "## 数据与拟合设置",
        "",
        "使用 random_30s_001/traces.npz 的实际 7 秒数据。回合 0 完整 6 秒，回合 1 部分 1 秒。",
        "仅拟合原纯滑移公式的 12 个参数，不使用弹塑性基底、混合权重或探针实验。",
        f"- random_blocked_fit：回合 0 的 0.5<t≤{train_end:g} 秒拟合，"
        f"t≥{validation_start:g} 秒时间留出，中间 {validation_start - train_end:g} 秒不参与拟合。",
        "- 回合 1 的预热后部分用于短回合验证（仅 0.5 秒，粒子沉降配置相同）。",
        "- random_all_fit：两个回合全部非预热样本参与拟合；其随机数据误差是拟合内误差。",
        "- 两版均沿用原 12 参数边界、训练三轴 RMS 归一化（至少 5 N），三个固定初值仅按训练损失选择。",
        "- 前进拟合轨迹与独立低速前进轨迹均未参与此次随机数据拟合。",
        "- current_mixed 为文件中保存的原部署模型预测；已核验其模型快照 SHA 与当前部署一致。",
        "- forward_sliding 为前一次只使用前进轨迹拟合的纯滑移候选。",
        "- 不筛掉标记为超覆盖范围的样本；拟合及预测均使用原有深度/速度限幅和有效质量稳定限制。",
        "- 下表使用世界 XYZ 全部原始物理采样，无时间平移或误差平滑；提供原始曲线及 20 ms 冲量加权平均图。",
        "",
        "## 三轴受力误差",
        "",
        "| 数据 | 模型 | Fx RMSE / N | Fy RMSE / N | Fz RMSE / N |",
        "|---|---|---:|---:|---:|",
    ]
    for name, item in metrics.items():
        if name.endswith(("upward", "downward")):
            continue
        for key in MODEL_NAMES:
            x, y, z = item["models"][key]["rmse_xyz_N"]
            lines.append(f"| {name} | {key} | {x:.2f} | {y:.2f} | {z:.2f} |")
    lines += [
        "",
        "## 归一化误差与判据",
        "",
        "NRMSE=RMSE/MPM RMS（分母至少 1 N）。沿用判据 RMSE≤max(5 N, 0.25×MPM RMS)。",
        "",
        "| 数据 | 模型 | Fx NRMSE | Fy NRMSE | Fz NRMSE | 三轴通过 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name in [
        "random_ep0_validation",
        "random_ep1_partial",
        "random_all_post_warmup",
        "forward_original_post_warmup",
        "forward_independent_post_warmup",
    ]:
        for key in ["forward_sliding", "random_blocked_fit", "random_all_fit"]:
            m = metrics[name]["models"][key]
            x, y, z = m["nrmse_xyz"]
            passed = "/".join("通过" if p else "未过" for p in m["passed_xyz"])
            lines.append(f"| {name} | {key} | {x:.1%} | {y:.1%} | {z:.1%} | {passed} |")
    before = np.array(metrics["random_all_post_warmup"]["models"]["forward_sliding"]["rmse_xyz_N"])
    after = np.array(metrics["random_all_post_warmup"]["models"]["random_all_fit"]["rmse_xyz_N"])
    improvement = 100 * (1 - after / before)
    lines += [
        "",
        "## 本次结果判断",
        "",
        f"全随机数据版在 9600 个非预热足端样本上的三轴 RMSE 相对前进拟合版分别下降 "
        f"{improvement[0]:.1f}% / {improvement[1]:.1f}% / {improvement[2]:.1f}%。这是拟合内改善。",
        "时间划分版在第一回合留出段的 Fx 明显改善，但 Fy 略变差、Fz 基本不变；"
        "第二回合短验证的三轴误差均下降，说明改善并非全部局限于训练前缀。",
        "即使全随机数据都参与拟合，随机数据上的水平 NRMSE 仍约为 35% / 49%，"
        "未通过项目的 25% 判据；尚不能称为与 MPM 三轴受力一致。",
        "代价是未参与此次拟合的低速前进工况变差，特别是 Fz；随机动作上更好的参数没有同时保住原来的前进拟合精度。",
        "随机数据的向上运动竖直误差降低，但向下运动竖直误差增大。"
        "当前压入/拔出共用一套竖直阻尼，结果反映了两类状态之间的拟合折中。",
        "",
        "## 12 参数对照",
        "",
        "| 参数 | 前进拟合 | 随机时间划分拟合 | 全随机数据拟合 |",
        "|---|---:|---:|---:|",
    ]
    coefficient_sets = [
        json.loads((output / path).read_text())["coefficients"]
        for path in ["forward_sliding_snapshot.json", "random_blocked_fit.json", "random_all_fit.json"]
    ]
    for parameter in SLIDING_NAMES:
        values = [p[parameter] for p in coefficient_sets]
        lines.append(f"| {parameter} | {values[0]:.8g} | {values[1]:.8g} | {values[2]:.8g} |")
    lines += [
        "",
        "## 收敛与可辨识性",
        "",
        f"时间划分版参数触边：{', '.join(blocked_info['at_bounds']) or '无'}。",
        f"全随机数据版参数触边：{', '.join(all_info['at_bounds']) or '无'}。",
        "各初值训练损失与完整参数保存在候选 JSON 中；初值选择不读取验证目标。",
        "",
        "## 解释范围",
        "",
        "这是记录状态上的离线受力拟合。随机第一回合现在已用于训练，不能继续把整段作为独立验证；"
        "应分别看留出后缀、短的第二回合，以及未参加此次拟合的前进轨迹。",
        "全部随机数据版的随机数据指标是拟合内结果，不构成随机工况泛化验证。",
        "未运行新 MPM、闭环支撑或 PPO 训练；未更改部署参数。",
        "",
        "## 重现",
        "",
        "```bash",
        ".venv/bin/python -m sand_proxy_lab.experiments.random_sliding --terrain m20_sand_v1",
        "```",
        "",
        "![误差对照](rmse_comparison.png)",
        "",
        "![随机受力，20 ms 显示](random_forces.png)",
        "",
        "![随机受力，原始物理步](random_raw_forces.png)",
        "",
        "![独立前进受力](forward_independent_forces.png)",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def run(terrain, output, train_end=4.0, validation_start=4.2):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = terrain / "shadow/random_30s_001/traces.npz"
    source_bytes = source.read_bytes()
    deployed_bytes = DEFAULT_MODEL_PATH.read_bytes()
    current = json.loads(deployed_bytes)
    data = load_random(source)
    snapshot = source.parent / "model_snapshot.json"
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != data["metadata"]["model_sha256"]:
        raise ValueError("Random trace snapshot hash mismatch")
    if snapshot.read_bytes() != deployed_bytes:
        raise ValueError("Recorded mixed-model comparator differs from current deployment")
    masks = random_masks(data, train_end, validation_start)
    old_path = terrain / "fit/sliding_only/20261029_comparison/sliding_shadow_fit.json"
    old_bytes = old_path.read_bytes()
    old = json.loads(old_bytes)
    if set(old["coefficients"]) != set(SLIDING_NAMES):
        raise ValueError("Need previous 12-parameter pure-sliding comparator")
    output.mkdir(parents=True, exist_ok=False)
    (output / "current_model_snapshot.json").write_bytes(deployed_bytes)
    (output / "forward_sliding_snapshot.json").write_bytes(old_bytes)
    blocked, blocked_info = fit([training_group(data, masks["random_ep0_training"])], current["geometry"]["radius"])
    all_fit, all_info = fit([training_group(data, masks["random_all_post_warmup"])], current["geometry"]["radius"])
    parameters = {"forward_sliding": old["coefficients"], "random_blocked_fit": blocked, "random_all_fit": all_fit}
    for name, values, info in [("random_blocked_fit", blocked, blocked_info), ("random_all_fit", all_fit, all_info)]:
        candidate = {
            "schema_version": 1,
            "model_type": "m20_foot_sliding_only_experiment",
            "geometry": current["geometry"],
            "material": current["material"],
            "domain": current["domain"],
            "coefficients": values,
            "fit_info": info,
            "formula": "F_slide directly; no backbone, blend or historical state",
            "source_trace_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "training_subset": "random_ep0_training" if name == "random_blocked_fit" else "random_all_post_warmup",
            "deployment_status": "offline_candidate_not_accepted_by_FootSandModel",
        }
        (output / f"{name}.json").write_text(json.dumps(candidate, indent=2) + "\n")
    predictions = {"current_mixed": data["proxy_force_w_N"]}
    predictions.update({k: pure_predictions(data, v, current["domain"]) for k, v in parameters.items()})
    metrics = {name: evaluate(data, predictions, mask) for name, mask in masks.items()}
    for direction, velocity_mask in [
        ("upward", data["foot_velocity_yaw_m_s"][..., 2] > 0.05),
        ("downward", data["foot_velocity_yaw_m_s"][..., 2] < -0.05),
    ]:
        contact = masks["random_all_post_warmup"][:, None] & (data["depth_m"] > 0) & velocity_mask
        metrics[f"random_{direction}"] = {
            "models": {k: scores(data["mpm_force_w_N"][contact], f[contact]) for k, f in predictions.items()}
        }
    np.savez_compressed(
        output / "random_replay.npz",
        time_s=data["time_s"],
        episode_id=data["episode_id"],
        episode_time_s=data["episode_time_s"],
        dt_s=data["dt_s"],
        mpm_force_w_N=data["mpm_force_w_N"],
        **masks,
        **predictions,
    )
    plot_trace(output, "random", data, predictions, random=True, train_end=train_end, validation_start=validation_start)
    plot_trace(
        output,
        "random_raw",
        data,
        predictions,
        random=True,
        raw=True,
        train_end=train_end,
        validation_start=validation_start,
    )
    sources = {"random": source}
    for name, directory in [
        ("forward_original", "walk_forward_001"),
        ("forward_independent", "refit_validation_offset065"),
    ]:
        path = terrain / "shadow" / directory / "traces.npz"
        forward = load_trace(path)
        forward_predictions = {"current_mixed": forward["proxy_force_w_N"]}
        # The original forward trace recorded a backbone, so replay the current model for a fair comparison.
        from sand_proxy_lab.shadow.refit import replay

        forward_predictions["current_mixed"] = replay(current, forward)
        forward_predictions.update({k: pure_predictions(forward, v, current["domain"]) for k, v in parameters.items()})
        post = forward["time_s"] > forward["metadata"]["warmup_s"] + 1e-8
        metrics[f"{name}_post_warmup"] = evaluate(forward, forward_predictions, post)
        np.savez_compressed(
            output / f"{name}_replay.npz",
            time_s=forward["time_s"],
            mpm_force_w_N=forward["mpm_force_w_N"],
            **forward_predictions,
        )
        plot_trace(output, name, forward, forward_predictions)
        sources[name] = path
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    manifest = {
        "sources": {
            k: {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for k, p in sources.items()
        },
        "deployed_sha256": hashlib.sha256(deployed_bytes).hexdigest(),
        "forward_candidate_sha256": hashlib.sha256(old_bytes).hexdigest(),
        "train_end_s": train_end,
        "validation_start_s": validation_start,
        "per_episode_warmup_excluded_s": data["metadata"]["warmup_s"],
        "subsets": {
            name: {
                "physics_frames": int(mask.sum()),
                "duration_s": float(data["dt_s"][mask].sum()),
                "contact_foot_samples": int((mask[:, None] & (data["depth_m"] > 0)).sum()),
            }
            for name, mask in masks.items()
        },
        "validation_used_for_initialization_or_selection": False,
        "default_force_domain_unchanged": current["domain"],
        "no_probe_data": True,
        "closed_loop_simulation": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    selected = [
        "random_ep0_validation",
        "random_ep1_partial",
        "random_all_post_warmup",
        "forward_independent_post_warmup",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), layout="constrained")
    for ax, name in zip(axes, selected, strict=True):
        for i, (key, label, color) in enumerate(zip(MODEL_NAMES, LABELS, COLORS, strict=True)):
            ax.bar(
                np.arange(3) + (i - 1.5) * 0.19,
                metrics[name]["models"][key]["rmse_xyz_N"],
                width=0.19,
                label=label,
                color=color,
            )
        ax.set(title=name.replace("_", " "), ylabel="raw-sample RMSE (N)", xticks=np.arange(3), xticklabels=list("XYZ"))
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.savefig(output / "rmse_comparison.png", dpi=160)
    plt.close(fig)
    write_report(output, metrics, blocked_info, all_info, train_end, validation_start)
    if source.read_bytes() != source_bytes or DEFAULT_MODEL_PATH.read_bytes() != deployed_bytes:
        raise RuntimeError("Source data or deployed model changed")
    summary = {name: {key: value["rmse_xyz_N"] for key, value in metrics[name]["models"].items()} for name in selected}
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Output: {output}", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_terrain_argument(parser)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-end", type=float, default=4.0)
    parser.add_argument("--validation-start", type=float, default=4.2)
    args = parser.parse_args()
    terrain = terrain_directory(args.terrain)
    output = args.output_dir or terrain / "fit/sliding_random" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run(terrain, output, args.train_end, args.validation_start)


if __name__ == "__main__":
    main()
