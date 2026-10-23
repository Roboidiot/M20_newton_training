"""Episode-aware, CPU-only report of RandomAgent MPM/proxy force differences."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .report import FEET, metrics


def evaluation_mask(data, metadata):
    """Exclude warmup separately in every episode, including partial last episodes."""
    return data["episode_time_s"] > metadata["warmup_s"] + 1e-8


def analyze(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files if key != "metadata"}
        metadata = json.loads(saved["metadata"].item())
    out = path.parent
    actual, proxy = data["mpm_force_w_N"], data["proxy_force_w_N"]
    valid = evaluation_mask(data, metadata)
    contact = np.linalg.norm(actual, axis=-1) > 1
    rows = []

    def aggregate(mask):
        result = {"foot_samples": int(mask.sum())}
        if mask.any():
            for axis, letter in enumerate("xyz"):
                durations = np.broadcast_to(data["dt_s"][:, None], mask.shape)[mask]
                result[letter] = metrics(actual[..., axis][mask], proxy[..., axis][mask], durations)
        return result

    eligible = np.broadcast_to(valid[:, None], contact.shape)
    pooled = {
        "all_post_warmup": aggregate(eligible),
        "mpm_contact": aggregate(eligible & contact),
        "in_domain_mpm_contact": aggregate(eligible & contact & ~data["out_of_domain"]),
    }
    for foot, name in enumerate(FEET):
        for subset, mask in (
            ("all_post_warmup", valid),
            ("mpm_contact", valid & contact[:, foot]),
            ("in_domain_mpm_contact", valid & contact[:, foot] & ~data["out_of_domain"][:, foot]),
        ):
            if not mask.any():
                continue
            for axis, letter in enumerate("xyz"):
                rows.append(
                    {
                        "foot": name,
                        "axis": letter,
                        "subset": subset,
                        **metrics(actual[mask, foot, axis], proxy[mask, foot, axis], data["dt_s"][mask]),
                    }
                )
    summary = {
        "status": metadata["status"],
        "agent": metadata["agent"],
        "proxy_force_applied": False,
        "physics_frames": len(actual),
        "recorded_duration_s": float(data["dt_s"].sum()),
        "evaluation_duration_s": float(data["dt_s"][valid].sum()),
        "episodes": len(np.unique(data["episode_id"])),
        "reset_counts": dict(
            Counter(ep.get("reset_reason", "incomplete") for ep in metadata["episodes"][:-1] if ep["status"] == "reset")
        ),
        "pooled": pooled,
        "out_of_domain_fraction_post_warmup": float(data["out_of_domain"][valid].mean()) if valid.any() else None,
        "out_of_domain_fraction_mpm_contact": float(data["out_of_domain"][eligible & contact].mean())
        if (eligible & contact).any()
        else None,
        "mean_total_mpm_Fz_N": float(actual[valid, :, 2].sum(-1).mean()) if valid.any() else None,
        "mean_total_proxy_Fz_N": float(proxy[valid, :, 2].sum(-1).mean()) if valid.any() else None,
        "mean_other_robot_mpm_Fz_N": float(data["mpm_other_robot_force_w_N"][valid, 2].mean()) if valid.any() else None,
        "per_foot": rows,
    }
    (out / "random_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        with (out / "random_metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (out / "forces.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "episode_id",
                "episode_time_s",
                "time_s",
                "post_warmup",
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
        for i in range(len(actual)):
            for foot, name in enumerate(FEET):
                writer.writerow(
                    [
                        data["episode_id"][i],
                        data["episode_time_s"][i],
                        data["time_s"][i],
                        valid[i],
                        name,
                        data["depth_m"][i, foot],
                        data["out_of_domain"][i, foot],
                        *actual[i, foot],
                        *proxy[i, foot],
                    ]
                )
    groups, starts, counts = np.unique(data["control_step"], return_index=True, return_counts=True)
    averaged = {"control_step": groups}
    for key in ("mpm_force_w_N", "proxy_force_w_N", "mpm_other_robot_force_w_N"):
        values = data[key]
        shape = (-1,) + (1,) * (values.ndim - 1)
        averaged[key] = np.add.reduceat(values * data["dt_s"].reshape(shape), starts, axis=0) / np.add.reduceat(
            data["dt_s"], starts
        ).reshape(shape)
    ends = starts + counts - 1
    for key in ("time_s", "episode_id", "episode_time_s", "depth_m"):
        averaged[key] = data[key][ends]
    np.savez_compressed(out / "control_windows.npz", **averaged)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    episode_ids = np.unique(averaged["episode_id"])
    titles = ("左前足 FL", "右前足 FR", "左后足 HL", "右后足 HR")

    def foot_plot(selected, filename, local=False):
        fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True, layout="constrained")
        for episode in selected:
            mask = averaged["episode_id"] == episode
            t = averaged["episode_time_s" if local else "time_s"][mask]
            origin = t[0] - averaged["episode_time_s"][mask][0]
            for foot in range(4):
                for axis, letter in enumerate("xyz"):
                    ax = axes[foot, axis]
                    for key, label, color in (
                        ("mpm_force_w_N", "MPM 实测", "#202e42"),
                        ("proxy_force_w_N", "修正后代理（影子预测）", "#227ab5"),
                    ):
                        ax.plot(
                            t,
                            averaged[key][mask, foot, axis],
                            label=label if episode == selected[0] else None,
                            color=color,
                            linewidth=0.9,
                            alpha=0.9,
                        )
                    ax.axvspan(
                        origin, min(t[-1], origin + metadata["warmup_s"]), color="#78828e", alpha=0.12, linewidth=0
                    )
                    if not local and episode != selected[0]:
                        ax.axvline(origin, color="#a36b40", linestyle=":", alpha=0.5, linewidth=0.7)
                    ax.set(title=f"{titles[foot]} · F{letter}", ylabel="力 / N")
                    ax.grid(alpha=0.2)
                    if foot == 3:
                        ax.set_xlabel("回合时间 / s" if local else "累计记录时间 / s（不含沙地沉降）")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
        label = f"回合 {selected[0]}" if local else f"{len(selected)} 个回合"
        fig.suptitle(
            f"RandomAgent：四足三轴力对照 · {label}\n"
            "世界 XYZ，Z 向上；20 ms 冲量加权平均；灰色为每回合预热，重置处不连接曲线",
            fontsize=13,
        )
        fig.savefig(out / filename, dpi=170)
        plt.close(fig)

    foot_plot(episode_ids, "random_foot_forces_20ms.png")
    for episode in episode_ids[:3]:
        foot_plot([episode], f"random_episode_{int(episode):03d}.png", local=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, layout="constrained")
    for episode in episode_ids:
        mask = averaged["episode_id"] == episode
        t = averaged["time_s"][mask]
        for key, label, color in (("mpm_force_w_N", "MPM 四足", "#202e42"), ("proxy_force_w_N", "代理四足", "#227ab5")):
            axes[0].plot(t, averaged[key][mask, :, 2].sum(-1), color=color, label=label if episode == 0 else None)
        axes[0].plot(
            t,
            averaged["mpm_other_robot_force_w_N"][mask, 2],
            color="#c38940",
            alpha=0.7,
            label="MPM 其他机器人部位" if episode == 0 else None,
        )
        for foot, title in enumerate(titles):
            axes[1].plot(t, averaged["depth_m"][mask, foot], label=title if episode == 0 else None)
        m = data["episode_id"] == episode
        surface = metadata["episodes"][int(episode)]["surface_height_m"]
        axes[2].plot(data["time_s"][m], data["base_pose_w"][m, 2] - surface, color="#447b69")
        start = t[0] - averaged["episode_time_s"][mask][0]
        for ax in axes:
            ax.axvspan(start, min(t[-1], start + metadata["warmup_s"]), color="#78828e", alpha=0.12, linewidth=0)
            ax.axvline(start, color="#a36b40", linestyle=":", alpha=0.5, linewidth=0.7)
            ax.grid(alpha=0.2)
    axes[0].axhline(metadata["robot_weight_N"], linestyle="--", color="#888888", label="机器人重量参考")
    axes[0].set_ylabel("竖直力 Fz / N")
    axes[0].legend(ncol=4, fontsize=9)
    axes[1].set_ylabel("足底压入深度 / m")
    axes[1].legend(ncol=4, fontsize=9)
    axes[2].set(ylabel="机身离参考沙面高度 / m", xlabel="累计记录时间 / s（不含沙地沉降）")
    fig.suptitle("RandomAgent 影子实验：支撑、深度及摔倒重置\n仅 MPM 施力；代理仅预测，不进行拟合", fontsize=14)
    fig.savefig(out / "random_support_and_resets.png", dpi=170)
    plt.close(fig)
    lines = [
        "# RandomAgent 单环境影子对照",
        "",
        f"状态：{metadata['status']}；{metadata.get('stop_reason', '')}",
        f"记录 {summary['recorded_duration_s']:.2f} s，排除各回合预热后 "
        f"{summary['evaluation_duration_s']:.2f} s；{summary['episodes']} 回合。",
        "策略直接复用 IsaacLab RandomAgent：16 维动作每 20 ms 独立采样 U(-1,1)，不加载训练策略。",
        "每回合先零动作预热；摔倒、越界或回合超时后恢复机器人、沙粒与求解器历史，并清空代理历史。",
        "仅 MPM 施加真实受力，修正后模型同步预测。数据未重新拟合。"
        "所有误差使用原始 2.5 ms 物理采样，图形采用 20 ms 冲量加权平均。",
        "",
        "| 样本范围 | 足端样本数 | Fx RMSE (N) | Fy RMSE (N) | Fz RMSE (N) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in pooled.items():
        if values["foot_samples"]:
            lines.append(
                f"| {name} | {values['foot_samples']} | "
                + " | ".join(f"{values[a]['rmse_N']:.3f}" for a in "xyz")
                + " |"
            )
    lines += [
        "",
        f"重置计数：`{summary['reset_counts']}`。",
        f"评估样本中超模型覆盖范围比例：{summary['out_of_domain_fraction_post_warmup']:.1%}。"
        if valid.any()
        else "无预热后的有效样本。",
        "随机动作不是稳定行走，足端以外的机身/小腿支撑不能计入足端预测目标；本实验测试跨动作泛化与误差，不能直接证明闭环代理训练稳定。",
        "",
        "![四足受力](random_foot_forces_20ms.png)",
        "![支撑和重置](random_support_and_resets.png)",
        "",
    ]
    (out / "random_report.md").write_text("\n".join(lines))
    print(json.dumps({key: value for key, value in summary.items() if key != "per_foot"}, indent=2), flush=True)
    return summary
