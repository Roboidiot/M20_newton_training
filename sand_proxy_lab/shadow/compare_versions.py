"""Reproduce plots from the unfitted new-condition shadow rollout, without simulation."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sand_proxy_lab.shadow.refit import comparison, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Default: extract backbone from the trace model snapshot")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    TRACE = args.trace.resolve()
    BASELINE = args.baseline.resolve() if args.baseline else TRACE.parent / "model_snapshot.json"
    OUT = args.output_dir or TRACE.parent / "comparison_plots"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    with np.load(TRACE, allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files if k != "metadata"}
        data["metadata"] = json.loads(saved["metadata"].item())
    metadata = data["metadata"]
    if "episode_id" in data and len(np.unique(data["episode_id"])) != 1:
        raise ValueError("Extract one continuous episode before comparing versions")
    assert not metadata["proxy_force_applied"]
    base_parameters = json.loads(BASELINE.read_text())
    base_parameters["domain"] = base_parameters.pop("intruder_domain", base_parameters["domain"])
    base_parameters.pop("shadow_contact", None)
    base_parameters["schema_version"] = 1
    before = replay(base_parameters, data)
    actual = data["mpm_force_w_N"]
    after = data["proxy_force_w_N"]
    assert np.isfinite(before).all() and np.isfinite(after).all()
    valid = data["time_s"] > metadata["warmup_s"] + 1e-8
    scores = comparison(data, before, after, valid)
    groups = np.unique(data["control_step"])
    times = np.array([data["time_s"][data["control_step"] == g][-1] for g in groups])

    def average(force):
        return np.stack(
            [
                np.average(force[data["control_step"] == g], axis=0, weights=data["dt_s"][data["control_step"] == g])
                for g in groups
            ]
        )

    curves = (actual, before, after)
    means = tuple(average(force) for force in curves)
    labels = ("MPM 实测", "修正前代理（同轨迹回放）", "修正后代理（影子预测）")
    colors = ("#202e42", "#d78b31", "#227ab5")
    styles = ("-", "--", "-")
    feet = ("左前足 FL", "右前足 FR", "左后足 HL", "右后足 HR")
    condition = f"command={metadata.get('command')}; start_xy={metadata.get('start_xy_m')}"

    def decorate(ax):
        ax.axvspan(0, metadata["warmup_s"], color="#78828e", alpha=0.13, linewidth=0)
        ax.axvline(metadata["warmup_s"], color="#78828e", linestyle=":", linewidth=1)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, data["time_s"][-1])

    def save(fig, name):
        fig.savefig(OUT / f"{name}.png", dpi=180, facecolor="white")
        fig.savefig(OUT / f"{name}.svg", facecolor="white")
        plt.close(fig)

    for name, selected, source, time, window in (
        ("before_after_20ms", (0, 1, 2), means, times, "20 ms 内按冲量加权平均"),
        ("refitted_vs_mpm_20ms", (0, 2), means, times, "20 ms 内按冲量加权平均"),
        ("refitted_vs_mpm_raw", (0, 2), curves, data["time_s"], "每 2.5 ms 的原始物理采样"),
    ):
        fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex=True, layout="constrained")
        for foot in range(4):
            for axis, letter in enumerate("xyz"):
                ax = axes[foot, axis]
                for index in selected:
                    ax.plot(
                        time,
                        source[index][:, foot, axis],
                        color=colors[index],
                        linestyle=styles[index],
                        linewidth=1.1 if index == 0 else 1.0,
                        alpha=0.9,
                        label=labels[index],
                    )
                decorate(ax)
                ax.set_title(f"{feet[foot]} · F{letter}", fontsize=11)
                ax.set_ylabel("力 / N")
                if foot == 3:
                    ax.set_xlabel("机器人释放后的时间 / s")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside lower center", ncol=len(selected), frameon=False)
        fig.suptitle(
            f"四足三轴受力对照（世界坐标系，Z 向上）\n{condition}\n{window}；"
            f"灰色区间为 0–{metadata['warmup_s']:g} s 预热",
            fontsize=13,
        )
        save(fig, name)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for index in range(3):
        axes[0].plot(
            times,
            means[index][..., 2].sum(-1),
            label=labels[index],
            color=colors[index],
            linestyle=styles[index],
            linewidth=1.5,
        )
    axes[0].axhline(metadata["robot_weight_N"], color="#666666", linestyle=":", label="机器人重量（仅作参考）")
    axes[0].set_ylabel("四足合计 Fz / N")
    axes[0].legend(loc="upper right", fontsize=9)
    for index in (1, 2):
        axes[1].plot(
            times,
            (means[index][..., 2] - means[0][..., 2]).sum(-1),
            label=labels[index] + " − MPM",
            color=colors[index],
            linestyle=styles[index],
            linewidth=1.5,
        )
    axes[1].axhline(0, color="#666666", linewidth=0.8)
    axes[1].set(ylabel="四足合计 Fz 误差 / N", xlabel="机器人释放后的时间 / s")
    axes[1].legend(loc="upper right", fontsize=9)
    for ax in axes:
        decorate(ax)
    fig.suptitle(
        f"四足总竖直力：修正前后对照\n{condition}\n20 ms 平均；灰色为预热；足端以外的机器人接触力未计入", fontsize=13
    )
    save(fig, "total_vertical_force_20ms")

    np.savez_compressed(
        OUT / "comparison_curves.npz",
        time_s=data["time_s"],
        dt_s=data["dt_s"],
        control_step=data["control_step"],
        mpm_force_w_N=actual,
        before_force_w_N=before,
        after_force_w_N=after,
        averaged_time_s=times,
        averaged_mpm_force_w_N=means[0],
        averaged_before_force_w_N=means[1],
        averaged_after_force_w_N=means[2],
    )
    scores["trace_sha256"] = hashlib.sha256(TRACE.read_bytes()).hexdigest()
    scores["baseline_sha256"] = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    scores["evaluation_window_s"] = [metadata["warmup_s"], float(data["time_s"][-1])]
    (OUT / "comparison_metrics.json").write_text(json.dumps(scores, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# 本次独立新工况影子实验曲线\n\n"
        "工况来源见 metadata.json；使用同一条轨迹回放各版本。\n"
        "仅 MPM 施力，修正后代理只预测；修正前模型在同一组状态上从轨迹起点推进历史回放。"
        "画图没有重新拟合或运行仿真；该轨迹是否曾参与拟合应由实验用途判断。\n\n"
        "- `refitted_vs_mpm_20ms.png`：四足三轴，修正后与 MPM，20 ms 冲量加权平均。\n"
        "- `before_after_20ms.png`：同时显示修正前、修正后和 MPM；各子图自动纵轴范围。\n"
        "- `refitted_vs_mpm_raw.png`：不平均的 2.5 ms 原始采样。\n"
        "- `total_vertical_force_20ms.png`：四足总竖直力及误差，非足端受力不计入。\n"
        "- 同名 SVG：矢量图；`comparison_curves.npz`：完整前后对照数据。\n\n"
        "灰色为 0–0.5 s 预热，RMSE 仅用之后 1 s 的原始物理采样计算，"
        "不通过平滑改善误差，不平移时间轴。坐标系为世界 XYZ，Z 向上，力为沙地作用于足端的力。\n\n"
        "从项目根目录重现：\n```bash\n"
        ".venv/bin/python -m sand_proxy_lab.shadow.compare_versions --trace <traces.npz>\n```\n"
    )
    print(json.dumps(scores, indent=2))
    print(f"Saved plots: {OUT}")


if __name__ == "__main__":
    main()
