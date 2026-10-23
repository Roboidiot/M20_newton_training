"""Plot collected foot force histories without fitting or changing the source data.

Usage: .venv/bin/python -m sand_proxy_lab.calibration.plot_forces
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "m20_sand_plot_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

from sand_proxy_lab.paths import add_terrain_argument, terrain_directory

from .common import identity
from .config import COL, COLUMNS, case_dict, cases, phase_start
from .data import remove_repeated_terminal_sample

MOTIONS = ("press", "pull", "forward", "lateral")
NAMES = {"press": "压入", "pull": "拔出", "forward": "前向平移", "lateral": "侧向平移"}
PALETTE = ("#2563EB", "#EA580C", "#0D9488", "#9333EA", "#DC2626", "#797D18")


def load_records(directory: Path, split: str):
    config = json.loads((directory / "config.json").read_text())
    config_id = identity(config)
    planned = [c for c in cases(config) if c.geometry == "foot" and c.motion in MOTIONS and c.split == split]
    records, missing = [], []
    # Snapshot the available, atomically completed files while collection continues independently.
    available = {p.name for p in directory.glob("foot_*.npz")}
    for case in planned:
        path = directory / f"{case.name}.npz"
        if path.name not in available:
            missing.append(case.name)
            continue
        source = path.read_bytes()
        with np.load(path, allow_pickle=False) as saved:
            data = saved["data"].copy()
            metadata = json.loads(saved["metadata"].item())
        if metadata.get("schema_version") != 2 or metadata.get("config_id") != config_id:
            raise ValueError(f"Different calibration configuration: {path}")
        if metadata["columns"] != list(COLUMNS) or metadata["case"] != case_dict(case):
            raise ValueError(f"Unexpected case or column layout: {path}")
        if data.ndim != 2 or data.shape[1] != len(COLUMNS) or not np.isfinite(data).all():
            raise ValueError(f"Invalid data: {path}")
        data, removed_terminal_rows = remove_repeated_terminal_sample(data)
        measured = data[data[:, COL["measure"]] > 0.5]
        if len(measured) < 2:
            raise ValueError(f"Insufficient measurement samples: {path}")
        records.append(
            {
                "case": case,
                "data": measured,
                "source": path.name,
                "sha256": hashlib.sha256(source).hexdigest(),
                "measurement_samples": len(measured),
                "removed_terminal_rows": removed_terminal_rows,
            }
        )
    if not records:
        raise ValueError(f"No completed foot press/pull/forward experiments for split {split!r}")
    return config, records, missing


def coordinates(record: dict, config: dict):
    case, data = record["case"], record["data"]
    elapsed = data[:, COL["time"]] - phase_start(config, case)
    if case.motion == "press":
        distance = data[:, COL["depth"]]
    elif case.motion == "pull":
        distance = case.depth - data[:, COL["depth"]]
    else:
        distance = data[:, COL["x" if case.motion == "forward" else "y"]] + config["travel"] / 2
    return distance * 100, elapsed


def distance_label(motion: str) -> str:
    return {
        "press": "足端最低点压入深度 (cm)",
        "pull": "从初始位置向上拔出的位移 (cm)",
        "forward": "从平移起点沿 +X 移动的位移 (cm)",
        "lateral": "从平移起点沿 +Y 移动的位移 (cm)",
    }[motion]


def describe(motion: str, depth: float) -> str:
    qualifier = {"press": "连续压入，目标深度", "pull": "初始深度", "forward": "固定深度", "lateral": "固定深度"}[
        motion
    ]
    return f"足端{NAMES[motion]}｜{qualifier} {depth * 100:g} cm"


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(color="#E2E8F0", linewidth=0.65)
    ax.axhline(0, color="#64748B", linewidth=0.65, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.margins(x=0.01, y=0.07)


def mark_surface(ax, motion: str, depth: float):
    if motion in ("press", "pull"):
        location = 0 if motion == "press" else depth * 100
        ax.axvline(location, linestyle="--", linewidth=0.8, color="#64748B", alpha=0.8)


def legend(fig, lines, y: float):
    fig.legend(
        handles=lines,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=6,
        frameon=False,
        fontsize=10,
        handlelength=2.6,
        columnspacing=1.6,
    )


def detail_figure(group: list[dict], config: dict, colors: dict):
    case = group[0]["case"]
    fig, axes = plt.subplots(3, 2, figsize=(13.4, 9.8), sharex="col", sharey="row")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.81, hspace=0.24, wspace=0.10)
    fig.suptitle(describe(case.motion, case.depth), fontsize=19, fontweight="bold", y=0.975)
    fig.text(
        0.5,
        0.928,
        f"{len(group)} 组原始 MPM 数据 · X 前向 / Y 侧向 / Z 向上 · 沙地对足端的合力",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    lines = []
    for component, field in enumerate(("fx", "fy", "fz")):
        for j, ax in enumerate(axes[component]):
            style_axis(ax)
            for record in group:
                x = coordinates(record, config)[j]
                speed = record["case"].speed
                (line,) = ax.plot(
                    x,
                    record["data"][:, COL[field]],
                    color=colors[speed],
                    linewidth=1.1,
                    alpha=0.9,
                    label=f"{speed:.2f} m/s",
                )
                if component == 0 and j == 0:
                    lines.append(line)
            if j == 0:
                ax.set_ylabel(f"$F_{field[-1]}$ (N)", fontsize=12)
                mark_surface(ax, case.motion, case.depth)
    axes[0, 0].set_title("按深度 / 位移对比", fontsize=11, pad=11)
    axes[0, 1].set_title("按动作时间对比", fontsize=11, pad=11)
    axes[-1, 0].set_xlabel(distance_label(case.motion), fontsize=11)
    axes[-1, 1].set_xlabel("测量动作开始后的时间 (s)", fontsize=11)
    legend(fig, lines, 0.90)
    note = "仅绘制 measure=1 的动作阶段；未平滑、未拟合、未取绝对值，保留起始瞬态。"
    if case.motion in ("press", "pull"):
        note += " 左列虚线：足端最低点到达参考沙面。"
    fig.text(0.5, 0.040, note, ha="center", fontsize=9, color="#64748B")
    removed = sum(r["removed_terminal_rows"] for r in group)
    if removed:
        fig.text(
            0.5,
            0.018,
            f"本图 {removed} 组实验各去除 1 个重复末端记录；原始 NPZ 文件未修改。",
            ha="center",
            fontsize=9,
            color="#64748B",
        )
    return fig


def overview_figure(motion: str, groups: list[list[dict]], config: dict, colors: dict):
    field = {"forward": "fx", "lateral": "fy"}.get(motion, "fz")
    columns = min(3, len(groups))
    rows = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12.8, 5.5 if rows == 1 else 8.5), squeeze=False)
    fig.subplots_adjust(
        left=0.075,
        right=0.98,
        bottom=0.20 if rows == 1 else 0.16,
        top=0.70 if rows == 1 else 0.80,
        hspace=0.62,
        wspace=0.30,
    )
    fig.suptitle(f"足端{NAMES[motion]}｜{field.title()} 原始受力曲线", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.90 if rows == 1 else 0.925,
        "相同工况、相同深度叠加不同速度 · 各深度采用独立纵轴范围",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    lines = []
    for i, group in enumerate(groups):
        case, ax = group[0]["case"], axes.flat[i]
        style_axis(ax)
        for record in group:
            speed = record["case"].speed
            (line,) = ax.plot(
                coordinates(record, config)[0],
                record["data"][:, COL[field]],
                color=colors[speed],
                linewidth=1.1,
                alpha=0.9,
                label=f"{speed:.2f} m/s",
            )
            if i == 0:
                lines.append(line)
        mark_surface(ax, motion, case.depth)
        qualifier = "连续压入至" if motion == "press" else "初始深度" if motion == "pull" else "固定深度"
        ax.set_title(f"{qualifier} {case.depth * 100:g} cm", fontsize=11, pad=8)
        ax.set_xlabel(distance_label(motion), fontsize=9)
        ax.set_ylabel(f"$F_{field[-1]}$ (N)", fontsize=10)
    for ax in axes.flat[len(groups) :]:
        ax.set_visible(False)
    legend(fig, lines, 0.86 if rows == 1 else 0.89)
    note = "世界坐标系力，保留原始符号；仅去除重复末端记录，未平滑或拟合。详图含完整 Fx / Fy / Fz 及时间曲线。"
    fig.text(0.5, 0.035, note, ha="center", fontsize=9, color="#64748B")
    return fig


def make_plots(directory: Path, output: Path, split: str):
    config, records, missing = load_records(directory, split)
    output.mkdir(parents=True, exist_ok=True)
    detail_dir = output / "by_depth"
    detail_dir.mkdir(exist_ok=True)
    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = "Noto Sans CJK JP" if "Noto Sans CJK JP" in installed else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    speeds = sorted({r["case"].speed for r in records})
    colors = {speed: PALETTE[i % len(PALETTE)] for i, speed in enumerate(speeds)}
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["case"].motion, record["case"].depth)].append(record)
    pages = []
    with PdfPages(output / "foot_force_curves.pdf", metadata={"Title": "M20 foot raw MPM force curves"}) as pdf:
        for motion in MOTIONS:
            motion_groups = []
            for (group_motion, depth), group in sorted(grouped.items()):
                if group_motion != motion:
                    continue
                group.sort(key=lambda r: (r["case"].speed, r["case"].seed))
                motion_groups.append(group)
                filename = f"foot_{motion}_d{round(depth * 1000):03d}.png"
                fig = detail_figure(group, config, colors)
                fig.savefig(detail_dir / filename, dpi=180)
                pdf.savefig(fig)
                plt.close(fig)
                pages.append(
                    {"motion": motion, "depth_m": depth, "experiments": len(group), "image": f"by_depth/{filename}"}
                )
            if motion_groups:
                fig = overview_figure(motion, motion_groups, config, colors)
                fig.savefig(output / f"foot_{motion}_overview.png", dpi=180)
                plt.close(fig)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "input_directory": str(directory.resolve()),
        "split": split,
        "config_id": identity(config),
        "experiment_count": len(records),
        "measurement_samples": sum(r["measurement_samples"] for r in records),
        "missing_cases": missing,
        "detail_figures": pages,
        "processing": (
            "Only measure=1 rows; drop the extra clamped terminal record when timestamps differ by <=1e-9 s. "
            "Retain the first endpoint sample. No smoothing, fitting, clipping or absolute force values."
        ),
        "records": [{k: (case_dict(v) if k == "case" else v) for k, v in r.items() if k != "data"} for r in records],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    readme = [
        "# 足端原始受力曲线",
        "",
        f"本次快照：{len(records)} 组 `{split}` 实验，{len(pages)} 张按工况和深度分组的详细图。",
        "每张详细图叠加不同速度，三行依次为 Fx、Fy、Fz；左列按深度/位移对齐，右列按测量动作开始时间对齐。",
        "",
        "[全部详细图 PDF](foot_force_curves.pdf)",
        "",
        "- [压入概览：Fz—深度](foot_press_overview.png)",
        "- [拔出概览：Fz—拔出位移](foot_pull_overview.png)",
        "- [前向平移概览：Fx—前向位移](foot_forward_overview.png)",
        "",
        "| 工况 | 深度 | 实验数 | Fx / Fy / Fz 详细图 |",
        "|---|---:|---:|---|",
    ]
    for page in pages:
        readme.append(
            f"| {NAMES[page['motion']]} | {page['depth_m'] * 100:g} cm | {page['experiments']} | "
            f"[查看]({page['image']}) |"
        )
    readme.extend(
        [
            "",
            "压入只有连续扫描到 20 cm 的 6 次实验；图中保留从参考沙面上方 2 cm 到压入 20 cm 的完整测量轨迹。",
            "拔出按初始深度分组，向上位移为初始深度减当前深度；虚线表示足端最低点到达参考沙面。",
            "平移按固定压入深度分组，横轴为从平移起点开始的 +X 位移，总行程 8 cm。",
            "深度均由足端碰撞体最低点相对沉降后的参考沙面计算，不是刚体原点高度。",
            "力为沙地施加在足端上的世界坐标系合力，单位 N，X 前向、Y 侧向、Z 向上；负号保留。",
            "仅选择 measure=1 的正式测量阶段，非压入工况的准备压入和保持阶段不画入曲线。",
            "仅去除采集循环多记录的重复末端点，保留首次到达终点的采样；未平滑、未拟合、未取绝对值。",
            "原始 npz 不变。源文件清单、SHA256 及每组剔除数量见 manifest.json。",
            "",
            "重新生成：",
            "",
            "```bash",
            ".venv/bin/python -m sand_proxy_lab.calibration.plot_forces",
            "```",
            "",
        ]
    )
    duplicates = [r for r in records if r["removed_terminal_rows"]]
    if duplicates:
        readme.extend(["本次以下实验各去除 1 个重复末端点（相邻时间差不超过 1e-9 s、终点进度/姿态相同）：", ""])
        readme.extend(f"- `{r['source']}`" for r in duplicates)
        readme.append("")
    if missing:
        readme.append(f"注意：本次选定的工况还有 {len(missing)} 组未完成，未用插值补齐；清单见 manifest.json。\n")
    (output / "README.md").write_text("\n".join(readme))
    print(
        json.dumps(
            {
                "experiments": len(records),
                "detail_figures": len(pages),
                "missing_cases": len(missing),
                "output_directory": str(output.resolve()),
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_terrain_argument(parser)
    parser.add_argument("--input-dir", type=Path, help="Override the terrain's intruder directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    args = parser.parse_args()
    args.input_dir = args.input_dir or terrain_directory(args.terrain) / "intruder"
    output = args.output_dir or args.input_dir / f"foot_force_curves_{args.split}"
    make_plots(args.input_dir, output, args.split)


if __name__ == "__main__":
    main()
