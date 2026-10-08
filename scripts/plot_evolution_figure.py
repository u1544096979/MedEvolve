"""绘制固定患者集上的五检查点自进化指标，不填补缺失数据。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_experiment_figures import style, DIMENSIONS, DIMENSION_LABELS

COLORS = ("#087F83", "#68B1BE", "#C45B25", "#52677B")
MARKERS = ("o", "s", "D", "^")


def template():
    return {"environment": None, "generator": None, "requested_tasks": None,
            "source_files": [], "interval": "patient-cluster 95% bootstrap CI",
            "checkpoints": [{"round": r, "quality": {d: None for d in DIMENSIONS},
                             "verification_rate": None, "high_order_valid_tasks": None,
                             "verified_tasks": None, "scored_tasks": None, "patients": None}
                            for r in range(5)]}


def draw(data):
    rows = data["checkpoints"]
    if [r["round"] for r in rows] != list(range(5)):
        raise ValueError("需要按顺序提供 0–4 五个检查点，缺失结果使用 null")
    if data.get("interval") != "patient-cluster 95% bootstrap CI":
        raise ValueError("区间定义必须与图注一致")
    observed = False
    fig, axes = plt.subplots(1, 3, figsize=(175 / 25.4, 76 / 25.4))
    fig.subplots_adjust(left=.08, right=.985, top=.77, bottom=.30, wspace=.44)
    def series(ax, values, color, marker, limits):
        nonlocal observed
        means = []
        for entry in values:
            if entry is None:
                means.append(np.nan)
                continue
            m = entry["mean"]
            if not np.isfinite(m) or not limits[0] <= m <= limits[1]:
                raise ValueError("指标超出允许范围")
            means.append(m); observed = True
        ax.plot(range(5), means, color=color, marker=marker, ms=3.8, lw=1.1)
        for x, entry in enumerate(values):
            if entry is None:
                continue
            low, high = entry.get("ci_low"), entry.get("ci_high")
            if (low is None) != (high is None):
                raise ValueError("区间两端须同时给出或缺失")
            if low is not None:
                if not limits[0] <= low <= entry["mean"] <= high <= limits[1]:
                    raise ValueError("区间须在允许范围内并包含均值")
                ax.errorbar(x, entry["mean"], yerr=[[entry["mean"]-low], [high-entry["mean"]]],
                            color=color, capsize=2, lw=.7)
    for ax, title in zip(axes, ("(a) Task quality", "(b) Verification", "(c) High-order tasks")):
        ax.set(title=title, xlim=(-.15, 4.15), xticks=range(5), xlabel="Evolution round")
        ax.grid(axis="y", color="#E4E9EE", lw=.5)
    axes[0].set(ylim=(1, 5), yticks=range(1, 6), ylabel="Quality score (1–5)")
    axes[1].set(ylim=(0, 100), yticks=(0, 25, 50, 75, 100), ylabel="Verification pass rate (%)")
    requested = data.get("requested_tasks")
    any_count = any(r["high_order_valid_tasks"] is not None for r in rows)
    if any_count and (not isinstance(requested, int) or requested <= 0):
        raise ValueError("计数结果需要实际请求题数作为上限")
    axes[2].set(ylabel="Verified L3–L4 tasks (count)", ylim=(0, requested or 1))
    if not requested:
        axes[2].set_yticks([])
    for dim, color, marker in zip(DIMENSIONS, COLORS, MARKERS):
        series(axes[0], [r["quality"][dim] for r in rows], color, marker, (1, 5))
    series(axes[1], [r["verification_rate"] for r in rows], COLORS[0], "o", (0, 100))
    series(axes[2], [r["high_order_valid_tasks"] for r in rows], COLORS[2], "D", (0, requested or 1))
    if observed and not (data.get("environment") and data.get("generator") and data.get("source_files")):
        raise ValueError("正式结果需要环境、模型与来源记录")
    if not observed:
        for ax in axes:
            ax.text(.5, .52, "Awaiting results", transform=ax.transAxes, ha="center",
                    color="#52677B", fontsize=8)
    fig.text(.08, .94, "Strategy evolution", fontsize=10, fontweight="bold")
    fig.text(.08, .865, " / ".join(filter(None, (data.get("environment"), data.get("generator"))))
             or "Layout preview", fontsize=8, color="#52677B")
    handles = [Line2D([], [], color=c, marker=m, ms=4, lw=1, label=l)
               for c, m, l in zip(COLORS, MARKERS, DIMENSION_LABELS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .105),
               ncol=4, frameon=False, fontsize=6.5, handlelength=1.5, columnspacing=1.2)
    fig.text(.08, .04, "0: initial strategy   |   Same held-out patients   |   Patient-cluster 95% bootstrap CI",
             fontsize=6.5, color="#52677B")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("papers/figures/experiments/template-strategy-evolution"))
    args = parser.parse_args()
    data = json.loads(args.data.read_text()) if args.data else template()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style()):
        figure = draw(data)
        for fmt in ("pdf", "svg", "png"):
            figure.savefig(args.output.with_suffix("."+fmt), dpi=300)
        plt.close(figure)
    args.output.with_suffix(".json").write_text(json.dumps(data, indent=2)+"\n")
