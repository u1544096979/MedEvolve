"""从独立汇总数据绘制实验 1 的有效性分布和四维质量图。"""

import argparse
import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

METHODS = ("medevolve", "taskcraft", "workflow", "singleshot", "selfgen", "envexplore")
LABELS = ("MedEvolve", "TaskCraft", "Workflow", "SingleShot", "SelfRefine", "EnvExplore")
METHOD_COLORS = ("#087F83", "#0072B2", "#7957A4", "#C45B25", "#88644E", "#52677B")
MARKERS = ("o", "s", "D", "^", "v", "P")
DIMENSIONS = ("factual_accuracy", "clinical_soundness", "evidence_sufficiency", "task_clarity")
DIMENSION_LABELS = ("Factual accuracy", "Clinical soundness", "Evidence sufficiency", "Task clarity")
OUTCOMES = ("invalid", "evaluation_missing", "runtime_error", "level_1", "level_2", "level_3", "level_4")
OUTCOME_LABELS = ("Verification failed", "Evaluation missing", "Upstream error", "L1: Retrieval", "L2: Single-step", "L3: Integration", "L4: Decision")
OUTCOME_COLORS = ("#B0B7BF", "#EBEEF1", "#6B717A", "#B7DCE3", "#68B1BE", "#087F83", "#C45B25")
HATCHES = ("//", "..", "xx", "", "--", "++", "oo")
Count = Annotated[int, Field(strict=True, ge=0)]
PositiveCount = Annotated[int, Field(strict=True, gt=0)]
Score = Annotated[float, Field(strict=True, ge=1, le=5, allow_inf_nan=False)]
Method = Literal["medevolve", "taskcraft", "workflow", "singleshot", "selfgen", "envexplore"]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Outcomes(InputModel):
    invalid: Count
    evaluation_missing: Count
    runtime_error: Count
    level_1: Count
    level_2: Count
    level_3: Count
    level_4: Count


class Dimension(InputModel):
    mean: Score
    ci_low: Score | None
    ci_high: Score | None
    n_tasks: PositiveCount
    n_patients: PositiveCount

    @model_validator(mode="after")
    def valid_summary(self):
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("both confidence interval endpoints must be present or null")
        if self.ci_low is not None and not self.ci_low <= self.mean <= self.ci_high:
            raise ValueError("confidence interval must contain the mean")
        if self.n_patients > self.n_tasks:
            raise ValueError("patient count exceeds scored task count")
        return self


class Quality(InputModel):
    factual_accuracy: Dimension | None
    clinical_soundness: Dimension | None
    evidence_sufficiency: Dimension | None
    task_clarity: Dimension | None


class MethodResult(InputModel):
    method: Method
    generated_tasks: Count | None = None
    generated_patients: Count | None = None
    outcomes: Outcomes | None = None
    quality: Quality | None = None

    @model_validator(mode="after")
    def valid_counts(self):
        if self.outcomes is not None:
            total = sum(getattr(self.outcomes, name) for name in OUTCOMES)
            if self.generated_tasks is None or total != self.generated_tasks:
                raise ValueError("outcome counts must partition generated_tasks")
        if self.generated_patients is not None and self.generated_tasks is not None:
            if self.generated_patients > self.generated_tasks:
                raise ValueError("generated patient count exceeds generated task count")
        if self.quality is not None:
            for name in DIMENSIONS:
                entry = getattr(self.quality, name)
                if entry is None:
                    continue
                if self.generated_tasks is not None and entry.n_tasks > self.generated_tasks:
                    raise ValueError("scored task count exceeds generated task count")
                if self.outcomes is not None:
                    possible = self.generated_tasks - self.outcomes.invalid
                    if entry.n_tasks > possible:
                        raise ValueError("quality count includes verification-failed tasks")
                if self.generated_patients is not None and entry.n_patients > self.generated_patients:
                    raise ValueError("scored patient count exceeds generated patient count")
        return self


class Group(InputModel):
    environment: str | None = None
    generator: str | None = None
    source_files: list[str] = Field(default_factory=list)
    results: list[MethodResult]

    @model_validator(mode="after")
    def complete_methods(self):
        names = [entry.method for entry in self.results]
        if len(names) != len(METHODS) or set(names) != set(METHODS):
            raise ValueError("each group requires the six methods exactly once")
        for value in (self.environment, self.generator):
            if value is not None and not value.strip():
                raise ValueError("environment and generator must be nonempty or null")
        populated = any(entry.generated_tasks is not None or entry.outcomes is not None
                        or entry.quality is not None for entry in self.results)
        if populated and (self.environment is None or self.generator is None):
            raise ValueError("observed data requires environment and generator labels")
        return self


class FigureData(InputModel):
    version: Literal[1] = 1
    groups: list[Group] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_groups(self):
        names = [group_stem(group) for group in self.groups]
        if len(names) != len(set(names)):
            raise ValueError("group output names collide")
        return self


def blank_data():
    return FigureData(groups=[Group(results=[MethodResult(method=name) for name in METHODS])])


def group_stem(group):
    if group.environment is None and group.generator is None:
        return "template"
    return re.sub(r"[^a-z0-9]+", "-", f"{group.environment or 'unspecified'}-{group.generator or 'unspecified'}".lower()).strip("-")


def group_label(group):
    return " / ".join(value for value in (group.environment, group.generator) if value)


def style():
    return {"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
            "axes.titlesize": 9, "axes.linewidth": .6, "axes.edgecolor": "#52677B",
            "axes.labelcolor": "#17324D", "text.color": "#17324D", "xtick.color": "#52677B",
            "ytick.color": "#17324D", "xtick.labelsize": 7, "ytick.labelsize": 8,
            "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
            "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
            "figure.facecolor": "white", "savefig.facecolor": "white"}


def ordered_results(group):
    lookup = {entry.method: entry for entry in group.results}
    return [lookup[name] for name in METHODS]


def setup_rows(ax):
    ax.set_yticks(range(6), LABELS)
    ax.set_ylim(5.6, -.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E4E9EE", linewidth=.5)
    ax.spines["left"].set_visible(False)
    ax.get_yticklabels()[0].set_fontweight("bold")


def draw_validity(group):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(175 / 25.4, 88 / 25.4))
    fig.subplots_adjust(left=.17, right=.87, top=.82, bottom=.30)
    setup_rows(ax)
    ax.set(xlim=(0, 100), xticks=(0, 25, 50, 75, 100), xlabel="Share of generated tasks (%)")
    ax.text(0, 1.21, "Task validity & reasoning requirements", transform=ax.transAxes,
            fontsize=10, fontweight="bold")
    ax.text(0, 1.10, group_label(group) or "Layout preview", transform=ax.transAxes,
            fontsize=8, color="#52677B")
    ax.text(1.04, 1.03, "Tasks", transform=ax.transAxes, fontsize=7, color="#52677B")
    for row, result in enumerate(ordered_results(group)):
        left = 0
        if result.outcomes is not None and result.generated_tasks:
            for name, color, hatch in zip(OUTCOMES, OUTCOME_COLORS, HATCHES):
                width = 100 * getattr(result.outcomes, name) / result.generated_tasks
                if width:
                    ax.barh(row, width, left=left, height=.60, color=color,
                            edgecolor="white", linewidth=.5, hatch=hatch)
                    left += width
        label = str(result.generated_tasks) if result.generated_tasks is not None else "—"
        ax.text(1.04, row, label, transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    handles = [Patch(facecolor=color, edgecolor="#52677B", linewidth=.3, hatch=hatch, label=label)
               for color, hatch, label in zip(OUTCOME_COLORS, HATCHES, OUTCOME_LABELS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .045),
               ncol=4, frameon=False, fontsize=7, columnspacing=1.1)
    return fig


def draw_quality(group):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(175 / 25.4, 126 / 25.4))
    fig.subplots_adjust(left=.17, right=.86, bottom=.13, top=.84, hspace=.53, wspace=1.15)
    fig.text(.17, .95, "Four-dimensional task quality", fontsize=10, fontweight="bold")
    fig.text(.17, .91, group_label(group) or "Layout preview", fontsize=8, color="#52677B")
    for index, (ax, dimension, title) in enumerate(zip(axes.flat, DIMENSIONS, DIMENSION_LABELS)):
        setup_rows(ax)
        ax.set(xlim=(1, 5), xticks=(1, 2, 3, 4, 5), xlabel="Quality score (1–5)")
        ax.set_title(f"({chr(97 + index)}) {title}", loc="left", pad=10)
        ax.text(1.05, 1.02, "n / P", transform=ax.transAxes, fontsize=6.5, color="#52677B")
        for row, (result, color, marker) in enumerate(zip(ordered_results(group), METHOD_COLORS, MARKERS)):
            entry = getattr(result.quality, dimension) if result.quality is not None else None
            if entry is None:
                label = "—"
            else:
                if entry.ci_low is not None:
                    ax.errorbar(entry.mean, row, xerr=[[entry.mean-entry.ci_low], [entry.ci_high-entry.mean]],
                                fmt="none", ecolor=color, elinewidth=1, capsize=2, capthick=.8)
                ax.plot(entry.mean, row, marker=marker, color=color, markersize=4,
                        markeredgecolor="white", markeredgewidth=.4)
                label = f"{entry.n_tasks} / {entry.n_patients}"
            ax.text(1.05, row, label, transform=ax.get_yaxis_transform(), fontsize=6.5, va="center")
    fig.text(.17, .035, "n: scored tasks   P: patients   Intervals: patient-cluster 95% bootstrap CI",
             fontsize=7, color="#52677B")
    return fig


def export_group(group, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = group_stem(group)
    with plt.rc_context(style()):
        for name, draw in (("task-validity", draw_validity), ("task-quality", draw_quality)):
            fig = draw(group)
            for extension in ("pdf", "svg", "png"):
                destination = output_dir / f"{stem}-{name}.{extension}"
                fig.savefig(destination, dpi=300, bbox_inches=None)
                print(destination)
            plt.close(fig)
    (output_dir / f"{stem}-data.json").write_text(
        FigureData(groups=[group]).model_dump_json(indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("papers/figures/experiments"))
    parser.add_argument("--write-schema", type=Path)
    parser.add_argument("--write-template", type=Path)
    args = parser.parse_args()
    for destination, payload in ((args.write_schema, FigureData.model_json_schema()),
                                  (args.write_template, blank_data().model_dump(mode="json"))):
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    data = FigureData.model_validate_json(args.input.read_text()) if args.input else blank_data()
    for group in data.groups:
        export_group(group, args.output_dir)


if __name__ == "__main__":
    main()
