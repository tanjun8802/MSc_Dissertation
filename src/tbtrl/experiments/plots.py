"""Paper figures from saved metrics; training data are never modified."""

import json
from pathlib import Path

import numpy as np

from .figure_style import (
    DEFAULT_STYLE,
    MODES,
    annotate_matrix,
    empty_panel,
    finish,
    goal_ticks,
    legend,
    numeric_axis,
    save_figure,
    styled,
    subplots,
)


def _records(output_dir):
    records = json.loads((Path(output_dir) / "summary.json").read_text())
    if not records:
        raise ValueError("The run has no task records to plot.")
    return records


@styled
def plot_run(output_dir, *, style=DEFAULT_STYLE, export=True):
    """Stopping cost and all-goal success; export run-overview under figures/."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    records = _records(output_dir)
    seeds = sorted({row["seed"] for row in records})
    fig, axes = subplots(len(seeds), 2, style=style, height=3.8 * len(seeds), squeeze=False)
    for i, seed in enumerate(seeds):
        rows = [row for row in records if row["seed"] == seed]
        tasks = sorted({row["task_id"] for row in rows})
        ax = axes[i, 0]
        has_failed = has_skipped = False
        for mode, (label, color, marker, linestyle) in MODES.items():
            subset = sorted(
                (row for row in rows if row["mode"] == mode), key=lambda row: row["task_id"]
            )
            if not subset:
                continue
            ids = [r["task_id"] for r in subset]
            values = [r["steps_to_threshold"] if r.get("solved") else r["steps"] for r in subset]
            ax.plot(ids, values, color=color, linestyle=linestyle, label=label)
            for task, value, row in zip(ids, values, subset):
                skipped = row.get("recovery_performed") is False
                failed = not row.get("solved")
                ax.scatter(
                    task,
                    value,
                    marker="D" if skipped else "x" if failed else marker,
                    **(dict(facecolors="none", edgecolors=color) if skipped else dict(color=color)),
                    s=28,
                    zorder=3,
                    clip_on=False,
                )
                has_failed |= failed
                has_skipped |= skipped
        ax.set(
            title=f"(a) Training cost · seed {seed}",
            xlabel="Training goal ID",
            ylabel="Environment steps to stopping",
            xticks=tasks,
        )
        ax.set_ylim(bottom=0)
        goal_ticks(ax, tasks)
        numeric_axis(ax, y=True)
        if has_failed:
            ax.add_line(
                Line2D(
                    [],
                    [],
                    color="#333333",
                    marker="x",
                    linestyle="none",
                    label="Unsolved: spent budget",
                )
            )
        if has_skipped:
            ax.add_line(
                Line2D(
                    [],
                    [],
                    color="#333333",
                    marker="D",
                    markerfacecolor="none",
                    linestyle="none",
                    label="Recovery skipped",
                )
            )
        legend(ax, columns=1)
        sequence = sorted(
            (row for row in rows if row["mode"] == "sequential"), key=lambda row: row["task_id"]
        )
        ax = axes[i, 1]
        ax.set(
            title=f"(b) All-goal success · seed {seed}",
            xlabel="Evaluated goal ID",
            ylabel="After training goal ID",
        )
        if sequence:
            evaluated = sorted({item["task_id"] for row in sequence for item in row["evaluation"]})
            matrix = np.array(
                [
                    [
                        next(
                            (
                                item["success_rate"]
                                for item in row["evaluation"]
                                if item["task_id"] == task
                            ),
                            np.nan,
                        )
                        for task in evaluated
                    ]
                    for row in sequence
                ]
            )
            im = ax.imshow(
                matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis", interpolation="nearest"
            )
            ax.set(
                xticks=range(len(evaluated)),
                xticklabels=evaluated,
                yticks=range(len(sequence)),
                yticklabels=[r["task_id"] for r in sequence],
            )
            annotate_matrix(ax, matrix, percent=True)
            goal_ticks(ax, evaluated, matrix=True)
            goal_ticks(ax, [r["task_id"] for r in sequence], axis="y", matrix=True)
            fig.colorbar(
                im,
                ax=ax,
                label="Evaluation success",
                format=PercentFormatter(1, decimals=0),
                shrink=0.85,
            )
        else:
            empty_panel(ax, "No sequential-training records")
    finish(
        fig,
        title="Training cost and goal retention",
        note="Unsolved tasks show their consumed budget, not a successful threshold. Recovery trials each restart from the final sequential model. Success is an empirical evaluation rate.",
    )
    if export:
        save_figure(fig, Path(output_dir) / "figures/run-overview", style=style)
    return fig


@styled
def plot_comparison(output_dir, *, style=DEFAULT_STYLE, export=True):
    """Mean spent cost; ±1 sample SD only for n>=2, using paired seed coverage.

    Failed budgets count. Recovery sums independent trials, not a continued run.
    """
    records = _records(output_dir)
    fig, axes = subplots(2, 2, style=style, height=6.7)
    counts = []
    all_tasks = sorted({r["task_id"] for r in records})
    for mode, (label, color, marker, linestyle) in MODES.items():
        rows = [r for r in records if r["mode"] == mode]
        seeds = sorted({r["seed"] for r in rows})
        tasks = sorted({r["task_id"] for r in rows})
        if not tasks:
            continue
        lookup = {(r["seed"], r["task_id"]): r for r in rows}
        if len(lookup) != len(rows) or set(lookup) != {(s, t) for s in seeds for t in tasks}:
            import matplotlib.pyplot as plt

            plt.close(fig)
            raise ValueError(f"{mode}: comparison requires one record per seed and goal.")
        counts.append(f"{label}: n={len(seeds)}")
        for column, metric in enumerate(("steps", "elapsed_seconds")):
            values = np.array(
                [[lookup[seed, task][metric] for task in tasks] for seed in seeds], dtype=float
            )
            if column == 1:
                values /= 60
            for row, costs in enumerate((values, values.cumsum(axis=1))):
                mean = costs.mean(axis=0)
                ax = axes[row, column]
                ax.plot(tasks, mean, marker=marker, color=color, linestyle=linestyle, label=label)
                if len(seeds) > 1:
                    std = costs.std(axis=0, ddof=1)
                    ax.fill_between(
                        tasks, mean - std, mean + std, color=color, alpha=0.14, linewidth=0
                    )
    for row in range(2):
        for column in range(2):
            ax = axes[row, column]
            ax.set(
                xlabel="Goal ID",
                xticks=all_tasks,
                ylabel="Environment steps" if column == 0 else "Training wall time (min)",
                title=f"({chr(97 + row * 2 + column)}) {'Per-goal' if row == 0 else 'Accumulated'} {'sample cost' if column == 0 else 'wall time'}",
            )
            numeric_axis(ax, y=column == 0)
            goal_ticks(ax, all_tasks)
            legend(ax, columns=1)
    note = (
        "; ".join(counts)
        + ". Lines: seed mean; bands: ±1 seed SD only for n≥2 (not confidence intervals). Unsolved budgets count; skipped recovery costs zero. Accumulated recovery sums independent trials. Wall time depends on hardware."
    )
    finish(fig, title="Sample and computation costs", note=note)
    if export:
        save_figure(fig, Path(output_dir) / "figures/cost-comparison", style=style)
    return fig
