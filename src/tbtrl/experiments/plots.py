"""Plot saved metrics; matplotlib is required only for this optional module."""

import json
from pathlib import Path


def plot_run(output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    records = json.loads((Path(output_dir) / "summary.json").read_text())
    seeds = sorted({row["seed"] for row in records})
    fig, axes = plt.subplots(len(seeds), 2, figsize=(12, 4 * len(seeds)), squeeze=False)
    for i, seed in enumerate(seeds):
        rows = [row for row in records if row["seed"] == seed]
        for mode in ("scratch", "sequential", "recovery"):
            subset = [row for row in rows if row["mode"] == mode]
            if not subset:
                continue
            values = [
                row["steps_to_threshold"] if row["solved"] else row["steps"] for row in subset
            ]
            (line,) = axes[i, 0].plot(
                [row["task_id"] for row in subset], values, marker="o", label=mode
            )
            failed = [(row["task_id"], row["steps"]) for row in subset if not row["solved"]]
            if failed:
                axes[i, 0].scatter(*zip(*failed), marker="x", color=line.get_color(), s=90)
        axes[i, 0].set(
            title=f"Seed {seed}: steps to threshold (× = unsolved budget)",
            xlabel="Task",
            ylabel="Environment steps",
        )
        axes[i, 0].legend()
        sequence = [row for row in rows if row["mode"] == "sequential"]
        if sequence:
            matrix = np.array(
                [[item["success_rate"] for item in row["evaluation"]] for row in sequence]
            )
            im = axes[i, 1].imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
            axes[i, 1].set(
                title="Transfer and retention: success rate",
                xlabel="Evaluated task",
                ylabel="After training task",
            )
            fig.colorbar(im, ax=axes[i, 1])
    fig.tight_layout()
    return fig


def plot_comparison(output_dir):
    """Per-goal and cumulative step/time costs, averaged across configured seeds.

    Unsolved runs contribute their spent budget; they are never silently dropped.
    The standard deviation is across seeds (zero for a single-seed experiment).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    records = json.loads((Path(output_dir) / "summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for mode in ("scratch", "sequential", "recovery"):
        rows = [r for r in records if r["mode"] == mode]
        seeds = sorted({r["seed"] for r in rows})
        tasks = sorted({r["task_id"] for r in rows})
        if not tasks:
            continue
        for column, metric in enumerate(("steps", "elapsed_seconds")):
            values = np.array(
                [
                    [
                        next(r[metric] for r in rows if r["seed"] == seed and r["task_id"] == task)
                        for task in tasks
                    ]
                    for seed in seeds
                ]
            )
            for row, costs in enumerate((values, values.cumsum(axis=1))):
                mean = costs.mean(axis=0)
                std = costs.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(len(tasks))
                ax = axes[row, column]
                ax.plot(tasks, mean, marker="o", label=mode)
                ax.fill_between(tasks, mean - std, mean + std, alpha=0.2)
                ax.set(
                    xlabel="Goal index",
                    ylabel="Steps" if column == 0 else "Seconds",
                    title=("Per-goal" if row == 0 else "Cumulative")
                    + " training cost (mean ± seed SD)",
                )
                ax.legend()
    fig.tight_layout()
    return fig
