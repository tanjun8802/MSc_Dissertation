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
