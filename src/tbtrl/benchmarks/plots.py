"""Shared inline diagnostics and publication exports for discovery notebooks."""

import json
from pathlib import Path

import numpy as np
import torch

from tbtrl.experiments.figure_style import (
    DEFAULT_STYLE,
    MODES,
    finish,
    legend,
    numeric_axis,
    save_figure,
    styled,
    subplots,
)


def _curve(record, benchmark):
    path = Path(record["path"])
    if benchmark == "pmo":
        rows = json.loads((path / "molecules.json").read_text())
        return np.array([r["queries"] for r in rows]), np.array([r["top10"] for r in rows])
    rows = json.loads((path / "discovery.json").read_text())
    campaigns = sorted({r["campaign"] for r in rows})
    values = [[r["discoveries"] for r in rows if r["campaign"] == c] for c in campaigns]
    return np.arange(1, len(values[0]) + 1), np.mean(values, axis=0)


class NotebookDiagnostics:
    def __init__(self, benchmark, *, style=DEFAULT_STYLE, display=True):
        if benchmark not in ("pmo", "made"):
            raise ValueError("Choose pmo or made")
        self.benchmark, self.style, self.display = benchmark, style, display

    @styled
    def __call__(self, *, record, environment, learner):
        import matplotlib.pyplot as plt
        from IPython.display import SVG, display

        path = Path(record["path"])
        fig, axes = subplots(1, 2, style=self.style, height=3.2)
        if self.benchmark == "pmo":
            x, y = _curve(record, self.benchmark)
            axes[0].plot(x, y, label="Mean top-10 score")
            axes[0].set(xlabel="Unique oracle calls", ylabel="TDC objective score")
        else:
            for campaign in range(1, environment.campaign + 1):
                rows = [r for r in environment.discoveries if r["campaign"] == campaign]
                axes[0].plot(
                    [r["queries"] for r in rows],
                    [r["discoveries"] for r in rows],
                    label=f"Campaign {campaign}",
                )
            axes[0].set(
                xlabel="Proposal evaluations in campaign", ylabel="Novel stable structures (mSUN)"
            )
        for key, label in (
            ("td", "Current-task TD"),
            ("replay", "Past-task TD"),
            ("total", "Total loss"),
        ):
            if record["losses"]:
                axes[1].plot(
                    [v["step"] for v in record["losses"]],
                    [v[key] for v in record["losses"]],
                    label=label,
                )
        axes[1].set(xlabel="Environment transitions", ylabel="Training loss", yscale="symlog")
        for ax in axes:
            numeric_axis(ax, x=True)
            legend(ax)
        finish(fig, title=f"{record['task']} · {MODES[record['mode']][0]} · seed {record['seed']}")
        save_figure(fig, path / "training-diagnostics", style=self.style)
        if self.display:
            display(fig)
        plt.close(fig)
        if self.benchmark == "made" and record["losses"]:
            losses = record["losses"]
            if "weighted_covariance" in losses[0]:
                fig, axes = subplots(1, 2, style=self.style, height=3.2)
                steps = [row["step"] for row in losses]
                for key, label in (
                    ("covariance", "Covariance"),
                    ("separation", "Task separation"),
                    ("norm", "Norm"),
                ):
                    axes[0].plot(steps, [r[f"weighted_{key}"] for r in losses], label=label)
                for key, label in (
                    ("td", "Current TD"),
                    ("replay", "Historical TD"),
                    ("covariance", "Covariance"),
                    ("separation", "Separation"),
                    ("norm", "Norm"),
                ):
                    if f"grad_{key}" in losses[0]:
                        axes[1].plot(steps, [r[f"grad_{key}"] for r in losses], label=label)
                axes[0].set(
                    xlabel="Training transitions",
                    ylabel="Weighted regularisation loss",
                    yscale="symlog",
                )
                axes[1].set(
                    xlabel="Training transitions",
                    ylabel="Weighted gradient L2 norm",
                    yscale="symlog",
                )
                for ax in axes:
                    ax.set_yscale("symlog", linthresh=1e-4)
                    ax.set_ylim(bottom=0)
                    numeric_axis(ax, x=True)
                    legend(ax)
                finish(
                    fig,
                    title=f"{record['task']} · objective contributions",
                    note="Gradients are measured before global clipping. Zero norm penalty means its threshold is inactive.",
                )
                save_figure(fig, path / "objective-diagnostics", style=self.style)
                if self.display:
                    display(fig)
                plt.close(fig)
        domain = environment.plot_domain()
        if self.benchmark == "pmo":
            svg = domain.data if hasattr(domain, "data") else domain
            (path / "top-molecules.svg").write_text(svg)
            if self.display:
                display(SVG(svg))
        else:
            # Pymatgen supplies the native hull; normalise its large hard-coded
            # fonts to the same physical figure style used throughout the notebooks.
            from matplotlib.text import Text

            domain.set_size_inches(self.style.width, self.style.width * 0.65)
            domain.set_layout_engine("constrained")
            for text in domain.findobj(match=Text):
                text.set_fontsize(self.style.font_size)
                text.set_fontweight("normal")
            for ax in domain.axes:
                for line in ax.lines:
                    line.set_linewidth(1.3)
                    line.set_markersize(5)
                if len(environment.env.observed_pd.elements) == 2:
                    element = environment.env.observed_pd.elements[-1]
                    ax.set_xlabel(f"Atomic fraction of {element}")
                    ax.set_ylabel("Formation energy (eV/atom)")
            finish(
                domain,
                title=f"{record['task']} · observed phase diagram · campaign {environment.campaign}",
                note=f"Hull line: strictly stable phases. Discovery counts include structures within {environment.env.stability_tolerance:g} eV/atom above the observed hull.",
            )
            save_figure(domain, path / "observed-phase-diagram", style=self.style)
            if self.display:
                display(domain)
            plt.close(domain)
        with torch.no_grad():
            embeddings = (
                learner.network.encode_goal(learner.tensor(list(learner.seen.values())))
                .cpu()
                .numpy()
            )
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
        fig, ax = subplots(style=self.style, height=3.1)
        matrix = embeddings @ embeddings.T
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        if len(matrix) <= 10:
            for row in range(len(matrix)):
                for col in range(len(matrix)):
                    ax.text(
                        col,
                        row,
                        f"{matrix[row, col]:.3f}",
                        ha="center",
                        va="center",
                        color="white" if abs(matrix[row, col]) > 0.6 else "black",
                    )
        ids = list(learner.seen)
        ax.set(
            xticks=range(len(ids)),
            yticks=range(len(ids)),
            xticklabels=ids,
            yticklabels=ids,
            xlabel="Task ID",
            ylabel="Task ID",
            title="Learned task representation cosine similarity",
        )
        fig.colorbar(im, ax=ax, label="Cosine similarity")
        save_figure(fig, path / "task-similarity", style=self.style)
        if self.display:
            display(fig)
        plt.close(fig)
        print("Replay memory:", json.dumps(record["replay_memory"], indent=2), flush=True)


@styled
def plot_comparison(records, benchmark, output_dir, *, style=DEFAULT_STYLE, display=True):
    """Aggregate paired seeds; MADE averages campaigns WITHIN each seed first."""
    import matplotlib.pyplot as plt
    from IPython.display import display as show

    tasks = sorted({r["task_id"] for r in records})
    fig, axes = subplots((len(tasks) + 1) // 2, 2, style=style, squeeze=False)
    for task, ax in zip(tasks, axes.flat):
        selected = [r for r in records if r["task_id"] == task]
        for mode in ("scratch", "sequential"):
            curves = [_curve(r, benchmark) for r in selected if r["mode"] == mode]
            # No extrapolation of early-aborted PMO runs as if they used the full budget.
            stop = min(int(x[-1]) for x, _ in curves)
            x = np.arange(1, stop + 1)
            values = np.asarray([np.interp(x, xx, yy) for xx, yy in curves])
            label, color, _, line = MODES[mode]
            mean = values.mean(axis=0)
            ax.plot(x, mean, color=color, linestyle=line, label=label)
            if len(values) > 1:
                sd = values.std(axis=0, ddof=1)
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15)
        ax.set(
            title=f"Task {task}: {selected[0]['task']}",
            xlabel="Unique oracle calls"
            if benchmark == "pmo"
            else "Proposal evaluations per campaign",
            ylabel="Mean top-10 score" if benchmark == "pmo" else "Novel stable structures (mSUN)",
        )
        numeric_axis(ax, x=True)
        legend(ax)
    for ax in list(axes.flat)[len(tasks) :]:
        ax.set_visible(False)
    finish(
        fig,
        title="Scratch vs sequential TBTRL",
        note=(
            "Online optimisation, matched task budgets. Lines: means across seeds; shading: ±1 sample SD when multiple seeds. "
            + (
                "Campaigns are averaged within each seed. Reference setup costs are reported separately."
                if benchmark == "made"
                else "Only newly scored canonical molecules spend oracle budget. Revisited molecules use cached scores."
            )
        ),
    )
    save_figure(fig, Path(output_dir) / "comparison", style=style)
    if display:
        show(fig)
    plt.close(fig)
    if benchmark == "made":
        fig, axes = subplots((len(tasks) + 1) // 2, 2, style=style, squeeze=False)
        for task, ax in zip(tasks, axes.flat):
            selected = [r for r in records if r["task_id"] == task]
            for mode in ("scratch", "sequential"):
                values = []
                for record in selected:
                    if record["mode"] == mode:
                        metrics = json.loads((Path(record["path"]) / "benchmark.json").read_text())[
                            "campaign_metrics"
                        ]
                        values.append([m["num_newly_discovered_stable"] for m in metrics])
                values = np.asarray(values)
                label, color, _, line = MODES[mode]
                x = np.arange(1, values.shape[1] + 1)
                mean = values.mean(axis=0)
                ax.plot(x, mean, color=color, linestyle=line, marker="o", label=label)
                if len(values) > 1:
                    sd = values.std(axis=0, ddof=1)
                    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15)
            ax.set(
                title=f"Task {task}: {selected[0]['task']}",
                xlabel="Campaign number",
                ylabel="Final discoveries (mSUN)",
                xticks=x,
            )
            legend(ax)
        for ax in list(axes.flat)[len(tasks) :]:
            ax.set_visible(False)
        finish(
            fig,
            title="Adaptation across campaigns",
            note="Environment/candidate pool resets each campaign; model keeps learning. Means ±1 sample SD across seeds, not independent campaign replicates.",
        )
        save_figure(fig, Path(output_dir) / "campaign-comparison", style=style)
        if display:
            show(fig)
        plt.close(fig)
    fig, axes = subplots(1, 2, style=style, height=3.1)
    for seed in sorted({r["seed"] for r in records}):
        rows = [r for r in records if r["mode"] == "sequential" and r["seed"] == seed]
        axes[0].plot(
            [r["task_id"] for r in rows],
            [r["replay_memory"]["archive_samples"] for r in rows],
            marker="o",
            label=f"Seed {seed}",
        )
        axes[1].plot(
            [r["task_id"] for r in rows],
            [r["replay_memory"]["payload_bytes"] / 1024**2 for r in rows],
            marker="o",
            label=f"Seed {seed}",
        )
    axes[0].set(xlabel="Completed task ID", ylabel="Archived transitions")
    axes[1].set(xlabel="Completed task ID", ylabel="Replay array payload (MiB)")
    for ax in axes:
        numeric_axis(ax)
        legend(ax)
    finish(
        fig,
        title="Bounded sequential replay",
        note="Completed-task buffer discarded after tail admission. Benchmark caches, results and Python metadata are separate.",
    )
    save_figure(fig, Path(output_dir) / "replay-memory", style=style)
    if display:
        show(fig)
    plt.close(fig)
