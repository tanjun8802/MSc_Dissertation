"""Seed-level discovery, retention and recovery summaries for controlled studies."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tbtrl.benchmarks.plots import NotebookDiagnostics, _curve
from tbtrl.benchmarks.study import METHODS
from tbtrl.experiments.figure_style import (
    COLORS,
    DEFAULT_STYLE,
    finish,
    legend,
    save_figure,
    styled,
    subplots,
)


def summarize_study(records):
    """One row per seed/task/method; campaigns never count as independent seeds."""
    rows = []
    for record in records:
        metrics = json.loads((Path(record["path"]) / "benchmark.json").read_text())[
            "campaign_metrics"
        ]
        rows.append(
            dict(
                seed=record["seed"],
                method=record["method"],
                task=record["task"],
                task_id=record["task_id"],
                mean_mSUN=float(np.mean([m["novelty_stable_unique_novel_count"] for m in metrics])),
                mean_AUDC=float(
                    np.mean([m["discovery_curve_area_under_discovery_curve"] for m in metrics])
                ),
                proposal_evaluations=record["final"]["queries"],
                reference_evaluations=record["final"]["reference_evaluations"],
                hours=record["seconds"] / 3600,
                archive_samples=record["replay_memory"]["archive_samples"],
                replay_MiB=record["replay_memory"]["payload_bytes"] / 1024**2,
                **{
                    k: record[k]
                    for k in ("after_task", "recovered_task", "recovery_steps")
                    if k in record
                },
            )
        )
    return rows


def _export(fig, path, style, display):
    import matplotlib.pyplot as plt
    from IPython.display import display as show

    save_figure(fig, Path(path), style=style)
    if display:
        show(fig)
    plt.close(fig)


def _line(ax, x, values, *, label, color, marker=None):
    values = np.asarray(values)
    mean = values.mean(axis=0)
    ax.plot(x, mean, label=label, color=color, marker=marker)
    if len(values) > 1:
        sd = values.std(axis=0, ddof=1)
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.13, color=color)


class StudyDiagnostics(NotebookDiagnostics):
    """Existing native plots plus candidate-ranking and representation diagnostics."""

    def __init__(self, *, style=DEFAULT_STYLE, display=True):
        super().__init__("made", style=style, display=display)

    def show_cached(self, record):
        """Redisplay completed-task diagnostics after a kernel restart, with no oracle."""
        if not self.display:
            return
        from IPython.display import SVG, Image, display

        for name in (
            "training-diagnostics",
            "objective-diagnostics",
            "observed-phase-diagram",
            "task-similarity",
            "selection-diagnostics",
        ):
            path = Path(record["path"]) / name
            if path.with_suffix(".svg").exists():
                display(SVG(filename=str(path.with_suffix(".svg"))))
            elif path.with_suffix(".png").exists():
                display(Image(filename=str(path.with_suffix(".png"))))
            else:
                print(f"Saved figure unavailable: {path.name} (training results are complete)")

    @styled
    def __call__(self, *, record, environment, learner):
        super().__call__(record=record, environment=environment, learner=learner)
        fig, axes = subplots(2, 2, style=self.style)
        history = record["history"]
        axes[0, 0].scatter(
            [r["predicted_q"] for r in history],
            [r["realised_return"] for r in history],
            s=8,
            alpha=0.4,
        )
        axes[0, 0].set(
            xlabel="Predicted Q of selected candidate", ylabel="Realised remaining campaign return"
        )
        spectrum = record.get("representation", {})
        if spectrum:
            values = spectrum["covariance_eigenvalues"]
            axes[0, 1].semilogy(
                range(1, len(values) + 1), np.maximum(values, 1e-10), label="Observed"
            )
            axes[0, 1].axhline(
                spectrum["covariance_target_variance"],
                color=COLORS[1],
                linestyle="--",
                label="Variance target",
            )
            legend(axes[0, 1])
        axes[0, 1].set(xlabel="Covariance eigenvalue rank", ylabel="Feature variance")
        losses = record["losses"]
        if losses and "td_replay_grad_cosine" in losses[0]:
            axes[1, 0].plot(
                [r["step"] for r in losses], [r["td_replay_grad_cosine"] for r in losses]
            )
        axes[1, 0].axhline(0, color="gray", linewidth=0.7)
        axes[1, 0].set(
            xlabel="Training transitions",
            ylabel="Current / historical TD gradient cosine",
            ylim=(-1.05, 1.05),
        )
        formulas = defaultdict(lambda: [0, 0.0])
        for proposal, row in zip(environment.discoveries, history):
            formulas[proposal["formula"]][0] += 1
            formulas[proposal["formula"]][1] += row["reward"]
        labels = list(formulas)
        x = np.arange(len(labels))
        axes[1, 1].bar(x - 0.18, [formulas[k][0] for k in labels], 0.36, label="Proposals")
        axes[1, 1].bar(
            x + 0.18, [formulas[k][1] for k in labels], 0.36, label="Net discovery reward"
        )
        axes[1, 1].set(xticks=x, xticklabels=labels, ylabel="Count", xlabel="Proposed composition")
        legend(axes[1, 1])
        finish(
            fig,
            title=f"{record['task']} · {METHODS[record['method']]} · selection diagnostics",
            note="Returns follow the changing exploratory policy, not an optimal-policy Q target. Spectrum uses at most 128 replay samples; cosine < 0 indicates opposing gradients.",
        )
        _export(fig, Path(record["path"]) / "selection-diagnostics", self.style, self.display)


@styled
def plot_study(records, output_dir, *, style=DEFAULT_STYLE, display=True):
    """Acquisition curves and campaign endpoints; shade variation ACROSS seeds."""
    summary = Path(output_dir) / "summary.json"
    if summary.exists():
        records = json.loads(summary.read_text())
    if not records:
        return
    methods = list(dict.fromkeys(r["method"] for r in records))
    tasks = sorted({r["task_id"] for r in records})
    for campaign_plot in (False, True):
        fig, axes = subplots((len(tasks) + 1) // 2, 2, style=style, squeeze=False)
        for task, ax in zip(tasks, axes.flat):
            for method in methods:
                selected = [r for r in records if r["task_id"] == task and r["method"] == method]
                if not selected:
                    continue
                if campaign_plot:
                    values = []
                    for record in selected:
                        metrics = json.loads((Path(record["path"]) / "benchmark.json").read_text())[
                            "campaign_metrics"
                        ]
                        values.append([m["novelty_stable_unique_novel_count"] for m in metrics])
                    x = np.arange(1, len(values[0]) + 1)
                else:
                    curves = [_curve(r, "made") for r in selected]
                    stop = min(len(y) for _, y in curves)
                    x, values = np.arange(1, stop + 1), [y[:stop] for _, y in curves]
                _line(
                    ax,
                    x,
                    values,
                    label=f"{METHODS[method]} (n={len(selected)})",
                    color=COLORS[list(METHODS).index(method) % len(COLORS)],
                    marker="o" if campaign_plot else None,
                )
            ax.set(
                title=next(r["task"] for r in records if r["task_id"] == task),
                xlabel="Campaign number" if campaign_plot else "Proposal evaluations per campaign",
                ylabel="Final mSUN" if campaign_plot else "Mean mSUN",
            )
            legend(ax)
        for ax in list(axes.flat)[len(tasks) :]:
            ax.set_visible(False)
        finish(
            fig,
            title="Adaptation across campaigns" if campaign_plot else "Discovery efficiency",
            note="Mean ±1 sample SD across seeds (n shown); campaigns are averaged within seeds. Campaigns include online learning, not frozen evaluation.",
        )
        _export(
            fig,
            Path(output_dir) / ("study-campaigns" if campaign_plot else "study-discovery"),
            style,
            display,
        )
    fig, axes = subplots(1, 2, style=style, height=3.2)
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        for seed in sorted({r["seed"] for r in rows}):
            selected = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["task_id"])
            for ax, field, scale in (
                (axes[0], "archive_samples", 1),
                (axes[1], "payload_bytes", 1024**2),
            ):
                ax.plot(
                    [r["task_id"] for r in selected],
                    [r["replay_memory"][field] / scale for r in selected],
                    label=f"{METHODS[method]}, seed {seed}",
                    marker="o",
                )
    axes[0].set(xlabel="Completed task ID", ylabel="Archived transitions")
    axes[1].set(xlabel="Completed task ID", ylabel="Replay array payload (MiB)")
    for ax in axes:
        legend(ax)
    finish(
        fig,
        title="Replay use",
        note="Payload excludes Python overhead, oracle memory and checkpoint files on disk.",
    )
    _export(fig, Path(output_dir) / "study-memory", style, display)


@styled
def plot_retention(records, output_dir, *, style=DEFAULT_STYLE, display=True):
    rows = summarize_study(records)
    for method in dict.fromkeys(r["method"] for r in rows):
        selected = [r for r in rows if r["method"] == method]
        count = max(r["after_task"] for r in selected) + 1
        matrix = np.full((count, count), np.nan)
        for stage in range(count):
            for task in range(count):
                values = [
                    r["mean_mSUN"]
                    for r in selected
                    if r["after_task"] == stage and r["task_id"] == task
                ]
                if values:
                    matrix[stage, task] = np.mean(values)
        fig, ax = subplots(style=style, height=4)
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis", vmin=0)
        for (stage, task), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                ax.text(
                    task,
                    stage,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1),
                )
        names = {r["task_id"]: r["task"] for r in selected}
        ax.set(
            xticks=range(count),
            yticks=range(count),
            xticklabels=[names.get(i, str(i)) for i in range(count)],
            xlabel="Evaluated task",
            ylabel="After training task ID",
        )
        fig.colorbar(im, ax=ax, label="Held-out mean final mSUN")
        finish(
            fig,
            title=f"Retention · {METHODS[method]}",
            note="Frozen greedy policy, held-out campaigns. Cells average seeds; missing/unlearned cells are blank. Scratch evaluates each independent model only on its own task.",
        )
        _export(fig, Path(output_dir) / f"retention-{method}", style, display)


def recovery_table(probes, retention, fraction=0.9):
    """First measured boundary regaining a fraction of own post-acquisition performance."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    references = {
        (r["seed"], r["method"], r["task_id"]): r["mean_mSUN"]
        for r in summarize_study(retention)
        if r["after_task"] == r["task_id"]
    }
    groups = defaultdict(list)
    for row in summarize_study(probes):
        if row["task_id"] == row["recovered_task"]:
            groups[(row["seed"], row["method"], row["task_id"])].append(row)
    rows = []
    for key, values in groups.items():
        reference = references.get(key)
        reached = [
            r["recovery_steps"]
            for r in values
            if reference is not None and reference > 0 and r["mean_mSUN"] >= fraction * reference
        ]
        rows.append(
            dict(
                seed=key[0],
                method=key[1],
                task_id=key[2],
                reference_mSUN=reference,
                threshold=None if reference is None else fraction * reference,
                proposals_to_recover=min(reached) if reached else None,
                status="reference unavailable/zero"
                if not reference
                else "recovered"
                if reached
                else "not reached within budget",
            )
        )
    return rows


@styled
def plot_recovery(probes, output_dir, *, style=DEFAULT_STYLE, display=True):
    rows = summarize_study(probes)
    for method in dict.fromkeys(r["method"] for r in rows):
        selected = [r for r in rows if r["method"] == method]
        for recovered in sorted({r["recovered_task"] for r in selected}):
            fig, ax = subplots(style=style, height=3.4)
            branch = [r for r in selected if r["recovered_task"] == recovered]
            for task in sorted({r["task_id"] for r in branch}):
                values = [r for r in branch if r["task_id"] == task]
                steps = sorted({r["recovery_steps"] for r in values})
                seeds = sorted({r["seed"] for r in values})
                # Only complete seed trajectories; never impute missing evaluations.
                trajectories = [
                    [
                        next(
                            (
                                r["mean_mSUN"]
                                for r in values
                                if r["seed"] == seed and r["recovery_steps"] == step
                            ),
                            np.nan,
                        )
                        for step in steps
                    ]
                    for seed in seeds
                ]
                trajectories = [v for v in trajectories if np.isfinite(v).all()]
                if trajectories:
                    _line(
                        ax,
                        steps,
                        trajectories,
                        label=values[0]["task"],
                        color=COLORS[task % len(COLORS)],
                        marker="o",
                    )
            ax.set(
                xlabel="Additional training proposals on recovered task",
                ylabel="Held-out mean final mSUN",
            )
            legend(ax)
            finish(
                fig,
                title=f"{METHODS[method]} · recovering task {recovered}",
                note="All tasks probed with frozen policies at each boundary. Mean ±1 sample SD across seeds. Probe oracle calls are additional to the training budget.",
            )
            _export(fig, Path(output_dir) / f"recovery-{method}-task-{recovered}", style, display)
