"""Controlled bounded-replay presets and figures regenerated from saved DQN runs."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .figure_style import (
    COLORS,
    DEFAULT_STYLE,
    empty_panel,
    finish,
    legend,
    numeric_axis,
    save_figure,
    styled,
    subplots,
)


def configure_k_replay(
    config,
    *,
    max_tasks=4,
    sample_budget=256,
    selector="basis",
    weighting="uniform",
    candidate_capacity=32,
    refresh_updates=100,
    selector_seed=0,
    capacity=100000,
    recent_capacity=50000,
    max_groups=16,
):
    """An opt-in DQN condition; original configs and SAC are unchanged.

    Recovery uses acquisition exploration/update cadence, starting independently
    from the final sequential model. Its steps are not a single joint recovery.
    """
    if config.algorithm != "dqn":
        raise ValueError("Bounded K replay is a DQN study.")
    result = deepcopy(config)
    result.name = f"{config.name}-k{max_tasks}-{selector}-{weighting}-soft-ball"
    result.model = dict(result.model, embedding_constraint="soft")
    result.replay_mode, result.replay_schedule = "global", "constant"
    result.global_replay_capacity, result.recent_replay_capacity = capacity, recent_capacity
    result.replay_max_groups, result.similarity_threshold = max_groups, 0.8
    result.training.pop("replay_tasks_per_batch", None)
    result.training.update(
        replay_ratio=0,
        goal_separation_coef=0,
        sigreg_coef=0,
        k_replay=dict(
            max_tasks=max_tasks,
            sample_budget=sample_budget,
            selector=selector,
            weighting=weighting,
            candidate_capacity=candidate_capacity,
            refresh_updates=refresh_updates,
            seed=selector_seed,
        ),
    )
    result.scratch_seen_goals = False
    result.scratch = dict(replay_ratio=0, replay_loss_coef=0)
    result.first_task = dict(replay_ratio=0)
    result.recovery = dict(total_steps=result.training["total_steps"])
    result.recovery_threshold = None
    result.recovery_return_threshold = result.training.get("early_stop_reward", 0.99)
    return result.validate()


def load_study(paths):
    """Return saved cases, retaining failures and identifying incomplete seed/modes."""
    cases = {}
    for label, directory in paths.items():
        directory = Path(directory)
        records = json.loads((directory / "summary.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        config = manifest["config"]
        cases[label] = dict(directory=directory, records=records, config=config)
    return cases


def study_table(paths):
    """Per-seed acquisition, final retention and independent-recovery accounting."""
    table = []
    for label, case in load_study(paths).items():
        count = len(case["config"]["goals"])
        for seed in case["config"]["seeds"]:
            for mode in sorted({r["mode"] for r in case["records"]}):
                rows = sorted(
                    (r for r in case["records"] if r["seed"] == seed and r["mode"] == mode),
                    key=lambda r: r["task_id"],
                )
                complete = [r["task_id"] for r in rows] == list(range(count))
                last = rows[-1] if rows else {}
                old = [
                    r["success_rate"]
                    for r in last.get("evaluation", [])
                    if r["task_id"] < count - 1
                ]
                table.append(
                    dict(
                        case=label,
                        seed=seed,
                        mode=mode,
                        complete=complete,
                        tasks=len(rows),
                        steps=sum(r["steps"] for r in rows),
                        solved=sum(r["solved"] for r in rows),
                        final_old_success=float(np.mean(old))
                        if old and mode == "sequential"
                        else None,
                    )
                )
    return table


def _complete_sequences(case, mode):
    count = len(case["config"]["goals"])
    sequences = []
    for seed in case["config"]["seeds"]:
        rows = sorted(
            (r for r in case["records"] if r["seed"] == seed and r["mode"] == mode),
            key=lambda r: r["task_id"],
        )
        if [r["task_id"] for r in rows] == list(range(count)):
            sequences.append(rows)
    return sequences


def _band(ax, values, color, label):
    values = np.asarray(values, dtype=float)
    x = np.arange(values.shape[1])
    mean = values.mean(0)
    ax.plot(x, mean, color=color, marker="o", label=f"{label} (n={len(values)})")
    if len(values) > 1:
        deviation = values.std(0, ddof=1)
        ax.fill_between(x, mean - deviation, mean + deviation, color=color, alpha=0.12)


@styled
def plot_study(paths, destination, *, style=DEFAULT_STYLE):
    """Acquisition, retention, recovery and sampled update cost with explicit n."""
    cases = load_study(paths)
    fig, axes = subplots(2, 2, height=7.5, style=style)
    incomplete = 0
    for index, (label, case) in enumerate(cases.items()):
        color = COLORS[index % len(COLORS)]
        for mode in ("scratch", "sequential"):
            sequences = _complete_sequences(case, mode)
            if not sequences:
                continue
            incomplete += len(case["config"]["seeds"]) - len(sequences)
            costs = [[r["steps"] for r in rows] for rows in sequences]
            _band(
                axes[0, 0],
                np.cumsum(costs, axis=1),
                color,
                label + (" scratch" if mode == "scratch" and label != "scratch" else ""),
            )
            if mode == "sequential":
                retention = [
                    [
                        np.mean(
                            [
                                v["success_rate"]
                                for v in r["evaluation"]
                                if v["task_id"] <= r["task_id"]
                            ]
                        )
                        for r in rows
                    ]
                    for rows in sequences
                ]
                _band(axes[0, 1], retention, color, label)
        recovery = _complete_sequences(case, "recovery")
        if recovery:
            values = [sum(r["steps"] for r in rows) for rows in recovery]
            axes[1, 0].scatter([index] * len(values), values, color=color, alpha=0.6, s=20)
            axes[1, 0].scatter(index, np.mean(values), color=color, marker="_", s=220)
            failed = sum(not r["solved"] for rows in recovery for r in rows)
            axes[1, 0].annotate(
                f"{failed} failed",
                (index, max(values)),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        timings = [
            v["update_seconds"] * 1000
            for r in case["records"]
            if r["mode"] == "sequential"
            for v in r.get("replay_diagnostics", [])
        ]
        if timings:
            q = np.quantile(timings, [0.25, 0.5, 0.75])
            axes[1, 1].errorbar(
                index, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], color=color, fmt="o", capsize=3
            )
    axes[0, 0].set(
        title="(a) Acquisition cost",
        xlabel="Introduced task ID",
        ylabel="Cumulative environment steps",
    )
    axes[0, 1].set(
        title="(b) Retention across\nintroduced tasks",
        xlabel="Introduced task ID",
        ylabel="Mean success",
        ylim=(-0.03, 1.05),
    )
    axes[1, 0].set(title="(c) Independent recovery trials", ylabel="Summed environment steps")
    axes[1, 1].set(title="(d) Refresh-update host timing", ylabel="Milliseconds · median and IQR")
    for ax in axes[0]:
        numeric_axis(ax)
        ax.legend(loc="best", fontsize=6.5, frameon=True, facecolor="white", framealpha=0.8)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xticks(range(len(cases)), [name.replace("-", "\n", 1) for name in cases], fontsize=7)
        ax.set_ylim(bottom=0)
        numeric_axis(ax)
    finish(
        fig,
        title="Bounded task replay · acquisition, retention and recovery",
        note=f"Bands: ±1 sample SD across complete seed sequences; no band for n=1. {incomplete} incomplete acquisition seed/modes omitted from curves; inspect the table. Failed runs retain spent budgets. Recovery branches are independent. Timings sample refresh updates and include selection/weighting; GPU execution is not synchronised.",
    )
    save_figure(fig, Path(destination) / "k-replay-study", style=style)
    return fig


@styled
def plot_replay_allocation(directory, *, seed=None, style=DEFAULT_STYLE):
    """Task-weight timeline, measured budgets and candidate-span coverage."""
    directory = Path(directory)
    records = json.loads((directory / "summary.json").read_text())
    config = json.loads((directory / "manifest.json").read_text())["config"]
    seed = config["seeds"][0] if seed is None else seed
    rows = sorted(
        (r for r in records if r["mode"] == "sequential" and r["seed"] == seed),
        key=lambda r: r["task_id"],
    )
    snapshots, boundaries, offset = [], [], 0
    for row in rows:
        for item in row.get("replay_diagnostics", []):
            snapshots.append(dict(item, sequence_step=offset + item["step"]))
        offset += row["steps"]
        boundaries.append(offset)
    fig, axes = subplots(2, 2, height=6.5, style=style)
    if not snapshots:
        for ax in axes.flat:
            empty_panel(ax, "No saved bounded-replay diagnostics")
    else:
        ids = sorted({i for s in snapshots for i in s["selected_tasks"]})
        if ids:
            weights = np.zeros((len(ids), len(snapshots)))
            lookup = {task: i for i, task in enumerate(ids)}
            for j, s in enumerate(snapshots):
                for task, weight in zip(s["selected_tasks"], s["replay_weights"]):
                    weights[lookup[task], j] = weight
            im = axes[0, 0].imshow(
                weights, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis"
            )
            axes[0, 0].set_yticks(range(len(ids)), ids)
            fig.colorbar(im, ax=axes[0, 0], label="Replay weight", shrink=0.75)
        else:
            empty_panel(axes[0, 0], "No historical objectives (K=0 or first task)")
        axes[0, 0].set(
            title="(a) Selected task weights",
            xlabel="Refresh snapshot",
            ylabel="Historical task ID",
        )
        x = [s["sequence_step"] for s in snapshots]
        axes[0, 1].plot(
            x, [len(s["selected_tasks"]) for s in snapshots], label="Historical objectives"
        )
        axes[0, 1].plot(
            x,
            [s["regularised_goals"] for s in snapshots],
            label="Regularised goals",
            linestyle="--",
        )
        k = config["training"]["k_replay"]["max_tasks"]
        axes[0, 1].axhline(k, color="#777777", linestyle=":", label=f"Replay cap K={k}")
        axes[0, 1].set(
            title="(b) Objective counts", xlabel="Sequence environment steps", ylabel="Count"
        )
        axes[1, 0].plot(x, [s.get("max_residual", 0) for s in snapshots], color=COLORS[2])
        axes[1, 0].plot(
            x,
            [s.get("mean_value_drift", 0) for s in snapshots],
            color=COLORS[3],
            linestyle="--",
            label="Common-probe Q drift",
        )
        axes[1, 0].lines[0].set_label("Embedding residual")
        axes[1, 0].set(
            title="(c) Candidate coverage and value drift",
            xlabel="Sequence environment steps",
            ylabel="Residual / mean absolute Q change",
        )
        axes[1, 1].plot(
            x,
            [s["replay_samples"] for s in snapshots],
            label="Processed historical samples",
            color=COLORS[1],
        )
        axes[1, 1].axhline(
            config["training"]["k_replay"]["sample_budget"],
            color="#777777",
            linestyle=":",
            label="Sample budget R",
        )
        axes[1, 1].set(
            title="(d) Replay sample budget",
            xlabel="Sequence environment steps",
            ylabel="Transition-task pairs",
        )
        for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
            for boundary in boundaries[:-1]:
                ax.axvline(boundary, linewidth=0.6, alpha=0.25, color="#555555")
            numeric_axis(ax, x=True)
            legend(ax)
    finish(
        fig,
        title=f"Replay allocation and constraints · seed {seed}",
        note="Snapshots are recorded at selector refreshes. Coverage describes candidate embeddings, not behavioural retention. Zero weight means a task was not selected; task-boundary lines mark cumulative training steps.",
    )
    save_figure(
        fig,
        directory
        / f"figures/replay-allocation{'-' + str(seed) if seed != config['seeds'][0] else ''}",
        style=style,
    )
    return fig


@styled
def plot_soft_ball_norms(directory, *, seed=None, style=DEFAULT_STYLE):
    """Audit soft radii and penalty activation from saved task snapshots/losses."""
    directory = Path(directory)
    case = load_study({"run": directory})["run"]
    config = case["config"]
    seed = config["seeds"][0] if seed is None else seed
    rows = sorted(
        (r for r in case["records"] if r["mode"] == "sequential" and r["seed"] == seed),
        key=lambda r: r["task_id"],
    )
    snapshots, losses, boundaries, offset = [], [], [], 0
    for row in rows:
        snapshots.extend(
            dict(s, sequence_step=offset + s["step"])
            for s in row.get("replay_diagnostics", [])
            if s.get("norm_constraint") == "soft"
        )
        path = directory / f"seed_{seed}/sequential/task_{row['task_id']}/losses.jsonl"
        if path.exists():
            losses.extend(
                dict(s, sequence_step=offset + s["step"])
                for s in map(json.loads, path.read_text().splitlines())
            )
        offset += row["steps"]
        boundaries.append(offset)
    fig, axes = subplots(2, 2, height=6.5, style=style)
    if not snapshots:
        for ax in axes.flat:
            empty_panel(ax, "No saved soft-ball diagnostics")
    else:
        x = [s["sequence_step"] for s in snapshots]
        for branch, ax, letter in (("phi", axes[0, 0], "a"), ("psi", axes[0, 1], "b")):
            ax.plot(x, [s[f"{branch}_norm_mean"] for s in snapshots], label="Mean norm")
            ax.plot(
                x,
                [s[f"{branch}_norm_max"] for s in snapshots],
                label="Maximum norm",
                color=COLORS[1],
            )
            ax.axhline(
                config["model"][f"{branch}_max_norm"],
                color="#555555",
                linestyle=":",
                label="Soft radius",
            )
            ax.set(
                title=f"({letter}) {'State-action' if branch == 'phi' else 'Task'} factor norms",
                ylabel="Euclidean norm",
            )
        for branch, color in (("phi", COLORS[0]), ("psi", COLORS[1])):
            axes[1, 0].plot(
                x,
                [s[f"{branch}_radius_excess_fraction"] for s in snapshots],
                label=branch,
                color=color,
            )
        axes[1, 0].set(
            title="(c) Samples outside soft radii", ylabel="Exceedance fraction", ylim=(-0.03, 1.05)
        )
        lx = [s["sequence_step"] for s in losses]
        for key, label, color, coefficient in (
            ("td", "Current TD", COLORS[2], 1),
            (
                "phi_norm",
                "State-action norm penalty",
                COLORS[0],
                config["training"]["phi_raw_norm_coef"],
            ),
            ("psi_norm", "Task norm penalty", COLORS[1], config["training"]["psi_raw_norm_coef"]),
        ):
            axes[1, 1].plot(lx, [s[key] * coefficient for s in losses], label=label, color=color)
        axes[1, 1].set(title="(d) Weighted sampled losses", ylabel="Loss contribution")
        axes[1, 1].set_yscale("symlog", linthresh=1e-6)
        axes[1, 1].set_ylim(bottom=0)
        for ax in axes.flat:
            ax.set_xlabel("Sequence environment steps")
            for boundary in boundaries[:-1]:
                ax.axvline(boundary, color="#777777", alpha=0.2, linewidth=0.6)
            numeric_axis(ax, x=True)
            legend(ax)
    finish(
        fig,
        title=f"Soft-ball representation diagnostics · seed {seed}",
        note="Radii are soft preferences, not hard bounds. Norms cover current/replayed state-actions and selected goals. Snapshots precede each recorded update; weighted loss sizes do not measure gradient influence.",
    )
    save_figure(
        fig,
        directory
        / f"figures/soft-ball-norms{'-' + str(seed) if seed != config['seeds'][0] else ''}",
        style=style,
    )
    return fig
