"""Saved-result analysis for matched-budget acquisition and recovery probes."""

import csv
import json
from pathlib import Path

import numpy as np

from tbtrl.continuation import atomic_json

from .dqn_study import load_study
from .figure_style import COLORS, DEFAULT_STYLE, finish, legend, panel_grid, save_figure, styled


def followup_table(paths, destination=None):
    """One row per seed and protocol; missing trials stay visible as incomplete."""
    table = []
    for name, case in load_study(paths).items():
        manifest = json.loads((case["directory"] / "manifest.json").read_text())
        config = case["config"]
        count = len(config["goals"])
        for seed in config["seeds"]:
            for mode in manifest["modes"]:
                rows = sorted(
                    (r for r in case["records"] if r["seed"] == seed and r["mode"] == mode),
                    key=lambda r: r["task_id"],
                )
                complete = [r["task_id"] for r in rows] == list(range(count))
                options = config["training"]["k_replay"]
                row = dict(
                    case=name,
                    seed=seed,
                    mode=mode,
                    protocol=manifest.get("recovery_protocol", "acquisition"),
                    k=options["max_tasks"],
                    selector=options["selector"],
                    complete=complete,
                    tasks=len(rows),
                    threshold_met=sum(r["solved"] for r in rows),
                    failures=sum(not r["solved"] for r in rows),
                    steps=sum(r["steps"] for r in rows),
                    hours=sum(r["elapsed_seconds"] for r in rows) / 3600,
                )
                if rows and mode != "recovery":
                    row["threshold_cost"] = float(
                        np.mean(
                            [
                                r["steps_to_threshold"]
                                if r["steps_to_threshold"] is not None
                                else r["steps"]
                                for r in rows
                            ]
                        )
                    )
                    if mode == "sequential" and complete and count > 1:
                        row["old_success"] = float(
                            np.mean(
                                [
                                    e["success_rate"]
                                    for e in rows[-1]["evaluation"]
                                    if e["task_id"] < count - 1
                                ]
                            )
                        )
                if rows and mode == "recovery":
                    active = [r for r in rows if r["recovery_performed"]]
                    row["retained_fraction"] = (len(rows) - len(active)) / len(rows)
                    row["recovery_cost"] = row["steps"] / len(rows)
                    row["adaptation_cost"] = (
                        float(np.mean([r["steps"] for r in active])) if active else None
                    )
                    deltas = []
                    for r in active:
                        before = {e["task_id"]: e["success_rate"] for e in r["initial_evaluation"]}
                        changes = [
                            e["success_rate"] - before[e["task_id"]]
                            for e in r["evaluation"]
                            if e["task_id"] != r["task_id"]
                        ]
                        if changes:
                            deltas.append(float(np.mean(changes)))
                    row["other_task_delta"] = float(np.mean(deltas)) if deltas else None
                table.append(row)
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "outcomes.json", table)
        keys = sorted(set().union(*(r.keys() for r in table)))
        with (destination / "outcomes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(table)
    return table


def _series(ax, rows, field, *, color, label, linestyle="-", scale=1):
    points = {}
    for row in rows:
        if row["complete"] and row.get(field) is not None:
            points.setdefault(row["k"], []).append(row[field] * scale)
    if not points:
        return
    xs = sorted(points)
    means = [np.mean(points[k]) for k in xs]
    deviations = [np.std(points[k], ddof=1) if len(points[k]) > 1 else 0 for k in xs]
    ax.errorbar(
        xs,
        means,
        yerr=deviations,
        color=color,
        marker="o",
        linestyle=linestyle,
        capsize=3,
        label=label,
    )
    ax.set_xticks(xs)


@styled
def plot_followup_acquisition(paths, destination, *, style=DEFAULT_STYLE):
    """K sensitivity with fixed training budgets and threshold failures retained."""
    table = followup_table(paths, destination)
    fig, axes = panel_grid(4, style=style, panel_height=3.8)
    fields = ("old_success", "threshold_cost", "threshold_fraction", "hours")
    titles = (
        "Final old-task retention",
        "Acquisition threshold cost",
        "Acquisition threshold reached",
        "Recorded training time",
    )
    labels = ("Success (%)", "Interactions (failure = full budget)", "Tasks (%)", "Hours / seed")
    for row in table:
        row["threshold_fraction"] = row["threshold_met"] / max(1, row["tasks"])
    for ax, field, title, ylabel in zip(axes, fields, titles, labels):
        scale = 100 if field in ("old_success", "threshold_fraction") else 1
        for selector, color in zip(("random", "basis"), COLORS):
            selected = [r for r in table if r["selector"] == selector and r["k"] > 0]
            _series(ax, selected, field, color=color, label=selector.title(), scale=scale)
        for name, color in (("fine-tuning", COLORS[2]), ("scratch", COLORS[4])):
            values = [
                r[field] * scale
                for r in table
                if r["case"] == name and r["complete"] and r.get(field) is not None
            ]
            if values:
                ax.axhline(np.mean(values), color=color, linestyle="--", label=name)
        ax.set(title=title, xlabel="Historical objective cap K", ylabel=ylabel)
        ax.grid(alpha=0.18)
        legend(ax)
    finish(
        fig,
        title="Fixed-budget Maze DQN · basis versus random replay",
        note="Error bars: seed SD; only complete sequences plotted. Threshold cost includes failures. "
        "Full per-task training continues after the first threshold crossing.",
    )
    save_figure(fig, Path(destination) / "fixed-budget-acquisition", style=style)
    return fig


@styled
def plot_followup_recovery(paths, destination, *, style=DEFAULT_STYLE):
    """Separate recovery cost, zero-update retention, and collateral behaviour change."""
    table = followup_table(paths, destination)
    fig, axes = panel_grid(6, style=style, panel_height=3.8)
    fields = (
        "retained_fraction",
        "recovery_cost",
        "adaptation_cost",
        "other_task_delta",
        "failure_fraction",
        "hours",
    )
    titles = (
        "Already retained before updates",
        "Recovery cost including retained tasks",
        "Cost for tasks requiring adaptation",
        "Other-task change after adaptation",
        "Recovery threshold not reached",
        "Recorded recovery time",
    )
    labels = (
        "Tasks (%)",
        "Interactions / task",
        "Interactions / adapted task",
        "Success change (percentage points)",
        "Tasks (%)",
        "Hours / seed",
    )
    for row in table:
        row["failure_fraction"] = row["failures"] / max(1, row["tasks"])
    for ax, field, title, ylabel in zip(axes, fields, titles, labels):
        scale = 100 if field in ("retained_fraction", "other_task_delta", "failure_fraction") else 1
        for selector, color in zip(("random", "basis"), COLORS):
            for protocol, linestyle in (("no-replay", "-"), ("method-replay", "--")):
                selected = [
                    r
                    for r in table
                    if r["selector"] == selector and r["k"] > 0 and r["protocol"] == protocol
                ]
                _series(
                    ax,
                    selected,
                    field,
                    color=color,
                    label=f"{selector} · {protocol}",
                    linestyle=linestyle,
                    scale=scale,
                )
        values = [
            r[field] * scale
            for r in table
            if r["k"] == 0
            and r["complete"]
            and r["protocol"] == "no-replay"
            and r.get(field) is not None
        ]
        if values:
            ax.axhline(np.mean(values), color=COLORS[2], linestyle=":", label="fine-tuning")
        if field == "other_task_delta":
            ax.axhline(0, color="gray", linewidth=0.7)
        ax.set(title=title, xlabel="Acquisition historical objective cap K", ylabel=ylabel)
        ax.grid(alpha=0.18)
        legend(ax)
    finish(
        fig,
        title="Independent recovery · common adaptation versus method replay",
        note="Each trial starts from its condition's same final checkpoint. Error bars: seed SD. "
        "Failures retain full cost; conditional panels exclude zero-step skips. "
        "Negative other-task change means forgetting during recovery.",
    )
    save_figure(fig, Path(destination) / "recovery-protocols", style=style)
    return fig
