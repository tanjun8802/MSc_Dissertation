"""Baseline-first Fetch experiments; controls are explicit and results are never overwritten."""

import hashlib
import json
import platform
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .config import training_settings
from .figure_style import DEFAULT_STYLE, save_figure, styled, subplots
from .runner import run_experiment

EXPLORATION = ("auto", "bounded", "fixed")
ARCHITECTURES = ("plain_sac", "factorised_critic", "factorised_actor_critic", "tbtrl")
RETENTION = (
    "control",
    "critic_replay",
    "policy_anchor",
    "capacity_archive",
    "gentle_transition",
    "combined",
)


def _merge_settings(base, overrides):
    """Recursively apply explicit JSON case overrides; an empty dict clears a field."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and value and isinstance(merged.get(key), dict):
            merged[key] = _merge_settings(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def configure_fetch_study(
    base, *, exploration="bounded", architecture="plain_sac", retention="control"
):
    """Resolve base JSON followed by its selected exploration/architecture/retention cases.

    No numerical study presets live in Python. Per-phase settings remain intact
    and are applied by training_settings after these explicit case overrides.
    """
    if (
        exploration not in EXPLORATION
        or architecture not in ARCHITECTURES
        or retention not in RETENTION
    ):
        raise ValueError("Unknown Fetch study case.")
    if not base.environment.startswith("fetch-") or base.algorithm != "sac":
        raise ValueError("Use a Fetch SAC config.")
    if retention != "control" and architecture != "tbtrl":
        raise ValueError("Retention interventions require architecture='tbtrl'.")
    values = base.to_dict()
    for section, case in (
        ("exploration", exploration),
        ("architecture", architecture),
        ("retention", retention),
    ):
        try:
            overrides = base.study_overrides[section][case]
        except KeyError as exc:
            raise ValueError(
                f"Define study_overrides.{section}.{case} in the experiment JSON."
            ) from exc
        values = _merge_settings(values, overrides)
    values["name"] = f"{base.name}-study-{exploration}-{architecture}-{retention}"
    return type(base)(**values).validate()


def _fingerprint(config, modes, device):
    source = Path(__file__).resolve().parents[1]
    runtime = {
        name: version(name)
        for name in ("torch", "numpy", "gymnasium", "gymnasium-robotics", "mujoco")
    }
    digest = hashlib.sha256(
        json.dumps(
            [config.to_dict(), modes, str(device), platform.python_version(), runtime],
            sort_keys=True,
        ).encode()
    )
    for file in sorted(source.rglob("*.py")):
        digest.update(str(file.relative_to(source)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def run_study_job(config, directory, *, modes=("scratch",), device="cpu", on_task_end=None):
    """Reuse completed tasks and append a validated goal suffix from full state.

    An interrupted task restarts from its preceding committed task boundary.
    """
    directory = Path(directory)
    print(f"Study job: {directory}", flush=True)
    for mode in modes:
        for task_id in range(min(len(config.goals), 2)):
            print(
                f"Resolved {mode} task={task_id}: "
                + json.dumps(training_settings(config, mode, task_id), sort_keys=True),
                flush=True,
            )
    print(
        f"Held-out evaluation episodes: {config.post_eval_episodes or config.eval_episodes}",
        flush=True,
    )
    fingerprint = _fingerprint(config, modes, device)
    receipt = directory / "study-complete.json"
    if directory.exists() and not (directory / "continuation.json").exists():
        if not receipt.exists():
            raise RuntimeError(f"Incomplete legacy job at {directory}; use a new output directory.")
        saved = json.loads(receipt.read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError(
                f"Source/config changed for {directory}; choose a new study directory."
            )
        print(f"Reusing completed legacy job: {directory}", flush=True)
        return json.loads((directory / "summary.json").read_text())
    # Histories belong to this job, never to a previous entropy/architecture case.
    from .diagnostics import NotebookDiagnostics

    if isinstance(on_task_end, NotebookDiagnostics):
        on_task_end = NotebookDiagnostics(
            display=on_task_end.display,
            rollout_episodes=on_task_end.rollout_episodes,
            gif_episodes=on_task_end.gif_episodes,
            style=on_task_end.style,
        )
    run_experiment(
        config, directory, device=device, modes=modes, on_task_end=on_task_end, resume=True
    )
    receipt.write_text(json.dumps(dict(fingerprint=fingerprint), indent=2))
    return json.loads((directory / "summary.json").read_text())


def run_baselines(
    base,
    directory,
    *,
    explorations=("auto", "bounded", "fixed"),
    architectures=("plain_sac",),
    seeds=(42,),
    smoke=False,
    device="cpu",
    on_task_end=None,
):
    """One independently seeded goal per job; completed jobs survive notebook restarts."""
    if not explorations or not architectures or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply cases and distinct training seeds.")
    records = []
    for exploration in explorations:
        for architecture in architectures:
            for seed in seeds:
                for goal_id, goal in enumerate(base.goals[:2] if smoke else base.goals):
                    c = configure_fetch_study(
                        base, exploration=exploration, architecture=architecture
                    )
                    # Match the runner's per-goal acquisition seed, independently of previous jobs.
                    c.goals, c.seeds = [goal], [seed + base.task_seed_stride * goal_id]
                    if smoke:
                        c = c.smoke()
                    job = (
                        Path(directory)
                        / f"{exploration}-{architecture}"
                        / f"seed_{seed}"
                        / f"goal_{goal_id}"
                    )
                    rows = run_study_job(c, job, device=device, on_task_end=on_task_end)
                    record = dict(
                        rows[0],
                        study_seed=seed,
                        study_goal_id=goal_id,
                        exploration=exploration,
                        architecture=architecture,
                        smoke=smoke,
                        post_eval_episodes=c.post_eval_episodes,
                        directory=str(job),
                    )
                    records.append(record)
                    Path(directory, "baseline-results.json").write_text(
                        json.dumps(records, indent=2)
                    )
    return records


def require_baseline(
    records, *, exploration, architecture, goal_count, seeds=(42, 43, 44), threshold=0.9
):
    """Fail closed on missing, smoke, failed-training or weak held-out baseline results."""
    rows = [
        r for r in records if r["exploration"] == exploration and r["architecture"] == architecture
    ]
    required = {(s, g) for s in seeds for g in range(goal_count)}
    actual = {(r["study_seed"], r["study_goal_id"]) for r in rows}
    if len(set(seeds)) < 3 or not required.issubset(actual):
        raise ValueError(
            "Confirm every goal on at least three training seeds before interpreting transfer."
        )
    selected = [r for r in rows if (r["study_seed"], r["study_goal_id"]) in required]
    if any(
        r["smoke"]
        or r["post_eval_episodes"] < 100
        or not r["solved"]
        or r["evaluation"][0]["hold_success_rate"] < threshold
        for r in selected
    ):
        raise ValueError(
            "Baseline gate failed: require solved training and >=90% held-out hold success on 100 episodes for each goal/seed."
        )
    return True


@styled
def plot_baselines(records, directory, *, style=DEFAULT_STYLE):
    """Individual-seed evidence: no confidence bands inferred from one training seed."""
    from matplotlib.ticker import PercentFormatter

    saved = Path(directory) / "baseline-results.json"
    if saved.exists():
        records = json.loads(saved.read_text())
    labels = list(dict.fromkeys(f"{r['exploration']} / {r['architecture']}" for r in records))
    fig, axes = subplots(1, 3, style=replace(style, width=13), height=3.5)
    for i, label in enumerate(labels):
        rows = [r for r in records if f"{r['exploration']} / {r['architecture']}" == label]
        for j, metric in enumerate(("ever_success_rate", "hold_success_rate")):
            values = [r["evaluation"][0][metric] for r in rows]
            x = i + np.linspace(-0.12, 0.12, len(values))
            axes[j].scatter(x, values, s=22)
            axes[j].plot([i - 0.18, i + 0.18], [np.mean(values)] * 2, color="black", lw=2)
        for k, row in enumerate(rows):
            axes[2].scatter(
                i + np.linspace(-0.12, 0.12, len(rows))[k],
                row["steps"],
                marker="o" if row["solved"] else "x",
                color="C0",
            )
    for ax, title in zip(
        axes, ("Reached at any time", "Held at episode end", "Training cost (× = unsolved)")
    ):
        ax.set(title=title, xticks=range(len(labels)), xticklabels=labels)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.15, axis="y")
    for ax in axes[:2]:
        ax.set(ylim=(-0.03, 1.03), ylabel="Held-out episode success")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_ylabel("Environment steps")
    fig.suptitle("Fetch scratch controls · each point is one goal × training seed")
    save_figure(fig, Path(directory) / "baseline-comparison", style=style)
    return fig
