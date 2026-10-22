"""Baseline-first Fetch experiments; controls are explicit and results are never overwritten."""

import hashlib
import json
import platform
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np

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


def configure_fetch_study(
    base, *, exploration="bounded", architecture="plain_sac", retention="control"
):
    """New protocol, independent of historical notebook configs.

    Automatic/bounded/fixed temperature cases share initial alpha and all other
    settings. Bounded adds a floor, ceiling and delayed adaptation as one named
    intervention. Actor controls share the environment, data budget and rates;
    network parameter counts are not matched across architectures.
    """
    if (
        exploration not in EXPLORATION
        or architecture not in ARCHITECTURES
        or retention not in RETENTION
    ):
        raise ValueError("Unknown Fetch study case.")
    if not base.environment.startswith("fetch-") or base.algorithm != "sac":
        raise ValueError("Use a Fetch SAC config.")
    c = deepcopy(base)
    c.name = f"{base.name}-study-{exploration}-{architecture}-{retention}"
    c.entropy_transfer = "reset" if exploration == "fixed" else "carry"
    c.eval_episodes, c.post_eval_episodes = 32, 100
    c.scratch, c.first_task, c.recovery = {}, {}, {"total_steps": 100000}
    c.carry_optimizer_state = False
    c.independent_model_initialization = True
    c.recovery_threshold = 0.9
    c.training.update(
        total_steps=500000,
        warmup_steps=10000,
        batch_size=256,
        eval_freq=5000,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_ent_coef=1e-4,
        initial_alpha=0.03,
        ent_coef=0.03 if exploration == "fixed" else "auto",
        min_alpha=0.01 if exploration == "bounded" else None,
        max_alpha=0.2 if exploration == "bounded" else None,
        entropy_tuning_start=50000 if exploration == "bounded" else 0,
        target_entropy=-4.0,
        warmup_action="random",
        critic_loss="huber",
        max_abs_q=1000.0,
        relabel_rewards=True,
        normalize_state_inputs=False,
        normalize_goal_inputs=False,
        q_target_min=None,
        q_target_max=None,
        eval_success_metric="hold",
        eval_hold_steps=10,
        early_stop_success_rate=0.9,
        early_stop_patience=3,
        enable_early_stop=True,
        sigreg_coef=0.0,
        goal_separation_coef=0.0,
        phi_raw_norm_coef=0.0,
        psi_raw_norm_coef=0.0,
        actor_tbtrl=None,
        replay_ratio=0.5,
        replay_loss_coef=0.05,
        transfer_probe_steps=[0, 9999, 10000, 10100, 11000, 15000],
        actor_gradient_diagnostics=True,
    )
    c.critic_type = "mlp" if architecture == "plain_sac" else "factorised"
    c.actor_type = "factorised" if architecture in ("factorised_actor_critic", "tbtrl") else "mlp"
    if c.actor_type == "mlp":
        c.actor_model = {}
    c.replay_mode = "per_task" if c.critic_type == "mlp" else "global"
    c.similarity_threshold = None if c.critic_type == "mlp" else 0.8
    c.archive_compression = "similarity"
    if architecture == "tbtrl":
        c.training.update(
            sigreg_coef=1e-5,
            goal_separation_coef=1e-5,
            phi_raw_norm_coef=0.003,
            psi_raw_norm_coef=0.003,
            phi_norm_target=8.0,
            psi_norm_target=8.0,
        )
        rank = c.actor_model.get("latent_dim", 16)
        c.training["actor_tbtrl"] = dict(
            replay_loss_coef=1.0,
            retention_kl_coef=0.0,
            sigreg_coef=1e-5,
            covariance_target_variance=2.0 / rank,
            sketch_dim=rank,
            goal_separation_coef=1e-5,
            goal_separation_target_cosine=0.85,
            phi_norm_coef=0.003,
            psi_norm_coef=0.003,
            phi_norm_target=2.0,
            psi_norm_target=2.0,
        )
    if retention != "control" and architecture != "tbtrl":
        raise ValueError("Retention interventions require architecture='tbtrl'.")
    if retention in ("critic_replay", "combined"):
        c.training["replay_loss_coef"] = 1.0
    if retention in ("policy_anchor", "combined"):
        c.training["actor_tbtrl"]["retention_kl_coef"] = 0.01
    if retention in ("capacity_archive", "combined"):
        c.archive_compression = "capacity"
    if retention in ("gentle_transition", "combined"):
        c.carry_optimizer_state = True
        c.training.update(
            warmup_action="policy", lr_actor=1e-4, lr_critic=1e-4, reset_target_from_critic=False
        )
    # Scratch and the first sequential task use the same acquisition settings.
    acquisition = dict(
        warmup_action="random", lr_actor=3e-4, lr_critic=3e-4, reset_target_from_critic=True
    )
    c.scratch.update(acquisition)
    c.first_task.update(acquisition)
    return c.validate()


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
    """Reuse a completed job only if source and config match. Never resume partial training.

    Interrupted jobs require a new output directory; no replay/optimiser/RNG state
    is reconstructed from evaluation-only checkpoints.
    """
    directory = Path(directory)
    fingerprint = _fingerprint(config, modes, device)
    receipt = directory / "study-complete.json"
    if directory.exists():
        if not receipt.exists():
            raise RuntimeError(
                f"Incomplete job at {directory}. Use a new study directory; this job cannot resume mid-training."
            )
        saved = json.loads(receipt.read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError(
                f"Source/config changed for {directory}; choose a new study directory."
            )
        print(f"Reusing completed job: {directory}", flush=True)
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
    run_experiment(config, directory, device=device, modes=modes, on_task_end=on_task_end)
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
        axes, ("Reached at any time", "Held for final 10 steps", "Training cost (× = unsolved)")
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
