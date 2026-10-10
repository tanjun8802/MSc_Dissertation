"""Rendering only: reusable task diagnostics, independent of TBTRL updates."""

import numpy as np

from .figure_style import (
    BRANCHES,
    COLORS,
    annotate_matrix,
    empty_panel,
    finish,
    legend,
    numeric_axis,
    subplots,
)

TERMS = {
    "current": "Current-goal objective",
    "replay": "Old-goal replay",
    "retention_kl": "Policy retention (KL)",
    "sigreg": "State covariance",
    "goal_separation": "Goal separation",
    "phi_norm": "State-factor norm",
    "psi_norm": "Goal-factor norm",
}
TERM_STYLES = {
    name: dict(color=COLORS[i], linestyle=("-", "--", "-.")[i % 3]) for i, name in enumerate(TERMS)
}


def goal_style(goal_id):
    return dict(
        color=COLORS[goal_id % len(COLORS)],
        marker=("o", "s", "^", "D")[goal_id % 4],
        linestyle=("-", "--", "-.", ":")[(goal_id // len(COLORS)) % 4],
    )


def plot_learning(result, config, record, *, style):
    fig, axes = subplots(2, 2, style=style, height=6.4)
    ax = axes[0, 0]
    if result.evaluations:
        ax.plot(*zip(*result.evaluations), marker="o", color=COLORS[0])
    else:
        empty_panel(ax, "No evaluation checkpoints recorded")
    ax.set(
        title="(a) Current-goal return", xlabel="Environment steps", ylabel="Mean episodic return"
    )
    numeric_axis(ax, x=True)
    ax = axes[0, 1]
    success = getattr(result, "success_rates", [])
    if success:
        ax.plot(*zip(*success), marker="o", color=COLORS[0])
    else:
        empty_panel(ax, "Per-checkpoint success was not recorded\nfor this trainer")
    ax.set(
        title="(b) Current-goal success", xlabel="Environment steps", ylabel="Evaluation success"
    )
    numeric_axis(ax, x=True, percent=True)
    ax = axes[1, 0]
    names = (
        {"td": "Current goal", "replay": "Old-goal replay"}
        if config.algorithm == "dqn"
        else {
            "td1": "Critic 1 · current",
            "td2": "Critic 2 · current",
            "replay1": "Critic 1 · replay",
            "replay2": "Critic 2 · replay",
        }
    )
    for i, (key, label) in enumerate(names.items()):
        ax.plot(
            [r["step"] for r in result.losses],
            [r[key] for r in result.losses],
            label=label,
            color=COLORS[i % 2],
            linestyle="--" if "replay" in key else "-",
        )
    ax.set(
        title="(c) Critic TD losses",
        xlabel="Environment steps",
        ylabel="Mean squared TD error",
    )
    numeric_axis(ax, x=True)
    legend(ax)
    ax = axes[1, 1]
    rows = sorted(record["evaluation"], key=lambda row: row["task_id"])
    for row in rows:
        task = row["task_id"]
        ax.bar(
            task,
            row["success_rate"],
            color=goal_style(task)["color"],
            width=0.65,
            edgecolor="#222222" if task == record["task_id"] else "white",
            linewidth=1.2,
            hatch="//" if task > record["task_id"] and record["mode"] == "sequential" else None,
        )
        ax.text(
            task,
            row["success_rate"] + 0.025,
            f"{row['success_rate']:.1%}".replace(".0%", "%"),
            ha="center",
            fontsize=7,
        )
    ax.set(
        title="(d) All-goal success",
        xlabel="Evaluated goal ID",
        ylabel="Evaluation success",
        xticks=[r["task_id"] for r in rows],
    )
    numeric_axis(ax, percent=True)
    finish(
        fig,
        note=f"Empirical means over {config.eval_episodes} evaluation episode(s); no confidence interval shown. TD losses are unweighted. Outlined bar: current goal; hatched bars: future goals in a sequential run.",
    )
    return fig


def plot_critic_objective(result, config, *, style):
    fig, ax = subplots(style=style, height=3.4)
    names = (
        {"total": "DQN critic"}
        if config.algorithm == "dqn"
        else {"total1": "SAC critic 1", "total2": "SAC critic 2"}
    )
    for i, (key, label) in enumerate(names.items()):
        ax.plot(
            [r["step"] for r in result.losses],
            [r[key] for r in result.losses],
            label=label,
            color=COLORS[i],
            linestyle=("-", "--")[i],
        )
    ax.set(title="Total critic objective", xlabel="Environment steps", ylabel="Weighted loss")
    numeric_axis(ax, x=True)
    legend(ax)
    finish(
        fig,
        note="Total logged objective, including configured current-task, replay and regularisation coefficients. Unweighted TD losses are shown separately in the learning figure.",
    )
    return fig


def plot_actor_losses(result, config, *, style):
    options = config.training.get("actor_tbtrl") or {}
    fig, axes = subplots(1, 2, style=style, height=3.7)
    steps = [r["step"] for r in result.losses]
    for key, coefficient, label, color, ls in (
        ("actor_current", 1, "Current goal", COLORS[0], "-"),
        ("actor_replay", options.get("replay_loss_coef", 0), "Weighted replay", COLORS[1], "--"),
        ("actor", 1, "Total actor loss", "#222222", "-."),
    ):
        if coefficient and all(key in r for r in result.losses):
            axes[0].plot(
                steps,
                [r[key] * coefficient for r in result.losses],
                label=label,
                color=color,
                linestyle=ls,
            )
    active = 0
    for name in ("sigreg", "goal_separation", "phi_norm", "psi_norm", "retention_kl"):
        coefficient = options.get(f"{name}_coef", 0)
        if coefficient and all(f"actor_{name}" in r for r in result.losses):
            active += 1
            axes[1].plot(
                steps,
                [r[f"actor_{name}"] * coefficient for r in result.losses],
                label=TERMS[name],
                **TERM_STYLES[name],
            )
    axes[0].set(title="(a) Actor objective", ylabel="Loss contribution")
    axes[1].set(
        title="(b) Weighted actor penalties", ylabel="Loss contribution (symlog)", yscale="symlog"
    )
    axes[1].set_yscale("symlog", linthresh=1e-7)
    if not active:
        empty_panel(axes[1], "No active actor penalties recorded")
    for ax in axes:
        ax.set_xlabel("Environment steps")
        numeric_axis(ax, x=True)
        legend(ax, columns=1)
    finish(
        fig,
        note="Replay and penalties include their configured loss coefficients. Total also includes active penalties. Symlog retains zeros; its linear region is |loss| ≤ 10⁻⁷.",
    )
    return fig


def plot_transfer(result, task_ids, *, style):
    probes = result.transfer_probes
    fig, axes = subplots(3, 1, style=style, height=8.1)
    steps = [p["step"] for p in probes]
    axes[0].semilogy(steps, [p["alpha"] for p in probes], marker="o", color=COLORS[0])
    axes[0].set(
        title="(a) Entropy temperature",
        xlabel="Environment steps",
        ylabel=r"Temperature $\alpha$ (log scale)",
    )
    numeric_axis(axes[0], x=True)
    for task in task_ids:
        values = [
            next(r["success_rate"] for r in p["evaluation"] if r["task_id"] == task) for p in probes
        ]
        axes[1].plot(steps, values, label=f"Goal {task}", **goal_style(task))
        early = [(p["updates"], value) for p, value in zip(probes, values) if p["updates"] <= 5001]
        if early:
            axes[2].plot(*zip(*early), label=f"Goal {task}", **goal_style(task))
    axes[1].set(
        title="(b) Current and previous goals", xlabel="Environment steps", ylabel="Probe success"
    )
    axes[2].set(
        title="(c) Early adaptation (first 5,001 updates)",
        xlabel="Completed optimiser updates",
        ylabel="Probe success",
    )
    for ax in axes[1:]:
        numeric_axis(ax, x=True, percent=True)
        legend(ax, columns=4)
    finish(
        fig,
        note="Diagnostic evaluation seeds differ from the standard evaluation. Probes do not affect stopping. Multiple observations can share update zero during random-action warm-up.",
    )
    return fig


def plot_gradients(rows, *, style):
    fig, axes = subplots(1, 2, style=style, height=4.1)
    steps = [r["step"] for r in rows]
    for name, label in TERMS.items():
        values = [r[f"actor_{name}_grad_norm"] for r in rows]
        if name in ("current", "replay") or any(values):
            axes[0].plot(steps, values, label=label, **TERM_STYLES[name])
    axes[0].set_yscale("symlog", linthresh=1e-6)
    axes[0].set(title="(a) Weighted actor gradients", ylabel=r"Gradient $L_2$ norm (symlog)")
    legend(axes[0], columns=1)
    values = [
        r["actor_current_replay_grad_cosine"]
        if r["actor_current_grad_norm"] > 0 and r["actor_replay_grad_norm"] > 0
        else np.nan
        for r in rows
    ]
    axes[1].plot(steps, values, color=COLORS[0], marker="o")
    axes[1].axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    axes[1].axhspan(-1, 0, color=COLORS[1], alpha=0.06)
    axes[1].set(
        title="(b) Current–replay alignment",
        ylabel="Gradient cosine similarity",
        ylim=(-1.05, 1.05),
        yticks=[-1, -0.5, 0, 0.5, 1],
    )
    if not np.isfinite(values).any():
        empty_panel(axes[1], "Undefined: at least one gradient is zero")
    for ax in axes:
        ax.set_xlabel("Environment steps")
        numeric_axis(ax, x=True)
    finish(
        fig,
        note="Negative cosine indicates conflicting local update directions. Undefined cosines are omitted, not plotted as zero. Symlog linear region: gradient norm ≤ 10⁻⁶.",
    )
    return fig


def plot_embeddings(name, data, task_ids, *, style):
    fig, axes = subplots(2, 2, style=style, height=6.5)
    im = axes[0, 0].imshow(
        data["psi_cosine"], vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest"
    )
    fig.colorbar(im, ax=axes[0, 0], label="Cosine similarity", ticks=[-1, 0, 1], shrink=0.85)
    axes[0, 0].set(
        title="(a) Goal-factor similarity",
        xlabel="Goal ID",
        ylabel="Goal ID",
        xticks=range(len(task_ids)),
        xticklabels=task_ids,
        yticks=range(len(task_ids)),
        yticklabels=task_ids,
    )
    annotate_matrix(axes[0, 0], data["psi_cosine"])
    values = np.concatenate([data["psi"], data["phi_fixed"]])
    centered = values - values.mean(0)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    points = centered @ basis[:2].T
    points = np.pad(points, ((0, 0), (0, max(0, 2 - points.shape[1]))))
    variance = np.pad(singular**2 / max(float((singular**2).sum()), 1e-20), (0, 2))
    count = len(task_ids)
    axes[0, 1].scatter(
        *points[count:].T,
        color="#999999",
        marker=".",
        alpha=0.65,
        label="Fixed state factors" if name.startswith("actor") else "Fixed state–action factors",
    )
    for i, task in enumerate(task_ids):
        axes[0, 1].scatter(
            *points[i],
            color=goal_style(task)["color"],
            marker="*",
            s=70,
            edgecolor="#222222",
            linewidth=0.4,
        )
        axes[0, 1].annotate(
            f"G{task}", points[i], xytext=(4, 4), textcoords="offset points", fontsize=7
        )
    axes[0, 1].set(
        title="(b) Joint factor projection",
        xlabel=f"PC 1 ({variance[0]:.1%} variance)",
        ylabel=f"PC 2 ({variance[1]:.1%} variance)",
    )
    numeric_axis(axes[0, 1])
    legend(axes[0, 1], columns=1)
    for i, (key, label) in enumerate(
        (("psi", "Goal factors"), ("phi_fixed", "Fixed probes"), ("phi_replay", "Replay probes"))
    ):
        norms = np.sort(np.linalg.norm(data[key], axis=-1))
        axes[1, 0].step(
            np.r_[norms[0], norms],
            np.r_[0, np.arange(1, len(norms) + 1) / len(norms)],
            where="post",
            color=COLORS[i],
            linestyle=("-", "--", "-.")[i],
            label=f"{label} (n={len(norms)})",
        )
    axes[1, 0].set(
        title="(c) Raw factor norms",
        xlabel=r"Embedding $L_2$ norm",
        ylabel="Fraction of samples ≤ norm",
    )
    numeric_axis(axes[1, 0], percent=True)
    legend(axes[1, 0], columns=1)
    eigenvalues = np.sort(data["covariance_eigenvalues"])[::-1]
    axes[1, 1].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker="o", color=COLORS[0])
    axes[1, 1].set(
        title="(d) Replay covariance spectrum",
        xlabel="Component rank (descending)",
        ylabel="Covariance eigenvalue",
    )
    from matplotlib.ticker import MaxNLocator

    axes[1, 1].xaxis.set_major_locator(MaxNLocator(5, integer=True))
    numeric_axis(axes[1, 1])
    finish(
        fig,
        note=f"{BRANCHES[name]}. Stars denote goal factors. PCA is refitted to this panel's probes; axes are not aligned across tasks. Empirical CDFs make different sample counts comparable. Actor covariance averages action heads.",
    )
    return fig


def plot_drift(data, task_ids, *, style):
    from matplotlib.ticker import MaxNLocator

    fig, axes = subplots(1, 2, style=style, height=3.4)
    for ax, key, xlabel, title in zip(
        axes,
        ("psi_drift", "phi_fixed_drift"),
        ("Previously seen goal ID", "Fixed probe index"),
        ("(a) Goal-factor change", "(b) Fixed-probe factor change"),
    ):
        values = data[key]
        ids = task_ids[: len(values)] if key == "psi_drift" else np.arange(len(values))
        ax.plot(ids, values, marker="o", color=COLORS[0])
        ax.set(title=title, xlabel=xlabel, ylabel=r"Embedding change ($L_2$)")
        if key == "psi_drift":
            ax.set_xticks(ids)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        numeric_axis(ax)
    finish(
        fig,
        note="Difference from the preceding sequential task on corresponding goals/probes. Actor probe indices flatten state and action-head indices. Changes need not imply a change in task performance.",
    )
    return fig


def plot_parameters(changes, *, style):
    from textwrap import fill

    keys = list(changes)
    fig, axes = subplots(1, 2, style=style, height=max(3.5, 0.25 * len(keys) + 1.8), sharey=True)
    for ax, metric, title, xlabel in zip(
        axes,
        ("l2_change", "relative_change"),
        ("(a) Absolute parameter change", "(b) Relative parameter change"),
        (r"$\|\theta_{end}-\theta_{start}\|_2$", r"$\|\Delta\theta\|_2 / \|\theta_{start}\|_2$"),
    ):
        ax.barh(range(len(keys)), [changes[k][metric] for k in keys], color=COLORS[0])
        ax.set(title=title, xlabel=xlabel, yticks=range(len(keys)))
        numeric_axis(ax)
    axes[0].set_yticklabels([fill(k.replace("_", " "), 25) for k in keys], fontsize=7)
    axes[0].invert_yaxis()
    finish(
        fig,
        note="Parameter groups retain their code names for traceability. Start/end refer to this training call. Relative change uses a denominator floor of 10⁻⁸.",
    )
    return fig
