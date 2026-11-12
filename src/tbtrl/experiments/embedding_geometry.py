"""Offline DQN factor geometry from saved diagnostics, without model execution.

All linear algebra uses detached JSON arrays on CPU. No float64 tensor is ever
created on MPS, and no training RNG, checkpoint, or replay state is accessed.
"""

import json
from pathlib import Path

import numpy as np

from .diagnostic_plots import goal_style
from .figure_style import (
    DEFAULT_STYLE,
    finish,
    goal_ticks,
    legend,
    numeric_axis,
    panel_grid,
    save_figure,
    styled,
)


def factor_geometry(vectors, *, raw_vectors=None, basis=None, center=None, tolerance=1e-8):
    """True norms/angles plus an orthogonal 2D display projection.

    Zero-length factors have undefined angles (NaN), including with themselves.
    Only display coordinates subtract ``center``; original norms/angles never do.
    Supplying a reference center and basis keeps views aligned across stages.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or min(vectors.shape) == 0 or not np.isfinite(vectors).all():
        raise ValueError("Expected a nonempty finite matrix of factor vectors.")
    raw = vectors if raw_vectors is None else np.asarray(raw_vectors, dtype=np.float64)
    if raw.shape != vectors.shape or not np.isfinite(raw).all():
        raise ValueError("Raw and used factors must have matching finite shapes.")
    centered = center is not None
    center = np.zeros(vectors.shape[1]) if center is None else np.asarray(center, dtype=float)
    if center.shape != (vectors.shape[1],) or not np.isfinite(center).all():
        raise ValueError("Projection center must be finite and match the factor dimension.")
    display_vectors = vectors - center
    if basis is None:
        _, _, vt = np.linalg.svd(display_vectors, full_matrices=True)
        basis = vt[:2].T
        # Resolve sign ambiguity for reproducible reference axes.
        for column in range(basis.shape[1]):
            pivot = np.argmax(np.abs(basis[:, column]))
            if basis[pivot, column] < 0:
                basis[:, column] *= -1
    basis = np.asarray(basis, dtype=np.float64)
    if (
        basis.shape != (vectors.shape[1], min(2, vectors.shape[1]))
        or not np.isfinite(basis).all()
        or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8)
    ):
        raise ValueError("Projection basis must have orthonormal columns and matching dimension.")
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > tolerance
    unit = np.zeros_like(vectors)
    unit[valid] = vectors[valid] / norms[valid, None]
    cosine = np.clip(unit @ unit.T, -1, 1)
    cosine[~(valid[:, None] & valid[None, :])] = np.nan
    angles = np.degrees(np.arccos(cosine))
    angles[np.diag_indices_from(angles)] = np.where(valid, 0, np.nan)
    projected = display_vectors @ basis
    projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
    energy = float(np.square(display_vectors).sum())
    return dict(
        vectors=vectors,
        raw_vectors=raw,
        norms=norms,
        raw_norms=np.linalg.norm(raw, axis=1),
        angles_degrees=angles,
        cosine=cosine,
        basis=basis,
        center=center,
        centered=centered,
        projected=projected,
        retained_energy=float(np.square(projected).sum() / energy) if energy else None,
    )


def load_saved_task_geometry(directory, *, seed, task_id=None, reference_task_id=None):
    """Load diagnostics and fixed, branch-specific centred SVD reference planes.

    The last available diagnostic stage is the default display/reference stage.
    Final-stage axes are an offline visualisation choice, never a training input.
    """
    directory = Path(directory)
    config = json.loads((directory / "manifest.json").read_text())["config"]
    if config["algorithm"] != "dqn":
        raise ValueError("This saved-factor view currently supports DQN diagnostics.")
    source = directory / f"seed_{seed}/sequential"
    stages = {
        int(path.parent.name.removeprefix("task_")): path
        for path in source.glob("task_*/diagnostics.json")
    }
    if not stages:
        raise ValueError(
            f"No saved embedding diagnostics for {directory.name}, seed {seed}. "
            "Choose a condition recorded with NotebookDiagnostics (the notebook's DETAIL_CASE)."
        )
    task_id = max(stages) if task_id is None else task_id
    reference_task_id = max(stages) if reference_task_id is None else reference_task_id
    if task_id not in stages or reference_task_id not in stages:
        raise ValueError(f"Requested stage unavailable; saved stages are {sorted(stages)}.")
    diagnostic = json.loads(stages[task_id].read_text())
    reference = json.loads(stages[reference_task_id].read_text())
    data, ref = diagnostic["branches"]["q"], reference["branches"]["q"]
    # The historical key 'bounded_*' contains the actual forward-pass factors,
    # including unprojected factors for soft-mode DQN runs.
    required = ("psi", "bounded_psi", "phi_fixed", "bounded_phi")
    if any(key not in data or key not in ref for key in required):
        raise ValueError("Saved diagnostics must include both raw and forward-pass factors.")

    def project(used_key, raw_key):
        center = np.asarray(ref[used_key], dtype=float).mean(axis=0)
        basis = factor_geometry(ref[used_key], center=center)["basis"]
        return factor_geometry(
            data[used_key], raw_vectors=data[raw_key], basis=basis, center=center
        )

    psi, phi = project("bounded_psi", "psi"), project("bounded_phi", "phi_fixed")
    task_ids = diagnostic["task_ids"]
    if len(task_ids) != len(psi["norms"]) or len(set(task_ids)) != len(task_ids):
        raise ValueError("Saved goal IDs must identify each task vector exactly once.")
    replay = None
    summary = directory / "summary.json"
    if summary.exists():
        rows = [
            r
            for r in json.loads(summary.read_text())
            if r["seed"] == seed and r["mode"] == "sequential" and r["task_id"] == task_id
        ]
        if len(rows) > 1:
            raise ValueError("Multiple saved acquisition records for the requested seed/task.")
        if rows and rows[0].get("replay_diagnostics"):
            row = rows[0]
            snapshot = max(row["replay_diagnostics"], key=lambda item: item["step"])
            selected = snapshot["selected_tasks"]
            if not set(selected).issubset(task_ids) or task_id in selected:
                raise ValueError("Saved replay task IDs do not match historical factor IDs.")
            weights = snapshot["replay_weights"]
            if len(weights) != len(selected):
                raise ValueError("Saved replay weights must match selected task IDs.")
            replay = dict(
                selected_tasks=selected,
                weights=weights,
                step=snapshot["step"],
                training_steps=row["steps"],
                selector=config["training"]["k_replay"]["selector"],
            )
    return dict(
        case=directory.name,
        seed=seed,
        task_id=task_id,
        reference_task_id=reference_task_id,
        task_ids=task_ids,
        psi=psi,
        phi=phi,
        model=config["model"],
        training=config.get("training", {}),
        replay=replay,
        source=str(stages[task_id].resolve()),
    )


def _vector_panel(
    ax,
    geometry,
    ids,
    *,
    prefix,
    title,
    radius,
    constraint,
    selected=(),
    current=None,
):
    from matplotlib.patches import Circle

    points = geometry["projected"]
    centered = geometry.get("centered", False)
    extent = max(float(np.abs(points).max()), 0 if centered else radius or 0, 1e-3) * 1.12
    if radius is not None and not centered:
        label = "Hard radius" if constraint == "projected" else "Soft radius (not a bound)"
        ax.add_patch(Circle((0, 0), radius, facecolor="#E8EEF6", edgecolor="#A7BDDA", alpha=0.6))
        ax.plot([], [], color="#A7BDDA", label=f"{label}: {radius:g}")
    ax.axhline(0, color="#999999", linewidth=0.6)
    ax.axvline(0, color="#999999", linewidth=0.6)
    for point, identity in zip(points, ids):
        appearance = goal_style(identity)
        ax.annotate(
            "",
            xy=point,
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="->",
                color=appearance["color"],
                linewidth=2.8 if identity in selected else 1.0,
                alpha=1 if identity in selected or not selected else 0.55,
            ),
        )
        ax.scatter(*point, color=appearance["color"], marker=appearance["marker"], s=24, zorder=3)
        if identity in selected:
            ax.scatter(
                *point, facecolors="none", edgecolors="black", s=100, linewidths=1.2, zorder=4
            )
        if identity == current:
            ax.scatter(*point, color="#FFD166", edgecolors="black", marker="*", s=120, zorder=5)
    if selected:
        ax.scatter(
            [], [], facecolors="none", edgecolors="black", s=70, label="Selected replay tasks"
        )
    if current is not None:
        ax.scatter(
            [], [], color="#FFD166", edgecolors="black", marker="*", s=90, label="Current task"
        )
    # Put labels outside the circle, ordered by endpoint height on each side.
    # This keeps almost coincident task vectors individually identifiable.
    for right in (False, True):
        indices = [i for i, point in enumerate(points) if (point[0] >= 0) == right]
        indices.sort(key=lambda i: points[i, 1])
        positions = np.linspace(-extent * 0.85, extent * 0.85, len(indices))
        for i, y in zip(indices, positions):
            if len(indices) == 1:
                y = points[i, 1]
            ax.annotate(
                f"{prefix}{ids[i]}" + (" [replay]" if ids[i] in selected else ""),
                xy=points[i],
                xytext=(extent * (1.08 if right else -1.08), y),
                ha="left" if right else "right",
                va="center",
                color=goal_style(ids[i])["color"],
                fontweight="bold" if ids[i] in selected else "normal",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", linewidth=0.5),
            )
    retained = geometry["retained_energy"]
    quantity = "reference-centred energy retained" if centered else "squared norm retained"
    description = (
        f"{retained:.1%} {quantity}" if retained is not None else "no variation to project"
    )
    ax.set(
        title=f"{title}\n2D projection: {description}",
        xlabel="Centred SVD direction 1" if centered else "Reference direction 1",
        ylabel="Centred SVD direction 2" if centered else "Reference direction 2",
        xlim=(-1.4 * extent, 1.4 * extent),
        ylim=(-1.05 * extent, 1.05 * extent),
        aspect="equal",
    )
    numeric_axis(ax)
    legend(ax, columns=1)


@styled
def plot_saved_task_geometry(
    directory,
    *,
    seed,
    task_id=None,
    reference_task_id=None,
    style=DEFAULT_STYLE,
    columns=1,
    export=True,
):
    """Vector views, true task norms and true pairwise angles from saved JSON only."""
    data = load_saved_task_geometry(
        directory, seed=seed, task_id=task_id, reference_task_id=reference_task_id
    )
    psi, phi, ids, model = data["psi"], data["phi"], data["task_ids"], data["model"]
    constraint = model.get("embedding_constraint", "projected")
    replay = data["replay"]
    selected = replay["selected_tasks"] if replay else []
    fig, axes = panel_grid(4, columns=columns, panel_height=4.6, style=style)
    for ax, geometry, identities, branch, prefix, title in (
        (axes[0], psi, ids, "psi", "G", r"(a) Task factors $\psi(g)$ used by Q"),
        (
            axes[1],
            phi,
            list(range(len(phi["norms"]))),
            "phi",
            "P",
            r"(b) Saved state–action probes $\phi(s,a)$",
        ),
    ):
        _vector_panel(
            ax,
            geometry,
            identities,
            prefix=prefix,
            title=title,
            radius=model.get(f"{branch}_max_norm"),
            constraint=constraint,
            selected=selected if branch == "psi" else (),
            current=data["task_id"] if branch == "psi" else None,
        )
    x = np.arange(len(ids))
    axes[2].plot(x, psi["raw_norms"], "s--", label="Raw encoder output", color="#D55E00")
    axes[2].plot(x, psi["norms"], "o-", label="Factor used by Q", color="#0072B2")
    radius = model.get("psi_max_norm")
    if radius is not None:
        axes[2].axhline(
            radius,
            color="#777777",
            linestyle=":",
            label="Hard radius" if constraint == "projected" else "Soft radius",
        )
    axes[2].set(
        title=f"(c) Actual task-factor norms in {psi['vectors'].shape[1]} dimensions",
        xlabel="Goal ID",
        ylabel=r"Original-space $L_2$ norm",
        ylim=(0, None),
    )
    axes[2].set_xticks(x, ids)
    if selected:
        positions = [ids.index(task) for task in selected]
        axes[2].scatter(
            positions,
            psi["norms"][positions],
            facecolors="none",
            edgecolors="black",
            s=100,
            zorder=5,
            label="Selected replay tasks",
        )
    numeric_axis(axes[2])
    legend(axes[2])
    angles = psi["angles_degrees"]
    im = axes[3].imshow(
        np.ma.masked_invalid(angles),
        vmin=0,
        vmax=180,
        cmap="RdBu_r",
        interpolation="nearest",
        aspect="auto",
    )
    fig.colorbar(
        im, ax=axes[3], label="Original-space angle (degrees)", ticks=[0, 45, 90, 135, 180]
    )
    axes[3].set(
        title="(d) True pairwise task angles · factors used by Q",
        xlabel="Goal ID",
        ylabel="Goal ID",
    )
    goal_ticks(axes[3], ids, matrix=True)
    goal_ticks(axes[3], ids, matrix=True, axis="y")
    for labels in (axes[3].get_xticklabels(), axes[3].get_yticklabels()):
        for label, identity in zip(labels, ids):
            if identity in selected:
                label.set_fontweight("bold")
    if len(ids) <= 12:
        for (row, col), angle in np.ndenumerate(angles):
            axes[3].text(
                col,
                row,
                f"{angle:.0f}°" if np.isfinite(angle) else "—",
                ha="center",
                va="center",
                fontsize=style.font_size - 2,
                color="white" if np.isfinite(angle) and (angle < 40 or angle > 140) else "black",
            )
    selection_note = "Replay selection was not saved."
    if replay:
        weighted_ids = ", ".join(
            f"G{task}={weight:.3f}" for task, weight in zip(selected, replay["weights"])
        )
        selection_note = (
            f"Last logged {replay['selector']} selection at step {replay['step']:,}/{replay['training_steps']:,}: "
            f"{weighted_ids or 'no historical tasks'}. Vectors are task-end factors, not selection-time factors."
        )
    finish(
        fig,
        title=f"{data['case']} · seed {seed} · embedding geometry after goal {data['task_id']}",
        note=f"Separate centred SVD planes for ψ and φ, fitted after goal {data['reference_task_id']}; no normalisation. Origins are reference means; branch axes are not comparable. Display energy equals explained variance at the reference stage. Panels (c–d) retain original-space norms/angles. Saved protocol: {constraint}, SIGReg={data['training'].get('sigreg_coef', 'unknown')}. {selection_note} Highlighted vectors identify replay tasks, not PCA directions or a guarantee of retention.",
    )
    if export:
        destination = (
            Path(directory)
            / "figures"
            / f"task-geometry-seed-{seed}-after-{data['task_id']}-ref-{data['reference_task_id']}"
        )
        save_figure(fig, destination, style=style)

        # Strict JSON: undefined angles are null, never non-standard NaN values.
        def serialise(value):
            if isinstance(value, np.ndarray):
                return serialise(value.tolist())
            if isinstance(value, dict):
                return {key: serialise(item) for key, item in value.items()}
            if isinstance(value, list):
                return [serialise(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        destination.with_suffix(".json").write_text(
            json.dumps(serialise(data), indent=2, allow_nan=False)
        )
    return fig
