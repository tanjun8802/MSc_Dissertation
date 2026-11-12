"""Centred display geometry, original-space measurements and logged replay IDs."""

import json
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tbtrl.experiments.embedding_geometry import (
    factor_geometry,
    load_saved_task_geometry,
    plot_saved_task_geometry,
)
from tbtrl.experiments.figure_style import FigureStyle


def test_angles_norms_and_zero_factors_are_measured_before_projection():
    vectors = np.array([[1, 0, 0], [0, 2, 0], [-1, 0, 0], [0, 0, 3], [0, 0, 0]])
    basis = np.eye(3)[:, :2]
    data = factor_geometry(vectors, basis=basis)
    np.testing.assert_allclose(data["norms"], [1, 2, 1, 3, 0])
    np.testing.assert_allclose(
        data["angles_degrees"][:4, :4],
        [[0, 90, 180, 90], [90, 0, 90, 90], [180, 90, 0, 90], [90, 90, 90, 0]],
    )
    assert np.isnan(data["angles_degrees"][4]).all()
    np.testing.assert_allclose(data["projected"], vectors[:, :2])
    assert data["norms"][3] == 3 and np.linalg.norm(data["projected"][3]) == 0
    assert data["retained_energy"] == pytest.approx(6 / 15)


def test_uncentred_projection_preserves_origin_and_does_not_force_unit_norms():
    data = factor_geometry([[2, 0], [0, 3], [0, 0]])
    np.testing.assert_allclose(np.linalg.norm(data["projected"], axis=1), [2, 3, 0])
    np.testing.assert_allclose(data["projected"][2], [0, 0])
    assert data["retained_energy"] == pytest.approx(1)
    assert np.isfinite(factor_geometry([[0], [0]])["projected"]).all()
    with pytest.raises(ValueError, match="orthonormal"):
        factor_geometry([[1, 2]], basis=[[1, 1], [1, 1]])


def test_centering_changes_only_display_and_reports_explained_variance():
    vectors = np.array([[3, 1, 1], [5, 1, 1], [4, 2, 1], [4, 0, 1]])
    original = factor_geometry(vectors)
    centered = factor_geometry(vectors, center=vectors.mean(axis=0))
    np.testing.assert_allclose(centered["projected"].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(centered["norms"], original["norms"])
    np.testing.assert_allclose(centered["angles_degrees"], original["angles_degrees"])
    assert centered["retained_energy"] == pytest.approx(1)
    singular_values = np.linalg.svd(vectors - vectors.mean(axis=0), compute_uv=False)
    assert centered["retained_energy"] == pytest.approx(
        sum(singular_values[:2] ** 2) / sum(singular_values**2)
    )


def saved_run(root, constraint="projected"):
    root.mkdir()
    model = dict(phi_max_norm=1, psi_max_norm=1, embedding_constraint=constraint)
    training = dict(sigreg_coef=0, k_replay=dict(selector="basis"))
    (root / "manifest.json").write_text(
        json.dumps(dict(config=dict(algorithm="dqn", model=model, training=training)))
    )
    raw_scale = 2 if constraint == "projected" else 1
    for task, ids, vectors in [
        (0, [0], [[0, 0, 0.5]]),
        (1, [0, 1], [[1, 0, 0], [0, 0.8, 0]]),
    ]:
        path = root / f"seed_42/sequential/task_{task}"
        path.mkdir(parents=True)
        (path / "diagnostics.json").write_text(
            json.dumps(
                dict(
                    task_ids=ids,
                    branches=dict(
                        q=dict(
                            psi=(np.asarray(vectors) * raw_scale).tolist(),
                            bounded_psi=vectors,
                            phi_fixed=[[0, 0, raw_scale], [0, raw_scale, 0]],
                            bounded_phi=[[0, 0, 1], [0, 1, 0]],
                        )
                    ),
                )
            )
        )
    return root


def test_stages_share_reference_axes_without_using_future_vectors_as_current(tmp_path):
    path = saved_run(tmp_path / "run")
    early = load_saved_task_geometry(path, seed=42, task_id=0)
    late = load_saved_task_geometry(path, seed=42)
    np.testing.assert_equal(early["psi"]["basis"], late["psi"]["basis"])
    np.testing.assert_equal(early["psi"]["center"], late["psi"]["center"])
    assert early["task_ids"] == [0] and late["task_ids"] == [0, 1]
    np.testing.assert_allclose(
        early["psi"]["projected"],
        (early["psi"]["vectors"] - late["psi"]["center"]) @ late["psi"]["basis"],
    )
    np.testing.assert_allclose(late["psi"]["projected"].mean(axis=0), 0, atol=1e-12)
    assert early["psi"]["norms"][0] == 0.5
    assert early["reference_task_id"] == 1
    with pytest.raises(ValueError, match="No saved embedding diagnostics"):
        load_saved_task_geometry(path, seed=99)
    with pytest.raises(ValueError, match="stage unavailable"):
        load_saved_task_geometry(path, seed=42, task_id=9)


@pytest.mark.parametrize("constraint", ["projected", "soft"])
def test_render_uses_saved_arrays_without_loading_models_or_changing_results(tmp_path, constraint):
    path = saved_run(tmp_path / constraint, constraint)
    original = {p: p.read_bytes() for p in path.rglob("*.json")}
    rng = np.random.get_state()
    # This path must also work for results produced on MPS, without GPU access.
    with patch("torch.load", side_effect=AssertionError("No model loading")):
        fig = plot_saved_task_geometry(
            path,
            seed=42,
            task_id=0,
            columns=1,
            style=FigureStyle(width=9.5, dpi=60, formats=("png",)),
        )
    np.testing.assert_allclose(
        fig.axes[2].lines[0].get_ydata(), [1 if constraint == "projected" else 0.5]
    )
    np.testing.assert_allclose(fig.axes[2].lines[1].get_ydata(), [0.5])
    label = "Hard radius" if constraint == "projected" else "Soft radius"
    assert not fig.axes[0].patches  # The centred origin is not the constraint origin.
    assert label in fig.axes[2].get_legend_handles_labels()[1]
    fig.canvas.draw()
    plt.close(fig)
    export = path / "figures/task-geometry-seed-42-after-0-ref-1.json"
    data = json.loads(export.read_text())
    assert data["psi"]["norms"] == [0.5]
    assert data["replay"] is None
    assert data["source"] == str((path / "seed_42/sequential/task_0/diagnostics.json").resolve())
    assert export.with_suffix(".png").exists()
    assert all(p.read_bytes() == raw for p, raw in original.items())
    after = np.random.get_state()
    assert rng[0] == after[0] and rng[2:] == after[2:]
    np.testing.assert_equal(rng[1], after[1])


def test_highlights_use_actual_logged_selection_and_report_its_time(tmp_path):
    path = saved_run(tmp_path / "run")
    row = dict(
        seed=42,
        mode="sequential",
        task_id=1,
        steps=10,
        replay_diagnostics=[
            dict(step=7, selected_tasks=[0], replay_weights=[1.0]),
            dict(step=3, selected_tasks=[], replay_weights=[]),
        ],
    )
    (path / "summary.json").write_text(json.dumps([row]))
    data = load_saved_task_geometry(path, seed=42)
    assert data["replay"]["selected_tasks"] == [0]
    assert data["replay"]["step"] == 7 and data["replay"]["training_steps"] == 10
    fig = plot_saved_task_geometry(path, seed=42, export=False)
    assert "Selected replay tasks" in fig.axes[0].get_legend_handles_labels()[1]
    assert "Selected replay tasks" in fig.axes[2].get_legend_handles_labels()[1]
    assert any(text.get_text() == "G0 [replay]" for text in fig.axes[0].texts)
    assert any("step 7/10" in text.get_text() for text in fig.texts)
    plt.close(fig)
    row["replay_diagnostics"][0]["selected_tasks"] = [1]
    (path / "summary.json").write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="historical factor IDs"):
        load_saved_task_geometry(path, seed=42)
