"""Scientific plot semantics, exports, and rendering isolation."""

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from tbtrl.environments.registry import make_environment
from tbtrl.experiments.config import load_config
from tbtrl.experiments.diagnostic_plots import TERMS, plot_gradients, plot_learning
from tbtrl.experiments.diagnostics import NotebookDiagnostics
from tbtrl.experiments.figure_style import MODES, FigureStyle, save_figure
from tbtrl.experiments.plots import plot_comparison, plot_run
from tbtrl.experiments.render import render_saved_run
from tbtrl.experiments.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]
FAST = FigureStyle(dpi=70, formats=("png",))


def records(seeds=(1,)):
    rows = []
    for seed in seeds:
        for mode in MODES:
            for task in (2, 5):
                rows.append(
                    dict(
                        seed=seed,
                        mode=mode,
                        task_id=task,
                        steps=100 * seed + task,
                        elapsed_seconds=60 * (seed + task),
                        steps_to_threshold=None if task == 5 else 80,
                        solved=task != 5,
                        evaluation=[
                            dict(task_id=5, success_rate=0.25),
                            dict(task_id=2, success_rate=1.0),
                        ],
                    )
                )
    return rows


def write_run(tmp_path, rows):
    (tmp_path / "summary.json").write_text(json.dumps(rows))
    return tmp_path


def test_summary_goal_alignment_mode_colours_and_single_seed_uncertainty(tmp_path):
    rows = records()
    row = next(r for r in rows if r["mode"] == "recovery" and r["task_id"] == 2)
    row.update(
        recovery_performed=False, steps=0, steps_to_threshold=0, elapsed_seconds=0, solved=True
    )
    path = write_run(tmp_path, rows[::-1])
    before = matplotlib.rcParams["font.size"]
    fig = plot_run(path, style=FAST, export=False)
    assert matplotlib.rcParams["font.size"] == before
    heat = next(ax for ax in fig.axes if ax.images)
    np.testing.assert_equal(heat.images[0].get_array(), [[1, 0.25], [1, 0.25]])
    assert [label.get_text() for label in heat.get_xticklabels()] == ["2", "5"]
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert "Unsolved: spent budget" in labels and "Recovery skipped" in labels
    assert fig.axes[0].collections[4].get_facecolors().size == 0
    plt.close(fig)
    fig = plot_comparison(path, style=FAST, export=False)
    assert all(not ax.collections for ax in fig.axes)  # no fake SD ribbon for n=1
    for ax in fig.axes:
        for line, mode in zip(ax.lines, MODES):
            assert line.get_color() == MODES[mode][1]
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [3, 6])
    assert "n=1" in fig._supxlabel.get_text()
    plt.close(fig)


def test_cost_bands_are_seed_sd_and_incomplete_seed_coverage_is_rejected(tmp_path):
    rows = records((1, 3))
    path = write_run(tmp_path, rows)
    fig = plot_comparison(path, style=FAST, export=False)
    assert len(fig.axes[0].collections) == 3
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [202, 205])
    np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), [202, 407])
    plt.close(fig)
    write_run(path, rows[:-1])
    with pytest.raises(ValueError, match="one record per seed and goal"):
        plot_comparison(path, style=FAST, export=False)
    plt.close("all")


def test_separate_scratch_control_is_used_in_all_cost_panels(tmp_path):
    rows = records((1, 3))
    baseline = tmp_path / "scratch"
    condition = tmp_path / "condition"
    baseline.mkdir()
    condition.mkdir()
    scratch = [dict(r, steps=r["steps"] * 2) for r in rows if r["mode"] == "scratch"]
    write_run(baseline, scratch)
    write_run(condition, [r for r in rows if r["mode"] != "scratch"])
    before = {p: p.read_bytes() for p in tmp_path.rglob("summary.json")}
    style = FigureStyle(width=9.5, font_size=11, dpi=70, formats=("png",))
    fig = plot_comparison(condition, baseline_dir=baseline, columns=1, style=style, export=False)
    for ax in fig.axes:
        assert ax.get_legend_handles_labels()[1] == [m[0] for m in MODES.values()]
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [404, 410])
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [4, 7])
    np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), [404, 814])
    np.testing.assert_allclose(fig.axes[3].lines[0].get_ydata(), [4, 11])
    assert all(ax.get_subplotspec().get_gridspec().ncols == 1 for ax in fig.axes)
    plt.close(fig)
    fig = plot_run(condition, baseline_dir=baseline, columns=1, seed=3, style=style, export=False)
    assert len([ax for ax in fig.axes if ax.images]) == 1
    assert "From scratch" in fig.axes[0].get_legend_handles_labels()[1]
    assert "seed 3" in fig.axes[0].get_title()
    plt.close(fig)
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("mismatch", ["seed", "goal", "duplicate", "missing"])
def test_separate_scratch_control_rejects_incompatible_results(tmp_path, mismatch):
    baseline, condition = tmp_path / "scratch", tmp_path / "condition"
    baseline.mkdir()
    condition.mkdir()
    rows = records()
    scratch = [r for r in rows if r["mode"] == "scratch"]
    write_run(condition, [r for r in rows if r["mode"] != "scratch"])
    if mismatch == "seed":
        scratch[0]["seed"] = 99
    elif mismatch == "goal":
        scratch[0]["goal"] = [99, 99]
    elif mismatch == "duplicate":
        scratch.append(scratch[0])
    else:
        scratch.pop()
    write_run(baseline, scratch)
    with pytest.raises(ValueError, match="Scratch baseline"):
        plot_comparison(condition, baseline_dir=baseline, export=False)


def test_study_timing_includes_scratch_and_excludes_recovery(tmp_path):
    from tbtrl.experiments.dqn_study import plot_study

    paths = {}
    for label, modes, seconds in (
        ("scratch", ["scratch"], 0.02),
        ("basis-uniform", ["sequential", "recovery"], 0.05),
    ):
        path = tmp_path / label
        path.mkdir()
        paths[label] = path
        rows = []
        for task in (0, 1):
            for mode in modes:
                rows.append(
                    dict(
                        seed=1,
                        task_id=task,
                        mode=mode,
                        steps=100,
                        solved=True,
                        evaluation=[dict(task_id=task, success_rate=1)],
                        replay_diagnostics=[
                            dict(update_seconds=9 if mode == "recovery" else seconds)
                        ],
                    )
                )
        write_run(path, rows)
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "config": {
                        "goals": [[0, 0], [1, 1]],
                        "seeds": [1],
                    }
                }
            )
        )
    fig = plot_study(paths, tmp_path / "figures", columns=1, style=FAST)
    points = [line for line in fig.axes[3].lines if line.get_marker() == "o"]
    assert len(points) == 2
    np.testing.assert_allclose([line.get_ydata()[0] for line in points], [20, 50])
    assert "From scratch" in fig.axes[3].get_xticklabels()[0].get_text()
    assert len(fig.axes[0].lines) == 2
    assert len(fig.axes[1].lines) == 1  # scratch is not a retention trajectory
    plt.close(fig)


def test_return_success_and_undefined_gradient_cosines_are_not_conflated():
    result = SimpleNamespace(
        evaluations=[(10, -20), (20, 50)], success_rates=[(10, 0.25), (20, 1)], losses=[]
    )
    config = SimpleNamespace(algorithm="sac", eval_episodes=8)
    record = dict(task_id=0, mode="sequential", evaluation=[dict(task_id=0, success_rate=1)])
    fig = plot_learning(result, config, record, style=FAST)
    np.testing.assert_equal(fig.axes[0].lines[0].get_ydata(), [-20, 50])
    np.testing.assert_equal(fig.axes[1].lines[0].get_ydata(), [0.25, 1])
    assert len(fig.axes[0].lines) == 1
    plt.close(fig)
    row = {f"actor_{term}_grad_norm": 0 for term in TERMS}
    row.update(step=1, actor_current_grad_norm=1, actor_current_replay_grad_cosine=0)
    fig = plot_gradients([row], style=FAST)
    assert np.isnan(fig.axes[1].lines[0].get_ydata()).all()
    plt.close(fig)


def test_paper_exports_include_high_resolution_raster_and_editable_vectors(tmp_path):
    fig = plot_run(write_run(tmp_path, records()), export=False)
    paths = save_figure(fig, tmp_path / "paper")
    assert {p.suffix for p in paths} == {".png", ".pdf", ".svg"}
    with Image.open(tmp_path / "paper.png") as im:
        assert im.info["dpi"][0] == pytest.approx(300, abs=0.1)
        assert im.width >= 2000
    assert (tmp_path / "paper.pdf").read_bytes().startswith(b"%PDF")
    assert "<text" in (tmp_path / "paper.svg").read_text()
    plt.close(fig)


def test_spatial_action_directions_follow_maze_coordinates():
    class AlwaysPositiveY(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def q_val_for_argmax_action(self, obs, goals):
            return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(obs), -1)

    config = load_config(ROOT / "configs/tbtrl_gridworldmaze_dqn.json").smoke()
    factory = partial(make_environment, config.environment, **config.environment_options)
    env = factory(goal=config.goals[0])
    layout = np.array(env.unwrapped.maze)
    env.close()
    captured = {}
    NotebookDiagnostics(display=False, rollout_episodes=1, style=FAST)._spatial(
        SimpleNamespace(network=AlwaysPositiveY()),
        config,
        dict(seed=42, task_id=0, mode="sequential"),
        factory,
        layout,
        lambda fig, name: captured.update({name: fig}),
    )
    from matplotlib.quiver import Quiver

    fig = captured["q-policy"]
    arrows = [
        collection
        for ax in fig.axes
        for collection in ax.collections
        if isinstance(collection, Quiver)
    ]
    assert len(arrows) == 1
    np.testing.assert_equal(arrows[0].V, 1)  # no display-only sign inversion
    maps = [ax.images[-1] for ax in fig.axes if ax.images]
    assert len({im.get_clim() for im in maps}) == 1
    for fig in captured.values():
        plt.close(fig)


def test_offline_rendering_preserves_saved_data_rng_and_never_trains(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs/tbtrl_gridworldmaze_sac_factorised_actor.json").smoke()
    config.goals = config.goals[:1]
    source = tmp_path / "run"
    run_experiment(config, source, modes=("sequential",), verbose=False)
    files = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}

    def no_training(*args, **kwargs):
        raise AssertionError("Offline rendering must not train")

    monkeypatch.setattr("tbtrl.experiments.runner._train", no_training)
    rng = torch.get_rng_state().clone()
    destination = render_saved_run(source, style=FAST)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert all(p.read_bytes() == data for p, data in files.items())
    assert (destination / "seed_42/sequential/task_0/q-policy.png").is_file()
    assert (destination / "figure-manifest.json").is_file()
