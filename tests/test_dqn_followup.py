"""Matched-budget training and independent recovery are distinct experiment contracts."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch

from tbtrl.experiments.config import load_config
from tbtrl.experiments.dqn_followup import configure_followup, run_recovery_probe
from tbtrl.experiments.dqn_followup_plots import (
    followup_table,
    plot_followup_acquisition,
    plot_followup_recovery,
)
from tbtrl.experiments.figure_style import FigureStyle
from tbtrl.experiments.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def configs():
    base = load_config(ROOT / "configs/tbtrl_gridworldmaze_dqn_followup.json").smoke()
    base.goals = [[5, 6], [3, 5], [3, 3]]
    base.training.update(early_stop_reward=-100, early_stop_patience=2)
    cases = configure_followup(
        base,
        k_values=(1, 2),
        sample_budget=8,
        candidate_capacity=4,
        refresh_updates=2,
        capacity=32,
        recent_capacity=16,
        max_groups=2,
    )
    for config in cases.values():
        config.recovery.update(total_steps=16)
        config.recovery_return_threshold = 2  # Unreachable: exercise all trials and failure costs.
    return cases


def test_followup_preserves_historical_regularisation_and_matches_budgets():
    cases = configs()
    assert set(cases) == {
        "random-k1",
        "basis-k1",
        "random-k2",
        "basis-k2",
        "scratch",
        "fine-tuning",
    }
    for config in cases.values():
        assert config.model["embedding_constraint"] == "projected"
        assert config.training["sigreg_coef"] == 0.01
        assert config.training["goal_separation_coef"] == 0.04
        assert config.training["phi_raw_norm_coef"] == 0.0002
        assert config.training["psi_raw_norm_coef"] == 0.0002
        assert config.training["total_steps"] == 24
        assert not config.training["enable_early_stop"]
        assert config.recovery["enable_early_stop"]
    a, b = deepcopy(cases["random-k2"]), deepcopy(cases["basis-k2"])
    a.name = b.name
    a.training["k_replay"]["selector"] = "basis"
    assert a == b


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_real_fixed_budget_and_independent_recovery_roundtrip(tmp_path, device):
    config = configs()["basis-k1"]
    source = tmp_path / "acquisition"
    acquired = run_experiment(config, source, device=device, modes=("sequential",), verbose=False)
    assert all(r["steps"] == 24 for r in acquired)
    # A threshold crossing records acquisition without truncating the matched budget.
    assert all(r["steps_to_threshold"] == 16 and r["solved"] for r in acquired)
    boundary = source / "seed_42/sequential/boundary.pt"
    before = hashlib.sha256(boundary.read_bytes()).hexdigest()
    source_summary = (source / "summary.json").read_bytes()
    paths = {}
    for protocol in ("no-replay", "method-replay"):
        out = tmp_path / protocol
        paths[protocol] = out
        rows = run_recovery_probe(source, out, protocol=protocol, device=device, verbose=False)
        assert len(rows) == 3
        assert all(r["steps"] == 16 and not r["solved"] for r in rows)
        assert all(r["source_boundary_sha256"] == before for r in rows)
        assert rows[0]["initial_evaluation"] == rows[1]["initial_evaluation"]
        counts = [len(d["selected_tasks"]) for r in rows for d in r["replay_diagnostics"]]
        assert max(counts) == (0 if protocol == "no-replay" else 1)
        if protocol == "no-replay":
            assert all(r["replay_memory"]["archive_samples"] == 0 for r in rows)
        receipt = out / "seed_42/recovery/after_3/task_0/complete.json"
        stamp = receipt.stat().st_mtime_ns
        again = run_recovery_probe(source, out, protocol=protocol, device=device, verbose=False)
        assert again == rows
        assert receipt.stat().st_mtime_ns == stamp
    assert hashlib.sha256(boundary.read_bytes()).hexdigest() == before
    assert (source / "summary.json").read_bytes() == source_summary
    table = followup_table(paths, tmp_path / "tables")
    assert len(table) == 2
    assert all(r["complete"] and r["failures"] == 3 for r in table)
    assert all(r["retained_fraction"] == 0 and r["adaptation_cost"] == 16 for r in table)
    for fn, inputs in (
        (plot_followup_acquisition, {"basis-k1": source}),
        (plot_followup_recovery, paths),
    ):
        figure = fn(inputs, tmp_path / fn.__name__, style=FigureStyle(dpi=60, formats=("png",)))
        plt.close(figure)
    assert len(list(tmp_path.rglob("*.png"))) == 2
    # Changed sources must not reuse results from a different starting checkpoint.
    with boundary.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Source/config changed"):
        run_recovery_probe(
            source, paths["no-replay"], protocol="no-replay", device=device, verbose=False
        )


def test_retained_tasks_skip_without_claiming_adaptation(tmp_path):
    config = configs()["fine-tuning"]
    config.recovery_return_threshold = -100
    source = tmp_path / "source"
    run_experiment(config, source, modes=("sequential",), verbose=False)
    rows = run_recovery_probe(source, tmp_path / "probe", protocol="no-replay", verbose=False)
    assert all(not r["recovery_performed"] and r["steps"] == 0 for r in rows)
    table = followup_table({"fine-tuning": tmp_path / "probe"})
    assert table[0]["retained_fraction"] == 1
    assert table[0]["adaptation_cost"] is None
    assert table[0]["other_task_delta"] is None
    assert table[0]["failures"] == 0


def test_incomplete_acquisition_is_rejected(tmp_path):
    config = configs()["basis-k1"]
    source = tmp_path / "source"
    run_experiment(config, source, modes=("sequential",), verbose=False)
    path = source / "summary.json"
    rows = json.loads(path.read_text())
    path.write_text(json.dumps(rows[:-1]))
    with pytest.raises(ValueError, match="Incomplete acquisition"):
        run_recovery_probe(source, tmp_path / "probe", protocol="no-replay", verbose=False)
