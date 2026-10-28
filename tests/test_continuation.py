"""Task-boundary continuation must preserve updates, replay and mode isolation."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from tbtrl.continuation import load_boundary
from tbtrl.experiments.config import load_config
from tbtrl.experiments.continuation import load_sequence
from tbtrl.experiments.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def assert_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree(x, y)
    else:
        assert a == b


def config(algorithm, global_replay):
    suffix = "factorised_actor" if algorithm == "sac" else None
    name = "tbtrl_gridworldmaze_" + algorithm + ("_" + suffix if suffix else "")
    full = load_config(ROOT / "configs" / (name + ".json"))
    c = full.smoke()
    c.goals = full.goals[:3]
    if global_replay:
        c.replay_mode = "global"
        c.similarity_threshold = 0.8
        c.global_replay_capacity, c.recent_replay_capacity = 64, 32
        c.replay_max_groups = 4
    if algorithm == "sac":
        c.entropy_transfer = "carry"
        c.carry_optimizer_state = True
    return c.validate()


@pytest.mark.parametrize(
    "algorithm,global_replay", [("dqn", False), ("dqn", True), ("sac", False), ("sac", True)]
)
def test_append_matches_uninterrupted_and_keeps_scratch_independent(
    algorithm, global_replay, tmp_path, monkeypatch
):
    from tbtrl.experiments import runner

    c = config(algorithm, global_replay)
    full, split = tmp_path / "full", tmp_path / "split"
    modes = ("scratch", "sequential")
    run_experiment(c, full, modes=modes, verbose=False)
    prefix = deepcopy(c)
    prefix.goals = c.goals[:2]
    run_experiment(prefix, split, modes=modes, verbose=False, resume=True)
    original = runner._train
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_train", counted)
    records = run_experiment(c, split, modes=modes, verbose=False, resume=True)
    assert calls == [2, 2]  # Only the new scratch task and the new TBTRL task.
    assert len(records) == 6
    for mode in modes:
        for task in range(3):
            rel = f"seed_42/{mode}/task_{task}"
            a = torch.load(full / rel / "checkpoint.pt", weights_only=True)
            b = torch.load(split / rel / "checkpoint.pt", weights_only=True)
            for component in (
                ("network", "target")
                if algorithm == "dqn"
                else ("actor", "critic", "target", "state_normalizer", "goal_normalizer")
            ):
                assert_tree(a[component], b[component])
            assert (full / rel / "losses.jsonl").read_text() == (
                split / rel / "losses.jsonl"
            ).read_text()
    a, b = [load_sequence(p / "seed_42/sequential/boundary.pt", "cpu") for p in (full, split)]
    if algorithm == "sac":
        assert_tree(a["result"].optimizer_state, b["result"].optimizer_state)
        assert a["result"].entropy_coefficient == b["result"].entropy_coefficient
    if global_replay:
        memory = b["memory"]
        assert np.shares_memory(memory.recent.obs, memory.storage.obs)
        assert all(np.shares_memory(v.obs, memory.storage.obs) for v in memory.buffers.values())
        assert a["memory"].task_groups == b["memory"].task_groups
        for field in memory.storage.fields:
            assert_tree(getattr(a["memory"].storage, field), getattr(memory.storage, field))
    calls.clear()
    assert run_experiment(c, split, modes=modes, verbose=False, resume=True) == records
    assert calls == []
    assert len(json.loads((split / "summary.json").read_text())) == 6


def test_interruption_after_committed_task_and_incompatible_append(tmp_path, monkeypatch):
    from tbtrl.experiments import runner

    c = config("sac", True)

    def interrupt(**kw):
        if kw["record"]["mode"] == "sequential":
            raise RuntimeError("interrupt after task")

    with pytest.raises(RuntimeError, match="interrupt after task"):
        run_experiment(
            c,
            tmp_path / "run",
            modes=("sequential",),
            on_task_end=interrupt,
            resume=True,
            verbose=False,
        )
    assert load_boundary(tmp_path / "run/seed_42/sequential/boundary.pt")["completed"] == 1
    original = runner._train
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_train", counted)
    run_experiment(c, tmp_path / "run", modes=("sequential",), resume=True, verbose=False)
    assert calls == [1, 2]
    for changed in ("goals", "training"):
        bad = deepcopy(c)
        if changed == "goals":
            bad.goals.reverse()
        else:
            bad.training["eval_freq"] += 1
        with pytest.raises(ValueError):
            run_experiment(bad, tmp_path / "run", modes=("sequential",), resume=True, verbose=False)


@pytest.mark.parametrize("controlled", [False, True])
def test_candidate_task_append_and_no_completed_oracle_calls(controlled, tmp_path):
    from test_discovery_study import TASKS, factory
    from test_discovery_study import config as candidate_config

    from tbtrl.benchmarks.candidates import run_experiment as candidate_run
    from tbtrl.benchmarks.study import _load, run_study

    tasks = TASKS + [dict(name="C", goal=[0.5, 0.5])]
    all_calls, append_calls = [], []
    full, split = tmp_path / "full", tmp_path / "split"
    kwargs = dict(config=candidate_config(), seeds=[7])
    run = run_study if controlled else candidate_run
    if controlled:
        kwargs.update(checkpoint_campaigns=False)
    run(tasks, factory(all_calls), full, **kwargs)
    run(tasks[:2], factory(append_calls), split, **kwargs)
    previous = len(append_calls)
    records = run(tasks, factory(append_calls), split, resume=True, **kwargs)
    # Two new acquisition jobs (scratch and TBTRL), each with eight oracle calls.
    assert len(append_calls) - previous == 16
    assert len(records) == 6
    for mode in ("scratch", "sequential"):
        a = json.loads((full / f"seed_7/{mode}/task_2/results.json").read_text())
        b = json.loads((split / f"seed_7/{mode}/task_2/results.json").read_text())
        assert a["history"] == b["history"]
        assert a["losses"] == b["losses"]
    path = "seed_7/sequential/boundary.pt"
    a, b = [_load(p / path)["learner"] for p in (full, split)]
    for key in ("network", "target", "optimizer", "regularizer", "seen"):
        assert_tree(a[key], b[key])
    for key in ("recent", "groups"):
        # Ragged original-task transitions, including reward/termination, survive.
        def arrays(value):
            if isinstance(value, dict):
                return {k: arrays(v) for k, v in value.items()}
            return [vars(t) for t in value]

        assert_tree(arrays(a[key]), arrays(b[key]))
    count = len(append_calls)
    run(tasks, factory(append_calls), split, resume=True, **kwargs)
    assert len(append_calls) == count
    assert not list(split.rglob("resume.pt"))


def test_recovery_is_bound_to_endpoint_and_does_not_mutate_continuation(tmp_path):
    c = config("sac", True)
    output = tmp_path / "run"
    prefix = deepcopy(c)
    prefix.goals = c.goals[:2]
    run_experiment(prefix, output, resume=True, verbose=False)
    old = output / "seed_42/recovery/after_2"
    before = {str(p): p.read_bytes() for p in old.rglob("metrics.json")}
    run_experiment(c, output, resume=True, verbose=False)
    assert all(Path(p).read_bytes() == contents for p, contents in before.items())
    assert (output / "seed_42/recovery/after_3/task_2/complete.json").exists()
    state = load_sequence(output / "seed_42/sequential/boundary.pt", "cpu")
    source = torch.load(output / "seed_42/sequential/task_2/checkpoint.pt", weights_only=True)
    assert_tree(state["result"].actor.state_dict(), source["actor"])
    rows = json.loads((output / "summary.json").read_text())
    assert len(rows) == 9
    # Recovery is an optional evaluation branch, not part of acquisition identity.
    rows = run_experiment(c, output, modes=("scratch", "sequential"), resume=True, verbose=False)
    assert len(rows) == 6
    assert all(row["mode"] != "recovery" for row in rows)


def test_partial_task_restarts_from_previous_boundary(tmp_path, monkeypatch):
    from tbtrl.experiments import runner

    c = config("dqn", True)
    original = runner._train
    failed = [False]

    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[3] == 1 and not failed[0]:
            failed[0] = True
            raise RuntimeError("incomplete task")
        return result

    monkeypatch.setattr(runner, "_train", broken)
    with pytest.raises(RuntimeError, match="incomplete task"):
        run_experiment(c, tmp_path / "split", modes=("sequential",), resume=True, verbose=False)
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_train", counted)
    run_experiment(c, tmp_path / "split", modes=("sequential",), resume=True, verbose=False)
    assert calls == [1, 2]
    run_experiment(c, tmp_path / "full", modes=("sequential",), verbose=False)
    a, b = [
        load_sequence(tmp_path / p / "seed_42/sequential/boundary.pt", "cpu")
        for p in ("full", "split")
    ]
    assert_tree(a["result"].network.state_dict(), b["result"].network.state_dict())


def test_pmo_reserved_goal_slots_preserve_task_prefix():
    from tbtrl.benchmarks.pmo import make_tasks

    original = make_tasks(["QED", "Albuterol_Similarity"], goal_capacity=128)
    extended = make_tasks(
        ["QED", "Albuterol_Similarity", "Mestranol_Similarity"], goal_capacity=128
    )
    assert original == extended[:2]
    assert all(len(t["goal"]) == 128 for t in extended)
    with pytest.raises(ValueError, match="goal_capacity"):
        make_tasks(["QED", "Albuterol_Similarity"], goal_capacity=1)


def test_controlled_boundary_repairs_receipt_without_repeating_oracle_calls(tmp_path, monkeypatch):
    from test_discovery_study import TASKS, factory
    from test_discovery_study import config as candidate_config

    from tbtrl.benchmarks import study

    ledger, failed = [], [False]
    original = study._atomic_torch

    def interrupted_receipt(path, value):
        if path.name == "complete.pt" and not failed[0]:
            failed[0] = True
            raise OSError("simulated receipt interruption")
        return original(path, value)

    monkeypatch.setattr(study, "_atomic_torch", interrupted_receipt)
    kwargs = dict(
        config=candidate_config(), seeds=[7], methods=["sequential"], checkpoint_campaigns=False
    )
    with pytest.raises(OSError, match="receipt interruption"):
        study.run_study(TASKS, factory(ledger), tmp_path, **kwargs)
    assert len(ledger) == 8
    study.run_study(TASKS, factory(ledger), tmp_path, **kwargs)
    assert len(ledger) == 16  # The first task committed before its receipt failed.
    assert len(list(tmp_path.rglob("boundary.pt"))) == 1
    for path in tmp_path.glob("seed_7/sequential/task_*/complete.pt"):
        assert "groups" not in study._load(path)["learner"]
    assert study._load(tmp_path / "seed_7/sequential/boundary.pt")["learner"]["groups"]


def test_append_plot_uses_saved_prefix_and_marks_unmeasured_goals(tmp_path):
    import matplotlib.pyplot as plt

    from tbtrl.experiments.plots import plot_run

    c = config("dqn", False)
    prefix = deepcopy(c)
    prefix.goals = c.goals[:2]
    run_experiment(prefix, tmp_path / "run", modes=("sequential",), resume=True, verbose=False)
    run_experiment(c, tmp_path / "run", modes=("sequential",), resume=True, verbose=False)
    fig = plot_run(tmp_path / "run", export=False)
    matrix = fig.axes[1].images[0].get_array()
    assert matrix.shape == (3, 3)
    assert np.ma.getmaskarray(matrix)[0, 2]
    assert not np.ma.getmaskarray(matrix)[2, 2]
    plt.close(fig)
