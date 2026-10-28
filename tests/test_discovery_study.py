"""Interrupt/restart equivalence, budget isolation and controlled-method contracts."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tbtrl.benchmarks.candidates import CandidateLearner
from tbtrl.benchmarks.study import (
    _load,
    made_study_config,
    method_config,
    run_recovery,
    run_retention,
    run_study,
)
from tbtrl.benchmarks.study_plots import (
    plot_recovery,
    plot_retention,
    plot_study,
    recovery_table,
)
from tbtrl.experiments.figure_style import FigureStyle

TASKS = [dict(name="A", goal=[1, 0]), dict(name="B", goal=[0, 1])]


def config():
    return replace(
        made_study_config(),
        hidden_dim=8,
        rep_dim=2,
        warmup=2,
        batch_size=2,
        regularizer_batch_size=4,
        capacity=30,
        recent_capacity=10,
        byte_budget=8192,
        covariance_target_variance=0.5,
        log_frequency=4,
    )


class ToyCampaigns:
    def __init__(self, seed, campaigns, ledger, interrupt=None, checkpoint=None):
        self.seed, self.campaigns, self.ledger, self.interrupt = seed, campaigns, ledger, interrupt
        self.campaign = self.total_queries = self.in_campaign = self.discoveries = 0
        self.campaign_metrics = []
        if checkpoint:
            self.__dict__.update(deepcopy(checkpoint))

    @property
    def finished(self):
        return self.total_queries >= self.campaigns * 4

    @property
    def progress(self):
        return self.total_queries / (self.campaigns * 4)

    def observation(self):
        return np.array([self.in_campaign / 4, self.discoveries / 4], np.float32), np.eye(
            2, dtype=np.float32
        )

    def reset(self):
        self.campaign += 1
        self.in_campaign = self.discoveries = 0
        return self.observation()

    def step(self, index):
        if self.interrupt and self.interrupt(self):
            raise RuntimeError("simulated interruption")
        self.ledger.append((self.seed, self.campaign, self.in_campaign, index))
        self.total_queries += 1
        self.in_campaign += 1
        reward = float(index == (self.seed + self.in_campaign) % 2)
        self.discoveries += reward
        done = self.in_campaign == 4
        if done:
            self.campaign_metrics.append(
                dict(
                    novelty_stable_unique_novel_count=self.discoveries,
                    discovery_curve_area_under_discovery_curve=self.discoveries * 2,
                )
            )
        state, actions = self.observation()
        return state, np.empty((0, 2), np.float32) if done else actions, reward, done

    def metrics(self):
        return dict(
            queries=self.total_queries,
            query_budget=self.campaigns * 4,
            campaign=self.campaign,
            campaign_queries=self.in_campaign,
            discoveries=self.discoveries,
            oracle_evaluations=self.total_queries,
            reference_evaluations=0,
        )

    def checkpoint(self):
        return {k: deepcopy(v) for k, v in vars(self).items() if k not in {"ledger", "interrupt"}}

    def save(self, path):
        (path / "benchmark.json").write_text(
            json.dumps(dict(campaign_metrics=self.campaign_metrics))
        )
        # Plotting uses per-query discoveries; synthetic monotone fixture only.
        rows = [
            dict(campaign=c, discoveries=i, queries=i)
            for c in range(1, self.campaign + 1)
            for i in range(1, 5)
        ]
        (path / "discovery.json").write_text(json.dumps(rows))


def factory(ledger, interrupt=None):
    def create(task, seed, *, checkpoint=None, campaigns=None):
        return ToyCampaigns(seed, campaigns or 2, ledger, interrupt, checkpoint)

    return create


def test_interrupted_campaign_resume_matches_uninterrupted_model_optimizer_and_replay(tmp_path):
    reference_calls, interrupted_calls = [], []
    arguments = dict(config=config(), seeds=[7], methods=["sequential"], checkpoint_campaigns=True)
    reference = run_study(TASKS, factory(reference_calls), tmp_path / "reference", **arguments)
    failed = [False]

    def interrupt(env):
        if env.seed == 7 and env.campaign == 2 and env.in_campaign == 1 and not failed[0]:
            failed[0] = True
            return True
        return False

    output = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_study(TASKS, factory(interrupted_calls, interrupt), output, **arguments)
    assert len(interrupted_calls) == 5
    result = run_study(TASKS, factory(interrupted_calls), output, **arguments)
    assert len(interrupted_calls) == len(reference_calls) + 1  # only unfinished work repeated
    for expected, actual in zip(reference, result):
        assert expected["history"] == actual["history"]
        assert expected["losses"] == actual["losses"]
        assert expected["replay_memory"] == actual["replay_memory"]
        first = _load(
            tmp_path / "reference" / f"seed_7/sequential/task_{expected['task_id']}/complete.pt"
        )["learner"]
        second = _load(output / f"seed_7/sequential/task_{actual['task_id']}/complete.pt")[
            "learner"
        ]
        for key in ("network", "target"):
            for name in first[key]:
                torch.testing.assert_close(first[key][name], second[key][name], rtol=0, atol=0)
        for parameter, values in first["optimizer"]["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(
                    value, second["optimizer"]["state"][parameter][key], rtol=0, atol=0
                )
    count = len(interrupted_calls)
    assert len(run_study(TASKS, factory(interrupted_calls), output, **arguments)) == 2
    assert (
        len(interrupted_calls) == count
    )  # no environment construction/evaluation for complete tasks
    with pytest.raises(ValueError, match="configuration differs"):
        run_study(TASKS, factory([]), output, **dict(arguments, config=replace(config(), lr=0.01)))
    with pytest.raises(FileExistsError):
        run_study(TASKS, factory([]), output, **arguments, resume=False)


def test_exploration_is_independent_of_replay_and_regularizer_draws():
    a = CandidateLearner(2, 2, {0: [1, 0]}, config(), 4)
    b = CandidateLearner(2, 2, {0: [1, 0]}, config(), 4)
    state = np.zeros(2, np.float32)
    actions = np.eye(2, dtype=np.float32)
    for _ in range(50):
        b.rng.random(100)
        b.regularizer_rng.random(100)
        assert a.choose(state, actions, 0, 1) == b.choose(state, actions, 0, 1)


def test_methods_only_change_intended_controls():
    base = made_study_config()
    assert method_config(base, "scratch") == method_config(base, "sequential")
    assert method_config(base, "no_covariance") == replace(base, covariance_coef=0)
    assert method_config(base, "no_separation") == replace(base, separation_coef=0)
    assert method_config(base, "small_archive") == replace(base, capacity=600)
    assert method_config(base, "fine_tuning") == replace(
        base, replay_coef=0, covariance_coef=0, separation_coef=0
    )
    assert method_config(base, "uniform_replay") == replace(
        base, balanced_replay=False, covariance_coef=0, separation_coef=0
    )


def test_retention_and_recovery_are_frozen_isolated_and_resumable(tmp_path):
    calls = []
    records = run_study(
        TASKS,
        factory(calls),
        tmp_path,
        config=config(),
        seeds=[42],
        methods=["scratch", "fine_tuning", "uniform_replay", "sequential"],
    )
    for record in records:
        if record["method"] in ("scratch", "fine_tuning"):
            assert record["replay_memory"]["archive_samples"] == 0
            assert all(row["replay"] == 0 for row in record["losses"])
    original = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.glob("seed_*/*/task_*/complete.pt")
    }
    count = len(calls)
    retention = run_retention(tmp_path, factory(calls), campaigns=1)
    assert len(retention) == 11  # scratch two; three sequential methods with 1+2 evaluations each
    assert len(calls) - count == 11 * 4
    assert all(
        not r["losses"] and all(row["epsilon"] == 0 for row in r["history"]) for r in retention
    )
    for record in retention:
        parent = tmp_path / f"seed_42/{record['method']}/task_{record['after_task']}/complete.pt"
        before = _load(parent)["learner"]
        after = _load(Path(record["path"]) / "complete.pt")["learner"]
        for name in before["network"]:
            torch.testing.assert_close(
                before["network"][name], after["network"][name], rtol=0, atol=0
            )
    count = len(calls)
    recovered, probes = run_recovery(tmp_path, factory(calls), campaigns=1, evaluation_campaigns=1)
    assert len(recovered) == 3 and len(probes) == 12
    assert len(calls) - count == (3 + 12) * 4
    assert all(not r["losses"] for r in probes)
    for path, digest in original.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    count = len(calls)
    run_retention(tmp_path, factory(calls), campaigns=1)
    run_recovery(tmp_path, factory(calls), campaigns=1, evaluation_campaigns=1)
    assert len(calls) == count
    table = recovery_table(probes, retention)
    assert len(table) == 3 and all(
        r["status"] in ("recovered", "not reached within budget", "reference unavailable/zero")
        for r in table
    )
    style = FigureStyle(dpi=50, formats=("png",))
    plot_study(records, tmp_path, style=style, display=False)
    plot_retention(retention, tmp_path, style=style, display=False)
    plot_recovery(probes, tmp_path, style=style, display=False)
    assert len(list(tmp_path.glob("*.png"))) == 10


def test_native_made_checkpoint_reconstructs_next_campaign_without_oracle_calls(tmp_path):
    pytest.importorskip("made")
    from made.oracles import AnalyticOracle

    from tbtrl.benchmarks.made import MaterialsEnvironment, smoke_dataset

    dataset = smoke_dataset()
    env = MaterialsEnvironment(
        dataset,
        AnalyticOracle("lj", relax=False),
        budget=2,
        campaigns=2,
        pool_size=6,
        max_stoichiometry=4,
        start_with_all_stable=False,
        filter_by_smact=False,
    )
    env.reset()
    with pytest.raises(ValueError, match="completed campaign"):
        env.checkpoint()
    env.step(0)
    env.step(0)
    checkpoint = env.checkpoint()
    from tbtrl.benchmarks.study import _atomic_torch

    _atomic_torch(tmp_path / "native.pt", checkpoint)
    restored = MaterialsEnvironment.from_checkpoint(
        _load(tmp_path / "native.pt"), dataset, AnalyticOracle("lj", relax=False)
    )
    assert restored.metrics() == env.metrics()
    first = env.reset()
    second = restored.reset()
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    for _ in range(2):
        a = env.step(0)
        b = restored.step(0)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])
        assert a[2:] == b[2:]
    assert restored.metrics() == env.metrics()
