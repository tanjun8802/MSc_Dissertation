"""Candidate TD/replay contracts plus optional native benchmark integration tests."""

import json
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from tbtrl.benchmarks.candidates import (
    CandidateConfig,
    CandidateLearner,
    CandidateMemory,
    Transition,
    run_experiment,
)
from tbtrl.benchmarks.pmo import top_auc


def config(**kwargs):
    return replace(
        CandidateConfig(
            hidden_dim=8,
            rep_dim=2,
            batch_size=2,
            warmup=2,
            capacity=12,
            recent_capacity=6,
            train_frequency=1,
            byte_budget=8192,
            max_groups=2,
        ),
        **kwargs,
    )


def transition(task=0, count=2, done=False):
    return Transition(
        task,
        np.ones(2, np.float32),
        np.ones(3, np.float32),
        1.0,
        np.ones(2, np.float32),
        np.ones((count, 3), np.float32),
        done,
    )


def test_td_uses_original_labels_ragged_candidates_and_terminal_mask():
    learner = CandidateLearner(2, 3, {0: [1, 0], 1: [0, 1]}, config(), 42)
    for network in (learner.network, learner.target):
        with torch.no_grad():
            for p in network.parameters():
                p.zero_()
            network.sa_encoder[-1].bias.fill_(2)
            network.goal_encoder[-1].bias.fill_(1)
    terminal = replace(transition(done=True), next_actions=np.empty((0, 3), np.float32))
    continuing = transition(task=1, count=5)
    # Q=4. Terminal target=1, continuing target=1+4. MSE=(9+1)/2.
    assert float(learner.td_loss([terminal, continuing])[0].detach()) == pytest.approx(5.0)
    permuted = continuing.next_actions[::-1].copy()
    assert learner.td_loss([replace(continuing, next_actions=permuted)])[0].item() == pytest.approx(
        1.0
    )
    with pytest.raises(ValueError, match="no next candidates"):
        learner.td_loss([replace(terminal, done=False)])


def test_replay_stays_bounded_discards_individual_buffer_and_preserves_labels():
    cfg = config(byte_budget=2048)
    goals = {i: [1.0, 0.0] for i in range(20)}  # force similarity grouping
    learner = CandidateLearner(2, 3, goals, cfg, 42)
    memory = CandidateMemory(cfg)
    for task in range(20):
        memory.begin_task()
        for _ in range(10):
            memory.add(transition(task, count=3))
        memory.finish_task(task, learner.network, goals, "cpu")
        assert not memory.recent
        assert memory.stats()["payload_bytes"] <= cfg.byte_budget
        assert len(memory.archive) <= cfg.capacity - cfg.recent_capacity
        assert len(memory.groups) <= cfg.max_groups
        assert all(t.reward == 1.0 for t in memory.archive)
    assert memory.archive[-1].task == 19
    assert len({t.task for t in memory.archive}) > 1  # group membership never replaces task labels
    with pytest.raises(ValueError, match="exceeds"):
        memory.add(transition(count=1000))


def test_capacity_archive_does_not_discard_similar_tasks_before_full():
    cfg = config(
        capacity=100,
        recent_capacity=10,
        byte_budget=1024**2,
        archive_strategy="capacity_only",
    )
    goals = {i: [1.0, 0.0] for i in range(3)}
    learner = CandidateLearner(2, 3, goals, cfg, 42)
    for task in goals:
        for _ in range(10):
            learner.memory.add(transition(task))
        learner.memory.finish_task(task, learner.network, goals, "cpu")
        assert learner.memory.admission["admitted_candidates"] == 4
        assert learner.memory.admission["evicted_for_capacity"] == 0
        assert not learner.memory.recent
    assert learner.memory.stats()["original_task_counts"] == {0: 4, 1: 4, 2: 4}
    assert len(learner.memory.groups) == 1  # grouping is not premature deletion


@pytest.mark.parametrize("byte_budget", [8192, 400])
def test_capacity_archive_thins_only_under_pressure_and_preserves_task_labels(byte_budget):
    cfg = config(archive_strategy="capacity_only", keep_fraction=1, byte_budget=byte_budget)
    goals = {i: [1.0, 0.0] for i in range(8)}
    learner = CandidateLearner(2, 3, goals, cfg, 42)
    for task in goals:
        for _ in range(6):
            learner.memory.add(transition(task, count=3))
        learner.memory.finish_task(task, learner.network, goals, "cpu")
        archive = learner.memory.archive
        assert len(archive) <= 6
        assert sum(t.nbytes for t in archive) <= byte_budget // 2
        assert archive[-1].task == task
        assert all(t.reward == 1 and not t.done for t in archive)
    assert learner.memory.admission["evicted_for_capacity"] > 0


def test_balanced_replay_samples_original_tasks_even_inside_one_similarity_group():
    learner = CandidateLearner(
        2, 3, {0: [1, 0], 1: [0, 1]}, config(sample_without_replacement=True), 42
    )
    pool = [transition(0) for _ in range(100)] + [transition(1)]
    batch = learner.sample(pool, 16, balanced=True)
    assert sum(t.task == 0 for t in batch) == sum(t.task == 1 for t in batch) == 8
    unique = learner.sample(pool, len(pool))
    assert len({id(t) for t in unique}) == len(pool)


def test_double_dqn_selects_online_but_evaluates_target_without_terminal_bootstrap():
    class LinearQ(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, states, actions, goals):
            return self.scale * actions[:, :1] * goals[..., :1]

    learner = CandidateLearner(2, 3, {0: [1, 0]}, config(), 42)
    learner.network, learner.target = LinearQ(1), LinearQ(-2)
    continuing = replace(transition(), next_actions=np.array([[1, 0, 0], [3, 0, 0]], np.float32))
    # Online selects action 3; target evaluates it as -6. Target=1-6=-5, Q=1.
    assert learner.td_loss([continuing])[0].item() == pytest.approx(36)
    terminal = replace(continuing, done=True, next_actions=np.empty((0, 3), np.float32))
    assert learner.td_loss([terminal])[0].item() == pytest.approx(0)


def test_regulariser_and_gradient_diagnostics_match_weighted_objective():
    cfg = config(
        covariance_target_variance=0.5,
        regularizer_batch_size=8,
        gradient_diagnostics=True,
        separation_target_cosine=-0.9,
        covariance_coef=0.1,
        separation_coef=1.0,
        replay_coef=0.5,
        balanced_replay=True,
        sample_without_replacement=True,
    )
    learner = CandidateLearner(2, 3, {0: [1, 0], 1: [0, 1]}, cfg, 42)
    learner.seen = learner.goals.copy()
    for i in range(6):
        learner.memory.add(replace(transition(1), state=np.array([i, -i], np.float32)))
    learner.memory.groups[0] = [transition(0)]
    control = CandidateLearner(2, 3, learner.goals, replace(cfg, gradient_diagnostics=False), 42)
    control.seen = learner.seen.copy()
    control.memory.recent = list(learner.memory.recent)
    control.memory.groups = {0: list(learner.memory.groups[0])}
    previous_target = [p.detach().clone() for p in learner.target.parameters()]
    result = learner.update(True)
    control.update(True)
    for measured, unmeasured in zip(learner.network.parameters(), control.network.parameters()):
        torch.testing.assert_close(measured, unmeasured, rtol=0, atol=0)
    assert all(np.isfinite(v) for v in result.values())
    assert result["weighted_replay"] == pytest.approx(0.5 * result["replay"])
    assert result["weighted_covariance"] == pytest.approx(0.1 * result["covariance"])
    assert result["total"] == pytest.approx(
        sum(v for k, v in result.items() if k.startswith("weighted_"))
    )
    assert result["grad_td"] > 0 and result["grad_replay"] > 0
    assert result["grad_separation"] > 0 and result["grad_covariance"] > 0
    for before, target, online in zip(
        previous_target, learner.target.parameters(), learner.network.parameters()
    ):
        torch.testing.assert_close(target, before.lerp(online, cfg.tau))


def test_oracle_output_is_scoped_and_does_not_hide_exceptions(tmp_path, capsys):
    from tbtrl.benchmarks.made import CountingOracle

    class NoisyOracle:
        def evaluate(self, structure):
            print("FIRE: 0 energy fmax")
            print("oracle warning", file=sys.stderr)
            if structure == "bad":
                raise RuntimeError("oracle failed")
            return {"energy": 1.0}

        def compute_elemental_energies_from_structures(self, entries):
            for entry in entries:
                self.evaluate(entry)
            return {"Li": 1.0}

    path = tmp_path / "oracle.log"
    oracle = CountingOracle(NoisyOracle(), path)
    assert oracle.evaluate("ok")["energy"] == 1
    oracle.compute_elemental_energies_from_structures(["Li", "O"])
    with pytest.raises(RuntimeError, match="oracle failed"):
        oracle.evaluate("bad")
    print("Training step=1")
    output = capsys.readouterr()
    assert output.out == "Training step=1\n" and output.err == ""
    assert path.read_text().count("FIRE:") == 4
    assert "oracle warning" in path.read_text()
    assert oracle.evaluations == 4


def test_native_fire_output_goes_to_file(tmp_path, capsys):
    pytest.importorskip("made")
    from made.oracles import AnalyticOracle
    from pymatgen.core import Lattice, Structure

    from tbtrl.benchmarks.made import CountingOracle

    native = AnalyticOracle("lj", relax=True, relax_kwargs={"steps": 2, "relax_unit_cell": False})
    path = tmp_path / "oracle.log"
    oracle = CountingOracle(native, path)
    capsys.readouterr()  # constructor notices are outside the evaluation scope
    result = oracle.evaluate(Structure(Lattice.cubic(4), ["Li", "Li"], [[0, 0, 0], [0.3, 0, 0]]))
    assert np.isfinite(result["energy"])
    assert "FIRE:" in path.read_text()
    assert "FIRE:" not in capsys.readouterr().out


def test_pmo_auc_matches_upstream_log_frequency_convention():
    assert top_auc([0.2, 0.6, 0.8, 1.0], 4, frequency=2, top_n=1) == pytest.approx(0.55)
    assert top_auc([0.5], 10, finish=True, top_n=10) == pytest.approx(0.475)
    assert top_auc([], 10) == 0


def test_incremental_auc_matches_official_prefix_calculation(monkeypatch):
    pytest.importorskip("rdkit")
    from tbtrl.benchmarks.pmo import MolecularEnvironment

    # Fake only canonicalisation to test 250 distinct labels without building large molecules.
    monkeypatch.setattr("tbtrl.benchmarks._vendor.moldqn.canonicalize", lambda s: s)
    scores = np.random.default_rng(7).random(250).tolist()
    env = MolecularEnvironment("test", budget=250, evaluator=lambda s: scores[int(s)])
    for i in range(250):
        env._score(str(i))
        assert env.metrics()["top10_auc"] == pytest.approx(top_auc(scores[: i + 1], 250))


def test_pmo_native_edits_cache_budget_and_all_default_oracles(tmp_path):
    pytest.importorskip("rdkit")
    pytest.importorskip("tdc")
    from tbtrl.benchmarks.pmo import DEFAULT_ORACLES, MolecularEnvironment, load_oracle

    for name in DEFAULT_ORACLES:
        assert np.isfinite(load_oracle(name)("CCO"))
    assert "rdkit.six" not in sys.modules  # compatibility alias does not leak
    calls = []
    env = MolecularEnvironment(
        "test", budget=2, evaluator=lambda smiles: calls.append(smiles) or 0.5
    )
    state, candidates = env.reset()
    assert state.shape == (1026,) and candidates.shape[1] == 1024
    for _ in range(2):
        env.step(env.candidates.index("C"))
    assert len(calls) == 1  # no-op revisits are free in native PMO
    _, candidates, _, done = env.step(env.candidates.index("CC"))
    assert done and env.finished and len(calls) == 2 and len(candidates) == 0
    env.save(tmp_path)
    assert len(json.loads((tmp_path / "molecules.json").read_text())) == 2


def test_candidate_runner_native_pmo_pairing_and_replay(tmp_path):
    pytest.importorskip("rdkit")
    from tbtrl.benchmarks.pmo import MolecularEnvironment

    tasks = [dict(name=f"task-{i}", goal=[float(i == j) for j in range(2)]) for i in range(2)]
    calls = []

    def factory(task, seed):
        return MolecularEnvironment(
            task["name"],
            budget=6,
            episode_length=3,
            max_steps=30,
            evaluator=lambda s: calls.append(s) or len(s) / 30,
        )

    records = run_experiment(
        tasks, factory, tmp_path / "run", config=config(byte_budget=4 * 1024**2), seeds=[1]
    )
    assert len(records) == 4
    scratch, sequential = records[0], records[2]
    assert scratch["history"] == sequential["history"]
    assert scratch["losses"] == sequential["losses"]
    assert all(r["losses"] for r in records)
    assert all(r["replay_memory"]["archive_samples"] == 0 for r in records[:2])
    assert any(row["replay"] > 0 for row in records[-1]["losses"])
    assert len(calls) == sum(
        r["final"]["queries"] for r in records
    )  # training/replay spend no extra calls


def test_made_native_budget_reward_and_reference_isolation(tmp_path):
    pytest.importorskip("made")
    from made.oracles import AnalyticOracle

    from tbtrl.benchmarks.made import MaterialsEnvironment, smoke_dataset

    env = MaterialsEnvironment(
        smoke_dataset(),
        AnalyticOracle("lj", relax=False),
        budget=3,
        campaigns=1,
        pool_size=8,
        max_stoichiometry=4,
        start_with_all_stable=False,
        filter_by_smact=False,
    )
    state, actions = env.reset()
    assert state.shape == (148,) and actions.shape[1] == 146
    reference_calls = env.oracle.evaluations
    # Fail if the learner's state/reward accesses evaluation-only ground truth metrics.
    env.env.get_latest_metrics = lambda: (_ for _ in ()).throw(AssertionError("evaluation-only"))
    state2, actions2 = env._observation()
    np.testing.assert_allclose(state2, state)
    np.testing.assert_allclose(actions2, actions)
    del env.env.get_latest_metrics  # save native metrics at campaign end
    total = 0
    for _ in range(3):
        _, _, reward, done = env.step(0)
        total += reward
    assert done and env.finished
    assert total == env.metrics()["discoveries"]
    assert env.oracle.evaluations == reference_calls + 3
    env.save(tmp_path)
    assert len(list(tmp_path.glob("*.cif"))) == 3
    assert json.loads((tmp_path / "native-metrics.json").read_text())
