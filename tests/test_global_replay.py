from pathlib import Path

import numpy as np
import pytest
import torch

from tbtrl.environments.registry import make_environment
from tbtrl.experiments.config import load_config
from tbtrl.experiments.runner import run_experiment
from tbtrl.global_replay import GlobalReplayMemory
from tbtrl.replay import ReplayBuffer

ROOT = Path(__file__).resolve().parents[1]


def collect(memory, offset=0, count=20):
    recent = memory.begin_task()
    for i in range(offset, offset + count):
        recent.add_transition([i, 0], [0.25, -0.1], 0, [i + 1, 0], False)


def test_global_budget_covers_recent_and_archive_across_many_novel_tasks():
    memory = GlobalReplayMemory(30, 10, 2, 2, max_groups=3)
    pointers = [getattr(memory.storage, name).ctypes.data for name in ReplayBuffer.fields]
    for task in range(100):
        collect(memory, 100 * task)
        # Every new task is dissimilar; bounded groups must consolidate repeatedly.
        memory.finish_task(task, np.eye(len(memory.representatives) + 1))
        stats = memory.stats()
        assert stats["recent_samples"] + stats["archive_samples"] <= 30
        assert stats["allocated_transition_bytes"] == 30 * 9 * 4
        assert stats["groups"] <= 3
        assert len(memory.buffers) == task + 1
        assert pointers == [
            getattr(memory.storage, name).ctypes.data for name in ReplayBuffer.fields
        ]
        for buffer in memory.buffers.values():
            assert np.shares_memory(buffer.obs, memory.storage.obs)
            assert len(buffer) > 0
            assert buffer.sample(32).obs.shape == (32, 2)
    assert memory.last_admission["merged_groups"] is not None


def test_tail_selection_similar_tasks_share_and_recovery_does_not_mutate_source():
    memory = GlobalReplayMemory(40, 10, 2, 2, max_groups=3)
    collect(memory)
    memory.finish_task(0, [[1]])
    np.testing.assert_equal(memory.buffers[0].obs[:, 0], [16, 17, 18, 19])
    collect(memory, 100)
    memory.finish_task(1, [[1, 1], [1, 1]])
    assert memory.buffers[0] is memory.buffers[1]
    np.testing.assert_equal(memory.buffers[0].obs[:, 0], [16, 17, 18, 19])
    collect(memory, 200)
    memory.finish_task(2, [[1, 0.9], [0.9, 1]])
    assert memory.buffers[2] is memory.buffers[0]
    assert 219 in memory.buffers[0].obs[:, 0]
    fork = memory.fork()
    assert fork.buffers[0] is fork.buffers[1]
    assert np.shares_memory(fork.buffers[0].obs, fork.storage.obs)
    assert not np.shares_memory(fork.storage.obs, memory.storage.obs)
    fork.buffers[0].obs[:] = -999
    assert (memory.buffers[0].obs[:, 0] >= 16).all()


def test_group_limit_merges_most_similar_old_pair_and_preserves_all_task_views():
    memory = GlobalReplayMemory(40, 10, 2, 2, max_groups=3)
    for task in range(3):
        collect(memory, task * 100)
        memory.finish_task(task, np.eye(task + 1))
    collect(memory, 300)
    similarities = np.eye(4)
    similarities[1, 2] = similarities[2, 1] = 0.95
    memory.finish_task(3, similarities)
    assert memory.last_admission["merged_groups"] == [1, 2]
    assert memory.buffers[1] is memory.buffers[2]
    assert memory.buffers[0] is not memory.buffers[1]
    assert memory.buffers[3] is not memory.buffers[1]
    assert set(memory.task_groups) == {0, 1, 2, 3}


@pytest.mark.parametrize("mode", ["simple", "shaped"])
def test_continuous_relabelling_matches_environment_and_ignores_stored_labels(mode):
    env = make_environment("maze-continuous", [3.5, 1.5], reward_mode=mode)
    buffer = ReplayBuffer(4, 2, 2)
    transitions = [
        ([1.5, 1.5], [1.5, 1.5]),  # blocked
        ([1.5, 1.5], [1.8, 1.5]),  # moving
        ([3.0, 1.5], [3.25, 1.5]),  # within goal radius
        ([7.0, 7.0], [7.25, 7.25]),
    ]
    for state, following in transitions:
        buffer.add_transition(state, [0.25, 0.0], 999, following, True, True)
    # Fixed ordered batch, including the radius boundary for the second goal.
    from tbtrl.replay import ReplayBatch

    batch = ReplayBatch(**{name: torch.as_tensor(getattr(buffer, name)) for name in buffer.fields})
    for goal in ([3.5, 1.5], [7.5, 7.5]):
        reward, terminated = env.relabel_transitions(batch, goal)
        compute = env.compute_simple_reward if mode == "simple" else env.compute_shaped_reward
        expected = [
            compute(s, a, sp, goal) for s, a, sp in zip(buffer.obs, buffer.actions, buffer.next_obs)
        ]
        np.testing.assert_allclose(reward.numpy().ravel(), expected, atol=1e-6)
        np.testing.assert_equal(
            terminated.numpy().ravel(),
            np.linalg.norm(buffer.next_obs - goal, axis=-1) <= env.goal_radius,
        )
    assert (batch.truncated == 1).all() and (batch.rewards == 999).all()
    env.close()


@pytest.mark.parametrize("algorithm", ["dqn", "sac_factorised_actor"])
def test_global_experiment_trains_replays_recovers_and_reports_memory(algorithm, tmp_path):
    config = load_config(ROOT / f"configs/tbtrl_gridworldmaze_{algorithm}.json").smoke()
    config.replay_mode = "global"
    config.global_replay_capacity, config.recent_replay_capacity = 20, 8
    config.replay_max_groups, config.similarity_threshold = 2, 0.8
    config.goals = config.goals + config.goals[:1]
    if config.algorithm == "sac":
        config.training.update(normalize_state_inputs=True, normalize_goal_inputs=True)
    rows = run_experiment(config, tmp_path / algorithm, verbose=False)
    assert len(rows) == 9
    for row in rows:
        stats = row["replay_memory"]
        assert stats["capacity"] == 20
        assert stats["archive_samples"] + stats["recent_samples"] <= 20
    sequence = [row for row in rows if row["mode"] == "sequential"]
    assert sequence[-1]["replay_memory"]["tasks"] == 3
    recovery = [row for row in rows if row["mode"] == "recovery"]
    assert all(
        row["replay_memory"]["archive_samples"] == sequence[-1]["replay_memory"]["archive_samples"]
        for row in recovery
    )
    assert "Global replay:" in (tmp_path / algorithm / "training.log").read_text()


def test_global_config_rejects_budget_that_prevents_updates():
    config = load_config(ROOT / "configs/tbtrl_gridworldmaze_sac.json").smoke()
    config.replay_mode, config.similarity_threshold = "global", 0.8
    config.recent_replay_capacity = 2
    with pytest.raises(ValueError, match="warmup"):
        config.validate()


def test_shared_sac_updates_ignore_wrong_stored_rewards_and_goal_termination(tmp_path, monkeypatch):
    config = load_config(ROOT / "configs/tbtrl_gridworldmaze_sac_factorised_actor.json").smoke()
    config.replay_mode, config.similarity_threshold = "global", 0.8
    config.global_replay_capacity, config.recent_replay_capacity = 20, 8
    config.replay_max_groups = 2
    config.training.update(normalize_state_inputs=True, normalize_goal_inputs=True)
    run_experiment(config, tmp_path / "correct", modes=("sequential",), verbose=False)
    original = ReplayBuffer.add_transition

    def wrong_labels(self, obs, action, reward, next_obs, terminated, truncated=False):
        return original(self, obs, action, 999.0, next_obs, not terminated, truncated)

    monkeypatch.setattr(ReplayBuffer, "add_transition", wrong_labels)
    run_experiment(config, tmp_path / "wrong", modes=("sequential",), verbose=False)
    relative = "seed_42/sequential/task_1/checkpoint.pt"
    expected = torch.load(tmp_path / "correct" / relative, weights_only=True)
    actual = torch.load(tmp_path / "wrong" / relative, weights_only=True)
    for component in ("actor", "critic", "target"):
        for key in expected[component]:
            torch.testing.assert_close(
                actual[component][key], expected[component][key], rtol=0, atol=0
            )
