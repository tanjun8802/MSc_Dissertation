import numpy as np
import pytest

from tbtrl.replay import ReplayBuffer, TaskReplayMemory


def filled(count=8, capacity=5, discrete=False):
    buffer = ReplayBuffer(capacity, 2, 1, discrete=discrete)
    for i in range(count):
        buffer.add_transition([i, 0], i % 4 if discrete else 0.25, i, [i + 1, 0], False, True)
    return buffer


def test_fractional_actions_survive_storage_and_sampling():
    buffer = filled()
    assert np.all(buffer.actions == 0.25)
    assert (buffer.sample(20).actions == 0.25).all()


def test_discrete_buffer_rejects_fractional_actions():
    with pytest.raises(ValueError, match="integers"):
        ReplayBuffer(4, 2, 1, discrete=True).add_transition([0, 0], 0.25, 0, [1, 0], False)


def test_tail_retention_after_wraparound():
    buffer = filled()
    np.testing.assert_equal(buffer.obs[buffer.chronological_indices(), 0], [3, 4, 5, 6, 7])
    tail = buffer.keep_last(0.6)
    np.testing.assert_equal(tail.obs[tail.chronological_indices(), 0], [5, 6, 7])
    buffer.append_from(tail, 0)
    np.testing.assert_equal(buffer.obs[buffer.chronological_indices(), 0], [3, 4, 5, 6, 7])


def test_identical_goals_share_existing_buffer_without_losing_task_mapping():
    memory = TaskReplayMemory(1, 0.8)
    memory.add(0, filled(discrete=True))
    before = memory.buffers[0].obs.copy()
    memory.add(1, filled(discrete=True), np.array([1.0]))
    assert memory.buffers[0] is memory.buffers[1]
    np.testing.assert_equal(before, memory.buffers[0].obs)
    memory.add(2, filled(discrete=True), np.array([0.4, 0.5]))
    assert memory.buffers[2] is not memory.buffers[0]


def test_similar_goals_merge_fraction_and_preserve_aliases():
    memory = TaskReplayMemory(1, 0.8)
    memory.add(0, filled(4, 4, True))
    memory.add(1, filled(12, 4, True), np.array([0.9]))
    assert memory.buffers[0] is memory.buffers[1]
    assert memory.buffers[0].obs[memory.buffers[0].chronological_indices()[-1], 0] == 11


def test_continuous_task_buffers_cannot_be_shared():
    memory = TaskReplayMemory(1, 0.8)
    memory.add(0, filled())
    with pytest.raises(ValueError, match="discrete"):
        memory.add(1, filled(), [1.0])
