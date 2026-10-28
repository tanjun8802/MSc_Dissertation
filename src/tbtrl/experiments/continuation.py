"""Full task-boundary state; evaluation-only checkpoint.pt stays backward compatible."""

from copy import copy

from tbtrl.continuation import atomic_torch, load_boundary, restore_rng, rng_state
from tbtrl.global_replay import GlobalReplayMemory
from tbtrl.replay import ReplayBuffer


def pack_memory(memory):
    if not isinstance(memory, GlobalReplayMemory):
        # Pickle preserves sharing between buffers aliased by task IDs.
        return memory
    # NumPy pickling does not preserve array-view aliasing. Persist the single
    # backing allocation and reconstruct every recent/archive view on restore.
    return dict(
        kind="global",
        attributes={
            k: v for k, v in vars(memory).items() if k not in ("storage", "recent", "buffers")
        },
        storage=memory.storage,
        recent_pos=memory.recent.pos,
        recent_size=memory.recent.size,
    )


def unpack_memory(saved):
    if not isinstance(saved, dict):
        return saved
    memory = GlobalReplayMemory.__new__(GlobalReplayMemory)
    memory.__dict__.update(saved["attributes"])
    memory.storage = saved["storage"]
    memory.recent = ReplayBuffer.view(
        memory.storage, 0, memory.recent_capacity, size=saved["recent_size"]
    )
    memory.recent.pos = saved["recent_pos"]
    memory._refresh_views()
    return memory


def save_sequence(path, *, result, memory, goals, completed):
    result = copy(result)
    # Current-task transitions already live in memory (or are no longer needed).
    result.buffer = None
    atomic_torch(
        path,
        dict(
            result=result,
            memory=pack_memory(memory),
            goals=goals,
            completed=completed,
            rng=rng_state(),
        ),
    )


def load_sequence(path, device):
    saved = load_boundary(path, device)
    saved["memory"] = unpack_memory(saved["memory"])
    restore_rng(saved["rng"])
    return saved
