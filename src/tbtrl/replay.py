"""Transition replay and bounded task memory used by the supported TBTRL trainers."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity ring buffer. Sampling is with replacement.

    The supported trainers consume single transitions, so episode metadata and
    future-goal sampling are deliberately outside this buffer's contract.
    """

    fields = ("obs", "actions", "rewards", "next_obs", "terminated", "truncated")

    def __init__(self, capacity, obs_dim, action_dim, device="cpu", *, discrete=False):
        if min(capacity, obs_dim, action_dim) < 1:
            raise ValueError("Buffer capacity and dimensions must be positive.")
        self.capacity, self.obs_dim, self.action_dim = capacity, obs_dim, action_dim
        self.device, self.discrete = device, discrete
        self.pos = self.size = 0
        for name in self.fields:
            width = (
                obs_dim if name in ("obs", "next_obs") else action_dim if name == "actions" else 1
            )
            dtype = np.int64 if name == "actions" and discrete else np.float32
            setattr(self, name, np.zeros((capacity, width), dtype=dtype))

    def __len__(self):
        return self.size

    def add_transition(self, obs, action, reward, next_obs, terminated, truncated=False):
        action = np.asarray(action).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError("Action shape does not match the replay buffer.")
        if self.discrete and not np.equal(action, np.floor(action)).all():
            raise ValueError("Discrete actions must be integers.")
        values = (obs, action, reward, next_obs, terminated, truncated)
        for name, value in zip(self.fields, values):
            getattr(self, name)[self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if not self.size or batch_size < 1:
            raise ValueError("Sampling requires a nonempty buffer and positive batch size.")
        indices = np.random.randint(0, self.size, size=batch_size)
        return ReplayBatch(
            **{
                name: torch.as_tensor(getattr(self, name)[indices], device=self.device)
                for name in self.fields
            }
        )

    def chronological_indices(self):
        return (self.pos - self.size + np.arange(self.size)) % self.capacity

    def keep_last(self, fraction):
        if not 0 < fraction <= 1:
            raise ValueError("Retained fraction must be in (0, 1].")
        retained = ReplayBuffer(
            self.capacity, self.obs_dim, self.action_dim, self.device, discrete=self.discrete
        )
        retained.append_from(self, fraction)
        return retained

    def append_from(self, other, fraction=1.0):
        if not 0 <= fraction <= 1:
            raise ValueError("Merged fraction must be in [0, 1].")
        if (self.obs_dim, self.action_dim, self.discrete) != (
            other.obs_dim,
            other.action_dim,
            other.discrete,
        ):
            raise ValueError("Cannot merge incompatible buffers.")
        if fraction == 0 or len(other) == 0:
            return
        count = max(1, int(len(other) * fraction))
        indices = other.chronological_indices()[-count:]
        # Copy first: self-merges must not overwrite yet-to-be-read transitions.
        arrays = [getattr(other, name)[indices].copy() for name in self.fields]
        for values in zip(*arrays):
            self.add_transition(*values)

    def state_dict(self):
        indices = self.chronological_indices()
        return {name: torch.from_numpy(getattr(self, name)[indices].copy()) for name in self.fields}


class TaskReplayMemory:
    """Own task buffers, optionally sharing DQN buffers by goal similarity.

    Shared buffers require reward relabelling by the receiving task's goal.
    SAC uses independent buffers containing their original task rewards.
    """

    def __init__(self, keep_fraction=0.4, similarity_threshold=None):
        if not 0 < keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1].")
        if similarity_threshold is not None and not -1 <= similarity_threshold < 1:
            raise ValueError("similarity_threshold must be in [-1, 1).")
        self.keep_fraction = keep_fraction
        self.similarity_threshold = similarity_threshold
        self.buffers = {}

    def add(self, task_id, buffer, similarities=None):
        retained = buffer.keep_last(self.keep_fraction)
        threshold = self.similarity_threshold
        if threshold is None or not self.buffers:
            self.buffers[task_id] = retained
            return
        if not buffer.discrete:
            raise ValueError("Similarity sharing requires discrete reward relabelling.")
        old_ids = sorted(self.buffers)
        if similarities is None or np.shape(similarities) != (len(old_ids),):
            raise ValueError("Supply similarities in sorted previous-task order.")
        if not np.isfinite(similarities).all():
            raise ValueError("Similarities must be finite.")
        best = int(np.argmax(similarities))
        similarity = float(np.clip(similarities[best], -1, 1))
        if similarity <= threshold:
            self.buffers[task_id] = retained
        else:
            shared = self.buffers[old_ids[best]]
            fraction = 0.0 if abs(similarity - 1.0) < 1e-6 else (1 - similarity) / (1 - threshold)
            shared.append_from(retained, fraction)
            self.buffers[task_id] = shared
