"""
replay_buffer.py
================
Experience replay buffer with uniform random sampling.

A replay buffer stores transitions (s, a, r, s', done) collected during
environment interaction and provides mini-batch samples for off-policy
learning algorithms (DQN, SAC, TD3, etc.).

In reward-free RL the buffer also acts as the raw data store for the
planning phase where transitions are relabelled with a task reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    """Mini-batch of transitions returned by :meth:`ReplayBuffer.sample`."""

    observations: np.ndarray       # (B, *obs_shape)
    actions: np.ndarray            # (B, *act_shape)
    rewards: np.ndarray            # (B,)
    next_observations: np.ndarray  # (B, *obs_shape)
    terminated: np.ndarray         # (B,) bool
    truncated: np.ndarray          # (B,) bool


class ReplayBuffer:
    """Fixed-capacity circular experience replay buffer.

    Supports uniform random sampling of mini-batches.

    Parameters
    ----------
    capacity :
        Maximum number of transitions to store.  When the buffer is full,
        the oldest transitions are overwritten.
    obs_shape :
        Shape of a single observation (e.g. ``(84, 84, 3)`` for images or
        ``(4,)`` for CartPole).
    action_shape :
        Shape of a single action (e.g. ``()`` for a discrete scalar or
        ``(2,)`` for a 2-D continuous action).
    seed :
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self._rng = np.random.default_rng(seed)

        self._observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=bool)
        self._truncated = np.zeros(capacity, dtype=bool)

        self._ptr: int = 0       # write pointer (circular)
        self._size: int = 0      # current number of stored transitions

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray | int | float,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store one transition in the buffer.

        Parameters
        ----------
        obs, next_obs :
            Observations before and after the step.
        action :
            Action taken.
        reward :
            Scalar reward received.
        terminated, truncated :
            Episode termination / truncation flags.
        """
        self._observations[self._ptr] = np.asarray(obs, dtype=np.float32)
        self._actions[self._ptr] = np.asarray(action, dtype=np.float32)
        self._rewards[self._ptr] = float(reward)
        self._next_observations[self._ptr] = np.asarray(next_obs, dtype=np.float32)
        self._terminated[self._ptr] = bool(terminated)
        self._truncated[self._ptr] = bool(truncated)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Sample a mini-batch of transitions uniformly at random.

        Parameters
        ----------
        batch_size :
            Number of transitions to sample.

        Returns
        -------
        batch : Batch
            Named tuple containing arrays of shape (batch_size, …).

        Raises
        ------
        ValueError
            If the buffer contains fewer transitions than ``batch_size``.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Not enough transitions to sample (have {self._size}, "
                f"need {batch_size})."
            )
        indices = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=self._next_observations[indices],
            terminated=self._terminated[indices],
            truncated=self._truncated[indices],
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """Return ``True`` once at least one transition has been stored."""
        return self._size > 0

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, "
            f"size={self._size}, "
            f"obs_shape={self.obs_shape})"
        )
