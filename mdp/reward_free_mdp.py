"""
reward_free_mdp.py
==================
Reward-free MDP wrapper.

In reward-free RL the agent explores without access to a task-specific reward
signal during training.  Rewards are revealed *after* exploration and used to
relabel stored transitions offline (the "planning phase").

This wrapper:
  1. Zeros out reward signals passed from the underlying environment.
  2. Stores every (s, a, s') transition in an internal buffer for offline
     relabelling.
  3. Exposes :meth:`relabel` to retroactively assign a reward function and
     return relabelled transitions.

References
----------
* Jin et al. (2020) "Reward-Free Exploration for Reinforcement Learning"
* Zhang et al. (2020) "Task-Agnostic Exploration in Policy Space"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from mdp.base_mdp import BaseMDP


@dataclass
class Transition:
    """Single (s, a, s') experience tuple."""

    state: Any
    action: Any
    next_state: Any
    original_reward: float  # stored but not returned to the agent
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


class RewardFreeMDP(BaseMDP):
    """Wraps another MDP and suppresses its reward signal.

    The agent interacts with this wrapper in exactly the same way as a
    standard MDP, but it always receives ``reward = 0.0``.

    Parameters
    ----------
    mdp :
        The underlying MDP (must implement :class:`BaseMDP`).
    buffer_capacity :
        Maximum number of transitions to retain in memory.
        Older transitions are evicted using a circular buffer.
    """

    def __init__(
        self,
        mdp: BaseMDP,
        buffer_capacity: int = 100_000,
    ) -> None:
        super().__init__(gamma=mdp.gamma)
        self._mdp = mdp
        self.buffer_capacity = buffer_capacity
        self._buffer: list[Transition] = []
        self._buffer_ptr: int = 0  # circular write pointer

    # ------------------------------------------------------------------
    # BaseMDP interface — delegate to inner MDP
    # ------------------------------------------------------------------

    @property
    def n_states(self) -> int:
        return self._mdp.n_states

    @property
    def n_actions(self) -> int:
        return self._mdp.n_actions

    def transition(self, state: Any, action: Any) -> Any:
        """Sample next state from the inner MDP."""
        return self._mdp.transition(state, action)

    def reward(
        self,
        state: Any,
        action: Any,
        next_state: Any,
    ) -> float:
        """Always returns 0.0 — reward is withheld during exploration."""
        return 0.0

    def is_terminal(self, state: Any) -> bool:
        return self._mdp.is_terminal(state)

    # ------------------------------------------------------------------
    # Exploration API
    # ------------------------------------------------------------------

    def step_and_record(
        self,
        state: Any,
        action: Any,
        terminated: bool = False,
        truncated: bool = False,
        info: dict | None = None,
    ) -> tuple[Any, float, bool, bool, dict]:
        """Execute one step, record the transition, and return reward=0.

        Parameters
        ----------
        state :
            Current state before the transition.
        action :
            Action selected by the exploration policy.
        terminated, truncated :
            Episode termination flags from the environment step.
        info :
            Auxiliary info dict from the environment.

        Returns
        -------
        next_state, reward=0.0, terminated, truncated, info
        """
        next_state = self._mdp.transition(state, action)
        original_reward = self._mdp.reward(state, action, next_state)

        t = Transition(
            state=state,
            action=action,
            next_state=next_state,
            original_reward=original_reward,
            terminated=terminated,
            truncated=truncated,
            info=info or {},
        )
        self._store(t)

        return next_state, 0.0, terminated, truncated, info or {}

    # ------------------------------------------------------------------
    # Relabelling (planning phase)
    # ------------------------------------------------------------------

    def relabel(
        self,
        reward_fn: Callable[[Any, Any, Any], float],
    ) -> list[Transition]:
        """Relabel stored transitions with a user-supplied reward function.

        Parameters
        ----------
        reward_fn :
            A callable ``(state, action, next_state) → float`` that
            implements the task-specific reward.

        Returns
        -------
        list[Transition]
            New Transition objects with ``original_reward`` set to the
            relabelled reward value.
        """
        relabelled = []
        for t in self._buffer:
            r = reward_fn(t.state, t.action, t.next_state)
            relabelled.append(
                Transition(
                    state=t.state,
                    action=t.action,
                    next_state=t.next_state,
                    original_reward=r,
                    terminated=t.terminated,
                    truncated=t.truncated,
                    info=t.info,
                )
            )
        return relabelled

    @property
    def buffer_size(self) -> int:
        """Current number of stored transitions."""
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store(self, transition: Transition) -> None:
        """Insert a transition into the circular buffer."""
        if len(self._buffer) < self.buffer_capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._buffer_ptr] = transition
        self._buffer_ptr = (self._buffer_ptr + 1) % self.buffer_capacity
