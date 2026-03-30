"""
base_agent.py
=============
Abstract interface for all RL agents.

Every agent in this codebase inherits from :class:`BaseAgent` and must
implement:
  * :meth:`select_action` — given an observation, return an action.
  * :meth:`update`        — incorporate a new experience tuple.
  * :meth:`reset`         — reset agent state between episodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAgent(ABC):
    """Abstract RL agent.

    Parameters
    ----------
    n_actions :
        Number of discrete actions |A|.
    gamma :
        Discount factor γ ∈ [0, 1).
    seed :
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_actions: int,
        gamma: float = 0.99,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.np_random = np.random.default_rng(seed)
        self._total_steps: int = 0
        self._total_episodes: int = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def select_action(self, observation: Any) -> Any:
        """Choose an action given an observation.

        Parameters
        ----------
        observation :
            The current environment observation.

        Returns
        -------
        action :
            The selected action.
        """

    @abstractmethod
    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> dict:
        """Update the agent's internal state from one experience tuple.

        Returns
        -------
        metrics : dict
            Dictionary of training metrics (e.g. ``{"loss": 0.42}``).
            Return an empty dict if no update is performed this step.
        """

    def reset(self) -> None:
        """Reset episode-level agent state (called at the start of each episode)."""
        self._total_episodes += 1

    # ------------------------------------------------------------------
    # Book-keeping helpers
    # ------------------------------------------------------------------

    def _increment_step(self) -> None:
        """Increment the global step counter."""
        self._total_steps += 1

    @property
    def total_steps(self) -> int:
        """Total environment steps taken across all episodes."""
        return self._total_steps

    @property
    def total_episodes(self) -> int:
        """Total number of episodes completed."""
        return self._total_episodes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_actions={self.n_actions}, gamma={self.gamma})"
        )
