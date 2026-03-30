"""
random_agent.py
===============
Uniformly-random exploration agent.

This agent selects actions uniformly at random from the discrete action set.
It serves as a lower-bound baseline and a useful sanity-check to verify that
the environment and experiment infrastructure are working correctly.

In reward-free RL it also models the simplest possible exploration policy,
whose coverage can be compared against more sophisticated intrinsic-motivation
approaches (count-based, curiosity-driven, skill-discovery, etc.).
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects actions uniformly at random.

    No learning takes place; :meth:`update` is a no-op.

    Parameters
    ----------
    n_actions :
        Number of discrete actions available.
    seed :
        Random seed for reproducibility.
    """

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        super().__init__(n_actions=n_actions, gamma=0.0, seed=seed)

    def select_action(self, observation: Any) -> int:
        """Return a uniformly-random action index."""
        self._increment_step()
        return int(self.np_random.integers(0, self.n_actions))

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
        """No-op: random agent does not learn."""
        return {}
