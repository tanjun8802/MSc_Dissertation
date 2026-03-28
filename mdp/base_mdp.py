"""
base_mdp.py
===========
Abstract interface for an MDP.

An MDP is formally defined as the tuple (S, A, P, R, γ) where:
  S : state space
  A : action space
  P : transition probability kernel  P(s' | s, a)
  R : reward function                R(s, a, s')
  γ : discount factor                γ ∈ [0, 1)

Subclasses implement the tabular and function-approximation variants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMDP(ABC):
    """Abstract Markov Decision Process.

    Parameters
    ----------
    gamma :
        Discount factor γ ∈ [0, 1).
    """

    def __init__(self, gamma: float = 0.99) -> None:
        if not (0.0 <= gamma < 1.0):
            raise ValueError(f"Discount factor gamma must be in [0, 1); got {gamma}.")
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of states |S|."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of actions |A|."""

    @abstractmethod
    def transition(self, state: Any, action: Any) -> Any:
        """Sample the next state from P(· | state, action).

        Parameters
        ----------
        state :
            Current state.
        action :
            Action taken.

        Returns
        -------
        next_state :
            The sampled next state.
        """

    @abstractmethod
    def reward(self, state: Any, action: Any, next_state: Any) -> float:
        """Return the scalar reward R(state, action, next_state).

        In reward-free settings this may always return 0.0 during the
        exploration phase.
        """

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Return ``True`` if *state* is a terminal (absorbing) state."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_states={self.n_states}, "
            f"n_actions={self.n_actions}, "
            f"gamma={self.gamma})"
        )
