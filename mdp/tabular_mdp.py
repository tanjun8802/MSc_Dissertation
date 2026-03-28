"""
tabular_mdp.py
==============
Finite, tabular MDP with explicit transition and reward tables.

Stores the full transition kernel P[s, a, s'] and reward table R[s, a, s']
as numpy arrays, enabling exact dynamic programming solutions (value iteration,
policy iteration) alongside sample-based RL methods.

Example
-------
>>> import numpy as np
>>> from mdp.tabular_mdp import TabularMDP
>>> P = np.ones((3, 2, 3)) / 3          # uniform random transitions
>>> R = np.zeros((3, 2, 3))
>>> mdp = TabularMDP(P, R, gamma=0.95)
>>> s_next = mdp.transition(0, 1)
>>> r = mdp.reward(0, 1, s_next)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mdp.base_mdp import BaseMDP


class TabularMDP(BaseMDP):
    """MDP backed by explicit probability and reward arrays.

    Parameters
    ----------
    P :
        Transition probability array of shape (|S|, |A|, |S|).
        ``P[s, a, s']`` = P(s' | s, a).  Each row ``P[s, a, :]`` must
        form a valid probability distribution (sum to 1, non-negative).
    R :
        Reward array of shape (|S|, |A|, |S|).
        ``R[s, a, s']`` is the scalar reward for the (s, a, s') triple.
    gamma :
        Discount factor.
    terminal_states :
        Set of terminal state indices. An empty set means there are no
        absorbing states (e.g. continuing tasks).
    """

    def __init__(
        self,
        P: np.ndarray,
        R: np.ndarray,
        gamma: float = 0.99,
        terminal_states: set[int] | None = None,
    ) -> None:
        super().__init__(gamma=gamma)

        if P.ndim != 3:
            raise ValueError(f"P must be 3-dimensional (|S|, |A|, |S|); got shape {P.shape}.")
        if R.shape != P.shape:
            raise ValueError(f"R must have the same shape as P; got {R.shape} vs {P.shape}.")

        self.P = P.astype(np.float64)
        self.R = R.astype(np.float64)
        self._n_states, self._n_actions, _ = P.shape
        self.terminal_states: set[int] = terminal_states or set()

    # ------------------------------------------------------------------
    # BaseMDP interface
    # ------------------------------------------------------------------

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def transition(self, state: int, action: int) -> int:
        """Sample next state from P(· | state, action)."""
        probs = self.P[state, action]
        return int(self.np_random.choice(self._n_states, p=probs))

    def reward(self, state: int, action: int, next_state: int) -> float:
        """Return the immediate reward R[state, action, next_state]."""
        return float(self.R[state, action, next_state])

    def is_terminal(self, state: int) -> bool:
        return state in self.terminal_states

    # ------------------------------------------------------------------
    # Dynamic-programming helpers
    # ------------------------------------------------------------------

    def expected_reward(self) -> np.ndarray:
        """Compute the expected reward table of shape (|S|, |A|).

        ``E_R[s, a] = sum_{s'} P[s, a, s'] * R[s, a, s']``
        """
        return np.einsum("ijk,ijk->ij", self.P, self.R)

    def value_iteration(
        self,
        tol: float = 1e-6,
        max_iter: int = 10_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run value iteration to find the optimal value function V*.

        Parameters
        ----------
        tol :
            Convergence threshold (sup-norm on the Bellman error).
        max_iter :
            Maximum number of sweeps.

        Returns
        -------
        V : np.ndarray, shape (|S|,)
            Optimal state-value function.
        policy : np.ndarray, shape (|S|,)
            Greedy policy derived from V.
        """
        E_R = self.expected_reward()  # (|S|, |A|)
        V = np.zeros(self._n_states)

        for _ in range(max_iter):
            # Q(s, a) = E_R[s, a] + γ · Σ_{s'} P[s, a, s'] · V[s']
            Q = E_R + self.gamma * np.einsum("ijk,k->ij", self.P, V)
            V_new = Q.max(axis=1)
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new

        policy = np.argmax(
            E_R + self.gamma * np.einsum("ijk,k->ij", self.P, V),
            axis=1,
        )
        return V, policy

    # ------------------------------------------------------------------
    # RNG (inherited from BaseMDP via BaseEnv-style pattern)
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @property
    def np_random(self) -> np.random.Generator:
        try:
            return self._np_random
        except AttributeError:
            self._np_random = np.random.default_rng()
            return self._np_random

    @np_random.setter
    def np_random(self, rng: np.random.Generator) -> None:
        self._np_random = rng
