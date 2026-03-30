"""
reward_conditioned_agent.py
===========================
Tabular reward-conditioned (return-conditioned) RL agent.

Based on the approach in:
    "Reward Conditioned Policies" (Emmons et al., 2021)
    arXiv:2112.13629

Key ideas
---------
* The policy is conditioned on the *desired return* (sum of future rewards):
  π(a | s, g_return) — given state s, what action achieves desired return g?
* **Behavioral cloning from hindsight**: collect episodes with any exploration
  policy, compute the achieved return G_t from each time-step, then treat
  (s_t, G_t, a_t) as a supervised training example.  The agent learns to
  imitate whichever actions led to each return level.
* Returns are **discretized** into bins so the tabular representation stays
  tractable.  The maximum observed return is tracked and used to scale bins
  dynamically.
* At **test time**, condition on the highest return bin to elicit the best
  learned behavior.

Training phases
---------------
1. **Exploration**: run episodes with a random policy (or ε-greedy) to collect
   diverse (state, action, return) data.
2. **Exploitation**: set ``desired_return`` to the maximum observed return and
   let the return-conditioned policy act greedily.

Usage (tabular GridWorld)
-------------------------
    agent = RewardConditionedAgent(
        n_states=25, n_actions=4, n_return_bins=10, seed=42
    )
    obs, _ = env.reset()
    while not done:
        action = agent.select_action(obs, desired_return=agent.max_observed_return)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated, truncated, info)
        obs = next_obs
    agent.finish_episode()
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.base_agent import BaseAgent


class RewardConditionedAgent(BaseAgent):
    """Tabular return-conditioned agent via behavioral cloning.

    Parameters
    ----------
    n_states :
        Total number of discrete states |S|.
    n_actions :
        Number of discrete actions |A|.
    n_return_bins :
        Number of discrete bins into which achieved returns are bucketed.
    gamma :
        Discount factor γ used when computing G_t from episode trajectories.
    alpha :
        Learning rate for soft count updates (0 < α ≤ 1).
        ``alpha=1`` sets the count to the new sample directly (no smoothing);
        values closer to 0 give a running average.
    epsilon :
        Initial ε for ε-greedy exploration during the data-collection phase.
    epsilon_min :
        Minimum ε after annealing.
    epsilon_decay :
        Multiplicative ε decay per episode.
    seed :
        Random seed.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n_return_bins: int = 10,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_actions=n_actions, gamma=gamma, seed=seed)
        self.n_states = n_states
        self.n_return_bins = n_return_bins
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Conditional action-count table: shape (n_states, n_return_bins, n_actions)
        # Initialized with Laplace (add-1) smoothing so every action has non-zero prob.
        self.action_counts = np.ones(
            (n_states, n_return_bins, n_actions), dtype=np.float64
        )

        # Per-episode buffer: list of (state, action, reward)
        self._episode_buffer: list[tuple[int, int, float]] = []

        # Highest return observed so far — used for dynamic bin scaling and
        # as the default desired_return at test time.
        self.max_observed_return: float = 1.0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        observation: Any,
        desired_return: float | None = None,
        *,
        greedy: bool = False,
    ) -> int:
        """Sample an action from the return-conditioned policy π(a | s, g).

        Parameters
        ----------
        observation :
            Array-like with the flat state index at position 0.
        desired_return :
            The return level to condition on.  If ``None``, defaults to the
            maximum observed return (optimistic conditioning).
        greedy :
            If ``True``, take the *argmax* action instead of sampling.
            Useful for deterministic evaluation.
        """
        self._increment_step()
        state = int(np.asarray(observation).flat[0])

        # During exploration, use ε-greedy to collect diverse trajectories
        if not greedy and self.np_random.random() < self.epsilon:
            return int(self.np_random.integers(0, self.n_actions))

        g = desired_return if desired_return is not None else self.max_observed_return
        return_bin = self._return_to_bin(g)
        counts = self.action_counts[state, return_bin]

        if greedy:
            return int(np.argmax(counts))

        probs = counts / counts.sum()
        return int(self.np_random.choice(self.n_actions, p=probs))

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
        """Buffer one (state, action, reward) triple for post-episode processing.

        The actual count update is deferred to :meth:`finish_episode` so we
        can compute discounted returns over the full trajectory.
        """
        state = int(np.asarray(observation).flat[0])
        self._episode_buffer.append((state, int(action), float(reward)))
        return {}

    def reset(self) -> None:
        """Called at the start of each episode."""
        super().reset()
        self._episode_buffer = []

    # ------------------------------------------------------------------
    # Post-episode return relabeling (behavioral cloning step)
    # ------------------------------------------------------------------

    def finish_episode(self) -> dict:
        """Update the conditional policy from the episode just completed.

        Steps
        -----
        1. Compute discounted return G_t for every time-step t.
        2. Update ``max_observed_return`` if needed.
        3. For each (s_t, a_t, G_t) triple, increment
           ``action_counts[s_t, bin(G_t), a_t]`` by ``alpha`` (soft update).
        4. Decay ε.

        Returns
        -------
        metrics : dict
            ``{"episode_return": float, "n_transitions": int}``
        """
        buf = self._episode_buffer
        if not buf:
            self._episode_buffer = []
            return {"episode_return": 0.0, "n_transitions": 0}

        # --- compute discounted returns ------------------------------------
        n = len(buf)
        returns = np.zeros(n, dtype=np.float64)
        G = 0.0
        for t in range(n - 1, -1, -1):
            _, _, r = buf[t]
            G = r + self.gamma * G
            returns[t] = G

        episode_return = float(returns[0])

        # --- update max observed return ------------------------------------
        self.max_observed_return = max(self.max_observed_return, episode_return)

        # --- soft-count behavioral cloning update -------------------------
        for t, (state, action, _) in enumerate(buf):
            rb = self._return_to_bin(returns[t])
            # Soft update: nudge count toward +1 for the observed action
            self.action_counts[state, rb, action] += self.alpha

        # --- anneal ε ------------------------------------------------------
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episode_buffer = []

        return {"episode_return": episode_return, "n_transitions": n}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _return_to_bin(self, return_value: float) -> int:
        """Map a scalar return to a discrete bin index in [0, n_return_bins).

        Bins are defined over [0, max_observed_return] with equal width.
        Returns below 0 are clamped to bin 0; returns above
        max_observed_return are clamped to the top bin.
        """
        upper = max(self.max_observed_return, 1e-8)
        normalized = float(return_value) / upper
        normalized = max(0.0, min(1.0, normalized))
        bin_idx = int(normalized * self.n_return_bins)
        # Clamp to valid range (edge case: normalized == 1.0)
        return min(bin_idx, self.n_return_bins - 1)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RewardConditionedAgent("
            f"n_states={self.n_states}, n_actions={self.n_actions}, "
            f"n_return_bins={self.n_return_bins}, "
            f"epsilon={self.epsilon:.3f}, "
            f"max_return={self.max_observed_return:.2f})"
        )
