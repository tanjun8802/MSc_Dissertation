"""
goal_conditioned_agent.py
=========================
Tabular goal-conditioned agent using Contrastive RL.

Based on:
    "A Single Goal is All You Need: Skills and Exploration Emerge from
    Contrastive RL without Rewards, Demonstrations, or Subgoals"
    (Liu, Tang & Eysenbach, 2024)

Key ideas
---------
* The critic C(s, a, sf) — the logit of (s, a) predicting future state sf —
  is learned via an infoNCE contrastive objective with LogSumExp
  regularisation (Eq. 3 in the paper).  No reward function is used.
* **Single-goal exploration**: during data collection the policy is ALWAYS
  conditioned on the single hard target goal s*.  This is the central
  insight of the paper and differs from prior work that samples easy/medium
  goals for exploration.
* The critic is trained on (s, a, sf) pairs where sf is a geometrically-
  discounted future state sampled from the trajectory, so multiple goals
  arise naturally in training (any visited state can serve as sf).
* Policy: entropy-regularised (softmax) over C[s, :, g], corresponding to
  the actor in contrastive RL (Eq. 4 in the paper).

Algorithm (Algorithm 1 in the paper)
-------------------------------------
    Initialise critic C[s, a, sf], replay buffer B, target goal s*.
    while not converged:
        Collect one trajectory using π(a | s, g=s*), add (s, a, sf) to B.
        Update critic C using infoNCE + LogSumExp reg (Eq. 3 in paper).
    Return policy π(a | s, g=s*).

Usage (tabular GridWorld)
-------------------------
    agent = GoalConditionedAgent(
        n_states=25, n_actions=4, gamma=0.99,
        alpha=0.1, temperature=1.0, seed=42,
    )
    agent.set_goal(24)  # bottom-right corner — the single hard target goal
    obs, _ = env.reset()
    while not done:
        action = agent.select_action(obs)
        next_obs, _, terminated, truncated, info = env.step(action)
        agent.update(obs, action, 0.0, next_obs, terminated, truncated, info)
        obs = next_obs
    agent.finish_episode_with_contrastive_update()
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.base_agent import BaseAgent


class GoalConditionedAgent(BaseAgent):
    """Tabular goal-conditioned agent with Contrastive RL.

    Parameters
    ----------
    n_states :
        Total number of discrete states |S|.
    n_actions :
        Number of discrete actions |A|.
    gamma :
        Discount factor γ ∈ [0, 1).  Also controls the geometric
        distribution for future-state sampling: Δ ~ Geom(1-γ).
    alpha :
        Step size for contrastive critic updates.
    temperature :
        Softmax temperature τ for the entropy-regularised policy.
        π(a|s,g) ∝ exp(C[s, a, g] / τ).  Smaller τ → more greedy.
    n_negatives :
        Number of negative future-state examples per infoNCE update
        (N-1 in Eq. 3 of the paper).
    logsumexp_reg :
        Coefficient of the LogSumExp regularisation term (0.01 in the
        paper).
    buffer_capacity :
        Maximum number of (s, a, sf) triples stored in the replay buffer.
    seed :
        Random seed.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        contrastive_gamma: float | None = None,
        alpha: float = 0.1,
        temperature: float = 1.0,
        n_negatives: int = 16,
        logsumexp_reg: float = 0.01,
        buffer_capacity: int = 10000,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_actions=n_actions, gamma=gamma, seed=seed)
        self.n_states = n_states
        self.alpha = alpha
        self.temperature = temperature
        self.n_negatives = n_negatives
        self.logsumexp_reg = logsumexp_reg
        self.buffer_capacity = buffer_capacity
        # contrastive_gamma controls geometric future-state sampling in the
        # infoNCE objective (Δ ~ Geom(1-contrastive_gamma) - 1).  It should be
        # chosen so that the mean offset E[Δ] = contrastive_gamma/(1-contrastive_gamma)
        # is comparable to the typical episode length.  Using the MDP discount
        # gamma=0.99 on short episodes (e.g. 5×5 grid, ~8 steps to goal) makes
        # sf=goal for >95% of sampled pairs, causing positive and negative
        # samples to be identical and the infoNCE gradient to cancel to zero.
        # Default: falls back to gamma so existing callers are unaffected.
        self._contrastive_gamma: float = contrastive_gamma if contrastive_gamma is not None else gamma

        # Contrastive critic: C[s, a, sf] — logit that (s, a) reaches sf
        self.C = np.zeros((n_states, n_actions, n_states), dtype=np.float64)

        # Replay buffer of (s, a, sf) triples — no rewards stored
        self._replay: list[tuple[int, int, int]] = []

        # Single hard target goal s* — used for ALL data collection
        self._target_goal: int = 0

        # Per-episode buffers for building (s, a, sf) pairs
        self._episode_states: list[int] = []
        self._episode_actions: list[int] = []

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def set_goal(self, goal: int) -> None:
        """Set the single hard target goal s*.

        Parameters
        ----------
        goal :
            Flat state index of the target goal.
        """
        self._target_goal = int(goal)
        self._episode_states = []
        self._episode_actions = []

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, observation: Any) -> int:
        """Entropy-regularised action selection, ALWAYS conditioned on s*.

        The policy is π(a | s, s*) ∝ exp(C[s, a, s*] / temperature).
        Conditioning on the single hard goal throughout training is the
        key exploration strategy described in the paper.

        Parameters
        ----------
        observation :
            Array-like containing the flat state index at position 0.
        """
        self._increment_step()
        state = int(np.asarray(observation).flat[0])
        g = self._target_goal

        logits = self.C[state, :, g]
        # Numerically stable softmax
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted / self.temperature)
        probs /= probs.sum()
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
        """Record the current state (and terminal next-state) in the episode buffer.

        Contrastive RL is reward-free, so the reward argument is
        intentionally ignored.  The critic update happens at episode end
        via :meth:`finish_episode_with_contrastive_update`.

        When ``terminated`` is ``True`` the goal state (``next_observation``)
        is appended as the final entry in ``_episode_states``.  This is
        essential: without it, the goal state can never appear as a future
        state ``sf`` in the geometric future-sampling step, so the contrastive
        critic would receive no signal about goal reachability.

        Returns
        -------
        metrics : dict
            Empty dict — no per-step metrics for reward-free CRL.
        """
        state = int(np.asarray(observation).flat[0])
        self._episode_states.append(state)
        self._episode_actions.append(int(action))
        # Append the goal (terminal next-state) so sf=goal can appear in pairs.
        if terminated:
            next_state = int(np.asarray(next_observation).flat[0])
            self._episode_states.append(next_state)
        return {}

    def reset(self) -> None:
        """Called at the start of each episode — clears the episode buffer."""
        super().reset()
        self._episode_states = []
        self._episode_actions = []

    # ------------------------------------------------------------------
    # Contrastive RL update (Algorithm 1 / Eq. 3 in the paper)
    # ------------------------------------------------------------------

    def finish_episode_with_contrastive_update(self) -> None:
        """Generate (s, a, sf) pairs from the episode and update the critic.

        Steps
        -----
        1. For each *transition* t in the episode, sample Δ ~ Geom(1-γ) and
           set sf = s_{t+Δ} (capped at the end of ``_episode_states``).
           When the episode terminated at the goal, ``_episode_states`` has
           one extra entry beyond ``_episode_actions`` (the goal state appended
           by :meth:`update`), so sf can equal the goal — providing the
           essential reachability signal for the contrastive critic.
        2. Append new pairs to the circular replay buffer.
        3. Sample a mini-batch and apply the infoNCE + LogSumExp reg update
           (Eq. 3 in the paper) to the tabular critic C[s, a, sf].
        """
        n_transitions = len(self._episode_actions)
        n_states = len(self._episode_states)   # may be n_transitions + 1 if goal appended
        if n_transitions == 0:
            return

        # Step 1: generate (s, a, sf) pairs using geometric future sampling.
        # Loop over transitions only; future_t can reach the appended goal state.
        new_pairs: list[tuple[int, int, int]] = []
        for t in range(n_transitions):
            s = self._episode_states[t]
            a = self._episode_actions[t]
            # Δ ~ Geom(1-contrastive_gamma): np.random.geometric returns number of trials ≥ 1,
            # subtract 1 for a 0-indexed offset so Δ ∈ {0, 1, 2, …}.
            # Use _contrastive_gamma (not the MDP gamma) so the mean offset
            # E[Δ] = cγ/(1-cγ) matches the typical episode length rather than
            # being dominated by the MDP discount (which can be 0.99 → E[Δ]=99,
            # far exceeding short episodes and collapsing all sf to the goal).
            delta = int(self.np_random.geometric(1.0 - self._contrastive_gamma)) - 1
            future_t = min(t + delta, n_states - 1)
            sf = self._episode_states[future_t]
            new_pairs.append((s, a, sf))

        # Step 2: add to replay buffer (FIFO, bounded capacity)
        for pair in new_pairs:
            if len(self._replay) >= self.buffer_capacity:
                self._replay.pop(0)
            self._replay.append(pair)

        # Step 3: contrastive update if enough data
        if len(self._replay) >= self.n_negatives + 1:
            self._contrastive_update()

        self._episode_states = []
        self._episode_actions = []

    def _contrastive_update(self) -> None:
        """One mini-batch of infoNCE + LogSumExp reg critic updates (Eq. 3).

        For each sampled positive pair (s, a, sf_pos), N-1 negative future
        states sf_neg are drawn from the replay buffer marginal.  The
        infoNCE gradient encourages C[s, a, sf_pos] to be high while the
        LogSumExp regularisation prevents the logits from growing without
        bound (as shown to be necessary in prior CRL analysis).
        """
        n_buf = len(self._replay)
        batch_size = min(64, n_buf)
        pos_indices = self.np_random.integers(0, n_buf, size=batch_size)

        for pos_idx in pos_indices:
            s, a, sf_pos = self._replay[pos_idx]

            # Sample n_negatives negatives from the replay buffer marginal
            neg_indices = self.np_random.integers(0, n_buf, size=self.n_negatives)
            sf_negs = [self._replay[i][2] for i in neg_indices]

            # Index 0 = positive, 1..n_negatives = negatives
            all_sf = [sf_pos] + sf_negs
            logits = np.array(
                [self.C[s, a, sf] for sf in all_sf], dtype=np.float64
            )

            # Numerically stable softmax
            logits_shifted = logits - logits.max()
            exp_logits = np.exp(logits_shifted)
            sum_exp = exp_logits.sum()
            softmax = exp_logits / sum_exp

            # log-sum-exp in original scale for LogSumExp regularisation.
            # Recover original-scale LSE: max + log(sum(exp(shifted))) = log(sum(exp(original)))
            lse = logits.max() + np.log(sum_exp)

            # Apply gradient for each sf_i:
            #   infoNCE term:      softmax_i - 1{i == 0}
            #   LogSumExp reg:     logsumexp_reg * 2 * lse * softmax_i
            for i, sf in enumerate(all_sf):
                infonce_grad = softmax[i] - (1.0 if i == 0 else 0.0)
                reg_grad = self.logsumexp_reg * 2.0 * lse * softmax[i]
                self.C[s, a, sf] -= self.alpha * (infonce_grad + reg_grad)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GoalConditionedAgent("
            f"n_states={self.n_states}, n_actions={self.n_actions}, "
            f"alpha={self.alpha}, temperature={self.temperature}, "
            f"n_negatives={self.n_negatives}, "
            f"contrastive_gamma={self._contrastive_gamma})"
        )
