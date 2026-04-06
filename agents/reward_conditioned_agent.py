"""
reward_conditioned_agent.py
===========================
Tabular reward-conditioned Q-learning agent.

Based on:
    "Reward-Conditioned Reinforcement Learning" (Nauman, Cygan & Abbeel, 2026)
    arXiv:2603.05066

Key ideas
---------
* The reward is decomposed into k *components* c₁(s,a), …, cₖ(s,a) that the
  environment exposes at each step.  A parameterized aggregation function
  combines them into a scalar:

      rψ = f(ψ, c₁, …, cₖ)   (linear: rψ = Σᵢ ψᵢ · cᵢ)

* Both the policy and the Q-function are conditioned on the reward
  parameterization ψ:   Q(s, a, ψ)  and  π(a | s, ψ).

* During **training**, for each transition a parameterization ψ is sampled
  from a mixture

      PΨ = α · δ(ψ*) + (1−α) · pΨ

  where ψ* is the nominal (target) parameterization and pΨ is a distribution
  over alternatives.  The reward is recomputed as rψ and both Q and π are
  updated conditioned on ψ.  Because Q-learning is off-policy, all updates
  are valid regardless of which ψ was used to collect the transition.

* At **test time** the agent conditions on ψ* to maximise the nominal task
  reward.  Training on diverse ψ improves sample efficiency under ψ* and
  enables zero-shot adaptation to alternative reward functions.

Tabular adaptation for GridWorld
---------------------------------
With k = 2 reward components:

    c₁(s, a) = 1  if the step ended the episode by reaching the goal, else 0
    c₂(s, a) = 1  always  (step indicator — weighted by a cost coefficient)

    rψ = ψ₁ · c₁ + ψ₂ · c₂   with ψ₁ = 1.0 fixed

Nominal:     ψ* = (1.0, 0.0)  — pure sparse goal-reaching reward.
Alternatives: ψ₂ ∈ [psi_min, 0]  (increasingly negative step-cost penalty).

The ψ₂ axis is discretized into ``n_psi_bins`` evenly-spaced values so that
the Q-table  Q[state, psi_bin, action]  stays tractable.

Training (per environment step)
--------------------------------
1. Extract components c₁, c₂ from the transition.
2. Sample ψ-bin from PΨ  (nominal bin with prob ``psi_mix_alpha``,
   uniform over all bins otherwise).
3. Compute rψ = ψ₁ · c₁ + ψ₂ · c₂.
4. TD-update  Q[s, ψ-bin, a]  using  rψ + γ · max Q[s', ψ-bin, :].
5. Always act in the environment under the nominal ψ* (ε-greedy).

Usage (tabular GridWorld)
--------------------------
    agent = RewardConditionedAgent(
        n_states=25, n_actions=4, n_psi_bins=5, psi_min=-0.1, seed=42
    )
    obs, _ = env.reset()
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated, truncated, info)
        obs = next_obs
    agent.finish_episode()   # decays ε
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.base_agent import BaseAgent


class RewardConditionedAgent(BaseAgent):
    """Tabular reward-conditioned Q-learning agent (Nauman et al., 2026).

    The agent trains a single Q-table  Q[s, ψ-bin, a]  by sampling diverse
    reward parameterisations ψ for each transition while always acting under
    the nominal parameterisation ψ*.

    Parameters
    ----------
    n_states :
        Total number of discrete states |S|.
    n_actions :
        Number of discrete actions |A|.
    n_psi_bins :
        Number of discrete reward-parameterisation bins.
        Bin 0 is reserved for the nominal ψ* (ψ₂ = 0, no step penalty).
    psi_min :
        Most negative step-cost weight used for alternative parameterisations
        (must be ≤ 0).  The n_psi_bins values are evenly spaced from 0 to
        psi_min  (e.g. psi_min=-0.1, n_psi_bins=5 → [0, -0.025, -0.05,
        -0.075, -0.1]).
    psi_mix_alpha :
        Probability of drawing the nominal ψ* bin during a training update
        (α in the mixture  PΨ = α · δ(ψ*) + (1−α) · Uniform(Ψ)).
    gamma :
        Discount factor γ.
    alpha :
        Q-learning step size.
    epsilon :
        Initial ε for ε-greedy exploration.
    epsilon_min :
        Floor for ε after annealing.
    epsilon_decay :
        Multiplicative ε decay applied once per episode via
        :meth:`finish_episode`.
    seed :
        Random seed.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        n_psi_bins: int = 5,
        psi_min: float = -0.1,
        psi_mix_alpha: float = 0.5,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_actions=n_actions, gamma=gamma, seed=seed)
        self.n_states = n_states
        self.n_psi_bins = n_psi_bins
        self.psi_min = float(psi_min)
        self.psi_mix_alpha = float(psi_mix_alpha)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # Q-table conditioned on (state, psi_bin, action)
        self.Q = np.zeros((n_states, n_psi_bins, n_actions), dtype=np.float64)

        # Discrete ψ₂ values (step-cost weight) for each bin.
        # Bin 0 = nominal ψ* (ψ₂ = 0.0); higher bins add increasing step cost.
        self.psi_values: np.ndarray = np.linspace(0.0, self.psi_min, n_psi_bins)

        # Index of the nominal parameterisation ψ* (always bin 0)
        self.nominal_psi_bin: int = 0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        observation: Any,
        *,
        greedy: bool = False,
    ) -> int:
        """ε-greedy action selection under the nominal parameterisation ψ*.

        Parameters
        ----------
        observation :
            Array-like with the flat state index at position 0.
        greedy :
            If ``True``, always take the argmax action (no exploration).
        """
        self._increment_step()
        state = int(np.asarray(observation).flat[0])

        if not greedy and self.np_random.random() < self.epsilon:
            return int(self.np_random.integers(0, self.n_actions))

        return int(np.argmax(self.Q[state, self.nominal_psi_bin]))

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
        """One-step Q-learning update with diverse reward-parameterisation sampling.

        Steps
        -----
        1. Decompose the transition into reward components c₁ and c₂.
        2. Sample a ψ-bin from PΨ = psi_mix_alpha·δ(ψ*) + (1−α)·Uniform(Ψ).
        3. Compute the parameterised reward  rψ = ψ₁·c₁ + ψ₂·c₂  (ψ₁=1 fixed).
        4. TD-update  Q[s, ψ-bin, a]  with  rψ + γ · max Q[s', ψ-bin, :].

        Returns
        -------
        metrics : dict
            ``{"td_error": float, "psi_bin": int, "r_psi": float}``
        """
        state = int(np.asarray(observation).flat[0])
        next_state = int(np.asarray(next_observation).flat[0])
        a = int(action)

        # --- reward components ------------------------------------------------
        # c₁: goal-reaching indicator (1 if episode ended by reaching the goal)
        # c₂: step indicator (always 1; negative ψ₂ turns this into a step cost)
        c1 = 1.0 if terminated else 0.0
        c2 = 1.0

        # --- sample ψ from PΨ -------------------------------------------------
        if self.np_random.random() < self.psi_mix_alpha:
            psi_bin = self.nominal_psi_bin
        else:
            psi_bin = int(self.np_random.integers(0, self.n_psi_bins))

        psi2 = float(self.psi_values[psi_bin])

        # --- parameterised reward: rψ = ψ₁·c₁ + ψ₂·c₂  (ψ₁ = 1.0 fixed) ----
        r_psi = c1 + psi2 * c2

        # --- Q-learning TD update ---------------------------------------------
        if terminated:
            td_target = r_psi
        else:
            td_target = r_psi + self.gamma * float(
                np.max(self.Q[next_state, psi_bin])
            )

        td_error = td_target - self.Q[state, psi_bin, a]
        self.Q[state, psi_bin, a] += self.alpha * td_error

        return {
            "td_error": float(td_error),
            "psi_bin": psi_bin,
            "r_psi": float(r_psi),
        }

    def reset(self) -> None:
        """Called at the start of each episode."""
        super().reset()

    def finish_episode(self) -> dict:
        """Decay ε at the end of an episode.

        Returns
        -------
        metrics : dict
            ``{"epsilon": float}``
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {"epsilon": self.epsilon}

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RewardConditionedAgent("
            f"n_states={self.n_states}, n_actions={self.n_actions}, "
            f"n_psi_bins={self.n_psi_bins}, psi_min={self.psi_min}, "
            f"psi_mix_alpha={self.psi_mix_alpha}, "
            f"epsilon={self.epsilon:.3f})"
        )
