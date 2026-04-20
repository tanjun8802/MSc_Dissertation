"""
exdm_agent.py
=============
Tabular adaptation of the Exploratory Diffusion Model (ExDM) for
reward-free reinforcement learning on discrete GridWorld environments.

Reference
---------
Ying, C., Chen, H., Hao, Z., Zhou, X., Su, H., & Zhu, J. (2025).
"Exploratory Diffusion Model for Unsupervised Reinforcement Learning."
arXiv:2502.07279.

Algorithm overview
------------------
ExDM drives reward-free exploration by maintaining a **state diffusion model**
that continuously fits the empirical distribution of visited states stored in a
replay buffer.  The key insight is:

    Well-visited states are well-modelled → low reconstruction error.
    Rarely visited states are poorly modelled → high reconstruction error.

The reconstruction error under the diffusion model is used directly as an
**intrinsic reward**:

    R_score(s) = E_{ε,t} [ ‖ε̂_θ'(s_t | t) − ε‖² ]               (Eq. 8)

A Gaussian behaviour policy (here: tabular ε-greedy Q-learning) is trained to
maximise this intrinsic reward, automatically driving the agent toward
under-explored regions of the state space.

Tabular adaptation for GridWorld
---------------------------------
The original ExDM operates on continuous state spaces (R²).  Here we adapt
the algorithm to a finite discrete state space |S| = n_states:

1.  **State embedding**: flat state index  s ∈ {0, …, n_states-1}  →  one-hot
    vector  s_onehot ∈ R^{n_states}.

2.  **Forward diffusion** (DDPM, linear noise schedule):
        s_t = √ᾱ_t · s_onehot  +  √(1−ᾱ_t) · ε,   ε ~ N(0, I)

    where  ᾱ_t = Π_{k=1}^{t} (1 − β_k)  and  β_k  follows a linear schedule
    from  β_start  to  β_end.

3.  **Score model**  ε̂_θ'(s_t | t): a separate weight matrix W[t] per diffusion
    timestep, predicting the noise added to the one-hot state:
        ε̂  =  W[t] @ s_t,    W[t] ∈ R^{n_states × n_states}

4.  **Score model update**: online SGD on the MSE loss
        L  =  ‖W[t] @ s_t  −  ε‖²
    using the mini-batch of states sampled from the replay buffer.

5.  **Intrinsic reward** for state s:
        R_score(s) = (1/K) Σ_{k=1}^{K} ‖W[t_k] @ s_{t_k}  −  ε_k‖²
    averaged over K randomly sampled (t_k, ε_k) pairs.

6.  **Behaviour policy**: standard ε-greedy Q-learning with the intrinsic
    reward.  The Q-table Q[s, a] is updated via the TD rule:
        Q[s, a] ← Q[s, a] + α · (r_int + γ · max_{a'} Q[s', a']  −  Q[s, a])

Usage (tabular GridWorld)
--------------------------
    agent = ExDMAgent(n_states=100, n_actions=4, seed=42)
    obs, _ = env.reset()
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated, truncated, info)
        obs = next_obs
    agent.finish_episode()  # decays ε
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Linear DDPM noise schedule helper
# ---------------------------------------------------------------------------

def _linear_beta_schedule(
    n_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> np.ndarray:
    """Return β_1, …, β_T for a linear noise schedule.

    Parameters
    ----------
    n_steps :
        Number of diffusion timesteps T.
    beta_start, beta_end :
        Start and end values for the linear schedule.

    Returns
    -------
    betas : np.ndarray, shape (n_steps,)
    """
    return np.linspace(beta_start, beta_end, n_steps, dtype=np.float64)


# ---------------------------------------------------------------------------
# ExDM agent
# ---------------------------------------------------------------------------


class ExDMAgent(BaseAgent):
    """Tabular Exploratory Diffusion Model (ExDM) agent.

    The agent trains a score-based state diffusion model on the empirical
    distribution of visited states and uses the reconstruction error as an
    intrinsic reward for ε-greedy Q-learning.  This implements the
    unsupervised pre-training stage (Algorithm 1 of the paper) adapted to
    discrete GridWorld environments.

    Parameters
    ----------
    n_states :
        Total number of discrete states |S| (= height × width for a grid).
    n_actions :
        Number of discrete actions |A|.
    gamma :
        Discount factor γ for Q-learning.
    alpha :
        Q-learning step size.
    model_lr :
        Learning rate for the score model (SGD on the MSE loss).
    n_diffusion_steps :
        Number of DDPM diffusion timesteps T.  Larger T gives a richer
        noise schedule but is slightly slower to update and evaluate.
    beta_start, beta_end :
        Endpoints of the linear noise schedule for β_t.
    epsilon :
        Initial ε for ε-greedy exploration.
    epsilon_min :
        Floor for ε after annealing.
    epsilon_decay :
        Multiplicative ε decay applied once per episode.
    buffer_capacity :
        Maximum number of state indices stored in the replay buffer.
    batch_size :
        Mini-batch size for each score-model update.
    n_model_updates :
        Number of mini-batch gradient steps applied to the score model per
        environment step.  Increasing this value allows the model to track
        the changing data distribution more quickly.
    reward_samples :
        Number of (ε, t) samples used to estimate the intrinsic reward
        E_{ε,t}[‖ε̂_θ'(s_t|t) − ε‖²] for each state.
    seed :
        Random seed.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        model_lr: float = 1e-2,
        n_diffusion_steps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10_000,
        batch_size: int = 32,
        n_model_updates: int = 5,
        reward_samples: int = 10,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_actions=n_actions, gamma=gamma, seed=seed)

        self.n_states = n_states
        self.alpha = float(alpha)
        self.model_lr = float(model_lr)
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.buffer_capacity = int(buffer_capacity)
        self.batch_size = int(batch_size)
        self.n_model_updates = int(n_model_updates)
        self.reward_samples = int(reward_samples)

        # ── Noise schedule ─────────────────────────────────────────────────
        betas = _linear_beta_schedule(n_diffusion_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        # ᾱ_t = cumulative product of (1 - β_k) for k = 1…t
        self._alpha_bar: np.ndarray = np.cumprod(alphas)  # shape (T,)
        self._sqrt_alpha_bar: np.ndarray = np.sqrt(self._alpha_bar)
        self._sqrt_one_minus_alpha_bar: np.ndarray = np.sqrt(1.0 - self._alpha_bar)

        # ── Score model: one weight matrix W[t] per diffusion step ─────────
        # W[t] ∈ R^{n_states × n_states}
        # Initialised to zero (→ ε̂ = 0) so the initial intrinsic reward is
        # the squared norm of the noise ε, which is always > 0.  This ensures
        # early exploration without any pre-visited states.
        self.W: np.ndarray = np.zeros(
            (n_diffusion_steps, n_states, n_states), dtype=np.float64
        )

        # ── Q-table for the behaviour policy ───────────────────────────────
        self.Q: np.ndarray = np.zeros((n_states, n_actions), dtype=np.float64)

        # ── Replay buffer (circular, stores flat state indices) ─────────────
        self._buffer: list[int] = []

    # ------------------------------------------------------------------
    # Noise-schedule helpers
    # ------------------------------------------------------------------

    def _noisy_state(self, state_onehot: np.ndarray, t_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample a noisy state s_t and the corresponding noise ε.

        Forward diffusion (DDPM):
            s_t = √ᾱ_t · s_onehot + √(1−ᾱ_t) · ε,   ε ~ N(0, I)

        Parameters
        ----------
        state_onehot : np.ndarray, shape (n_states,)
            One-hot encoding of the original state.
        t_idx : int
            Diffusion timestep index (0-based, in range [0, T-1]).

        Returns
        -------
        s_t : np.ndarray, shape (n_states,)
            Noisy state embedding.
        eps : np.ndarray, shape (n_states,)
            Sampled standard Gaussian noise.
        """
        eps = self.np_random.standard_normal(self.n_states)
        s_t = self._sqrt_alpha_bar[t_idx] * state_onehot + self._sqrt_one_minus_alpha_bar[t_idx] * eps
        return s_t, eps

    # ------------------------------------------------------------------
    # Score model
    # ------------------------------------------------------------------

    def _score_model_update(self, states_batch: np.ndarray) -> float:
        """Apply one mini-batch SGD update to all score-model weight matrices.

        For each state s in the batch, samples a random diffusion timestep t,
        computes the noisy embedding s_t, and applies a gradient step on the
        MSE loss  L = ‖W[t] @ s_t − ε‖².

        Parameters
        ----------
        states_batch : np.ndarray of int, shape (B,)
            Flat state indices sampled from the replay buffer.

        Returns
        -------
        float
            Mean MSE loss over the batch.
        """
        total_loss = 0.0
        for s_idx in states_batch:
            # One-hot encoding of the state
            s_onehot = np.zeros(self.n_states, dtype=np.float64)
            s_onehot[s_idx] = 1.0

            # Random diffusion timestep
            t_idx = int(self.np_random.integers(0, self.n_diffusion_steps))

            # Forward diffusion: s_t and noise ε
            s_t, eps = self._noisy_state(s_onehot, t_idx)

            # Predicted noise: ε̂ = W[t] @ s_t
            eps_hat = self.W[t_idx] @ s_t               # (n_states,)

            # Residual: r = ε̂ − ε
            residual = eps_hat - eps                     # (n_states,)

            # Gradient of L = ‖r‖² w.r.t. W[t]: dL/dW[t] = residual ⊗ s_t
            # W[t] -= model_lr · residual ⊗ s_t
            self.W[t_idx] -= self.model_lr * np.outer(residual, s_t)

            total_loss += float(np.dot(residual, residual))

        return total_loss / max(1, len(states_batch))

    def _compute_intrinsic_reward(self, state_idx: int) -> float:
        """Estimate the score-based intrinsic reward for a given state.

        R_score(s) = (1/K) Σ_{k=1}^{K} ‖W[t_k] @ s_{t_k} − ε_k‖²   (Eq. 8)

        A higher value means the current score model fits this state poorly,
        indicating the state has been visited infrequently.

        Parameters
        ----------
        state_idx : int
            Flat state index.

        Returns
        -------
        float
            Estimated intrinsic reward R_score(s).
        """
        s_onehot = np.zeros(self.n_states, dtype=np.float64)
        s_onehot[state_idx] = 1.0

        total = 0.0
        for _ in range(self.reward_samples):
            t_idx = int(self.np_random.integers(0, self.n_diffusion_steps))
            s_t, eps = self._noisy_state(s_onehot, t_idx)
            eps_hat = self.W[t_idx] @ s_t
            residual = eps_hat - eps
            total += float(np.dot(residual, residual))

        return total / self.reward_samples

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, observation: Any) -> int:
        """ε-greedy action selection under the Q-table.

        Parameters
        ----------
        observation :
            Array-like containing the flat state index at position 0.

        Returns
        -------
        int
            Selected action.
        """
        self._increment_step()
        state = int(np.asarray(observation).flat[0])

        if self.np_random.random() < self.epsilon:
            return int(self.np_random.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

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
        """One-step update: add state to replay buffer, update score model,
        then perform a Q-learning TD update with the intrinsic reward.

        The extrinsic reward (``reward``) is intentionally ignored during
        unsupervised pre-training — consistent with Algorithm 1 of the paper.
        All learning signal comes from the score-based intrinsic reward.

        Steps
        -----
        1. Record the current state in the replay buffer.
        2. Compute score-based intrinsic reward R_score(s) for the current state.
        3. Apply n_model_updates mini-batch SGD steps to the score model.
        4. Q-learning TD update:
               Q[s, a] += α · (r_int + γ · max Q[s', :] − Q[s, a])

        Parameters
        ----------
        observation :
            Current observation (flat state index at position 0).
        action :
            Executed action.
        reward :
            Extrinsic reward (not used during pre-training).
        next_observation :
            Next observation.
        terminated :
            Whether the episode ended by reaching the goal.
        truncated :
            Whether the episode ended due to a time limit.
        info :
            Additional step info (unused).

        Returns
        -------
        dict
            ``{"intrinsic_reward": float, "td_error": float, "score_loss": float}``
        """
        state = int(np.asarray(observation).flat[0])
        next_state = int(np.asarray(next_observation).flat[0])
        a = int(action)

        # ── 1. Add state to replay buffer (circular, FIFO) ─────────────────
        if len(self._buffer) >= self.buffer_capacity:
            self._buffer.pop(0)
        self._buffer.append(state)

        # ── 2. Intrinsic reward for the current state ───────────────────────
        r_int = self._compute_intrinsic_reward(state)

        # ── 3. Score model update (multiple mini-batches) ───────────────────
        score_loss = 0.0
        if len(self._buffer) >= self.batch_size:
            for _ in range(self.n_model_updates):
                idxs = self.np_random.integers(0, len(self._buffer), size=self.batch_size)
                batch = np.array([self._buffer[i] for i in idxs], dtype=int)
                score_loss = self._score_model_update(batch)

        # ── 4. Q-learning TD update with intrinsic reward ──────────────────
        if terminated:
            td_target = r_int
        else:
            td_target = r_int + self.gamma * float(np.max(self.Q[next_state]))

        td_error = td_target - self.Q[state, a]
        self.Q[state, a] += self.alpha * td_error

        return {
            "intrinsic_reward": r_int,
            "td_error": float(td_error),
            "score_loss": float(score_loss),
        }

    def reset(self) -> None:
        """Called at the start of each episode."""
        super().reset()

    def finish_episode(self) -> dict:
        """Decay ε at the end of each episode.

        Returns
        -------
        dict
            ``{"epsilon": float}``
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return {"epsilon": self.epsilon}

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ExDMAgent("
            f"n_states={self.n_states}, n_actions={self.n_actions}, "
            f"n_diffusion_steps={self.n_diffusion_steps}, "
            f"model_lr={self.model_lr}, epsilon={self.epsilon:.3f})"
        )
