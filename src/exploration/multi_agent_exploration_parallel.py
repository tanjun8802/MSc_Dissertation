from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import math
import random

import numpy as np
import torch


Array = np.ndarray


def make_discrete_transition_model(
    action_effects: Dict[Any, Sequence[float]]
) -> Callable[[Array, Any], Array]:
    effects = {k: np.asarray(v, dtype=float) for k, v in action_effects.items()}

    def model(state: Array, action: Any) -> Array:
        return np.asarray(state, dtype=float) + effects[action]

    return model


class NumpyMultiAgentExplorer:
    """
    Single-process multi-agent explorer.

    - Keeps N env instances in one process.
    - Uses env.sample_initial_state() when available.
    - Supports three policy modes:
        * "goal": epsilon-greedy wrt heuristic goal progress + novelty.
        * "q": epsilon-greedy wrt Q-network.
        * "hybrid": epsilon-greedy wrt Q + goal + novelty.
    - Exposes reset() and collect_steps(...) to match a notebook DQN loop.
    """

    def __init__(
        self,
        n_agents: int,
        env_fn: Callable[[], Any],
        candidate_actions: Sequence[Any],
        state_low: Sequence[float],
        state_high: Sequence[float],
        goal: Sequence[float],
        transition_model: Callable[[Array, Any], Array],
        epsilon: float = 0.1,
        novelty_scale: float = 0.0,
        state_bins: Optional[Sequence[int]] = None,
        rng_seed: Optional[int] = None,
        score_fn: Optional[Callable[[Array, Any, Array, Array], float]] = None,
        done_fn: Optional[Callable[[Array, Array, Dict[str, Any]], bool]] = None,
        reset_fn: Optional[Callable[[Any, Array], Tuple[Any, Dict[str, Any]]]] = None,
        step_fn: Optional[Callable[[Any, Any], Tuple[Any, float, bool, Dict[str, Any]]]] = None,
        policy_mode: str = "goal",   # "goal", "q", "hybrid"
        goal_scale: float = 1.0,
        q_scale: float = 1.0,
        intrinsic_in_reward: bool = True,
        goal_tolerance: float = 1e-2,
    ):
        self.n_agents = int(n_agents)
        self.env_fn = env_fn
        self.candidate_actions = list(candidate_actions)
        self.state_low = np.asarray(state_low, dtype=float)
        self.state_high = np.asarray(state_high, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.transition_model = transition_model

        self.epsilon = float(epsilon)
        self.novelty_scale = float(novelty_scale)
        self.state_bins = tuple(state_bins) if state_bins is not None else None
        self.score_fn = score_fn
        self.done_fn = done_fn
        self.reset_fn = reset_fn
        self.step_fn = step_fn

        self.policy_mode = policy_mode
        self.goal_scale = float(goal_scale)
        self.q_scale = float(q_scale)
        self.intrinsic_in_reward = bool(intrinsic_in_reward)
        self.goal_tolerance = float(goal_tolerance)

        self.rng = np.random.default_rng(rng_seed)
        self.py_rng = random.Random(rng_seed)

        self.state_action_counts: Dict[Tuple[Any, ...], int] = {}

        self.envs: List[Any] = []
        self.states: List[Array] = []
        self.done_flags: List[bool] = []
        self.episode_ids: List[int] = []
        self.timesteps: List[int] = []
        self.next_episode_id: int = 0

        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive.")
        if self.state_low.shape != self.state_high.shape:
            raise ValueError("state_low and state_high must have the same shape.")
        if self.goal.shape != self.state_low.shape:
            raise ValueError("goal must have the same dimension as the state bounds.")
        if len(self.candidate_actions) == 0:
            raise ValueError("candidate_actions cannot be empty.")
        if self.policy_mode not in {"goal", "q", "hybrid"}:
            raise ValueError("policy_mode must be one of {'goal', 'q', 'hybrid'}.")

    # ---------------------------------------------------------------------
    # Initial state sampling
    # ---------------------------------------------------------------------

    def _sample_env_initial_state(self, env: Any) -> Array:
        """
        Prefer the env's own valid-state sampler when available.
        Fall back to uniform box sampling only if needed.
        """
        if hasattr(env, "sample_initial_state") and callable(env.sample_initial_state):
            s = env.sample_initial_state()
            return np.asarray(s, dtype=float)

        return self.rng.uniform(
            low=self.state_low,
            high=self.state_high,
            size=self.state_low.shape,
        ).astype(float)

    # ---------------------------------------------------------------------
    # Novelty
    # ---------------------------------------------------------------------

    def _digitize_state(self, state: Array) -> Tuple[Any, ...]:
        state = np.asarray(state, dtype=float)

        if self.state_bins is None:
            return tuple(np.round(state, 3).tolist())

        ratios = (state - self.state_low) / np.maximum(self.state_high - self.state_low, 1e-8)
        ratios = np.clip(ratios, 0.0, 0.999999)
        return tuple(int(r * b) for r, b in zip(ratios, self.state_bins))

    def _action_key(self, action: Any) -> Tuple[Any, ...]:
        if np.isscalar(action):
            return (action,)
        return tuple(np.round(np.asarray(action, dtype=float), 3).tolist())

    def _state_action_key(self, state: Array, action: Any) -> Tuple[Any, ...]:
        return self._digitize_state(state) + self._action_key(action)

    def novelty_bonus(self, state: Array, action: Any) -> float:
        if self.novelty_scale <= 0.0:
            return 0.0
        key = self._state_action_key(state, action)
        count = self.state_action_counts.get(key, 0)
        return self.novelty_scale / math.sqrt(count + 1.0)

    def update_novelty(self, state: Array, action: Any) -> None:
        if self.novelty_scale <= 0.0:
            return
        key = self._state_action_key(state, action)
        self.state_action_counts[key] = self.state_action_counts.get(key, 0) + 1

    def reset_novelty(self) -> None:
        self.state_action_counts.clear()

    # ---------------------------------------------------------------------
    # Goal scoring
    # ---------------------------------------------------------------------

    def default_score(self, predicted_next_state: Array) -> float:
        return -float(np.linalg.norm(np.asarray(predicted_next_state, dtype=float) - self.goal))

    def goal_score(self, state: Array, action: Any) -> float:
        predicted_next_state = np.asarray(self.transition_model(state, action), dtype=float)
        if self.score_fn is not None:
            return float(self.score_fn(state, action, predicted_next_state, self.goal))
        return self.default_score(predicted_next_state)

    # ---------------------------------------------------------------------
    # Env integration
    # ---------------------------------------------------------------------

    def _reset_one_env(self, env: Any, init_state: Array) -> Tuple[Array, Dict[str, Any]]:
        if self.reset_fn is not None:
            obs, info = self.reset_fn(env, init_state)
        else:
            try:
                obs, info = env.reset(options={"state": init_state})
            except TypeError:
                obs, info = env.reset()
        return np.asarray(obs, dtype=float), info

    def _step_one_env(self, env: Any, action: Any) -> Tuple[Array, float, bool, Dict[str, Any]]:
        if self.step_fn is not None:
            obs, reward, done, info = self.step_fn(env, action)
        else:
            out = env.step(action)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = out

        return np.asarray(obs, dtype=float), float(reward), bool(done), info

    def _goal_reached(self, state: Array, info: Dict[str, Any]) -> bool:
        if self.done_fn is not None:
            return bool(self.done_fn(state, self.goal, info))
        return bool(np.linalg.norm(np.asarray(state, dtype=float) - self.goal) < self.goal_tolerance)

    # ---------------------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------------------

    def select_actions_goal(self, active_mask: Optional[Sequence[bool]] = None) -> List[Any]:
        actions: List[Any] = []

        for i, state in enumerate(self.states):
            active = True if active_mask is None else bool(active_mask[i])
            if not active:
                actions.append(None)
                continue

            if self.py_rng.random() < self.epsilon:
                actions.append(self.py_rng.choice(self.candidate_actions))
                continue

            best_action = None
            best_score = -float("inf")

            for action in self.candidate_actions:
                score = self.goal_scale * self.goal_score(state, action)
                score += self.novelty_bonus(state, action)

                if score > best_score:
                    best_score = score
                    best_action = action

            actions.append(best_action)

        return actions

    def select_actions_from_q(
        self,
        q_network,
        device,
        active_mask: Optional[Sequence[bool]] = None,
    ) -> List[Any]:
        actions: List[Any] = []

        for i, state in enumerate(self.states):
            active = True if active_mask is None else bool(active_mask[i])
            if not active:
                actions.append(None)
                continue

            if self.py_rng.random() < self.epsilon:
                actions.append(self.py_rng.choice(self.candidate_actions))
                continue

            obs_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = q_network(obs_t)
                a_idx = int(q_vals.argmax(dim=-1).item())

            actions.append(self.candidate_actions[a_idx])

        return actions

    def select_actions_hybrid(
        self,
        q_network,
        device,
        active_mask: Optional[Sequence[bool]] = None,
    ) -> List[Any]:
        actions: List[Any] = []

        for i, state in enumerate(self.states):
            active = True if active_mask is None else bool(active_mask[i])
            if not active:
                actions.append(None)
                continue

            if self.py_rng.random() < self.epsilon:
                actions.append(self.py_rng.choice(self.candidate_actions))
                continue

            obs_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = q_network(obs_t).squeeze(0).detach().cpu().numpy()

            best_action = None
            best_score = -float("inf")

            for a_idx, action in enumerate(self.candidate_actions):
                score = self.q_scale * float(q_vals[a_idx])
                score += self.goal_scale * self.goal_score(state, action)
                score += self.novelty_bonus(state, action)

                if score > best_score:
                    best_score = score
                    best_action = action

            actions.append(best_action)

        return actions

    def _choose_actions(
        self,
        active_mask: Sequence[bool],
        q_network=None,
        device: str = "cpu",
    ) -> List[Any]:
        if self.policy_mode == "goal":
            return self.select_actions_goal(active_mask=active_mask)

        if self.policy_mode == "q":
            if q_network is None:
                raise ValueError("q_network must be provided when policy_mode='q'.")
            return self.select_actions_from_q(
                q_network=q_network,
                device=device,
                active_mask=active_mask,
            )

        if self.policy_mode == "hybrid":
            if q_network is None:
                raise ValueError("q_network must be provided when policy_mode='hybrid'.")
            return self.select_actions_hybrid(
                q_network=q_network,
                device=device,
                active_mask=active_mask,
            )

        raise ValueError(f"Unknown policy_mode: {self.policy_mode}")

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def reset(self, reset_novelty: bool = False) -> Array:
        """
        Reset all envs with fresh env-sampled initial states.
        """
        self.close()

        if reset_novelty:
            self.reset_novelty()

        self.envs = [self.env_fn() for _ in range(self.n_agents)]
        self.states = []
        self.done_flags = [False] * self.n_agents
        self.episode_ids = []
        self.timesteps = []

        for env in self.envs:
            init_state = self._sample_env_initial_state(env)
            obs, _ = self._reset_one_env(env, init_state)
            self.states.append(obs)
            self.episode_ids.append(self.next_episode_id)
            self.next_episode_id += 1
            self.timesteps.append(0)

        return np.asarray(self.states, dtype=float)

    def collect_steps(
        self,
        replay_buffer,
        num_steps: int = 1,
        flush_partial_episodes: bool = False,
        q_network=None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Step all active agents num_steps times and push transitions into replay_buffer.
        """
        if len(self.envs) == 0:
            self.reset()

        num_transitions = 0
        num_episodes_finished = 0

        for _ in range(num_steps):
            active_mask = [not d for d in self.done_flags]
            if not any(active_mask):
                break

            chosen_actions = self._choose_actions(
                active_mask=active_mask,
                q_network=q_network,
                device=device,
            )

            for agent_id, action in enumerate(chosen_actions):
                if action is None:
                    continue

                state = self.states[agent_id].copy()
                ep_id = self.episode_ids[agent_id]
                t = self.timesteps[agent_id]

                next_state, env_reward, env_done, info = self._step_one_env(self.envs[agent_id], action)

                intrinsic = self.novelty_bonus(state, action)
                stored_reward = env_reward + intrinsic if self.intrinsic_in_reward else env_reward

                self.update_novelty(state, action)

                done = bool(env_done or self._goal_reached(next_state, info))

                replay_buffer.add_transition(
                    obs=state,
                    action=action,
                    reward=stored_reward,
                    next_obs=next_state,
                    terminated=done,
                    truncated=False,
                    episode_id=ep_id,
                    timestep=t,
                )

                num_transitions += 1
                self.states[agent_id] = next_state
                self.timesteps[agent_id] += 1

                if done:
                    num_episodes_finished += 1
                    self.done_flags[agent_id] = True

                    if not flush_partial_episodes:
                        new_init_state = self._sample_env_initial_state(self.envs[agent_id])
                        reset_obs, _ = self._reset_one_env(self.envs[agent_id], new_init_state)

                        self.states[agent_id] = reset_obs
                        self.done_flags[agent_id] = False
                        self.episode_ids[agent_id] = self.next_episode_id
                        self.next_episode_id += 1
                        self.timesteps[agent_id] = 0

        return {
            "num_transitions": num_transitions,
            "num_episodes_finished": num_episodes_finished,
        }

    def close(self) -> None:
        for env in self.envs:
            try:
                if hasattr(env, "close"):
                    env.close()
            except Exception:
                pass

        self.envs = []
        self.states = []
        self.done_flags = []
        self.episode_ids = []
        self.timesteps = []