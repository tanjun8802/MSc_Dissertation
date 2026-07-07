from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

Array = np.ndarray


def unit_vector(v: Array, eps: float = 1e-8) -> Array:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return v / max(n, eps)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def running_mean_std(z_samples: Array, axis: int = 0, eps: float = 1e-8):
    z_samples = np.asarray(z_samples, dtype=np.float32)
    mu = np.mean(z_samples, axis=axis)
    sigma = np.std(z_samples, axis=axis) + eps
    return mu, sigma


def latin_hypercube_sampling(
    num_points: int,
    dim: int,
    low: float = -2.0,
    high: float = 2.0,
    rng: Optional[np.random.RandomState] = None,
) -> Array:
    rng = np.random.RandomState() if rng is None else rng
    cut = np.linspace(0.0, 1.0, num_points + 1, dtype=np.float32)
    u = rng.rand(num_points, dim).astype(np.float32)
    a = cut[:num_points]
    b = cut[1 : num_points + 1]
    rd_points = u * (b - a)[:, None] + a[:, None]
    H = np.zeros_like(rd_points)
    for j in range(dim):
        order = rng.permutation(num_points)
        H[:, j] = rd_points[order, j]
    return low + (high - low) * H


class IdentityEncoder:
    def __call__(self, states: Array) -> Array:
        return np.asarray(states, dtype=np.float32).copy()


def snap_to_nearest_free_cell(pos: Array, free_cells: Array) -> Array:
    centres = np.asarray(free_cells, dtype=np.float32) + 0.5
    dists = np.linalg.norm(centres - np.asarray(pos, dtype=np.float32)[None, :], axis=1)
    return centres[int(np.argmin(dists))]


class LatentSpaceManager:
    def __init__(self, latent_dim: int, init_c: float = 2.0, rng: Optional[np.random.RandomState] = None):
        self.latent_dim = latent_dim
        self.init_c = init_c
        self.rng = np.random.RandomState() if rng is None else rng
        self.mu = np.zeros(latent_dim, dtype=np.float32)
        self.sigma = np.ones(latent_dim, dtype=np.float32)

    def update_stats(self, z_batch: Array) -> None:
        self.mu, self.sigma = running_mean_std(z_batch)

    def to_standardized(self, z: Array) -> Array:
        return (np.asarray(z, dtype=np.float32) - self.mu) / self.sigma

    def from_standardized(self, z_tilde: Array) -> Array:
        return self.mu + self.sigma * np.asarray(z_tilde, dtype=np.float32)

    def lhs_initial_latents(self, num_agents: int) -> Array:
        z_tilde = latin_hypercube_sampling(
            num_points=num_agents,
            dim=self.latent_dim,
            low=-self.init_c,
            high=self.init_c,
            rng=self.rng,
        )
        return self.from_standardized(z_tilde)


class StateActionCoverageTracker:
    """
    Tracks:
      - state-region counts N(z_bin)
      - state-action counts N(z_bin, a_bin)
      - optional transition counts N(z_bin, a_bin, z_next_bin)

    Uses simple quantization over latent/state space.
    """

    def __init__(self, latent_dim: int, bin_size: float = 1.0, action_bins: int = 4):
        self.latent_dim = int(latent_dim)
        self.bin_size = float(bin_size)
        self.action_bins = int(action_bins)

        self.state_counts: Dict[Tuple[int, ...], int] = {}
        self.state_action_counts: Dict[Tuple[Tuple[int, ...], int], int] = {}
        self.transition_counts: Dict[Tuple[Tuple[int, ...], int, Tuple[int, ...]], int] = {}

    def state_bin(self, z: Array) -> Tuple[int, ...]:
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        return tuple(np.floor(z / self.bin_size).astype(np.int32).tolist())

    def action_bin_from_direction(self, direction: Array) -> int:
        d = np.asarray(direction, dtype=np.float32).reshape(-1)
        if d.shape[0] < 2:
            return 0
        dx, dy = float(d[0]), float(d[1])
        if self.action_bins == 4:
            if abs(dx) >= abs(dy):
                return 3 if dx > 0 else 2
            return 0 if dy > 0 else 1
        angle = np.arctan2(dy, dx)
        angle = (angle + 2.0 * np.pi) % (2.0 * np.pi)
        return int(np.floor(self.action_bins * angle / (2.0 * np.pi))) % self.action_bins

    def get_state_count(self, z: Array) -> int:
        return self.state_counts.get(self.state_bin(z), 0)

    def get_state_action_count(self, z: Array, action_bin: int) -> int:
        key = (self.state_bin(z), int(action_bin))
        return self.state_action_counts.get(key, 0)

    def get_transition_count(self, z: Array, action_bin: int, z_next: Array) -> int:
        key = (self.state_bin(z), int(action_bin), self.state_bin(z_next))
        return self.transition_counts.get(key, 0)

    def observe_state(self, z: Array) -> None:
        zb = self.state_bin(z)
        self.state_counts[zb] = self.state_counts.get(zb, 0) + 1

    def observe_state_action(self, z: Array, action_bin: int) -> None:
        key = (self.state_bin(z), int(action_bin))
        self.state_action_counts[key] = self.state_action_counts.get(key, 0) + 1

    def observe_transition(self, z: Array, action_bin: int, z_next: Array) -> None:
        key = (self.state_bin(z), int(action_bin), self.state_bin(z_next))
        self.transition_counts[key] = self.transition_counts.get(key, 0) + 1

    def observe_full(self, z: Array, action_bin: int, z_next: Array) -> None:
        self.observe_state(z)
        self.observe_state_action(z, action_bin)
        self.observe_transition(z, action_bin, z_next)


class StateActionDirectedPlanner:
    """
    Planner that chooses candidate action-branches using:
      - state novelty bonus via low N(z)
      - state-action novelty bonus via low N(z, a)
      - optional transition novelty via low N(z, a, z')
      - repulsion across agents
      - optional home / center / memory bias
      - optional goal bias
    """

    def __init__(
        self,
        latent_dim: int,
        tracker: Optional[StateActionCoverageTracker] = None,
        num_candidates: int = 12,
        center_weight: float = 0.0,
        home_weight: float = 0.05,
        memory_weight: float = 0.25,
        repel_weight: float = 0.30,
        state_novelty_weight: float = 0.50,
        state_action_novelty_weight: float = 1.00,
        transition_novelty_weight: float = 0.25,
        goal_weight: float = 0.0,
        jitter_scale: float = 0.10,
        memory_smooth: float = 0.7,
        repel_radius: float = 1.0,
        step_size: float = 0.5,
        count_eps: float = 1.0,
        goal_latent: Optional[Array] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.d = int(latent_dim)
        self.tracker = tracker or StateActionCoverageTracker(latent_dim=latent_dim, bin_size=1.0, action_bins=4)
        self.num_candidates = int(num_candidates)

        self.w_c = float(center_weight)
        self.w_h = float(home_weight)
        self.w_m = float(memory_weight)
        self.w_r = float(repel_weight)

        self.w_s = float(state_novelty_weight)
        self.w_sa = float(state_action_novelty_weight)
        self.w_t = float(transition_novelty_weight)
        self.w_g = float(goal_weight)

        self.jitter_scale = float(jitter_scale)
        self.gamma = float(memory_smooth)
        self.repel_radius = float(repel_radius)
        self.step_size = float(step_size)
        self.count_eps = float(count_eps)

        self.goal_latent = None if goal_latent is None else np.asarray(goal_latent, dtype=np.float32).reshape(-1)
        self.rng = np.random.RandomState() if rng is None else rng

        self.z_home: Optional[Array] = None
        self.memory: Optional[Array] = None

    def set_goal_latent(self, goal_latent: Optional[Array]) -> None:
        self.goal_latent = None if goal_latent is None else np.asarray(goal_latent, dtype=np.float32).reshape(-1)

    def reset_agents(self, z_init: Array) -> None:
        z_init = np.asarray(z_init, dtype=np.float32)
        self.z_home = z_init.copy()
        self.memory = np.zeros_like(z_init)
        for i in range(z_init.shape[0]):
            self.tracker.observe_state(z_init[i])

    def _latent_center(self, z_agents: Array) -> Array:
        return np.mean(z_agents, axis=0)

    def _repulsion_directions(self, z_agents: Array) -> Array:
        diff = z_agents[:, None, :] - z_agents[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        mask = (dist < self.repel_radius).astype(np.float32)
        safe_dist = np.where(np.isfinite(dist), np.clip(dist, 1e-8, None), 1.0)
        contrib = mask[:, :, None] * (diff / safe_dist[:, :, None])
        vec = np.sum(contrib, axis=1)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        out = np.zeros_like(vec)
        nonzero = norms.squeeze(-1) > 1e-8
        out[nonzero] = vec[nonzero] / norms[nonzero]
        return out

    def _count_bonus(self, n: int) -> float:
        return 1.0 / np.sqrt(float(n) + self.count_eps)

    def _goal_score(self, z_next: Array, z_curr: Array) -> float:
        if self.goal_latent is None:
            return 0.0
        d_curr = float(np.linalg.norm(self.goal_latent - z_curr))
        d_next = float(np.linalg.norm(self.goal_latent - z_next))
        return d_curr - d_next

    def _spacing_penalty(self, z_next: Array, all_next: List[Array]) -> float:
        penalty = 0.0
        for other in all_next:
            d = float(np.linalg.norm(z_next - other))
            if d < self.repel_radius:
                penalty += (self.repel_radius - d)
        return penalty

    def plan(self, z_agents: Array) -> Tuple[Array, Array, Array]:
        z_agents = np.asarray(z_agents, dtype=np.float32)
        n, d = z_agents.shape
        if d != self.d:
            raise ValueError(f"Expected latent dim {self.d}, got {d}")
        if self.z_home is None or self.memory is None:
            self.reset_agents(z_agents)

        directions = np.zeros_like(z_agents)
        action_bins = np.zeros((n,), dtype=np.int64)
        predicted_next = np.zeros_like(z_agents)

        z_center = self._latent_center(z_agents) if self.w_c != 0.0 else None
        repel = self._repulsion_directions(z_agents) if self.w_r != 0.0 else np.zeros_like(z_agents)

        chosen_nexts: List[Array] = []

        for i in range(n):
            base = np.zeros(self.d, dtype=np.float32)

            if self.w_c != 0.0 and z_center is not None:
                base += self.w_c * unit_vector(z_center - z_agents[i])
            if self.w_h != 0.0:
                base += self.w_h * unit_vector(self.z_home[i] - z_agents[i])
            if self.w_m != 0.0:
                base += self.w_m * unit_vector(self.memory[i])
            if self.w_r != 0.0:
                base += self.w_r * repel[i]
            if self.jitter_scale > 0.0:
                base += self.jitter_scale * unit_vector(self.rng.normal(size=self.d).astype(np.float32))

            if np.linalg.norm(base) > 1e-8:
                base = unit_vector(base)
            else:
                base = unit_vector(self.rng.normal(size=self.d).astype(np.float32))

            best_score = -1e18
            best_dir = base.copy()
            best_abin = self.tracker.action_bin_from_direction(best_dir)
            best_z_next = z_agents[i] + self.step_size * best_dir

            for _ in range(self.num_candidates):
                noise = unit_vector(self.rng.normal(size=self.d).astype(np.float32))
                cand_dir = unit_vector(0.7 * base + 0.3 * noise)
                abin = self.tracker.action_bin_from_direction(cand_dir)
                z_next = z_agents[i] + self.step_size * cand_dir

                n_s = self.tracker.get_state_count(z_next)
                n_sa = self.tracker.get_state_action_count(z_agents[i], abin)
                n_t = self.tracker.get_transition_count(z_agents[i], abin, z_next)

                score = 0.0
                score += self.w_s * self._count_bonus(n_s)
                score += self.w_sa * self._count_bonus(n_sa)
                score += self.w_t * self._count_bonus(n_t)
                score += self.w_g * self._goal_score(z_next, z_agents[i])
                score -= self.w_r * self._spacing_penalty(z_next, chosen_nexts)

                if score > best_score:
                    best_score = score
                    best_dir = cand_dir
                    best_abin = abin
                    best_z_next = z_next

            new_dir = unit_vector(self.gamma * best_dir + (1.0 - self.gamma) * self.memory[i])
            directions[i] = new_dir
            action_bins[i] = self.tracker.action_bin_from_direction(new_dir)
            predicted_next[i] = z_agents[i] + self.step_size * new_dir
            self.memory[i] = new_dir
            chosen_nexts.append(predicted_next[i].copy())

        return directions, action_bins, predicted_next

    def observe_transition(self, z: Array, action_bin: int, z_next: Array) -> None:
        self.tracker.observe_full(z, int(action_bin), z_next)


@dataclass
class CollectStats:
    num_transitions: int
    num_episodes_finished: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "num_transitions": self.num_transitions,
            "num_episodes_finished": self.num_episodes_finished,
        }


class MultiAgentExplorer:
    def __init__(
        self,
        envs: List[Any],
        encoder: Optional[Callable[[Array], Array]] = None,
        planner: Optional[StateActionDirectedPlanner] = None,
        latent_manager: Optional[LatentSpaceManager] = None,
        free_cells: Optional[Array] = None,
        seed: int = 0,
        goal_extractor: Optional[Callable[[Any], Optional[Array]]] = None,
    ):
        self.envs = list(envs)
        if len(self.envs) == 0:
            raise ValueError("envs must be non-empty")

        self.num_agents = len(self.envs)
        self.encoder = encoder or IdentityEncoder()
        self.rng = np.random.RandomState(seed)

        obs_dim = int(np.asarray(self.envs[0].observation_space.sample()).shape[0])
        self.latent_dim = obs_dim

        self.latent_manager = latent_manager or LatentSpaceManager(self.latent_dim, rng=self.rng)
        tracker = StateActionCoverageTracker(latent_dim=self.latent_dim, bin_size=1.0, action_bins=4)
        self.planner = planner or StateActionDirectedPlanner(
            latent_dim=self.latent_dim,
            tracker=tracker,
            rng=self.rng,
        )

        self.free_cells = None if free_cells is None else np.asarray(free_cells, dtype=np.float32)
        self.goal_extractor = goal_extractor

        self.current_obs: List[Array] = []
        self.ep_obs: List[List[Array]] = [[] for _ in range(self.num_agents)]
        self.ep_actions: List[List[int]] = [[] for _ in range(self.num_agents)]
        self.ep_rewards: List[List[float]] = [[] for _ in range(self.num_agents)]
        self.ep_next_obs: List[List[Array]] = [[] for _ in range(self.num_agents)]
        self.ep_terminated: List[List[float]] = [[] for _ in range(self.num_agents)]
        self.ep_truncated: List[List[float]] = [[] for _ in range(self.num_agents)]

    @classmethod
    def from_env(
        cls,
        env_fn: Callable[[], Any],
        num_agents: int,
        encoder: Optional[Callable[[Array], Array]] = None,
        planner_overrides: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        goal_extractor: Optional[Callable[[Any], Optional[Array]]] = None,
    ) -> "MultiAgentExplorer":
        envs = [env_fn() for _ in range(num_agents)]
        base_env = envs[0]

        free_cells = None
        raw_env = getattr(base_env, "env", base_env)
        if hasattr(raw_env, "free_cells"):
            free_cells = np.asarray(raw_env.free_cells, dtype=np.float32)

        obs_dim = int(np.asarray(base_env.observation_space.sample()).shape[0])
        rng = np.random.RandomState(seed)

        planner_cfg = dict(planner_overrides or {})
        bin_size = float(planner_cfg.pop("bin_size", 1.0))
        action_bins = int(planner_cfg.pop("action_bins", 4))

        tracker = StateActionCoverageTracker(
            latent_dim=obs_dim,
            bin_size=bin_size,
            action_bins=action_bins,
        )
        planner = StateActionDirectedPlanner(
            latent_dim=obs_dim,
            tracker=tracker,
            rng=rng,
            **planner_cfg,
        )
        latent_manager = LatentSpaceManager(latent_dim=obs_dim, rng=rng)

        return cls(
            envs=envs,
            encoder=encoder or IdentityEncoder(),
            planner=planner,
            latent_manager=latent_manager,
            free_cells=free_cells,
            seed=seed,
            goal_extractor=goal_extractor,
        )

    def _reset_env(self, env: Any):
        out = env.reset()
        if isinstance(out, tuple):
            obs = out[0]
        else:
            obs = out
        return np.asarray(obs, dtype=np.float32)

    def _maybe_extract_goal(self) -> None:
        if self.goal_extractor is None:
            return
        goal = self.goal_extractor(self.envs[0])
        if goal is None:
            return
        goal = np.asarray(goal, dtype=np.float32).reshape(1, -1)
        z_goal = np.asarray(self.encoder(goal), dtype=np.float32).reshape(-1)
        self.planner.set_goal_latent(z_goal)

    def _maybe_set_start_positions(self, obs_batch: Array) -> Array:
        if self.free_cells is None:
            return obs_batch
        self.latent_manager.update_stats(obs_batch)
        z_init = self.latent_manager.lhs_initial_latents(self.num_agents)
        snapped = np.stack([snap_to_nearest_free_cell(z, self.free_cells) for z in z_init], axis=0)
        for i, env in enumerate(self.envs):
            raw_env = getattr(env, "env", env)
            if hasattr(raw_env, "agent_position"):
                raw_env.agent_position = snapped[i].copy()
                obs_batch[i] = snapped[i].copy()
        return obs_batch

    def reset(self) -> Array:
        self.current_obs = [self._reset_env(env) for env in self.envs]
        obs_batch = np.stack(self.current_obs, axis=0).astype(np.float32)
        obs_batch = self._maybe_set_start_positions(obs_batch)
        self._maybe_extract_goal()

        z_batch = np.asarray(self.encoder(obs_batch), dtype=np.float32)
        if z_batch.ndim != 2 or z_batch.shape[0] != self.num_agents:
            raise ValueError(f"Encoder must return shape (num_agents, latent_dim), got {z_batch.shape}")

        self.planner.reset_agents(z_batch)

        for i in range(self.num_agents):
            self.current_obs[i] = obs_batch[i].copy()
            self.ep_obs[i].clear()
            self.ep_actions[i].clear()
            self.ep_rewards[i].clear()
            self.ep_next_obs[i].clear()
            self.ep_terminated[i].clear()
            self.ep_truncated[i].clear()

        return obs_batch

    def _direction_to_discrete_action(self, direction: Array) -> int:
        dx, dy = float(direction[0]), float(direction[1])
        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 2
        return 0 if dy > 0 else 1

    def _flush_episode(self, replay_buffer: Any, i: int) -> None:
        if len(self.ep_obs[i]) == 0:
            return
        replay_buffer.add_episode(
            dict(
                obs=self.ep_obs[i],
                actions=self.ep_actions[i],
                rewards=self.ep_rewards[i],
                next_obs=self.ep_next_obs[i],
                terminated=self.ep_terminated[i],
                truncated=self.ep_truncated[i],
            )
        )
        self.ep_obs[i].clear()
        self.ep_actions[i].clear()
        self.ep_rewards[i].clear()
        self.ep_next_obs[i].clear()
        self.ep_terminated[i].clear()
        self.ep_truncated[i].clear()

    def collect_steps(
        self,
        replay_buffer: Any,
        num_steps: int = 1,
        flush_partial_episodes: bool = False,
    ) -> Dict[str, int]:
        if not self.current_obs:
            self.reset()

        num_transitions = 0
        num_episodes_finished = 0

        for _ in range(num_steps):
            obs_batch = np.stack(self.current_obs, axis=0).astype(np.float32)
            z_batch = np.asarray(self.encoder(obs_batch), dtype=np.float32)
            if z_batch.ndim != 2 or z_batch.shape[0] != self.num_agents:
                raise ValueError(f"Encoder must return shape (num_agents, latent_dim), got {z_batch.shape}")

            directions, planned_action_bins, _pred_next = self.planner.plan(z_batch)

            for i, env in enumerate(self.envs):
                z_curr = z_batch[i].copy()
                action = self._direction_to_discrete_action(directions[i])

                out = env.step(action)
                if len(out) == 5:
                    next_obs, rew, terminated, truncated, _info = out
                else:
                    next_obs, rew, done, _info = out
                    terminated, truncated = bool(done), False

                next_obs = np.asarray(next_obs, dtype=np.float32)
                z_next = np.asarray(self.encoder(next_obs[None, :]), dtype=np.float32).reshape(-1)

                self.planner.observe_transition(z_curr, int(planned_action_bins[i]), z_next)

                self.ep_obs[i].append(np.asarray(self.current_obs[i], dtype=np.float32).copy())
                self.ep_actions[i].append(int(action))
                self.ep_rewards[i].append(float(rew))
                self.ep_next_obs[i].append(next_obs.copy())
                self.ep_terminated[i].append(float(terminated))
                self.ep_truncated[i].append(float(truncated))

                self.current_obs[i] = next_obs
                num_transitions += 1

                if terminated or truncated:
                    self._flush_episode(replay_buffer, i)
                    num_episodes_finished += 1
                    self.current_obs[i] = self._reset_env(env)

        if flush_partial_episodes:
            for i in range(self.num_agents):
                self._flush_episode(replay_buffer, i)

        return CollectStats(num_transitions, num_episodes_finished).as_dict()