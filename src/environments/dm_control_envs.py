# environments/dmcontrol_envs.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from dm_control import suite
except ImportError as e:
    raise ImportError(
        "dm_control is required for DMControlGymEnv. "
        "Install via `pip install dm_control`."
    ) from e


class DMControlGymEnv(gym.Env):
    """
    Wraps a dm_control (domain, task) into a Gymnasium Env with:
      - observation: flattened float32 vector
      - action: continuous Box
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        domain="cheetah",
        task="run",
        render_mode=None,
        max_episode_steps=1000,
        flatten_obs=True,
    ):
        super().__init__()
        self._env = suite.load(domain_name=domain, task_name=task)
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.flatten_obs = bool(flatten_obs)
        self.step_count = 0

        obs_spec = self._env.observation_spec()
        action_spec = self._env.action_spec()

        # Build observation space: concatenate all obs components into a single vector
        if self.flatten_obs:
            lows = []
            highs = []
            for key, spec in obs_spec.items():
                shape = int(np.prod(spec.shape))
                # dm_control specs may have min/max; fall back to [-inf, inf] if None
                low = spec.minimum if spec.minimum is not None else -np.inf
                high = spec.maximum if spec.maximum is not None else np.inf
                lows.append(np.full(shape, low, dtype=np.float32))
                highs.append(np.full(shape, high, dtype=np.float32))
            low_vec = np.concatenate(lows).astype(np.float32)
            high_vec = np.concatenate(highs).astype(np.float32)
            self.observation_space = spaces.Box(
                low=low_vec, high=high_vec, dtype=np.float32
            )
        else:
            # If you prefer dict observations, adapt here
            raise NotImplementedError("Non-flattened obs not implemented in this wrapper.")

        # Action space from dm_control spec
        act_low = action_spec.minimum
        act_high = action_spec.maximum
        self.action_space = spaces.Box(
            low=act_low.astype(np.float32),
            high=act_high.astype(np.float32),
            shape=action_spec.shape,
            dtype=np.float32,
        )

    def _flatten_obs(self, obs_dict):
        # obs_dict: OrderedDict of name -> np.array
        if not self.flatten_obs:
            return obs_dict
        flat_parts = [np.asarray(v, dtype=np.float32).ravel() for v in obs_dict.values()]
        return np.concatenate(flat_parts, axis=0)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # dm_control uses its own random seed
            self._env.reset()
        time_step = self._env.reset()
        self.step_count = 0
        obs = self._flatten_obs(time_step.observation)
        info = {}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        time_step = self._env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = float(time_step.reward or 0.0)
        terminated = bool(time_step.last())
        truncated = self.step_count >= self.max_episode_steps
        self.step_count += 1
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.physics.render(camera_id=0)
        return None

    def close(self):
        # dm_control does not need explicit close, but keep for API symmetry
        pass


def make_dmcontrol_env(
    domain="cheetah",
    task="run",
    max_episode_steps=1000,
):
    """
    Factory function for DMControlGymEnv, analogous to your makeenv(...) helper.
    """
    return DMControlGymEnv(
        domain=domain,
        task=task,
        max_episode_steps=max_episode_steps,
    )

class DMControlGoalWrapper(gym.Wrapper):
    """
    Goal-conditioned wrapper around DMControlGymEnv.

    Observation: concat(s, g) as 1D float32 vector.
    Reward:
      - simple: 1.0 if ||s - g||_2 <= goal_radius else 0.0
      - shaped: base +/- potential-based shaping like your MazeGoalWrapper.
    """

    def __init__(
        self,
        env: DMControlGymEnv,
        goal_state=None,
        goal_reward: float = 1.0,
        step_reward: float = 0.0,
        reward_mode: str = "simple",
        gamma: float = 0.99,
        goal_radius: float = 0.1,
    ):
        super().__init__(env)
        self.goal_reward = float(goal_reward)
        self.step_reward = float(step_reward)
        self.reward_mode = str(reward_mode).lower()
        self.gamma = float(gamma)
        self.goal_radius = float(goal_radius)

        base_dim = int(np.prod(self.env.observation_space.shape))
        low_base = self.env.observation_space.low.astype(np.float32).flatten()
        high_base = self.env.observation_space.high.astype(np.float32).flatten()

        self.observation_space = spaces.Box(
            low=np.concatenate([low_base, low_base]),
            high=np.concatenate([high_base, high_base]),
            dtype=np.float32,
        )

        self.goal = None
        if goal_state is not None:
            self.set_goal(goal_state)

        self.current_task = None  # for compatibility

    def set_goal(self, goal_state):
        """
        goal_state: same dimension as flattened DMControl state.
        """
        g = np.asarray(goal_state, dtype=np.float32).flatten()
        base_dim = int(np.prod(self.env.observation_space.shape))
        if g.shape[0] != base_dim:
            raise ValueError(
                f"Goal state dim {g.shape[0]} != base obs dim {base_dim}"
            )
        self.goal = g

    def sample_goal(self, num_steps=1_000, seed=None):
        """
        Sample a goal by rolling out random actions; pick a visited state.
        """
        rng = np.random.default_rng(seed)
        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32).flatten()
        last = obs
        for _ in range(num_steps):
            a = self.env.action_space.sample()
            next_obs, _, terminated, truncated, _ = self.env.step(a)
            next_obs = np.asarray(next_obs, dtype=np.float32).flatten()
            last = next_obs
            if terminated or truncated:
                break
        self.set_goal(last)
        return self.goal.copy()

    def _concat_obs_goal(self, obs):
        s = np.asarray(obs, dtype=np.float32).flatten()
        if self.goal is None:
            g = np.zeros_like(s)
        else:
            g = self.goal
        return np.concatenate([s, g], axis=0)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32).flatten()
        concat = self._concat_obs_goal(obs)
        info = dict(info)
        if self.goal is not None:
            info["goal_state"] = self.goal.copy()
        return concat, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32).flatten()
        s = obs
        g = self.goal if self.goal is not None else np.zeros_like(s)
        dist = float(np.linalg.norm(s - g))

        reached = dist <= self.goal_radius

        if self.reward_mode == "simple":
            reward = self.goal_reward if reached else self.step_reward
        else:
            # Potential-based shaping phi(s) = -||s - g||
            phi_s = -dist
            phi_prev = info.get("phi_prev", phi_s)
            base = self.goal_reward if reached else self.step_reward
            shaping = self.gamma * phi_s - phi_prev
            reward = base + shaping
            info = dict(info)
            info["phi_prev"] = phi_s

        if reached:
            terminated = True

        concat = self._concat_obs_goal(obs)
        info = dict(info)
        info["goal_dist"] = dist
        info["reached"] = reached

        return concat, reward, terminated, truncated, info