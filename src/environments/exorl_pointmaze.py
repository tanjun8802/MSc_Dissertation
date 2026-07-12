import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ExORLPointMazeWrapper(gym.Env):
    def __init__(
        self,
        base_env,
        goal_xy,
        goal_radius=0.2,
        step_penalty=0.0,
        goal_reward=1.0,
        max_episode_steps=500,
        include_goal=False,
    ):
        super().__init__()
        self.env = base_env
        self.goal_xy = np.asarray(goal_xy, dtype=np.float32)
        self.goal_radius = float(goal_radius)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.max_episode_steps = int(max_episode_steps)
        self.include_goal = include_goal
        self._t = 0

        act_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        act_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        obs0 = self._extract_obs(self.env.reset())
        obs_dim = obs0.shape[0] + (2 if include_goal else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _unwrap_reset(self, out):
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return obs, info

    def _unwrap_step(self, out):
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        elif len(out) == 4:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        else:
            raise ValueError("Unexpected step return format")
        return obs, reward, terminated, truncated, info

    def _extract_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.include_goal:
            obs = np.concatenate([obs, self.goal_xy], axis=0)
        return obs.astype(np.float32)

    def _get_xy(self, obs, info):
        if "position" in info:
            return np.asarray(info["position"], dtype=np.float32)[:2]
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return obs[:2]

    def reset(self, *, seed=None, options=None):
        self._t = 0
        if seed is not None:
            out = self.env.reset(seed=seed)
        else:
            out = self.env.reset()
        obs, info = self._unwrap_reset(out)
        obs = self._extract_obs(obs)
        return obs, info

    def step(self, action):
        self._t += 1
        action = np.asarray(action, dtype=np.float32)
        out = self.env.step(action)
        obs_raw, _, terminated, truncated, info = self._unwrap_step(out)

        xy = self._get_xy(obs_raw, info)
        dist = np.linalg.norm(xy - self.goal_xy)
        reached = dist <= self.goal_radius

        reward = self.goal_reward if reached else self.step_penalty
        terminated = bool(terminated or reached)
        truncated = bool(truncated or (self._t >= self.max_episode_steps))

        obs = self._extract_obs(obs_raw)
        info = dict(info)
        info["goal_xy"] = self.goal_xy.copy()
        info["distance_to_goal"] = float(dist)
        info["success"] = bool(reached)
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()