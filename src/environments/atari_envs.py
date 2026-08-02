# environments/atari_envs.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AtariEnvWrapper(gym.Env):
    """
    High-dimensional Atari env with pixel observations and discrete actions,
    wrapped to look like your existing MazeGridWorld envs.
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id="ALE/Breakout-v5",
        render_mode=None,
        frame_stack=4,
        max_episode_steps=108_000,  # standard long horizon
    ):
        super().__init__()

        base_env = gym.make(env_id, render_mode="rgb_array")
        # Standard Atari preprocessing: grayscale, resize, no frame skipping here
        base_env = gym.wrappers.AtariPreprocessing(
            base_env,
            grayscale_obs=True,
            scale_obs=False,
            frame_skip=4,
            screen_size=84,
        )
        # Stack last N frames to get temporal information
        base_env = gym.wrappers.FrameStack(base_env, num_stack=frame_stack)

        self.env = base_env
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.step_count = 0

        # Observation space: typically (84, 84, frame_stack)
        self.observation_space = self.env.observation_space
        # Action space: discrete joystick actions
        self.action_space = self.env.action_space

        # For compatibility with your other envs:
        self.action_names = [str(a) for a in range(self.action_space.n)]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.step_count = 0
        # Gymnasium AtariPreprocessing returns uint8; cast to float32 if you prefer
        obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if self.step_count >= self.max_episode_steps:
            truncated = True
        obs = np.asarray(obs, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def valid_action_mask(self, obs):
        """
        For compatibility with MazeGridWorld.valid_action_mask:
        all Atari actions are always available, so this is just all True.
        """
        return np.ones(self.action_space.n, dtype=bool)


def make_atari_env(
    env_id="ALE/Breakout-v5",
    frame_stack=4,
    max_episode_steps=108_000,
):
    """
    Factory to match the pattern used in your notebook's makeenv(...) helper.
    """
    return AtariEnvWrapper(
        env_id=env_id,
        frame_stack=frame_stack,
        max_episode_steps=max_episode_steps,
    )

class AtariGoalWrapper(gym.Wrapper):
    """
    Goal-conditioned wrapper around AtariEnvWrapper.

    Observation: concat(s, g) as a 1D float32 vector (flattened).
    Reward:
      - simple: 1.0 if ||s - g||_2 <= goal_radius else 0.0
      - shaped: goal_reward at goal, else step_reward + potential-based shaping.
    """

    def __init__(
        self,
        env: AtariEnvWrapper,
        goal_obs=None,
        goal_reward: float = 1.0,
        step_reward: float = 0.0,
        reward_mode: str = "simple",
        gamma: float = 0.99,
        goal_radius: float = 1e-3,
    ):
        super().__init__(env)
        self.goal_reward = float(goal_reward)
        self.step_reward = float(step_reward)
        self.reward_mode = str(reward_mode).lower()
        self.gamma = float(gamma)
        self.goal_radius = float(goal_radius)

        # Flatten base obs space to 1D vector and double it (state + goal)
        base_shape = int(np.prod(self.env.observation_space.shape))
        low_base = np.full(base_shape, 0.0, dtype=np.float32)
        high_base = np.full(
            base_shape,
            255.0 if self.env.observation_space.dtype == np.uint8 else 1.0,
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.concatenate([low_base, low_base]),
            high=np.concatenate([high_base, high_base]),
            dtype=np.float32,
        )

        self.goal = None
        if goal_obs is not None:
            self.set_goal(goal_obs)

        # Optional current task id, for tracking coverage if you want
        self.current_task = None

    def set_goal(self, goal_obs):
        """
        goal_obs can be an observation (same shape as env.obs).
        We flatten and store as float32.
        """
        g = np.asarray(goal_obs, dtype=np.float32)
        if g.shape != self.env.observation_space.shape:
            raise ValueError(
                f"Goal obs shape {g.shape} != base obs shape {self.env.observation_space.shape}"
            )
        self.goal = g.flatten()

    def sample_goal(self, num_steps=1_000, seed=None):
        """
        Simple heuristic for sampling a goal: run a random policy for some steps
        and pick a visited observation as goal.
        """
        rng = np.random.default_rng(seed)
        obs, _ = self.env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        last = obs
        for _ in range(num_steps):
            a = self.env.action_space.sample()
            next_obs, _, terminated, truncated, _ = self.env.step(a)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            last = next_obs
            if terminated or truncated:
                break
        self.set_goal(last)
        return self.goal.copy()

    def _concat_obs_goal(self, obs):
        s = np.asarray(obs, dtype=np.float32).flatten()
        if self.goal is None:
            # Default: use zero goal; you can force set_goal before training
            g = np.zeros_like(s)
        else:
            g = self.goal
        return np.concatenate([s, g], axis=0)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)
        concat = self._concat_obs_goal(obs)
        info = dict(info)
        if self.goal is not None:
            info["goal_obs"] = self.goal.copy()
        return concat, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        s_flat = obs.flatten()
        g_flat = self.goal if self.goal is not None else np.zeros_like(s_flat)
        dist = float(np.linalg.norm(s_flat - g_flat))

        reached = dist <= self.goal_radius

        if self.reward_mode == "simple":
            reward = self.goal_reward if reached else self.step_reward
        else:
            # Potential-based shaping: phi(s) = -||s - g||_2
            phi_s = -dist
            # Use last obs from info if provided, else treat s_prev ~ s
            # For simplicity we'll approximate with no shaping on first step
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

    def valid_action_mask(self, obs):
        # Delegate to wrapped env; for Atari we return all-True.
        if hasattr(self.env, "valid_action_mask"):
            return self.env.valid_action_mask(obs)
        return np.ones(self.action_space.n, dtype=bool)


def make_atari_goal_env(
    env_id="ALE/Breakout-v5",
    frame_stack=4,
    goal_obs=None,
    reward_mode="simple",
    goal_radius=1e-3,
    max_episode_steps=108_000,
):
    base = AtariEnvWrapper(
        env_id=env_id,
        frame_stack=frame_stack,
        max_episode_steps=max_episode_steps,
    )
    wrapper = AtariGoalWrapper(
        base,
        goal_obs=goal_obs,
        reward_mode=reward_mode,
        goal_radius=goal_radius,
    )
    return wrapper