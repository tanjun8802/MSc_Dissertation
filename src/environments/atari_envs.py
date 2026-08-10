# environments/atari_envs.py

import gymnasium as gym
import numpy as np


class AtariEnvWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id="ALE/Breakout-v5",
        render_mode=None,
        frame_stack=4,
        max_episode_steps=108_000,
        flatten_obs=True,
    ):
        super().__init__()

        base_env = gym.make(env_id, render_mode="rgb_array")
        base_env = gym.wrappers.AtariPreprocessing(
            base_env,
            grayscale_obs=True,
            scale_obs=False,
            frame_skip=1,
            screen_size=84,
        )
        base_env = gym.wrappers.FrameStackObservation(base_env, stack_size=frame_stack)

        self.env = base_env
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.flatten_obs = bool(flatten_obs)
        self.step_count = 0

        base_obs_space = self.env.observation_space
        self.base_observation_space = base_obs_space

        if self.flatten_obs:
            obsdim = int(np.prod(base_obs_space.shape))
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=255.0,
                shape=(obsdim,),
                dtype=np.float32,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=255.0,
                shape=base_obs_space.shape,
                dtype=np.float32,
            )

        self.action_space = self.env.action_space
        self.action_names = [str(a) for a in range(self.action_space.n)]

    def _format_obs(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        if self.flatten_obs:
            x = x.reshape(-1)
        return x

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.step_count = 0
        return self._format_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if self.step_count >= self.max_episode_steps:
            truncated = True
        return self._format_obs(obs), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def valid_action_mask(self, obs=None):
        return np.ones(self.action_space.n, dtype=bool)


def make_atari_env(
    env_id="ALE/Breakout-v5",
    frame_stack=4,
    max_episode_steps=108_000,
    flatten_obs=True,
):
    return AtariEnvWrapper(
        env_id=env_id,
        frame_stack=frame_stack,
        max_episode_steps=max_episode_steps,
        flatten_obs=flatten_obs,
    )


class AtariGoalWrapper(gym.Wrapper):
    def __init__(
        self,
        env: AtariEnvWrapper,
        goal_obs=None,
        goal_reward: float = 1.0,
        step_reward: float = 0.0,
        reward_mode: str = "simple",
        gamma: float = 0.99,
        goal_radius: float = 100.0,
    ):
        super().__init__(env)
        self.goal_reward = float(goal_reward)
        self.step_reward = float(step_reward)
        self.reward_mode = str(reward_mode).lower()
        self.gamma = float(gamma)
        self.goal_radius = float(goal_radius)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.goal = None
        self.prev_phi = None
        self.current_task = None

        if goal_obs is not None:
            self.set_goal(goal_obs)

    def set_goal(self, goal_obs):
        g = np.asarray(goal_obs, dtype=np.float32).reshape(-1)
        if g.shape != self.observation_space.shape:
            raise ValueError(
                f"Goal obs shape {g.shape} != obs shape {self.observation_space.shape}"
            )
        self.goal = g.copy()

    def sample_goal(self, num_steps=1000, seed=None):
        rng = np.random.default_rng(seed)
        obs, _ = self.env.reset(seed=seed)
        last = np.asarray(obs, dtype=np.float32).reshape(-1)
        for _ in range(num_steps):
            a = int(rng.integers(self.action_space.n))
            next_obs, _, terminated, truncated, _ = self.env.step(a)
            last = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            if terminated or truncated:
                break
        self.set_goal(last)
        return self.goal.copy()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_phi = None
        info = dict(info)
        if self.goal is not None:
            info["goal_obs"] = self.goal.copy()
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        s = np.asarray(obs, dtype=np.float32).reshape(-1)

        if self.goal is None:
            reward = float(env_reward)
            reached = False
            dist = np.nan
        else:
            dist = float(np.linalg.norm(s - self.goal))
            reached = dist <= self.goal_radius

            if self.reward_mode == "simple":
                reward = self.goal_reward if reached else self.step_reward
            elif self.reward_mode == "shaped":
                phi = -dist
                if self.prev_phi is None:
                    shaping = 0.0
                else:
                    shaping = self.gamma * phi - self.prev_phi
                base = self.goal_reward if reached else self.step_reward
                reward = base + shaping
                self.prev_phi = phi
            else:
                reward = float(env_reward)

        if reached:
            terminated = True

        info = dict(info)
        info["goal_dist"] = dist
        info["reached"] = reached
        if self.goal is not None:
            info["goal_obs"] = self.goal.copy()

        return obs, float(reward), terminated, truncated, info

    def valid_action_mask(self, obs=None):
        if hasattr(self.env, "valid_action_mask"):
            return self.env.valid_action_mask(obs)
        return np.ones(self.action_space.n, dtype=bool)


def make_atari_goal_env(
    env_id="ALE/Breakout-v5",
    frame_stack=4,
    goal_obs=None,
    reward_mode="simple",
    goal_radius=100.0,
    max_episode_steps=108_000,
    flatten_obs=False,
):
    base = AtariEnvWrapper(
        env_id=env_id,
        frame_stack=frame_stack,
        max_episode_steps=max_episode_steps,
        flatten_obs=flatten_obs,
    )
    env = AtariGoalWrapper(
        base,
        goal_obs=goal_obs,
        reward_mode=reward_mode,
        goal_radius=goal_radius,
    )
    return env