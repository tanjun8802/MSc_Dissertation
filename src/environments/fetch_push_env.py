import numpy as np
import gymnasium as gym

try:
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
except Exception:
    gymnasium_robotics = None


class FetchReachGoalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        env_id="FetchReach-v4",
        reward_type="sparse",
        max_episode_steps=50,
        render_mode=None,
        flatten_obs=False,
        goal=None,
        goal_radius=0.05,
        use_native_reward=True,
    ):
        super().__init__()
        self.env_id = env_id if env_id is not None else (
            "FetchReachDense-v4" if str(reward_type).lower() == "dense" else "FetchReach-v4"
        )
        self.reward_type = str(reward_type).lower()
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode
        self.flatten_obs = bool(flatten_obs)
        self.goal_radius = float(goal_radius)
        self.use_native_reward = bool(use_native_reward)
        self.current_goal = None
        if goal is not None:
            self.set_goal(goal)

        self._env = gym.make(
            self.env_id,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )

        self.action_space = self._env.action_space
        self.observation_space = self._build_observation_space(self._env.observation_space)
        self.step_count = 0

    def _build_observation_space(self, obs_space):
        if not isinstance(obs_space, gym.spaces.Dict):
            if self.flatten_obs:
                return obs_space
            raise TypeError("Expected dict observation space for FetchReach.")

        parts_low = []
        parts_high = []
        for k in ["observation", "achieved_goal", "desired_goal"]:
            if k in obs_space.spaces:
                sp = obs_space.spaces[k]
                parts_low.append(np.asarray(sp.low, dtype=np.float32).ravel())
                parts_high.append(np.asarray(sp.high, dtype=np.float32).ravel())

        low = np.concatenate(parts_low, axis=0)
        high = np.concatenate(parts_high, axis=0)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _flatten_obs(self, obs):
        if not isinstance(obs, dict):
            return np.asarray(obs, dtype=np.float32).ravel()
        parts = []
        for k in ["observation", "achieved_goal", "desired_goal"]:
            if k in obs:
                parts.append(np.asarray(obs[k], dtype=np.float32).ravel())
        for k, v in obs.items():
            if k not in {"observation", "achieved_goal", "desired_goal"}:
                parts.append(np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(parts, axis=0)

    def set_goal(self, goal):
        g = np.asarray(goal, dtype=np.float32).ravel()
        if g.shape[0] != 3:
            raise ValueError(f"FetchReach goal must be 3D, got shape {g.shape}")
        self.current_goal = g

    def sample_goal(self, low=(-0.2, 0.3, 0.42), high=(0.2, 0.9, 0.42), seed=None):
        rng = np.random.default_rng(seed)
        g = rng.uniform(low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32))
        self.set_goal(g)
        return g.copy()

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self.step_count = 0

        if self.current_goal is not None and isinstance(obs, dict) and "desired_goal" in obs:
            obs = dict(obs)
            obs["desired_goal"] = self.current_goal.copy()
            info = dict(info)
            info["desired_goal"] = self.current_goal.copy()

        if self.flatten_obs:
            obs = self._flatten_obs(obs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.step_count += 1

        if isinstance(obs, dict):
            achieved = np.asarray(obs["achieved_goal"], dtype=np.float32).ravel()
            desired = np.asarray(obs["desired_goal"], dtype=np.float32).ravel()

            if self.current_goal is not None:
                desired = self.current_goal.copy()

            dist = float(np.linalg.norm(achieved - desired))
            success = dist <= self.goal_radius

            if not self.use_native_reward:
                reward = 0.0
                if self.reward_type == "dense":
                    reward = -dist
                else:
                    reward = 1.0 if success else 0.0

            info = dict(info)
            info["achieved_goal"] = achieved.copy()
            info["desired_goal"] = desired.copy()
            info["goal_dist"] = dist
            info["success"] = success

            if success:
                terminated = True

        if self.flatten_obs:
            obs = self._flatten_obs(obs)

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def make_fetch_push_env(
    goal=None,
    reward_type="sparse",
    max_episode_steps=50,
    render_mode=None,
    flatten_obs=False,
    goal_radius=0.05,
    use_native_reward=True,
):
    return FetchReachGoalEnv(
        goal=goal,
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        flatten_obs=flatten_obs,
        goal_radius=goal_radius,
        use_native_reward=use_native_reward,
    )


def make_env(
    goal=None,
    reward_type="dense",
    max_episode_steps=50,
    render_mode=None,
    flatten_obs=True,
    goal_radius=0.05,
    use_native_reward=True,
    seed=None,
):
    def _init():
        env = make_fetch_push_env(
            goal=goal,
            reward_type=reward_type,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            flatten_obs=flatten_obs,
            goal_radius=goal_radius,
            use_native_reward=use_native_reward,
        )
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _init