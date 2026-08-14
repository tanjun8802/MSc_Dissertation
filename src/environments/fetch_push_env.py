import numpy as np
import gymnasium as gym
import os

from stable_baselines3.common.callbacks import BaseCallback

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

class FixedGoalWrapper(gym.Wrapper):
    """
    Forces FetchReach to use one selected target position.

    The returned observation remains:
        observation
        achieved_goal
        desired_goal
    """

    def __init__(self, env, goal):
        super().__init__(env)

        self.fixed_goal = np.asarray(
            goal,
            dtype=np.float32,
        ).reshape(-1)

        if self.fixed_goal.shape != (3,):
            raise ValueError(
                "Goal must have shape (3,), "
                f"got {self.fixed_goal.shape}"
            )

    def _set_goal(self, obs):
        self.env.unwrapped.goal = (
            self.fixed_goal.copy()
        )

        obs = dict(obs)
        obs["desired_goal"] = (
            self.fixed_goal.copy()
        )

        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs = self._set_goal(obs)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        obs = self._set_goal(obs)

        achieved_goal = np.asarray(
            obs["achieved_goal"],
            dtype=np.float32,
        )

        distance = float(
            np.linalg.norm(
                achieved_goal - self.fixed_goal
            )
        )

        info = dict(info)
        info["goal_dist"] = distance
        info["is_success"] = float(
            distance <= 0.05
        )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )


import math
import os

import numpy as np

from stable_baselines3.common.callbacks import (
    BaseCallback,
)


class SuccessRateEvalCallback(
    BaseCallback
):
    def __init__(
        self,
        eval_env,
        eval_freq,
        n_eval_episodes=100,
        success_threshold=0.95,
        return_threshold=None,
        required_checkpoints=3,
        log_path=None,
        deterministic=True,
        verbose=1,
    ):
        super().__init__(verbose)

        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = (
            n_eval_episodes
        )

        self.success_threshold = (
            success_threshold
        )

        # Minimum acceptable mean episode return.
        # For sparse FetchReach, -10 means
        # approximately reaching the goal within
        # 10 steps.
        self.return_threshold = (
            return_threshold
        )

        self.required_checkpoints = (
            required_checkpoints
        )

        self.log_path = log_path
        self.deterministic = deterministic

        self.eval_steps = []
        self.success_rates = []
        self.eval_returns = []

        self.qualified_success_rates = []

        self.consecutive_successful_evals = 0
        self.steps_to_target = None

        if self.log_path is not None:
            os.makedirs(
                self.log_path,
                exist_ok=True,
            )

    def _evaluate_success_rate(self):
        successes = []
        episode_returns = []
        qualified_successes = []

        for episode_idx in range(
            self.n_eval_episodes
        ):
            obs, info = (
                self.eval_env.reset(
                    seed=(
                        self.n_calls
                        + episode_idx
                    )
                )
            )

            terminated = False
            truncated = False

            episode_return = 0.0
            final_info = {}

            while not (
                terminated or truncated
            ):
                action, _ = (
                    self.model.predict(
                        obs,
                        deterministic=(
                            self.deterministic
                        ),
                    )
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    final_info,
                ) = self.eval_env.step(
                    action
                )

                episode_return += float(
                    reward
                )

            is_success = float(
                final_info.get(
                    "is_success",
                    0.0,
                )
            )

            successes.append(is_success)
            episode_returns.append(
                episode_return
            )

            # A qualified success must both:
            # 1. reach the goal
            # 2. meet the return requirement
            if self.return_threshold is None:
                is_qualified_success = (
                    is_success >= 1.0
                )
            else:
                is_qualified_success = (
                    is_success >= 1.0
                    and episode_return
                    >= self.return_threshold
                )

            qualified_successes.append(
                float(
                    is_qualified_success
                )
            )

        success_rate = float(
            np.mean(successes)
        )

        mean_return = float(
            np.mean(episode_returns)
        )

        qualified_success_rate = float(
            np.mean(qualified_successes)
        )

        return (
            success_rate,
            mean_return,
            qualified_success_rate,
        )

    def _save_results(self):
        if self.log_path is None:
            return

        save_path = os.path.join(
            self.log_path,
            "success_evaluations.npz",
        )

        np.savez(
            save_path,
            timesteps=np.asarray(
                self.eval_steps
            ),
            success_rates=np.asarray(
                self.success_rates
            ),
            returns=np.asarray(
                self.eval_returns
            ),
            qualified_success_rates=(
                np.asarray(
                    self.qualified_success_rates
                )
            ),
        )

    def _on_step(self):
        if (
            self.n_calls
            % self.eval_freq
            != 0
        ):
            return True

        (
            success_rate,
            mean_return,
            qualified_success_rate,
        ) = self._evaluate_success_rate()

        self.eval_steps.append(
            self.num_timesteps
        )

        self.success_rates.append(
            success_rate
        )

        self.eval_returns.append(
            mean_return
        )

        self.qualified_success_rates.append(
            qualified_success_rate
        )

        required_successes = math.ceil(
            self.success_threshold
            * self.n_eval_episodes
        )

        success_condition = (
            success_rate
            >= self.success_threshold
        )

        qualified_success_condition = (
            qualified_success_rate
            >= self.success_threshold
        )

        if self.return_threshold is None:
            return_condition = True
        else:
            return_condition = (
                mean_return
                >= self.return_threshold
            )

        passed = (
            success_condition
            and qualified_success_condition
            and return_condition
        )

        if passed:
            self.consecutive_successful_evals += 1
        else:
            self.consecutive_successful_evals = 0

        return_text = (
            "disabled"
            if self.return_threshold is None
            else f"{self.return_threshold:.3f}"
        )

        print(
            f"Step {self.num_timesteps:,} | "
            f"Success: {success_rate:.3f} | "
            f"Qualified: "
            f"{qualified_success_rate:.3f} | "
            f"Return: {mean_return:.3f} | "
            f"Return threshold: "
            f"{return_text} | "
            f"Required successes: "
            f"{required_successes}/{self.n_eval_episodes} | "
            f"Consecutive: "
            f"{self.consecutive_successful_evals}/"
            f"{self.required_checkpoints}"
        )

        if (
            self.consecutive_successful_evals
            >= self.required_checkpoints
        ):
            self.steps_to_target = (
                self.num_timesteps
            )

            print(
                "\nTarget performance reached:"
            )

            print(
                f"  success rate >= "
                f"{self.success_threshold:.2f}"
            )

            print(
                f"  qualified success rate >= "
                f"{self.success_threshold:.2f}"
            )

            if self.return_threshold is not None:
                print(
                    f"  mean return >= "
                    f"{self.return_threshold:.3f}"
                )

            print(
                "Stopping training."
            )

            self._save_results()

            return False

        self._save_results()

        return True

def sample_goals_from_resets(
    make_env,
    num_goals,
    seed,
):
    """
    Sample goals from the environment's own reset distribution.

    Assumes reset() either returns:
        obs["desired_goal"]

    or stores the sampled goal as:
        env.unwrapped.goal
    """

    env = make_env()

    sampled_goals = []

    for goal_idx in range(num_goals):
        obs, info = env.reset(
            seed=seed + goal_idx
        )

        if isinstance(obs, dict) and (
            "desired_goal" in obs
        ):
            goal = obs["desired_goal"]

        elif isinstance(info, dict) and (
            "goal" in info
        ):
            goal = info["goal"]

        elif hasattr(env.unwrapped, "goal"):
            goal = env.unwrapped.goal

        else:
            raise RuntimeError(
                "Could not identify the sampled goal. "
                "Expected obs['desired_goal'], "
                "info['goal'], or env.unwrapped.goal."
            )

        sampled_goals.append(
            np.asarray(
                goal,
                dtype=np.float32,
            ).copy()
        )

    env.close()

    return sampled_goals