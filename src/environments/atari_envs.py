# ============================================================
# Standalone Atari environment wrappers
# ============================================================

from __future__ import annotations

import gymnasium as gym
import numpy as np


class AtariEnvWrapper(gym.Env):
    """
    Atari preprocessing wrapper.

    Applies:
        1. AtariPreprocessing
        2. FrameStackObservation

    This is an ordinary Atari environment wrapper. It is
    independent of the manipulator wrappers and is not a
    goal-conditioned HER environment.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ]
    }

    def __init__(
        self,
        env_id: str = "ALE/Breakout-v5",
        render_mode: str | None = None,
        frame_stack: int = 4,
        max_episode_steps: int | None = 108_000,
        flatten_obs: bool = False,
    ):
        super().__init__()

        if frame_stack <= 0:
            raise ValueError(
                "frame_stack must be positive."
            )

        self.env_id = env_id
        self.render_mode = render_mode
        self.frame_stack = int(frame_stack)
        self.flatten_obs = bool(flatten_obs)

        # The underlying Gymnasium TimeLimit wrapper owns
        # episode truncation. No second manual horizon is used.
        base_env = gym.make(
            env_id,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            frameskip=1,
        )

        base_env = gym.wrappers.AtariPreprocessing(
            base_env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )

        base_env = gym.wrappers.FrameStackObservation(
            base_env,
            stack_size=self.frame_stack,
        )

        self.env = base_env

        self.action_space = (
            self.env.action_space
        )

        if not isinstance(
            self.env.observation_space,
            gym.spaces.Box,
        ):
            raise TypeError(
                "Expected a Box observation space after "
                "Atari preprocessing and frame stacking."
            )

        self.base_observation_space = (
            self.env.observation_space
        )

        self.observation_space = (
            self._build_observation_space(
                self.base_observation_space
            )
        )

        self.action_names = [
            str(action)
            for action in range(
                self.action_space.n
            )
        ]

    def _build_observation_space(
        self,
        base_space: gym.spaces.Box,
    ) -> gym.spaces.Box:
        low = np.asarray(
            base_space.low,
            dtype=np.float32,
        )

        high = np.asarray(
            base_space.high,
            dtype=np.float32,
        )

        if self.flatten_obs:
            low = low.reshape(-1)
            high = high.reshape(-1)

        return gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def _format_obs(
        self,
        obs,
    ) -> np.ndarray:
        obs = np.asarray(
            obs,
            dtype=np.float32,
        )

        if self.flatten_obs:
            obs = obs.reshape(-1)

        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        obs, info = self.env.reset(
            seed=seed,
            options=options,
        )

        obs = self._format_obs(
            obs
        )

        return obs, dict(info)

    def step(
        self,
        action,
    ):
        action = int(action)

        if not self.action_space.contains(
            action
        ):
            raise ValueError(
                f"Invalid Atari action: {action}"
            )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        obs = self._format_obs(
            obs
        )

        # Preserve the underlying terminated/truncated flags.
        return (
            obs,
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def valid_action_mask(
        self,
        obs=None,
    ):
        return np.ones(
            self.action_space.n,
            dtype=bool,
        )


def make_atari_env(
    env_id: str = "ALE/Breakout-v5",
    frame_stack: int = 4,
    max_episode_steps: int | None = 108_000,
    flatten_obs: bool = False,
    render_mode: str | None = None,
):
    """
    Creates a standalone preprocessed Atari environment.
    """

    return AtariEnvWrapper(
        env_id=env_id,
        render_mode=render_mode,
        frame_stack=frame_stack,
        max_episode_steps=max_episode_steps,
        flatten_obs=flatten_obs,
    )


class AtariScoreGoalWrapper(gym.Wrapper):
    """
    Adds a score threshold to an Atari environment.

    The observation remains an ordinary ndarray.
    This wrapper is independent of HER and does not expose
    achieved_goal/desired_goal dictionary observations.
    """

    def __init__(
        self,
        env: gym.Env,
        goal_score: float | None = None,
        goal_reward: float = 1.0,
        step_reward: float = 0.0,
        reward_mode: str = "simple",
        terminate_on_goal: bool = True,
    ):
        super().__init__(env)

        self.goal_score = (
            None
            if goal_score is None
            else float(goal_score)
        )

        self.goal_reward = float(
            goal_reward
        )

        self.step_reward = float(
            step_reward
        )

        self.reward_mode = str(
            reward_mode
        ).lower()

        if self.reward_mode not in {
            "simple",
            "native",
        }:
            raise ValueError(
                "reward_mode must be either "
                "'simple' or 'native'."
            )

        self.terminate_on_goal = bool(
            terminate_on_goal
        )

        self.current_score = 0.0
        self.goal_reached = False

    def set_goal(
        self,
        goal_score,
    ):
        goal_array = np.asarray(
            goal_score,
            dtype=np.float32,
        ).reshape(-1)

        if goal_array.shape != (1,):
            raise ValueError(
                "Atari score goal must have shape "
                "(1,)."
            )

        self.goal_score = float(
            goal_array[0]
        )

        self.goal_reached = False

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        obs, info = self.env.reset(
            seed=seed,
            options=options,
        )

        self.current_score = 0.0
        self.goal_reached = False

        info = dict(info)

        info["goal_score"] = (
            self.goal_score
        )

        info["current_score"] = (
            self.current_score
        )

        info["reached"] = False
        info["is_success"] = 0.0

        return obs, info

    def step(
        self,
        action,
    ):
        (
            obs,
            native_reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        native_reward = float(
            native_reward
        )

        info = dict(info)

        # Breakout reward is the score increment.
        self.current_score += (
            native_reward
        )

        if self.goal_score is None:
            reached = False
            reward = native_reward

        else:
            reached = (
                self.current_score
                >= self.goal_score
            )

            if self.reward_mode == "native":
                reward = native_reward

            else:
                first_reach = (
                    reached
                    and not self.goal_reached
                )

                reward = (
                    self.goal_reward
                    if first_reach
                    else self.step_reward
                )

        if reached:
            self.goal_reached = True

            if self.terminate_on_goal:
                terminated = True

        info["current_score"] = (
            self.current_score
        )

        info["goal_score"] = (
            self.goal_score
        )

        info["reached"] = bool(
            reached
        )

        info["is_success"] = float(
            reached
        )

        return (
            obs,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def valid_action_mask(
        self,
        obs=None,
    ):
        if hasattr(
            self.env,
            "valid_action_mask",
        ):
            return self.env.valid_action_mask(
                obs
            )

        return np.ones(
            self.action_space.n,
            dtype=bool,
        )


def make_atari_goal_env(
    env_id: str = "ALE/Breakout-v5",
    frame_stack: int = 4,
    goal_obs=None,
    reward_mode: str = "simple",
    max_episode_steps: int | None = 108_000,
    render_mode: str | None = None,
    terminate_on_goal: bool = True,
):
    """
    Creates an Atari environment with an optional score goal.

    Parameters
    ----------
    goal_obs:
        None, or a one-element array such as:
        np.array([25.0], dtype=np.float32)
    """

    base_env = make_atari_env(
        env_id=env_id,
        frame_stack=frame_stack,
        max_episode_steps=max_episode_steps,
        flatten_obs=False,
        render_mode=render_mode,
    )

    goal_score = None

    if goal_obs is not None:
        goal_array = np.asarray(
            goal_obs,
            dtype=np.float32,
        ).reshape(-1)

        if goal_array.shape != (1,):
            raise ValueError(
                "Breakout goal must have shape "
                "(1,), for example "
                "np.array([25.0])."
            )

        goal_score = float(
            goal_array[0]
        )

    return AtariScoreGoalWrapper(
        env=base_env,
        goal_score=goal_score,
        goal_reward=1.0,
        step_reward=0.0,
        reward_mode=reward_mode,
        terminate_on_goal=terminate_on_goal,
    )


