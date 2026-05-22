"""Gymnasium adapter for robosuite environments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RobosuiteGymWrapper(gym.Env[np.ndarray | dict[str, np.ndarray], np.ndarray]):
    """Expose robosuite tasks through the Gymnasium API."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_name: str = "Lift",
        robots: str | list[str] = "Panda",
        controller_configs: dict[str, Any] | None = None,
        render_mode: str | None = None,
        flatten_observation: bool = True,
        **robosuite_kwargs: Any,
    ) -> None:
        super().__init__()
        try:
            import robosuite as suite
        except ImportError as exc:
            raise ImportError(
                "robosuite is required for RobosuiteGymWrapper. Install project dependencies first."
            ) from exc

        self._suite = suite
        self.render_mode = render_mode
        self.flatten_observation = flatten_observation

        self._env = suite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=controller_configs,
            **robosuite_kwargs,
        )

        low, high = self._env.action_spec
        self.action_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

        initial_obs = self._env.reset()
        self.observation_space = self._build_observation_space(initial_obs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            try:
                self._env.seed(seed)
            except AttributeError:
                pass

        _ = options
        obs = self._env.reset()
        return self._process_observation(obs), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, done, info = self._env.step(clipped_action)

        terminated = bool(done)
        truncated = bool(info.get("TimeLimit.truncated", False)) if isinstance(info, dict) else False
        return self._process_observation(obs), float(reward), terminated, truncated, info

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def _process_observation(self, obs: Any) -> np.ndarray | dict[str, np.ndarray]:
        if not isinstance(obs, Mapping):
            return np.asarray(obs, dtype=np.float32)

        obs_dict = {key: np.asarray(value, dtype=np.float32).reshape(-1) for key, value in obs.items()}
        if self.flatten_observation:
            return np.concatenate([obs_dict[key] for key in sorted(obs_dict)], axis=0)
        return obs_dict

    def _build_observation_space(self, obs: Any) -> gym.Space:
        processed = self._process_observation(obs)
        if isinstance(processed, dict):
            return spaces.Dict(
                {
                    key: spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=np.float32)
                    for key, value in processed.items()
                }
            )
        return spaces.Box(low=-np.inf, high=np.inf, shape=processed.shape, dtype=np.float32)
