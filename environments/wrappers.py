"""
wrappers.py
===========
Lightweight environment wrappers that modify observations or episode length
without changing the underlying environment logic.

Each wrapper follows the Gymnasium ``Wrapper`` pattern: it wraps a
``BaseEnv`` (or a Gymnasium ``Env``) and forwards all calls to the
inner environment, overriding only the methods it needs to change.
"""

from __future__ import annotations

from typing import Any, SupportsFloat, Tuple

import numpy as np

from environments.base_env import BaseEnv


class _Wrapper(BaseEnv):
    """Thin pass-through wrapper; subclasses override as needed."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__()
        self.env = env

    # Forward all Gymnasium-style calls to the inner env by default.
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict]:
        return self.env.step(action)

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.env!r})"


class TimeLimit(_Wrapper):
    """Truncate an episode after ``max_steps`` steps.

    Useful when the underlying environment does not enforce a time limit
    (e.g. reward-free GridWorld with ``max_steps=None``).

    Parameters
    ----------
    env :
        The environment to wrap.
    max_steps :
        Maximum number of steps per episode before truncation.
    """

    def __init__(self, env: BaseEnv, max_steps: int = 1000) -> None:
        super().__init__(env)
        self.max_steps = max_steps
        self._elapsed_steps: int = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        self._elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_steps:
            truncated = True
        info["elapsed_steps"] = self._elapsed_steps
        return obs, reward, terminated, truncated, info


class NormaliseObservation(_Wrapper):
    """Normalise observations online using a running mean and variance.

    Uses Welford's algorithm for numerically stable incremental updates.
    Observations are normalised to approximately zero-mean, unit-variance.

    Parameters
    ----------
    env :
        The environment to wrap.
    epsilon :
        Small constant added to the standard deviation to prevent division
        by zero.
    clip :
        If not ``None``, clip normalised observations to ``[-clip, clip]``.
    """

    def __init__(
        self,
        env: BaseEnv,
        epsilon: float = 1e-8,
        clip: float | None = None,
    ) -> None:
        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip

        # Running statistics (initialised lazily on first observation)
        self._count: int = 0
        self._mean: np.ndarray | None = None
        self._M2: np.ndarray | None = None  # sum of squared deviations

    # ------------------------------------------------------------------
    # Running-statistics helpers (Welford's online algorithm)
    # ------------------------------------------------------------------

    def _init_stats(self, obs: np.ndarray) -> None:
        self._mean = np.zeros_like(obs, dtype=np.float64)
        self._M2 = np.zeros_like(obs, dtype=np.float64)

    def _update_stats(self, obs: np.ndarray) -> None:
        if self._mean is None:
            self._init_stats(obs)
        obs = obs.astype(np.float64)
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._M2 += delta * delta2

    @property
    def _var(self) -> np.ndarray:
        if self._count < 2:
            return np.ones_like(self._mean, dtype=np.float64)
        return self._M2 / (self._count - 1)

    def _normalise(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float64)
        normalised = (obs - self._mean) / (np.sqrt(self._var) + self.epsilon)
        if self.clip is not None:
            normalised = np.clip(normalised, -self.clip, self.clip)
        return normalised.astype(np.float32)

    # ------------------------------------------------------------------
    # Gymnasium-style overrides
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs_arr = np.asarray(obs)
        self._update_stats(obs_arr)
        return self._normalise(obs_arr), info

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_arr = np.asarray(obs)
        self._update_stats(obs_arr)
        return self._normalise(obs_arr), reward, terminated, truncated, info
