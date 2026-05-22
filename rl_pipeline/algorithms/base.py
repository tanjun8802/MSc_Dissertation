"""Base interfaces for custom RL algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(slots=True)
class Transition:
    """Single environment transition for replay/training."""

    observation: Any
    action: np.ndarray
    reward: float
    next_observation: Any
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class BaseAlgorithm(ABC):
    """Interface your custom algorithm should implement."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, seed: int | None = None) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_action(self, observation: Any, step: int, training: bool = True) -> np.ndarray:
        """Return a valid action for the current observation."""

    def observe(self, transition: Transition) -> None:
        """Store transition data (e.g., in replay buffer)."""

    def update(self, step: int) -> dict[str, float]:
        """Run one optimization/update step and return optional metrics."""
        return {}

    def on_episode_end(self, episode: int, episode_return: float, episode_length: int) -> None:
        """Optional callback after each completed episode."""
