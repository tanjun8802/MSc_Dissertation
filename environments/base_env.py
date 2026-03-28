"""
base_env.py
===========
Abstract base class that every custom environment must inherit from.

Design choice
-------------
We mirror the Gymnasium (formerly OpenAI Gym) interface so that custom
environments remain compatible with standard RL libraries (Stable-Baselines3,
CleanRL, etc.) while allowing us to add dissertation-specific helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, SupportsFloat, Tuple

import numpy as np


class BaseEnv(ABC):
    """Abstract base environment compatible with the Gymnasium API.

    Subclasses must implement :meth:`reset`, :meth:`step`, and expose the
    ``observation_space`` / ``action_space`` attributes.

    Attributes
    ----------
    observation_space : gym.Space
        Defines the structure and valid range of observations.
    action_space : gym.Space
        Defines the structure and valid range of actions.
    np_random : numpy.random.Generator
        Seeded random-number generator; set via :meth:`seed`.
    """

    metadata: dict = {}

    def __init__(self) -> None:
        self.np_random: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[Any, dict]:
        """Reset the environment to an initial state.

        Parameters
        ----------
        seed :
            Optional seed for reproducibility.
        options :
            Additional reset options (environment-specific).

        Returns
        -------
        observation :
            The initial observation.
        info :
            Auxiliary diagnostic information.
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        return None, {}  # type: ignore[return-value]

    @abstractmethod
    def step(
        self,
        action: Any,
    ) -> Tuple[Any, SupportsFloat, bool, bool, dict]:
        """Run one step of the environment dynamics.

        Parameters
        ----------
        action :
            An action provided by the agent.

        Returns
        -------
        observation :
            Agent's observation of the current environment.
        reward :
            Scalar reward signal (may be 0 for reward-free settings).
        terminated :
            Whether a terminal state has been reached.
        truncated :
            Whether the episode was truncated (e.g. time limit).
        info :
            Auxiliary diagnostic information.
        """

    @abstractmethod
    def render(self) -> Any:
        """Render the environment (optional visual output)."""

    def close(self) -> None:
        """Clean up resources. Override in subclasses if needed."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Set the random seed and return it (Gym legacy API)."""
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__}>"
