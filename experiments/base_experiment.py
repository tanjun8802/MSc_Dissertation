"""
base_experiment.py
==================
Abstract experiment runner that ties together an environment and an agent.

Subclasses override :meth:`train_step` (and optionally :meth:`eval_step`)
to implement algorithm-specific logic (Q-learning update, actor-critic step,
etc.) while inheriting the standard episode-loop book-keeping.

Design goals
------------
* **Separation of concerns** — environment interaction, agent updates, and
  logging live in separate methods so each can be replaced independently.
* **Reproducibility** — a global ``seed`` controls all randomness.
* **Minimal dependencies** — only numpy and the project's own modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from agents.base_agent import BaseAgent
from environments.base_env import BaseEnv
from utils.logger import Logger
from utils.metrics import EpisodeMetrics


class BaseExperiment(ABC):
    """Orchestrates the training loop for one (environment, agent) pair.

    Parameters
    ----------
    env :
        The training environment.
    agent :
        The RL agent.
    n_episodes :
        Total number of training episodes to run.
    eval_every :
        Run a greedy evaluation episode every *n* training episodes.
        Set to ``0`` to disable evaluation.
    seed :
        Global random seed.
    log_dir :
        Directory path for saving experiment logs.
    """

    def __init__(
        self,
        env: BaseEnv,
        agent: BaseAgent,
        n_episodes: int = 1000,
        eval_every: int = 100,
        seed: int = 0,
        log_dir: str = "logs",
    ) -> None:
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.eval_every = eval_every
        self.seed = seed
        self.logger = Logger(log_dir=log_dir)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Abstract hooks (algorithm-specific logic goes here)
    # ------------------------------------------------------------------

    @abstractmethod
    def train_step(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> dict:
        """Perform one agent update and return a dict of training metrics."""

    def eval_step(self, obs: Any) -> Any:
        """Select a (potentially greedy) action during evaluation.

        Default implementation re-uses the agent's own select_action.
        Override to implement greedy / deterministic action selection.
        """
        return self.agent.select_action(obs)

    # ------------------------------------------------------------------
    # Main loops
    # ------------------------------------------------------------------

    def run(self) -> list[EpisodeMetrics]:
        """Run all training episodes and return per-episode metric records."""
        all_metrics: list[EpisodeMetrics] = []

        for episode in range(1, self.n_episodes + 1):
            metrics = self._run_episode(episode, training=True)
            all_metrics.append(metrics)
            self.logger.log_episode(episode, metrics)

            if self.eval_every > 0 and episode % self.eval_every == 0:
                eval_metrics = self._run_episode(episode, training=False)
                self.logger.log_eval(episode, eval_metrics)

        return all_metrics

    def _run_episode(self, episode: int, training: bool) -> EpisodeMetrics:
        """Run a single episode in training or evaluation mode."""
        obs, info = self.env.reset(seed=int(self._rng.integers(0, 2**31)))
        self.agent.reset()

        total_reward = 0.0
        steps = 0
        step_metrics_list: list[dict] = []

        while True:
            if training:
                action = self.agent.select_action(obs)
            else:
                action = self.eval_step(obs)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            steps += 1

            if training:
                step_metrics = self.train_step(
                    obs, action, float(reward), next_obs, terminated, truncated, info
                )
                step_metrics_list.append(step_metrics)

            obs = next_obs

            if terminated or truncated:
                break

        return EpisodeMetrics(
            episode=episode,
            total_reward=total_reward,
            length=steps,
            training=training,
            step_metrics=step_metrics_list,
        )
