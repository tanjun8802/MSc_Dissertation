"""Generic RL training loop for Gymnasium-compatible environments."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym

from rl_pipeline.algorithms.base import BaseAlgorithm, Transition


@dataclass(slots=True)
class TrainingConfig:
    total_steps: int
    learning_starts: int = 1000
    update_every: int = 1
    max_episode_steps: int | None = None


@dataclass(slots=True)
class TrainingStats:
    episodes: int = 0
    total_steps: int = 0
    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)


def run_training_loop(env: gym.Env, algorithm: BaseAlgorithm, config: TrainingConfig) -> TrainingStats:
    """Run environment interaction and call algorithm hooks for storage and updates."""

    obs, _ = env.reset()
    stats = TrainingStats()

    episode_return = 0.0
    episode_length = 0

    for step in range(config.total_steps):
        action = algorithm.select_action(obs, step=step, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        transition = Transition(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        algorithm.observe(transition)

        if step >= config.learning_starts and (step + 1) % config.update_every == 0:
            algorithm.update(step=step)

        obs = next_obs
        episode_return += reward
        episode_length += 1
        stats.total_steps += 1

        reached_limit = config.max_episode_steps is not None and episode_length >= config.max_episode_steps
        if terminated or truncated or reached_limit:
            stats.episodes += 1
            stats.returns.append(episode_return)
            stats.lengths.append(episode_length)
            algorithm.on_episode_end(stats.episodes, episode_return, episode_length)
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

    return stats
