"""Example entrypoint for running the custom RL pipeline on robosuite."""

from __future__ import annotations

import argparse

import numpy as np

from rl_pipeline.algorithms.base import BaseAlgorithm
from rl_pipeline.envs.robosuite_gym import RobosuiteGymWrapper
from rl_pipeline.training.loop import TrainingConfig, run_training_loop


class RandomPolicyAlgorithm(BaseAlgorithm):
    """Minimal working algorithm baseline for pipeline validation."""

    def select_action(self, observation, step: int, training: bool = True) -> np.ndarray:
        del observation, step, training
        return self.action_space.sample()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a custom RL pipeline on robosuite.")
    parser.add_argument("--env", type=str, default="Lift", help="robosuite environment name")
    parser.add_argument("--robot", type=str, default="Panda", help="robot name")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = RobosuiteGymWrapper(
        env_name=args.env,
        robots=args.robot,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        horizon=args.max_episode_steps,
    )

    try:
        algorithm = RandomPolicyAlgorithm(
            observation_space=env.observation_space,
            action_space=env.action_space,
            seed=args.seed,
        )
        stats = run_training_loop(
            env,
            algorithm,
            TrainingConfig(
                total_steps=args.total_steps,
                learning_starts=args.learning_starts,
                update_every=args.update_every,
                max_episode_steps=args.max_episode_steps,
            ),
        )

        avg_return = float(np.mean(stats.returns)) if stats.returns else 0.0
        print(f"Episodes: {stats.episodes}")
        print(f"Total steps: {stats.total_steps}")
        print(f"Average return: {avg_return:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
