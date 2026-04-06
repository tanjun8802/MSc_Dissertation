"""
run_experiment.py
=================
CLI entry-point: run a random-agent baseline on the GridWorld environment.

Usage
-----
    python experiments/run_experiment.py [--episodes N] [--seed S] [--render]

This script wires together the GridWorld environment, the RandomAgent, and
the experiment runner to produce a minimal working example of the full
experiment pipeline.  It is intended as a *template* — swap in your own
agent and environment to run real experiments.
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.random_agent import RandomAgent
from environments.gridworld import GridWorld
from utils.config import load_config
from experiments.base_experiment import BaseExperiment
from utils.metrics import EpisodeMetrics


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")



# ---------------------------------------------------------------------------
# Concrete experiment: random baseline on GridWorld
# ---------------------------------------------------------------------------


class RandomBaselineExperiment(BaseExperiment):
    """Run a RandomAgent on GridWorld — no learning, just interaction."""

    def train_step(
        self,
        obs,
        action,
        reward: float,
        next_obs,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> dict:
        # The RandomAgent does nothing on update; return empty metrics.
        return self.agent.update(obs, action, reward, next_obs, terminated, truncated, info)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random-agent baseline on GridWorld.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--height", type=int, default=5, help="Grid height.")
    parser.add_argument("--width", type=int, default=5, help="Grid width.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--render", action="store_true", help="Print ASCII grid after each episode.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs.")


    pre_p = argparse.ArgumentParser(add_help=False)
    pre_p.add_argument("--config", default=_CONFIG_PATH)
    cfg_path = pre_p.parse_known_args(argv)[0].config
    cfg = load_config(cfg_path)
    if cfg:
        # Override defaults with config values.
        parser.set_defaults(**cfg)

    return parser.parse_args(argv)




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> list[EpisodeMetrics]:
    args = parse_args(argv)

    env = GridWorld(
        height=args.height,
        width=args.width,
        max_steps=args.max_steps,
    )
    agent = RandomAgent(n_actions=env.n_actions, seed=args.seed)

    experiment = RandomBaselineExperiment(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        eval_every=0,  # no evaluation for the baseline
        seed=args.seed,
        log_dir=args.log_dir,
    )

    print(f"Running random baseline: {env!r}")
    print(f"  Episodes : {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Seed     : {args.seed}")
    print()

    all_metrics = experiment.run()

    # Summary
    rewards = [m.total_reward for m in all_metrics]
    lengths = [m.length for m in all_metrics]
    print(f"Results over {args.episodes} episodes:")
    print(f"  Mean reward : {sum(rewards) / len(rewards):.2f}")
    print(f"  Mean length : {sum(lengths) / len(lengths):.1f}")
    print(f"  Total steps : {agent.total_steps}")

    if args.render:
        print("\nFinal grid state:")
        print(env.render())

    return all_metrics


if __name__ == "__main__":
    main()
