"""
run_rcrl.py
===========
CLI entry-point: Reward-Conditioned (Return-Conditioned) RL on GridWorld.

Background
----------
Reward-conditioned RL treats the desired *return* as an additional input to the
policy, transforming reward maximisation into a supervised learning problem:

    π(a | s, g_return) — what action achieves desired return g?

This script implements the core ideas from:
    "Reward Conditioned Policies" (Emmons et al., 2021)  arXiv:2112.13629

The approach has two phases:

1. **Exploration** (``--explore-episodes`` episodes):
   Collect diverse (state, action, achieved-return) data using an ε-greedy
   policy over the GridWorld.  The environment uses a fixed goal so that
   reaching it gives a +1 reward.

2. **Exploitation** (``--exploit-episodes`` episodes):
   Condition the return-conditioned policy on the *maximum observed return*
   and evaluate its performance.  ε is kept at ``epsilon_min`` so the policy
   acts nearly greedily.

The conditional policy is tabular:
    action_counts[s, bin(G_t), a] += α    (behavioral-cloning update)
    π(a | s, g) ∝ action_counts[s, bin(g), a]

Usage
-----
    # basic run
    python experiments/run_rcrl.py

    # custom settings
    python experiments/run_rcrl.py \\
        --explore-episodes 300 --exploit-episodes 100 \\
        --height 5 --width 5 --goal 24 --seed 0 --render \\
        --log-dir logs/rcrl
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.reward_conditioned_agent import RewardConditionedAgent
from environments.gridworld import GridWorld
from experiments.base_experiment import BaseExperiment
from utils.metrics import EpisodeMetrics


# ---------------------------------------------------------------------------
# Concrete experiment
# ---------------------------------------------------------------------------


class RCRLExperiment(BaseExperiment):
    """Reward-conditioned RL experiment in two phases.

    Exploration phase
    -----------------
    Run ``n_explore`` episodes with ε-greedy exploration.  After each episode
    call ``agent.finish_episode()`` to update the return-conditioned policy
    via behavioral cloning.

    Exploitation phase
    ------------------
    Run ``n_exploit`` episodes with ``desired_return = agent.max_observed_return``
    and ``greedy=True`` to evaluate the learned policy.
    """

    def __init__(
        self,
        env: GridWorld,
        agent: RewardConditionedAgent,
        n_explore: int,
        n_exploit: int,
        **kwargs,
    ) -> None:
        # total episodes = explore + exploit
        super().__init__(env=env, agent=agent, n_episodes=n_explore + n_exploit, **kwargs)
        self.n_explore = n_explore
        self.n_exploit = n_exploit

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
        """Buffer the transition (actual update happens in finish_episode)."""
        return self.agent.update(obs, action, reward, next_obs, terminated, truncated, info)

    def run(self) -> list[EpisodeMetrics]:
        """Two-phase training loop."""
        all_metrics: list[EpisodeMetrics] = []

        # --- Phase 1: Exploration -------------------------------------------
        print("Phase 1 — Exploration (collecting diverse trajectories) …")
        for episode in range(1, self.n_explore + 1):
            metrics = self._run_episode(episode, training=True)
            all_metrics.append(metrics)

            finish_info = self.agent.finish_episode()
            self.logger.log_episode(episode, metrics)

            if episode % max(1, self.n_explore // 5) == 0:
                print(
                    f"  [explore ep {episode:>4d}]  "
                    f"reward={metrics.total_reward:.2f}  "
                    f"length={metrics.length:>3d}  "
                    f"G0={finish_info['episode_return']:.3f}  "
                    f"max_G={self.agent.max_observed_return:.3f}  "
                    f"ε={self.agent.epsilon:.3f}"
                )

        # --- Phase 2: Exploitation -------------------------------------------
        print(
            f"\nPhase 2 — Exploitation "
            f"(desired_return={self.agent.max_observed_return:.3f}) …"
        )
        # Lock ε at minimum for near-greedy behaviour
        self.agent.epsilon = self.agent.epsilon_min

        exploit_rewards = []
        for ep_idx in range(1, self.n_exploit + 1):
            episode = self.n_explore + ep_idx
            # Override select_action to use desired_return = max observed
            desired = self.agent.max_observed_return
            metrics = self._run_exploit_episode(episode, desired_return=desired)
            all_metrics.append(metrics)
            exploit_rewards.append(metrics.total_reward)

            self.logger.log_eval(episode, metrics)

            if ep_idx % max(1, self.n_exploit // 5) == 0:
                last = exploit_rewards[-max(1, len(exploit_rewards) // 5):]
                print(
                    f"  [exploit ep {ep_idx:>3d}]  "
                    f"reward={metrics.total_reward:.2f}  "
                    f"length={metrics.length:>3d}  "
                    f"mean_reward(recent)={sum(last)/len(last):.3f}"
                )

        return all_metrics

    def _run_exploit_episode(
        self, episode: int, desired_return: float
    ) -> EpisodeMetrics:
        """Run one episode with the return-conditioned policy."""
        obs, info = self.env.reset(seed=int(self._rng.integers(0, 2**31)))
        self.agent.reset()

        total_reward = 0.0
        steps = 0

        while True:
            action = self.agent.select_action(obs, desired_return=desired_return, greedy=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            steps += 1
            obs = next_obs

            if terminated or truncated:
                break

        return EpisodeMetrics(
            episode=episode,
            total_reward=total_reward,
            length=steps,
            training=False,
            step_metrics=[],
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reward-Conditioned RL on GridWorld.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--height", type=int, default=5, help="Grid height.")
    parser.add_argument("--width", type=int, default=5, help="Grid width.")
    parser.add_argument(
        "--goal",
        type=int,
        default=None,
        help="Goal state index. Defaults to bottom-right cell.",
    )
    parser.add_argument("--explore-episodes", type=int, default=400, help="Exploration episodes.")
    parser.add_argument("--exploit-episodes", type=int, default=100, help="Exploitation episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--n-return-bins", type=int, default=10, help="Number of return bins.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Behavioral-cloning learning rate.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial ε.")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum ε.")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="ε decay per episode.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--render", action="store_true", help="Print ASCII grid after run.")
    parser.add_argument("--log-dir", type=str, default="logs/rcrl", help="Log directory.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> list[EpisodeMetrics]:
    args = parse_args(argv)

    goal_pos_flat = args.goal if args.goal is not None else args.height * args.width - 1
    goal_row, goal_col = divmod(goal_pos_flat, args.width)

    env = GridWorld(
        height=args.height,
        width=args.width,
        goal_pos=(goal_row, goal_col),
        max_steps=args.max_steps,
    )

    agent = RewardConditionedAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        n_return_bins=args.n_return_bins,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    experiment = RCRLExperiment(
        env=env,
        agent=agent,
        n_explore=args.explore_episodes,
        n_exploit=args.exploit_episodes,
        seed=args.seed,
        log_dir=args.log_dir,
    )

    print("=" * 60)
    print("Reward-Conditioned RL (RCRL)")
    print("=" * 60)
    print(f"  Environment : {env!r}")
    print(f"  Agent       : {agent!r}")
    print(f"  Goal state  : {goal_pos_flat} ({goal_row}, {goal_col})")
    print(f"  Explore eps : {args.explore_episodes}")
    print(f"  Exploit eps : {args.exploit_episodes}")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  Seed        : {args.seed}")
    print()

    all_metrics = experiment.run()

    # Split metrics into phases
    explore_metrics = [m for m in all_metrics if m.training]
    exploit_metrics = [m for m in all_metrics if not m.training]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    explore_rewards = [m.total_reward for m in explore_metrics]
    exploit_rewards = [m.total_reward for m in exploit_metrics]
    exploit_lengths = [m.length for m in exploit_metrics]

    print()
    print("Summary:")
    print(f"  Max observed return      : {agent.max_observed_return:.4f}")
    print(f"  Explore mean reward      : {_mean(explore_rewards):.4f}")
    print(f"  Exploit mean reward      : {_mean(exploit_rewards):.4f}")
    print(f"  Exploit mean length      : {_mean(exploit_lengths):.1f}")
    print(f"  Total env steps          : {agent.total_steps}")

    if args.render:
        obs, _ = env.reset()
        print("\nFinal grid state:")
        print(env.render())

    return all_metrics


if __name__ == "__main__":
    main()
