"""
compare_approaches.py
=====================
Side-by-side comparison of three RL approaches on the GridWorld environment:

    1. **Random Baseline**  — uniformly random action selection (no learning)
    2. **GCRL**             — Goal-Conditioned Q-Learning + Hindsight Experience Replay
    3. **RCRL**             — Reward-Conditioned Q-Learning (reward-parameterization-conditioned)

The script runs all three approaches under the same GridWorld configuration,
then prints a concise comparison table so you can immediately see the
differences in sample efficiency, final performance, and convergence speed.

Usage
-----
    # quick comparison (few episodes)
    python experiments/compare_approaches.py

    # longer run for more accurate results
    python experiments/compare_approaches.py \\
        --episodes 500 --height 5 --width 5 --seed 42

    # render final grid after each approach
    python experiments/compare_approaches.py --render
"""

from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass
from typing import Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.random_agent import RandomAgent
from agents.goal_conditioned_agent import GoalConditionedAgent
from agents.reward_conditioned_agent import RewardConditionedAgent
from environments.gridworld import GridWorld
from experiments.run_baseline import BaseExperiment
from experiments.run_gcrl import GCRLExperiment
from experiments.run_rcrl import RCRLExperiment
from utils.metrics import EpisodeMetrics


# ---------------------------------------------------------------------------
# Minimal random-baseline experiment (inline, mirrors run_experiment.py)
# ---------------------------------------------------------------------------


class RandomBaselineExperiment(BaseExperiment):
    """Random-agent baseline — no learning."""

    def train_step(self, obs, action, reward, next_obs, terminated, truncated, info):
        return self.agent.update(obs, action, reward, next_obs, terminated, truncated, info)


# ---------------------------------------------------------------------------
# Result summary dataclass
# ---------------------------------------------------------------------------


@dataclass
class ApproachResult:
    name: str
    all_rewards: list[float]
    eval_rewards: list[float]   # rewards from evaluation episodes (if any)
    total_steps: int
    n_episodes: int

    @property
    def mean_reward(self) -> float:
        return _mean(self.all_rewards)

    @property
    def mean_reward_last10pct(self) -> float:
        k = max(1, len(self.all_rewards) // 10)
        return _mean(self.all_rewards[-k:])

    @property
    def mean_eval_reward(self) -> float:
        return _mean(self.eval_rewards) if self.eval_rewards else float("nan")

    @property
    def mean_length(self) -> float:
        return float(self.n_episodes)


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def run_random_baseline(
    height: int, width: int, goal_pos: tuple[int, int], max_steps: int,
    episodes: int, seed: int, log_dir: str,
) -> ApproachResult:
    env = GridWorld(height=height, width=width, goal_pos=goal_pos, max_steps=max_steps)
    agent = RandomAgent(n_actions=env.n_actions, seed=seed)
    exp = RandomBaselineExperiment(
        env=env, agent=agent, n_episodes=episodes,
        eval_every=0, seed=seed,
        log_dir=os.path.join(log_dir, "random"),
    )
    metrics = exp.run()
    return ApproachResult(
        name="Random Baseline",
        all_rewards=[m.total_reward for m in metrics],
        eval_rewards=[],
        total_steps=agent.total_steps,
        n_episodes=episodes,
    )


def run_gcrl(
    height: int, width: int, max_steps: int,
    episodes: int, eval_goal: int, seed: int,
    alpha: float, epsilon: float, epsilon_min: float,
    epsilon_decay: float, her_k: int, gamma: float,
    eval_every: int, log_dir: str,
) -> ApproachResult:
    env = GridWorld(height=height, width=width, max_steps=max_steps)
    agent = GoalConditionedAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        gamma=gamma, alpha=alpha,
        epsilon=epsilon, epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        her_k=her_k, seed=seed,
    )
    exp = GCRLExperiment(
        env=env, agent=agent, eval_goal=eval_goal,
        n_episodes=episodes, eval_every=eval_every,
        seed=seed, log_dir=os.path.join(log_dir, "gcrl"),
    )
    metrics = exp.run()
    train_metrics = [m for m in metrics if m.training]
    eval_metrics = [m for m in metrics if not m.training]
    return ApproachResult(
        name="GCRL (Goal-Conditioned + HER)",
        all_rewards=[m.total_reward for m in train_metrics],
        eval_rewards=[m.total_reward for m in eval_metrics],
        total_steps=agent.total_steps,
        n_episodes=episodes,
    )


def run_rcrl(
    height: int, width: int, goal_pos: tuple[int, int], max_steps: int,
    explore_episodes: int, exploit_episodes: int,
    n_psi_bins: int, psi_min: float, psi_mix_alpha: float, alpha: float,
    epsilon: float, epsilon_min: float, epsilon_decay: float,
    gamma: float, seed: int, log_dir: str,
) -> ApproachResult:
    env = GridWorld(height=height, width=width, goal_pos=goal_pos, max_steps=max_steps)
    agent = RewardConditionedAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        n_psi_bins=n_psi_bins,
        psi_min=psi_min,
        psi_mix_alpha=psi_mix_alpha,
        gamma=gamma, alpha=alpha,
        epsilon=epsilon, epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    exp = RCRLExperiment(
        env=env, agent=agent,
        n_explore=explore_episodes,
        n_exploit=exploit_episodes,
        seed=seed, log_dir=os.path.join(log_dir, "rcrl"),
    )
    metrics = exp.run()
    explore_metrics = [m for m in metrics if m.training]
    exploit_metrics = [m for m in metrics if not m.training]
    return ApproachResult(
        name="RCRL (Reward-Conditioned)",
        all_rewards=[m.total_reward for m in explore_metrics],
        eval_rewards=[m.total_reward for m in exploit_metrics],
        total_steps=agent.total_steps,
        n_episodes=explore_episodes + exploit_episodes,
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _print_table(results: list[ApproachResult]) -> None:
    """Print a comparison table to stdout."""
    col_w = 32
    num_w = 14

    header = (
        f"{'Approach':<{col_w}}"
        f"{'Mean Reward':>{num_w}}"
        f"{'Last-10% Reward':>{num_w}}"
        f"{'Mean Eval Reward':>{num_w}}"
        f"{'Total Steps':>{num_w}}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  COMPARISON RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        eval_str = f"{r.mean_eval_reward:.4f}" if r.eval_rewards else "  N/A"
        print(
            f"{r.name:<{col_w}}"
            f"{r.mean_reward:>{num_w}.4f}"
            f"{r.mean_reward_last10pct:>{num_w}.4f}"
            f"{eval_str:>{num_w}}"
            f"{r.total_steps:>{num_w}}"
        )
    print("=" * len(header))
    print()

    # Narrative summary
    trained = [r for r in results if r.name != "Random Baseline"]
    baseline = next((r for r in results if r.name == "Random Baseline"), None)

    if baseline:
        print(f"Baseline mean reward: {baseline.mean_reward:.4f}")
    for r in trained:
        gain = (
            r.mean_eval_reward - baseline.mean_reward
            if (baseline and r.eval_rewards)
            else r.mean_reward_last10pct - (baseline.mean_reward if baseline else 0)
        )
        print(f"  {r.name}: improvement over baseline = {gain:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Random, GCRL, and RCRL on GridWorld.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--height", type=int, default=5, help="Grid height.")
    parser.add_argument("--width", type=int, default=5, help="Grid width.")
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes per approach.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial ε.")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Min ε.")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="ε decay.")
    parser.add_argument("--her-k", type=int, default=4, help="HER substitutions (GCRL).")
    parser.add_argument("--n-psi-bins", type=int, default=5, help="Reward-parameterisation bins (RCRL).")
    parser.add_argument("--psi-min", type=float, default=-0.1, help="Most negative step-cost weight (RCRL, ≤ 0).")
    parser.add_argument("--psi-mix-alpha", type=float, default=0.5, help="Fraction of nominal ψ* draws in training mixture (RCRL).")
    parser.add_argument("--eval-every", type=int, default=50, help="GCRL eval every N episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--render", action="store_true", help="Show grid after each approach.")
    parser.add_argument("--log-dir", type=str, default="logs/compare", help="Log directory.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    n_states = args.height * args.width
    eval_goal = n_states - 1  # bottom-right cell
    goal_row, goal_col = divmod(eval_goal, args.width)
    goal_pos = (goal_row, goal_col)

    # Split episodes: use 80% for RCRL exploration, 20% for exploitation
    rcrl_explore = max(10, int(args.episodes * 0.8))
    rcrl_exploit = max(5, args.episodes - rcrl_explore)

    print("=" * 60)
    print("Comparing RL Approaches on GridWorld")
    print("=" * 60)
    print(f"  Grid          : {args.height}×{args.width}")
    print(f"  Goal          : state {eval_goal} ({goal_row},{goal_col})")
    print(f"  Max steps     : {args.max_steps}")
    print(f"  Episodes      : {args.episodes} (per approach)")
    print(f"  RCRL explore  : {rcrl_explore}  /  exploit: {rcrl_exploit}")
    print(f"  Seed          : {args.seed}")
    print()

    results: list[ApproachResult] = []

    # --- 1. Random baseline ------------------------------------------------
    print("─" * 40)
    print("Running: Random Baseline …")
    r_random = run_random_baseline(
        height=args.height, width=args.width,
        goal_pos=goal_pos, max_steps=args.max_steps,
        episodes=args.episodes, seed=args.seed,
        log_dir=args.log_dir,
    )
    results.append(r_random)
    print(f"Done.  Mean reward = {r_random.mean_reward:.4f}")

    # --- 2. GCRL -----------------------------------------------------------
    print()
    print("─" * 40)
    print("Running: GCRL (Goal-Conditioned + HER) …")
    r_gcrl = run_gcrl(
        height=args.height, width=args.width,
        max_steps=args.max_steps,
        episodes=args.episodes, eval_goal=eval_goal,
        seed=args.seed, alpha=args.alpha,
        epsilon=args.epsilon, epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        her_k=args.her_k, gamma=args.gamma,
        eval_every=args.eval_every,
        log_dir=args.log_dir,
    )
    results.append(r_gcrl)
    print(f"Done.  Mean eval reward = {r_gcrl.mean_eval_reward:.4f}")

    # --- 3. RCRL -----------------------------------------------------------
    print()
    print("─" * 40)
    print("Running: RCRL (Reward-Conditioned) …")
    r_rcrl = run_rcrl(
        height=args.height, width=args.width,
        goal_pos=goal_pos, max_steps=args.max_steps,
        explore_episodes=rcrl_explore,
        exploit_episodes=rcrl_exploit,
        n_psi_bins=args.n_psi_bins,
        psi_min=args.psi_min,
        psi_mix_alpha=args.psi_mix_alpha,
        alpha=args.alpha,
        epsilon=args.epsilon, epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        gamma=args.gamma, seed=args.seed,
        log_dir=args.log_dir,
    )
    results.append(r_rcrl)
    print(f"Done.  Mean exploit reward = {r_rcrl.mean_eval_reward:.4f}")

    # --- Print comparison table -------------------------------------------
    _print_table(results)

    if args.render:
        env = GridWorld(
            height=args.height, width=args.width,
            goal_pos=goal_pos, max_steps=args.max_steps,
        )
        env.reset()
        print("Final grid layout (G = goal, . = free):")
        print(env.render())


if __name__ == "__main__":
    main()
