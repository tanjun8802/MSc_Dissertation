"""
compare_approaches.py
=====================
Side-by-side comparison of three RL approaches on the GridWorld environment:

    1. **Random Baseline**  — uniformly random action selection (no learning)
    2. **GCRL**             — Single-Goal Contrastive RL (Liu, Tang & Eysenbach, 2024)
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
from utils.config import load_config
from utils.metrics import EpisodeMetrics

# Default config shared across all three approaches
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


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
    alpha: float, temperature: float, n_negatives: int,
    logsumexp_reg: float, buffer_capacity: int, gamma: float,
    eval_every: int, log_dir: str,
) -> ApproachResult:
    # Training env: goal_pos is set so episodes terminate at the goal.
    # Algorithm 1 of Liu et al. (2024) requires the single hard target goal
    # to be a terminal state; without it the contrastive critic never receives
    # (s, a, sf=goal) pairs and learns nothing about goal reachability.
    goal_row, goal_col = divmod(eval_goal, width)
    env = GridWorld(height=height, width=width,
                    goal_pos=(goal_row, goal_col), max_steps=max_steps)
    agent = GoalConditionedAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        gamma=gamma, alpha=alpha,
        temperature=temperature,
        n_negatives=n_negatives,
        logsumexp_reg=logsumexp_reg,
        buffer_capacity=buffer_capacity,
        seed=seed,
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
        name="GCRL (Single-Goal Contrastive RL)",
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
    col_w = 36
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
    parser.add_argument("--config", type=str, default=_CONFIG_PATH, help="Path to YAML config file.")
    parser.add_argument("--height", type=int, default=5, help="Grid height.")
    parser.add_argument("--width", type=int, default=5, help="Grid width.")
    parser.add_argument("--episodes", type=int, default=300, help="Training episodes per approach.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    # GCRL (contrastive RL) hyperparameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature τ (GCRL).")
    parser.add_argument("--n-negatives", type=int, default=16, help="Negative examples per infoNCE update (GCRL).")
    parser.add_argument("--logsumexp-reg", type=float, default=0.01, help="LogSumExp regularisation coefficient (GCRL).")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="Replay buffer capacity (GCRL).")
    # RCRL hyperparameters
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial ε (RCRL).")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Min ε (RCRL).")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="ε decay (RCRL).")
    parser.add_argument("--n-psi-bins", type=int, default=5, help="Reward-parameterisation bins (RCRL).")
    parser.add_argument("--psi-min", type=float, default=-0.1, help="Most negative step-cost weight (RCRL, ≤ 0).")
    parser.add_argument("--psi-mix-alpha", type=float, default=0.5, help="Fraction of nominal ψ* draws in training mixture (RCRL).")
    parser.add_argument("--eval-every", type=int, default=50, help="GCRL eval every N episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--render", action="store_true", help="Show grid after each approach.")
    parser.add_argument("--log-dir", type=str, default="logs/compare", help="Log directory.")

    # --- Apply YAML config as defaults (CLI args override YAML) ---
    pre_p = argparse.ArgumentParser(add_help=False)
    pre_p.add_argument("--config", default=_CONFIG_PATH)
    cfg_path = pre_p.parse_known_args(argv)[0].config
    cfg = load_config(cfg_path)
    if cfg:
        env_cfg = cfg.get("env", {})
        agent_cfg = cfg.get("agent", {})
        training_cfg = cfg.get("training", {})
        mdp_cfg = cfg.get("mdp", {})
        log_cfg = cfg.get("logging", {})
        yaml_defaults: dict = {}
        if "height" in env_cfg:
            yaml_defaults["height"] = env_cfg["height"]
        if "width" in env_cfg:
            yaml_defaults["width"] = env_cfg["width"]
        if "max_steps" in env_cfg:
            yaml_defaults["max_steps"] = env_cfg["max_steps"]
        if "gamma" in mdp_cfg:
            yaml_defaults["gamma"] = mdp_cfg["gamma"]
        if "alpha" in agent_cfg:
            yaml_defaults["alpha"] = agent_cfg["alpha"]
        if "epsilon" in agent_cfg:
            yaml_defaults["epsilon"] = agent_cfg["epsilon"]
        if "epsilon_min" in agent_cfg:
            yaml_defaults["epsilon_min"] = agent_cfg["epsilon_min"]
        if "epsilon_decay" in agent_cfg:
            yaml_defaults["epsilon_decay"] = agent_cfg["epsilon_decay"]
        if "n_episodes" in training_cfg:
            yaml_defaults["episodes"] = training_cfg["n_episodes"]
        if "eval_every" in training_cfg:
            yaml_defaults["eval_every"] = training_cfg["eval_every"]
        if "seed" in training_cfg:
            yaml_defaults["seed"] = training_cfg["seed"]
        if "log_dir" in log_cfg:
            yaml_defaults["log_dir"] = log_cfg["log_dir"]
        parser.set_defaults(**yaml_defaults)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# compare_all — importable by the evaluation notebook
# ---------------------------------------------------------------------------


def compare_all(args: argparse.Namespace) -> list[ApproachResult]:
    """Run all three RL approaches and return their results.

    Parameters
    ----------
    args :
        Parsed argument namespace produced by :func:`parse_args`.
        All required fields (height, width, episodes, …) must be present.

    Returns
    -------
    list[ApproachResult]
        Results for Random Baseline, GCRL, and RCRL in that order.
    """
    n_states = args.height * args.width
    eval_goal = n_states - 1
    goal_row, goal_col = divmod(eval_goal, args.width)
    goal_pos = (goal_row, goal_col)

    rcrl_explore = max(10, int(args.episodes * 0.8))
    rcrl_exploit = max(5, args.episodes - rcrl_explore)

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
    print("Running: GCRL (Single-Goal Contrastive RL) …")
    r_gcrl = run_gcrl(
        height=args.height, width=args.width,
        max_steps=args.max_steps,
        episodes=args.episodes, eval_goal=eval_goal,
        seed=args.seed, alpha=args.alpha,
        temperature=args.temperature,
        n_negatives=args.n_negatives,
        logsumexp_reg=args.logsumexp_reg,
        buffer_capacity=args.buffer_capacity,
        gamma=args.gamma,
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

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    n_states = args.height * args.width
    eval_goal = n_states - 1
    goal_row, goal_col = divmod(eval_goal, args.width)
    goal_pos = (goal_row, goal_col)
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

    results = compare_all(args)

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
