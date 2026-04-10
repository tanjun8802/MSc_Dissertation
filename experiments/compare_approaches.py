"""
compare_approaches.py
=====================
Side-by-side comparison of three RL approaches on configurable GridWorld
environments:

    1. **Random Baseline**  — uniformly random action selection (no learning)
    2. **GCRL**             — Single-Goal Contrastive RL (Liu, Tang & Eysenbach, 2024)
    3. **RCRL**             — Reward-Conditioned Q-Learning (reward-parameterization-conditioned)
    4. **Optimal (BFS)**    — Shortest-path policy; serves as the theoretical upper bound

The script runs all approaches under the same environment configuration,
prints a concise comparison table, and (optionally) saves a cumulative-reward
plot that overlays the optimal baseline so sample efficiency can be judged at
a glance.

Usage
-----
    # quick comparison on open GridWorld (default)
    python experiments/compare_approaches.py

    # longer run with four-rooms environment
    python experiments/compare_approaches.py \\
        --env four_rooms --height 11 --width 11 --episodes 800 --seed 42

    # windy stochastic GridWorld with plot output
    python experiments/compare_approaches.py \\
        --env windy --episodes 600 --plot results/windy_comparison.png

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
from environments import make_env
from environments.gridworld import GridWorld
from experiments.run_baseline import BaseExperiment
from experiments.run_gcrl import GCRLExperiment
from experiments.run_rcrl import RCRLExperiment
from mdp.shortest_path import ShortestPathSolver
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
# Environment factory helpers
# ---------------------------------------------------------------------------


def _build_env(
    env_name: str,
    height: int,
    width: int,
    max_steps: int,
    goal_pos=None,
) -> GridWorld:
    """Build a GridWorld-compatible environment from the given parameters.

    Parameters
    ----------
    env_name :
        Short environment name (e.g. ``"gridworld"``, ``"four_rooms"``, ``"windy"``).
    height, width :
        Grid dimensions forwarded to the constructor.
    max_steps :
        Episode time limit forwarded to the constructor.
    goal_pos :
        Terminal goal ``(row, col)`` tuple, or ``None`` for reward-free mode
        (used by GCRL which trains without a terminal signal).
    """
    return make_env(
        env_name,
        height=height,
        width=width,
        goal_pos=goal_pos,
        max_steps=max_steps,
    )



# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def run_random_baseline(
    env_name: str, height: int, width: int, goal_pos: tuple[int, int], max_steps: int,
    episodes: int, seed: int, log_dir: str,
) -> ApproachResult:
    env = _build_env(env_name, height, width, max_steps, goal_pos=goal_pos)
    agent = RandomAgent(n_actions=env.n_actions, seed=seed)
    exp = RandomBaselineExperiment(
        env=env, agent=agent, n_episodes=episodes,
        eval_every=eval_every, seed=seed,
        log_dir=os.path.join(log_dir, "random"),
    )
    metrics = exp.run()
    train_metrics = [m for m in metrics if m.training]
    eval_metrics = [m for m in metrics if not m.training]
    return ApproachResult(
        name="Random Baseline",
        all_rewards=[m.total_reward for m in train_metrics],
        eval_rewards=[m.total_reward for m in eval_metrics],
        total_steps=agent.total_steps,
        n_episodes=episodes,
    )


def run_gcrl(
    env_name: str, height: int, width: int, max_steps: int,
    episodes: int, eval_goal: int, seed: int,
    alpha: float, temperature: float, n_negatives: int,
    logsumexp_reg: float, buffer_capacity: int, gamma: float,
    eval_every: int, log_dir: str,
    contrastive_gamma: float | None = None,
    n_critic_updates: int = 10,
) -> ApproachResult:
    env = _build_env(env_name, height, width, max_steps, goal_pos=None)
    agent = GoalConditionedAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        gamma=gamma, alpha=alpha,
        contrastive_gamma=contrastive_gamma,
        temperature=temperature,
        n_negatives=n_negatives,
        logsumexp_reg=logsumexp_reg,
        buffer_capacity=buffer_capacity,
        n_critic_updates=n_critic_updates,
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
    env_name: str, height: int, width: int, goal_pos: tuple[int, int], max_steps: int,
    explore_episodes: int, exploit_episodes: int,
    n_psi_bins: int, psi_min: float, psi_mix_alpha: float, alpha: float,
    epsilon: float, epsilon_min: float, epsilon_decay: float,
    gamma: float, seed: int, log_dir: str, eval_every: int = 0,
) -> ApproachResult:
    env = _build_env(env_name, height, width, max_steps, goal_pos=goal_pos)
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
        eval_every=eval_every,
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


def run_optimal_baseline(
    env_name: str, height: int, width: int, goal_pos: tuple[int, int],
    max_steps: int, episodes: int, gamma: float, seed: int,
) -> ApproachResult:
    """Compute the BFS-optimal policy and simulate *episodes* episodes.

    Returns
    -------
    ApproachResult
        ``all_rewards`` is a constant list of the undiscounted per-episode
        reward (1.0 if the goal is reachable, 0.0 otherwise).
        ``eval_rewards`` contains the same values (the optimal policy is
        evaluated greedily; there is no separate training phase).
        The ``optimal_return`` attribute (discounted) is printed to stdout.
    """
    env = _build_env(env_name, height, width, max_steps, goal_pos=goal_pos)
    solver = ShortestPathSolver(env, gamma=gamma)
    result = solver.solve()

    print(f"  BFS shortest-path distance  : {result.distance} steps")
    print(f"  Optimal discounted return   : {result.optimal_return:.6f}  (γ={gamma})")

    rewards = solver.simulate(n_episodes=episodes, seed=seed)

    return ApproachResult(
        name="Optimal (BFS Shortest Path)",
        all_rewards=rewards,
        eval_rewards=rewards,
        # total_steps is approximate (assumes every episode takes exactly
        # result.distance steps; stochastic environments may differ).
        total_steps=result.distance * episodes,
        n_episodes=episodes,
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
    optimal = next((r for r in results if "Optimal" in r.name), None)
    baseline = next((r for r in results if r.name == "Random Baseline"), None)
    trained = [r for r in results if r.name not in ("Random Baseline",) and "Optimal" not in r.name]

    if optimal:
        print(f"Optimal mean reward: {optimal.mean_reward:.4f}")
    if baseline:
        print(f"Baseline mean reward: {baseline.mean_reward:.4f}")
    for r in trained:
        ref = optimal or baseline
        gain = (
            r.mean_eval_reward - ref.mean_reward
            if (ref and r.eval_rewards)
            else r.mean_reward_last10pct - (ref.mean_reward if ref else 0)
        )
        print(f"  {r.name}: improvement over {'optimal' if optimal else 'baseline'} = {gain:+.4f}")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(
    results: list[ApproachResult],
    optimal_result: ApproachResult | None,
    output_path: str,
    window: int = 20,
    env_label: str = "GridWorld",
) -> None:
    """Save a cumulative-reward comparison plot to *output_path*.

    The plot shows per-episode reward (smoothed with a rolling mean of
    *window* episodes) for each approach, with the optimal BFS baseline
    drawn as a horizontal dashed line.  This makes it easy to see at a
    glance how quickly each algorithm converges toward the theoretical
    ceiling.

    Parameters
    ----------
    results :
        List of :class:`ApproachResult` for all non-optimal approaches.
    optimal_result :
        The BFS optimal result whose ``mean_reward`` is used for the
        horizontal reference line.  May be ``None`` to skip the line.
    output_path :
        File path for the saved figure (PNG, PDF, …).
    window :
        Rolling-mean window size (episodes).
    env_label :
        Short label shown in the plot title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed — skipping plot.")
        return

    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colour / style cycle (skip optimal and random for distinct colours)
    _STYLES: list[dict] = [
        {"color": "#2196F3", "lw": 2},   # blue  — GCRL
        {"color": "#FF5722", "lw": 2},   # orange — RCRL
        {"color": "#9E9E9E", "lw": 1.5, "linestyle": "--"},  # grey — random
    ]

    def _rolling_mean(vals: list[float], w: int) -> np.ndarray:
        arr = np.array(vals, dtype=float)
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        # 'valid' mode: output shorter by (w-1); prepend NaN to align with x
        smoothed = np.convolve(arr, kernel, mode="valid")
        pad = np.full(w - 1, np.nan)
        return np.concatenate([pad, smoothed])

    style_idx = 0
    for r in results:
        rewards = r.all_rewards
        if not rewards:
            continue
        x = np.arange(1, len(rewards) + 1)
        smoothed = _rolling_mean(rewards, window)

        style = _STYLES[style_idx % len(_STYLES)] if style_idx < len(_STYLES) else {}
        style_idx += 1

        ax.plot(x, smoothed, label=r.name, **style, alpha=0.9)
        # Light shading for raw data
        ax.fill_between(
            x,
            np.array(rewards, dtype=float),
            alpha=0.08,
            color=style.get("color", "grey"),
        )

    # Optimal BFS baseline — horizontal dashed line
    if optimal_result is not None:
        opt_val = optimal_result.mean_reward
        ax.axhline(
            opt_val,
            color="#4CAF50", lw=2, linestyle="-.",
            label=f"Optimal (BFS) = {opt_val:.2f}",
            zorder=5,
        )

    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel(f"Total Reward (rolling mean, w={window})", fontsize=13)
    ax.set_title(
        f"RL Approach Comparison — {env_label}\n"
        f"(higher is better; optimal line = BFS shortest-path ceiling)",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Random, GCRL, RCRL, and Optimal on configurable GridWorld.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=_CONFIG_PATH, help="Path to YAML config file.")
    # Environment selection
    parser.add_argument(
        "--env",
        type=str,
        default="gridworld",
        choices=["gridworld", "grid", "four_rooms", "fourrooms", "windy", "windy_gridworld"],
        help=(
            "Environment to use. "
            "'gridworld'/'grid' = open GridWorld; "
            "'four_rooms'/'fourrooms' = Four-Rooms benchmark; "
            "'windy'/'windy_gridworld' = stochastic Windy GridWorld."
        ),
    )
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
    parser.add_argument(
        "--n-critic-updates",
        type=int,
        default=10,
        help=(
            "Number of infoNCE mini-batch updates per episode (GCRL).  "
            "More updates per episode accelerates convergence of the "
            "contrastive critic without changing the total episode budget.  "
            "Default: 10 (empirically sufficient for grids up to 10×10)."
        ),
    )
    parser.add_argument(
        "--contrastive-gamma",
        type=float,
        default=None,
        help=(
            "Geometric future-state sampling gamma for the GCRL contrastive "
            "objective.  Controls the mean lookahead E[Δ] = cγ/(1-cγ); must "
            "satisfy E[Δ] ≥ minimum Manhattan distance from start to goal.  "
            "Defaults to (H+W-2)/(H+W-1) which exactly equals the minimum "
            "path length for a grid of height H and width W "
            "(e.g. 8/9≈0.89 for 5×5, 18/19≈0.947 for 10×10).  "
            "Override explicitly to fine-tune for grids with walls or "
            "non-corner start/goal positions."
        ),
    )
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
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Save a cumulative-reward comparison plot to PATH "
            "(e.g. 'results/comparison.png'). "
            "Requires matplotlib."
        ),
    )
    parser.add_argument(
        "--plot-window",
        type=int,
        default=20,
        help="Rolling-mean window size (episodes) for the comparison plot.",
    )

    # --- Apply YAML config as defaults (CLI args override YAML) ---
    pre_p = argparse.ArgumentParser(add_help=False)
    pre_p.add_argument("--config", default=_CONFIG_PATH)

    if argv is None:
        argv = []
        
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
# compare_all — importable entry point (used by notebooks)
# ---------------------------------------------------------------------------


def compare_all(args) -> list[ApproachResult]:
    """Run all approaches and return results (used by evaluation notebooks).

    Parameters
    ----------
    args :
        Parsed :class:`argparse.Namespace` (e.g. from :func:`parse_args`).

    Returns
    -------
    list of ApproachResult
        Results for [Random, GCRL, RCRL, Optimal] in that order.
    """
    n_states = args.height * args.width
    eval_goal = n_states - 1  # bottom-right cell
    goal_row, goal_col = divmod(eval_goal, args.width)
    goal_pos = (goal_row, goal_col)

    # Override goal for environments that set their own default goal
    env_name = getattr(args, "env", "gridworld")

    rcrl_explore = max(10, int(args.episodes * 0.8))
    rcrl_exploit = max(5, args.episodes - rcrl_explore)

    results: list[ApproachResult] = []

    r_random = run_random_baseline(
        env_name=env_name,
        height=args.height, width=args.width,
        goal_pos=goal_pos, max_steps=args.max_steps,
        episodes=args.episodes, seed=args.seed,
        log_dir=args.log_dir,
    )
    results.append(r_random)

    r_gcrl = run_gcrl(
        env_name=env_name,
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

    r_rcrl = run_rcrl(
        env_name=env_name,
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

    r_optimal = run_optimal_baseline(
        env_name=env_name,
        height=args.height, width=args.width,
        goal_pos=goal_pos, max_steps=args.max_steps,
        episodes=args.episodes, gamma=args.gamma,
        seed=args.seed,
    )
    results.append(r_optimal)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    env_name = args.env

    n_states = args.height * args.width
    goal_state = (
        args.goal_state
        if (hasattr(args, "goal_state") and args.goal_state is not None)
        else n_states - 1
    )
    start_state = getattr(args, "start_state", 0) or 0
    goal_row, goal_col = divmod(goal_state, args.width)
    start_row, start_col = divmod(start_state, args.width)
    goal_pos = (goal_row, goal_col)

    # Split episodes: use 80% for RCRL exploration, 20% for exploitation
    rcrl_explore = max(10, int(args.episodes * 0.8))
    rcrl_exploit = max(5, args.episodes - rcrl_explore)

    env_labels: dict[str, str] = {
        "gridworld": "Open GridWorld",
        "grid": "Open GridWorld",
        "four_rooms": "Four-Rooms GridWorld",
        "fourrooms": "Four-Rooms GridWorld",
        "windy": "Windy GridWorld (stochastic)",
        "windy_gridworld": "Windy GridWorld (stochastic)",
    }
    env_label = env_labels.get(env_name.lower(), env_name)

    print("=" * 60)
    print(f"Comparing RL Approaches on {env_label}")
    print("=" * 60)
    print(f"  Environment   : {env_label}")
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
        env_name=env_name,
        height=args.height, width=args.width,
        start_pos=start_pos, goal_pos=goal_pos, max_steps=args.max_steps,
        episodes=args.episodes, seed=args.seed,
        eval_every=args.eval_every,
        log_dir=args.log_dir,
    )
    results.append(r_random)
    print(f"Done.  Mean reward = {r_random.mean_reward:.4f}")

    # --- 2. GCRL -----------------------------------------------------------
    print()
    print("─" * 40)
    print("Running: GCRL (Single-Goal Contrastive RL) …")
    r_gcrl = run_gcrl(
        env_name=env_name,
        height=args.height, width=args.width,
        start_pos=start_pos, max_steps=args.max_steps,
        episodes=args.episodes, eval_goal=goal_state,
        seed=args.seed, alpha=args.alpha,
        temperature=args.temperature,
        n_negatives=args.n_negatives,
        logsumexp_reg=args.logsumexp_reg,
        buffer_capacity=args.buffer_capacity,
        gamma=args.gamma,
        eval_every=args.eval_every,
        log_dir=args.log_dir,
        contrastive_gamma=contrastive_gamma,
        n_critic_updates=args.n_critic_updates,
    )
    results.append(r_gcrl)
    print(f"Done.  Mean eval reward = {r_gcrl.mean_eval_reward:.4f}")

    # --- 3. RCRL -----------------------------------------------------------
    # All episodes used for training (with interleaved eval every eval_every).
    # No separate terminal exploitation block — keeps the episode budgets
    # and x-axis positions consistent with Random and GCRL.
    print()
    print("─" * 40)
    print("Running: RCRL (Reward-Conditioned) …")
    r_rcrl = run_rcrl(
        env_name=env_name,
        height=args.height, width=args.width,
        start_pos=start_pos, goal_pos=goal_pos, max_steps=args.max_steps,
        explore_episodes=args.episodes,
        exploit_episodes=0,
        n_psi_bins=args.n_psi_bins,
        psi_min=args.psi_min,
        psi_mix_alpha=args.psi_mix_alpha,
        alpha=args.alpha,
        epsilon=args.epsilon, epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        gamma=args.gamma, seed=args.seed,
        eval_every=args.eval_every,
        log_dir=args.log_dir,
    )
    results.append(r_rcrl)
    print(f"Done.  Mean exploit reward = {r_rcrl.mean_eval_reward:.4f}")

    # --- 4. Optimal BFS baseline -------------------------------------------
    print()
    print("─" * 40)
    print("Running: Optimal Baseline (BFS Shortest Path) …")
    r_optimal = run_optimal_baseline(
        env_name=env_name,
        height=args.height, width=args.width,
        goal_pos=goal_pos, max_steps=args.max_steps,
        episodes=args.episodes, gamma=args.gamma,
        seed=args.seed,
    )
    results.append(r_optimal)
    print(f"Done.  Mean reward = {r_optimal.mean_reward:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    n_states = args.height * args.width
    goal_state = args.goal_state if args.goal_state is not None else n_states - 1
    start_state = args.start_state if args.start_state is not None else 0
    goal_row, goal_col = divmod(goal_state, args.width)
    start_row, start_col = divmod(start_state, args.width)
    goal_pos = (goal_row, goal_col)
    start_pos = (start_row, start_col)

    print("=" * 60)
    print("Comparing RL Approaches on GridWorld")
    print("=" * 60)
    print(f"  Grid          : {args.height}×{args.width}")
    print(f"  Start         : state {start_state} ({start_row},{start_col})")
    print(f"  Goal          : state {goal_state} ({goal_row},{goal_col})")
    print(f"  Max steps     : {args.max_steps}")
    print(f"  Episodes      : {args.episodes} (per approach, all training)")
    print(f"  Eval every    : {args.eval_every} episodes")
    print(f"  Seed          : {args.seed}")
    print()

    results = compare_all(args)

    _print_table(results)

    # --- Plot (optional) ---------------------------------------------------
    if args.plot:
        plot_comparison(
            results=[r for r in results if "Optimal" not in r.name],
            optimal_result=r_optimal,
            output_path=args.plot,
            window=args.plot_window,
            env_label=env_label,
        )

    if args.render:
        env = _build_env(env_name, args.height, args.width, args.max_steps, goal_pos=goal_pos)
        env.reset()
        print("Final grid layout (G = goal, # = wall, . = free):")
        print(env.render())


if __name__ == "__main__":
    main()
