"""
run_rcrl.py
===========
CLI entry-point: Reward-Conditioned RL on GridWorld.

Background
----------
Reward-conditioned RL trains a single agent to optimise a *family* of reward
specifications by conditioning the policy and value function on a reward
parameterisation ψ:

    Q(s, a, ψ)   π(a | s, ψ)

This script implements the approach from:
    "Reward-Conditioned Reinforcement Learning" (Nauman, Cygan & Abbeel, 2026)
    arXiv:2603.05066

The reward is decomposed into k components stored per transition.  During
training each update draws ψ from a mixture

    PΨ = α · δ(ψ*) + (1−α) · pΨ

and recomputes the scalar reward rψ = Σᵢ ψᵢ·cᵢ without additional environment
interaction.  The agent always *acts* under the nominal ψ*, so sample
efficiency under the target task is preserved while the conditioned
representation gains robustness across reward variants.

The script has two phases:

1. **Training** (``--explore-episodes`` episodes):
   ε-greedy exploration; at every step a ψ is sampled from PΨ and a
   Q-learning TD update is applied to  Q[s, ψ-bin, a].

2. **Exploitation** (``--exploit-episodes`` episodes):
   Greedy evaluation under ψ*.  ε is locked at ``epsilon_min``.

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
from utils.config import load_config
from utils.metrics import EpisodeMetrics

# Path to the YAML config for this experiment (sibling configs/ directory)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "rcrl.yaml")


# ---------------------------------------------------------------------------
# Concrete experiment
# ---------------------------------------------------------------------------


class RCRLExperiment(BaseExperiment):
    """Reward-conditioned Q-learning experiment in two phases.

    Training phase
    --------------
    Run ``n_explore`` episodes with ε-greedy exploration under the nominal ψ*.
    At every step the agent samples ψ ~ PΨ, computes the parameterised reward
    rψ, and performs a Q-learning update on Q[s, ψ-bin, a].
    ε is decayed once per episode via ``agent.finish_episode()``.

    Exploitation phase
    ------------------
    Run ``n_exploit`` episodes with greedy action selection under ψ* to
    evaluate the policy learned by training on diverse parameterisations.
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

        # --- Phase 1: Training -------------------------------------------
        print("Phase 1 — Training (Q-learning with diverse ψ sampling) …")
        for episode in range(1, self.n_explore + 1):
            metrics = self._run_episode(episode, training=True)
            all_metrics.append(metrics)

            finish_info = self.agent.finish_episode()
            self.logger.log_episode(episode, metrics)

            if episode % max(1, self.n_explore // 5) == 0:
                mean_td = metrics.mean_step_metric("td_error")
                print(
                    f"  [train ep {episode:>4d}]  "
                    f"reward={metrics.total_reward:.2f}  "
                    f"length={metrics.length:>3d}  "
                    f"mean_td={mean_td or 0.0:.4f}  "
                    f"ε={self.agent.epsilon:.3f}"
                )

        # --- Phase 2: Exploitation -------------------------------------------
        print(
            f"\nPhase 2 — Exploitation "
            f"(greedy under nominal ψ*, ε={self.agent.epsilon_min}) …"
        )
        # Lock ε at minimum for near-greedy behaviour
        self.agent.epsilon = self.agent.epsilon_min

        exploit_rewards = []
        for ep_idx in range(1, self.n_exploit + 1):
            episode = self.n_explore + ep_idx
            metrics = self._run_exploit_episode(episode)
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

    def _run_exploit_episode(self, episode: int) -> EpisodeMetrics:
        """Run one episode with the nominal-ψ* greedy policy."""
        obs, info = self.env.reset(seed=int(self._rng.integers(0, 2**31)))
        self.agent.reset()

        total_reward = 0.0
        steps = 0

        while True:
            action = self.agent.select_action(obs, greedy=True)
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
    parser.add_argument("--config", type=str, default=_CONFIG_PATH, help="Path to YAML config file.")
    parser.add_argument("--height", type=int, default=5, help="Grid height.")
    parser.add_argument("--width", type=int, default=5, help="Grid width.")
    parser.add_argument(
        "--goal",
        type=int,
        default=None,
        help="Goal state index. Defaults to bottom-right cell.",
    )
    parser.add_argument("--explore-episodes", type=int, default=400, help="Training episodes.")
    parser.add_argument("--exploit-episodes", type=int, default=100, help="Exploitation episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--n-psi-bins", type=int, default=5, help="Number of reward-parameterisation bins.")
    parser.add_argument("--psi-min", type=float, default=-0.1, help="Most negative step-cost weight (≤ 0).")
    parser.add_argument("--psi-mix-alpha", type=float, default=0.5, help="Fraction of nominal ψ* draws during training.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Q-learning step size.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial ε.")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum ε.")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="ε decay per episode.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--render", action="store_true", help="Print ASCII grid after run.")
    parser.add_argument("--log-dir", type=str, default="logs/rcrl", help="Log directory.")

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
        # goal_pos in YAML is [row, col]; convert to flat index using the
        # YAML width (which may be overridden on the CLI, but we use YAML
        # width here as that's the grid width the goal was specified for).
        goal_pos = env_cfg.get("goal_pos")
        if goal_pos is not None:
            w = env_cfg.get("width", 5)
            yaml_defaults["goal"] = goal_pos[0] * w + goal_pos[1]
        if "gamma" in mdp_cfg:
            yaml_defaults["gamma"] = mdp_cfg["gamma"]
        if "n_psi_bins" in agent_cfg:
            yaml_defaults["n_psi_bins"] = agent_cfg["n_psi_bins"]
        if "psi_min" in agent_cfg:
            yaml_defaults["psi_min"] = agent_cfg["psi_min"]
        if "psi_mix_alpha" in agent_cfg:
            yaml_defaults["psi_mix_alpha"] = agent_cfg["psi_mix_alpha"]
        if "alpha" in agent_cfg:
            yaml_defaults["alpha"] = agent_cfg["alpha"]
        if "epsilon" in agent_cfg:
            yaml_defaults["epsilon"] = agent_cfg["epsilon"]
        if "epsilon_min" in agent_cfg:
            yaml_defaults["epsilon_min"] = agent_cfg["epsilon_min"]
        if "epsilon_decay" in agent_cfg:
            yaml_defaults["epsilon_decay"] = agent_cfg["epsilon_decay"]
        if "explore_episodes" in training_cfg:
            yaml_defaults["explore_episodes"] = training_cfg["explore_episodes"]
        if "exploit_episodes" in training_cfg:
            yaml_defaults["exploit_episodes"] = training_cfg["exploit_episodes"]
        if "seed" in training_cfg:
            yaml_defaults["seed"] = training_cfg["seed"]
        if "log_dir" in log_cfg:
            yaml_defaults["log_dir"] = log_cfg["log_dir"]
        parser.set_defaults(**yaml_defaults)

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
        n_psi_bins=args.n_psi_bins,
        psi_min=args.psi_min,
        psi_mix_alpha=args.psi_mix_alpha,
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
    print("Reward-Conditioned RL (RCRL) — Nauman et al. 2026")
    print("=" * 60)
    print(f"  Environment : {env!r}")
    print(f"  Agent       : {agent!r}")
    print(f"  Goal state  : {goal_pos_flat} ({goal_row}, {goal_col})")
    print(f"  Train eps   : {args.explore_episodes}")
    print(f"  Exploit eps : {args.exploit_episodes}")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  ψ bins      : {args.n_psi_bins}  (psi_min={args.psi_min})")
    print(f"  ψ mix α     : {args.psi_mix_alpha}")
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
    print(f"  Explore mean reward      : {_mean(explore_rewards):.4f}")
    print(f"  Exploit mean reward      : {_mean(exploit_rewards):.4f}")
    print(f"  Exploit mean length      : {_mean(exploit_lengths):.1f}")
    print(f"  Final ε                  : {agent.epsilon:.4f}")
    print(f"  Total env steps          : {agent.total_steps}")

    if args.render:
        obs, _ = env.reset()
        print("\nFinal grid state:")
        print(env.render())

    return all_metrics


if __name__ == "__main__":
    main()
