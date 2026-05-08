"""
run_gcrl.py
===========
CLI entry-point: Single-Goal Contrastive RL on the GridWorld environment.

Background
----------
Goal-conditioned RL (GCRL) extends standard RL by conditioning the policy on
a desired goal g.  The critic becomes C(s, a, sf), capturing the (log)
likelihood that taking action a from state s will eventually reach state sf.

This script implements the approach from:
    "A Single Goal is All You Need: Skills and Exploration Emerge from
    Contrastive RL without Rewards, Demonstrations, or Subgoals"
    (Liu, Tang & Eysenbach, 2024)

Key mechanism — **Single-Goal Contrastive RL** (Algorithm 1 in the paper):
* The critic C(s, a, sf) is learned via an infoNCE contrastive objective
  with LogSumExp regularisation (Eq. 3).  No reward function is used.
* During data collection, the policy is ALWAYS conditioned on the single
  hard target goal s*.  Skills and exploration emerge naturally without any
  curriculum, dense rewards, or subgoal generation.

Usage
-----
    # basic run
    python experiments/run_gcrl.py

    # with custom settings
    python experiments/run_gcrl.py \\
        --episodes 500 --height 5 --width 5 \\
        --goal 24 --seed 0 --render --log-dir logs/gcrl
"""

from __future__ import annotations

import argparse
from email import parser
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.goal_conditioned_agent import GoalConditionedAgent
from environments.gridworld import GridWorld
from experiments.base_experiment import BaseExperiment
from utils.config import load_config
from utils.metrics import EpisodeMetrics

# Path to the YAML config for this experiment (sibling configs/ directory)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "gcrl.yaml")


# ---------------------------------------------------------------------------
# Concrete experiment
# ---------------------------------------------------------------------------


class GCRLExperiment(BaseExperiment):
    """Single-goal contrastive RL on GridWorld.

    Each training episode:
    1. The policy is conditioned on the single hard target goal s* throughout.
    2. The agent runs one episode always targeting s*.
    3. After the episode, (s, a, sf) pairs are generated via geometric future
       sampling and the contrastive critic is updated (Eq. 3 in the paper).

    Evaluation episodes run a greedy policy (argmax over C[s, :, s*]) using
    a goal-embedded environment so that reaching s* gives a +1 reward.
    """

    def __init__(self, env: GridWorld, agent: GoalConditionedAgent, eval_goal: int, **kwargs) -> None:
        super().__init__(env=env, agent=agent, **kwargs)
        self.eval_goal = eval_goal

        # Set the single hard target goal on the agent once — it never changes
        self.agent.set_goal(eval_goal)

        # Separate evaluation env with goal embedded so success is measurable
        goal_row, goal_col = divmod(eval_goal, env.width)
        self._eval_env = GridWorld(
            height=env.height,
            width=env.width,
            start_pos=env.start_pos,
            goal_pos=(goal_row, goal_col),
            walls=env.walls,
            max_steps=env.max_steps,
        )

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
        """Delegate to the agent's (reward-free) update."""
        return self.agent.update(obs, action, reward, next_obs, terminated, truncated, info)

    def run(self) -> list[EpisodeMetrics]:
        """Override run() to apply the contrastive update after each episode."""
        all_metrics: list[EpisodeMetrics] = []
        last_eval_metrics: EpisodeMetrics | None = None

        # Pre-compute stage milestones for C-table snapshots (early/mid/late)
        _n = self.n_episodes
        _q_milestones = {
            max(1, _n // 3): "early",
            max(1, 2 * _n // 3): "mid",
            _n: "late",
        }

        for episode in range(1, self.n_episodes + 1):
            # Always use the single hard target goal (Algorithm 1 in paper)
            self.agent.set_goal(self.eval_goal)

            metrics = self._run_episode(episode, training=True)
            all_metrics.append(metrics)

            # Contrastive critic update after the episode
            self.agent.finish_episode_with_contrastive_update()

            self.logger.log_episode(episode, metrics)

            if self.eval_every > 0 and episode % self.eval_every == 0:
                eval_metrics = self._run_gcrl_eval(episode)
                all_metrics.append(eval_metrics)
                self.logger.log_eval(episode, eval_metrics)
                last_eval_metrics = eval_metrics
                print(
                    f"  [eval] goal={self.eval_goal}  "
                    f"reward={eval_metrics.total_reward:.2f}  "
                    f"length={eval_metrics.length}"
                )

            # Save C-table snapshot at early / mid / late milestones
            # C[:, :, eval_goal] acts as Q[state, action] for goal-reaching
            if episode in _q_milestones:
                stage = _q_milestones[episode]
                c_slice = self.agent.C[:, :, self.eval_goal].copy()
                np.save(
                    os.path.join(self.logger.log_dir, f"q_{stage}.npy"),
                    c_slice,
                )

        # Save the full C-table (shape: n_states × n_actions × n_states) so
        # that the evaluation notebook can use C[:, :, any_goal] for
        # zero-shot transfer-goal evaluation without retraining.
        np.save(
            os.path.join(self.logger.log_dir, "c_table.npy"),
            self.agent.C.copy(),
        )

        # Save trajectory of the last evaluation episode for visualisation
        if last_eval_metrics is not None and last_eval_metrics.trajectory:
            self.logger.log_trajectory(
                last_eval_metrics.episode, last_eval_metrics.trajectory
            )

        return all_metrics

    def _run_gcrl_eval(self, episode: int) -> EpisodeMetrics:
        """Run one evaluation episode using the softmax policy conditioned on eval_goal.

        The paper (Liu et al., 2024) uses the SAME softmax policy for both
        training and evaluation: π(a|s,g) ∝ exp(C[s,a,g]/τ).  Using hard
        argmax instead can cause the agent to get permanently stuck on
        blocked actions (wall NOPs) whose C-values are numerically similar to
        good-direction actions early in training — the softmax naturally
        recovers by exploring alternatives.
        """
        self.agent.set_goal(self.eval_goal)
        obs, info = self._eval_env.reset(seed=int(self._rng.integers(0, 2**31)))
        self.agent.reset()

        total_reward = 0.0
        steps = 0
        trajectory: list[tuple[int, int, int, float]] = []

        while True:
            state = int(np.asarray(obs).flat[0])
            # Softmax policy — consistent with training and the paper.
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = self._eval_env.step(action)
            total_reward += float(reward)
            steps += 1
            trajectory.append((steps, state, action, float(reward)))
            obs = next_obs

            if terminated or truncated:
                # Append the arrival state so the final arrow in trajectory
                # visualisations reaches the goal cell (action=-1 = terminal marker).
                if terminated:
                    next_state = int(np.asarray(next_obs).flat[0])
                    trajectory.append((steps + 1, next_state, -1, 0.0))
                break

        return EpisodeMetrics(
            episode=episode,
            total_reward=total_reward,
            length=steps,
            training=False,
            step_metrics=[],
            trajectory=trajectory,
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-Goal Contrastive RL on GridWorld.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=_CONFIG_PATH, help="Path to YAML config file.")
    parser.add_argument("--height", type=int, default=5, help="Grid height.")
    parser.add_argument("--width", type=int, default=5, help="Grid width.")
    parser.add_argument("--start-pos", nargs=2, type=int, metavar=("ROW", "COL"), default=None, help="Start position as: ROW COL")
    parser.add_argument("--goal-pos", nargs=2, type=int, metavar=("ROW", "COL"), default=None, help="Goal position as: ROW COL")
    parser.add_argument("--walls", nargs="*", type=int, default=None, help="Wall coordinates as flat list: r1 c1 r2 c2 ...")
    parser.add_argument(
        "--goal",
        type=int,
        default=None,
        help="Target goal (flat state index). Defaults to bottom-right cell.",
    )
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Contrastive critic step size.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature τ.")
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=0.0,
        help="Average actor entropy target; 0 anneals the softmax policy toward greedy.",
    )
    parser.add_argument(
        "--min-temperature",
        type=float,
        default=1e-6,
        help="Lower bound for the softmax temperature.",
    )
    parser.add_argument("--n-negatives", type=int, default=16, help="Negative examples per infoNCE update.")
    parser.add_argument("--logsumexp-reg", type=float, default=0.01, help="LogSumExp regularisation coefficient.")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="Replay buffer capacity.")
    parser.add_argument(
        "--samples-per-insert",
        type=int,
        default=256,
        help="Replay ratio: positive critic samples drawn per newly inserted transition.",
    )
    parser.add_argument(
        "--n-critic-updates",
        type=int,
        default=10,
        help="Legacy minimum number of critic mini-batches per episode.",
    )
    parser.add_argument(
        "--contrastive-gamma",
        type=float,
        default=None,
        help=(
            "Geometric future-state sampling parameter for the contrastive "
            "objective (Δ ~ Geom(1-cγ)-1, mean offset = cγ/(1-cγ)).  "
            "Should be chosen so the mean offset matches typical episode length. "
            "Defaults to --gamma if not set.  "
            "Rule of thumb: use ~0.9 for short episodes (5×5 grid) "
            "and ~0.99 for long episodes (15×15 grid)."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--eval-every", type=int, default=50, help="Eval every N episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--render", action="store_true", help="Print ASCII grid after run.")
    parser.add_argument("--log-dir", type=str, default="logs/gcrl", help="Log directory.")

    # --- Apply YAML config as defaults (CLI args override YAML) ---
    # Parse --config first with a minimal parser so we know which file to load.
    pre, _ = argparse.ArgumentParser(add_help=False).parse_known_args(argv)
    # Use --config value if provided on CLI, otherwise fall back to _CONFIG_PATH.
    cfg_path = None
    if argv:
        pre_p = argparse.ArgumentParser(add_help=False)
        pre_p.add_argument("--config", default=_CONFIG_PATH)
        cfg_path = pre_p.parse_known_args(argv)[0].config
    cfg = load_config(cfg_path or _CONFIG_PATH)
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
        if "start_pos" in env_cfg and env_cfg["start_pos"] is not None:
            yaml_defaults["start_pos"] = list(env_cfg["start_pos"])
        if "goal_pos" in env_cfg and env_cfg["goal_pos"] is not None:
            yaml_defaults["goal_pos"] = list(env_cfg["goal_pos"])
        if "walls" in env_cfg:
            yaml_defaults["walls"] = [x for pair in env_cfg["walls"] for x in pair]
        if "max_steps" in env_cfg:
            yaml_defaults["max_steps"] = env_cfg["max_steps"]
        if "gamma" in mdp_cfg:
            yaml_defaults["gamma"] = mdp_cfg["gamma"]
        if "contrastive_gamma" in agent_cfg:
            yaml_defaults["contrastive_gamma"] = agent_cfg["contrastive_gamma"]
        if "alpha" in agent_cfg:
            yaml_defaults["alpha"] = agent_cfg["alpha"]
        if "temperature" in agent_cfg:
            yaml_defaults["temperature"] = agent_cfg["temperature"]
        if "target_entropy" in agent_cfg:
            yaml_defaults["target_entropy"] = agent_cfg["target_entropy"]
        if "min_temperature" in agent_cfg:
            yaml_defaults["min_temperature"] = agent_cfg["min_temperature"]
        if "n_negatives" in agent_cfg:
            yaml_defaults["n_negatives"] = agent_cfg["n_negatives"]
        if "logsumexp_reg" in agent_cfg:
            yaml_defaults["logsumexp_reg"] = agent_cfg["logsumexp_reg"]
        if "buffer_capacity" in agent_cfg:
            yaml_defaults["buffer_capacity"] = agent_cfg["buffer_capacity"]
        if "samples_per_insert" in agent_cfg:
            yaml_defaults["samples_per_insert"] = agent_cfg["samples_per_insert"]
        if "n_critic_updates" in agent_cfg:
            yaml_defaults["n_critic_updates"] = agent_cfg["n_critic_updates"]
        if "epsilon" in agent_cfg:
            yaml_defaults["epsilon"] = agent_cfg["epsilon"]
        if "epsilon_min" in agent_cfg:
            yaml_defaults["epsilon_min"] = agent_cfg["epsilon_min"]
        if "epsilon_decay" in agent_cfg:
            yaml_defaults["epsilon_decay"] = agent_cfg["epsilon_decay"]
        if "her_k" in agent_cfg:
            yaml_defaults["her_k"] = agent_cfg["her_k"]
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
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> list[EpisodeMetrics]:
    args = parse_args(argv)

    n_states = args.height * args.width
    eval_goal = args.goal if args.goal is not None else n_states - 1
    goal_row, goal_col = divmod(eval_goal, args.width)

    # Training env: goal_pos is set so episodes terminate at the goal.
    # This is required by Algorithm 1 of Liu et al. (2024) — the single hard
    # target goal must be the terminal state so that the geometric future-state
    # sampling produces (s, a, sf=goal) pairs with strong causal signal.
    # No reward signal is used by the contrastive critic (reward-free learning).
    env = GridWorld(
        height=args.height,
        width=args.width,
        goal_pos=(goal_row, goal_col),
        max_steps=args.max_steps,
    )

    agent = GoalConditionedAgent(
        n_states=n_states,
        n_actions=env.n_actions,
        gamma=args.gamma,
        contrastive_gamma=args.contrastive_gamma,
        alpha=args.alpha,
        temperature=args.temperature,
        target_entropy=args.target_entropy,
        min_temperature=args.min_temperature,
        n_negatives=args.n_negatives,
        logsumexp_reg=args.logsumexp_reg,
        buffer_capacity=args.buffer_capacity,
        samples_per_insert=args.samples_per_insert,
        n_critic_updates=args.n_critic_updates,
        seed=args.seed,
    )

    experiment = GCRLExperiment(
        env=env,
        agent=agent,
        eval_goal=eval_goal,
        n_episodes=args.episodes,
        eval_every=args.eval_every,
        seed=args.seed,
        log_dir=args.log_dir,
    )

    print("=" * 60)
    print("Goal-Conditioned RL (GCRL) — Single-Goal Contrastive RL")
    print("=" * 60)
    print(f"  Environment : {env!r}")
    print(f"  Agent       : {agent!r}")
    print(f"  Target goal : state {eval_goal}")
    print(f"  Episodes    : {args.episodes}")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  Seed        : {args.seed}")
    print()

    all_metrics = experiment.run()

    train_metrics = [m for m in all_metrics if m.training]
    eval_metrics_list = [m for m in all_metrics if not m.training]
    rewards = [m.total_reward for m in train_metrics]
    n = len(rewards)
    last_10pct = max(1, n // 10)
    eval_rewards = [m.total_reward for m in eval_metrics_list]

    print()
    print(f"Results over {args.episodes} training episodes:")
    print(f"  Mean train reward (all)      : {sum(rewards) / max(n, 1):.4f}")
    print(f"  Mean train reward (last 10%) : {sum(rewards[-last_10pct:]) / last_10pct:.4f}")
    if eval_rewards:
        print(f"  Mean eval reward             : {sum(eval_rewards) / len(eval_rewards):.4f}")
    print(f"  Total env steps              : {agent.total_steps}")

    if args.render:
        agent.set_goal(eval_goal)
        obs, _ = env.reset()
        print("\nFinal grid state:")
        print(env.render())

    return all_metrics


if __name__ == "__main__":
    main()
