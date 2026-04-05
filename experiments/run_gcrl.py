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
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.goal_conditioned_agent import GoalConditionedAgent
from environments.gridworld import GridWorld
from experiments.base_experiment import BaseExperiment
from utils.metrics import EpisodeMetrics


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
                print(
                    f"  [eval] goal={self.eval_goal}  "
                    f"reward={eval_metrics.total_reward:.2f}  "
                    f"length={eval_metrics.length}"
                )

        return all_metrics

    def _run_gcrl_eval(self, episode: int) -> EpisodeMetrics:
        """Run one greedy evaluation episode using the goal-enabled eval env."""
        self.agent.set_goal(self.eval_goal)
        obs, info = self._eval_env.reset(seed=int(self._rng.integers(0, 2**31)))
        self.agent.reset()

        total_reward = 0.0
        steps = 0

        while True:
            state = int(np.asarray(obs).flat[0])
            # Greedy: argmax over the contrastive critic
            action = int(np.argmax(self.agent.C[state, :, self.eval_goal]))
            next_obs, reward, terminated, truncated, info = self._eval_env.step(action)
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
        description="Single-Goal Contrastive RL on GridWorld.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--height", type=int, default=5, help="Grid height.")
    parser.add_argument("--width", type=int, default=5, help="Grid width.")
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
    parser.add_argument("--n-negatives", type=int, default=16, help="Negative examples per infoNCE update.")
    parser.add_argument("--logsumexp-reg", type=float, default=0.01, help="LogSumExp regularisation coefficient.")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="Replay buffer capacity.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--eval-every", type=int, default=50, help="Eval every N episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--render", action="store_true", help="Print ASCII grid after run.")
    parser.add_argument("--log-dir", type=str, default="logs/gcrl", help="Log directory.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> list[EpisodeMetrics]:
    args = parse_args(argv)

    # Training env is reward-free (no goal_pos) — contrastive RL needs no reward
    env = GridWorld(
        height=args.height,
        width=args.width,
        max_steps=args.max_steps,
    )

    n_states = env.n_states
    eval_goal = args.goal if args.goal is not None else n_states - 1

    agent = GoalConditionedAgent(
        n_states=n_states,
        n_actions=env.n_actions,
        gamma=args.gamma,
        alpha=args.alpha,
        temperature=args.temperature,
        n_negatives=args.n_negatives,
        logsumexp_reg=args.logsumexp_reg,
        buffer_capacity=args.buffer_capacity,
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
