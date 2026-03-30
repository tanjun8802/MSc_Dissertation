"""
run_gcrl.py
===========
CLI entry-point: Goal-Conditioned Q-Learning with Hindsight Experience Replay
on the GridWorld environment.

Background
----------
Goal-conditioned RL (GCRL) extends standard RL by conditioning the policy on
a desired goal g.  The Q-function becomes Q(s, g, a), capturing the value of
taking action a from state s when the agent wants to reach g.

This script implements the approach from:
    "A Single Goal is All You Need: On the Elegance of Task-Conditioned RL"
    (Mezghani et al., 2023)

Key mechanism — **Hindsight Experience Replay (HER)**:
After every training episode the agent retroactively relabels transitions with
*achieved* future states as substitute goals.  This generates a dense learning
signal even when the intended goal was never reached, dramatically improving
sample efficiency.

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
    """Goal-conditioned Q-learning with HER on GridWorld.

    Each training episode:
    1. A goal state is sampled uniformly from all valid (non-wall) states.
    2. The agent runs one episode conditioned on that goal.
    3. After the episode, HER relabeling is applied to update Q(s, g', a) for
       additional substitute goals sampled from the trajectory.

    Evaluation episodes always use ``eval_goal`` (default: bottom-right cell).
    A separate goal-enabled environment is used for evaluation so that reaching
    the goal gives a +1 reward and terminates the episode.
    """

    def __init__(self, env: GridWorld, agent: GoalConditionedAgent, eval_goal: int, **kwargs) -> None:
        super().__init__(env=env, agent=agent, **kwargs)
        self.eval_goal = eval_goal
        self._all_states = env.get_all_states()

        # Build a separate evaluation env with the goal embedded so that
        # reaching eval_goal terminates the episode with +1 reward.
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
        """Delegate to the agent's Q-learning update."""
        return self.agent.update(obs, action, reward, next_obs, terminated, truncated, info)

    def run(self) -> list[EpisodeMetrics]:
        """Override run() to sample random goals and apply HER each episode."""
        rng = np.random.default_rng(self.seed)
        all_metrics: list[EpisodeMetrics] = []

        for episode in range(1, self.n_episodes + 1):
            # Sample a random goal for this training episode
            goal = int(rng.choice(self._all_states))
            self.agent.set_goal(goal)

            metrics = self._run_episode(episode, training=True)
            all_metrics.append(metrics)

            # Apply HER after episode completes
            self.agent.finish_episode_with_her()

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
            action = int(self.agent.Q[state, self.eval_goal].argmax())
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
        description="Goal-Conditioned Q-Learning with HER on GridWorld.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--height", type=int, default=5, help="Grid height.")
    parser.add_argument("--width", type=int, default=5, help="Grid width.")
    parser.add_argument(
        "--goal",
        type=int,
        default=None,
        help="Fixed evaluation goal (flat state index). Defaults to bottom-right cell.",
    )
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Q-learning step size.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial ε.")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum ε.")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="ε decay per episode.")
    parser.add_argument("--her-k", type=int, default=4, help="HER substitutions per transition.")
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
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        her_k=args.her_k,
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
    print("Goal-Conditioned RL (GCRL) with Hindsight Experience Replay")
    print("=" * 60)
    print(f"  Environment : {env!r}")
    print(f"  Agent       : {agent!r}")
    print(f"  Eval goal   : state {eval_goal}")
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
    print(f"  Final ε                      : {agent.epsilon:.4f}")

    if args.render:
        agent.set_goal(eval_goal)
        obs, _ = env.reset()
        print("\nFinal grid state:")
        print(env.render())

    return all_metrics


if __name__ == "__main__":
    main()
