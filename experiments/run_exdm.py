"""
run_exdm.py
===========
CLI entry-point: Exploratory Diffusion Model (ExDM) on GridWorld.

Background
----------
ExDM uses a score-based intrinsic reward derived from a state diffusion model
to drive reward-free exploration.  It was proposed in:

    Ying, C., Chen, H., Hao, Z., Zhou, X., Su, H., & Zhu, J. (2025).
    "Exploratory Diffusion Model for Unsupervised Reinforcement Learning."
    arXiv:2502.07279.

Key mechanism — **Score-based Intrinsic Reward** (Eq. 8 in the paper):

    R_score(s) = E_{ε,t} [ ‖ε̂_θ'(s_t | t) − ε‖² ]

where ε̂_θ' is a state diffusion score model trained online on the empirical
state distribution in the replay buffer.  States that are rarely visited have
high reconstruction error under the diffusion model, yielding high intrinsic
reward and encouraging further exploration.

Usage
-----
    # basic run
    python experiments/run_exdm.py

    # with custom settings
    python experiments/run_exdm.py \\
        --episodes 500 --height 10 --width 10 \\
        --goal 99 --seed 0 --render --log-dir logs/exdm
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.exdm_agent import ExDMAgent
from environments.gridworld import GridWorld
from experiments.base_experiment import BaseExperiment
from utils.config import load_config
from utils.metrics import EpisodeMetrics

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "exdm.yaml")


# ---------------------------------------------------------------------------
# Concrete experiment: ExDM on GridWorld
# ---------------------------------------------------------------------------


class ExDMExperiment(BaseExperiment):
    """ExDM pre-training on a GridWorld environment.

    The agent is trained purely with the score-based intrinsic reward;
    extrinsic rewards are ignored during this phase (reward-free pre-training).
    Greedy evaluation episodes are periodically run against the hard target
    goal so that performance can be compared to other approaches.

    Parameters
    ----------
    env :
        The GridWorld environment.
    agent :
        An :class:`~agents.exdm_agent.ExDMAgent` instance.
    eval_goal :
        Flat state index of the evaluation goal.
    n_episodes :
        Total number of training episodes.
    eval_every :
        Evaluate every N episodes (0 = no evaluation).
    seed :
        Random seed.
    log_dir :
        Directory for CSV and artefact output.
    """

    def __init__(
        self,
        env,
        agent: ExDMAgent,
        eval_goal: int,
        n_episodes: int = 300,
        eval_every: int = 50,
        seed: int = 42,
        log_dir: str = "logs/exdm",
    ) -> None:
        super().__init__(
            env=env,
            agent=agent,
            n_episodes=n_episodes,
            eval_every=eval_every,
            seed=seed,
            log_dir=log_dir,
        )
        self.eval_goal = eval_goal
        goal_row, goal_col = divmod(eval_goal, env.width)
        # Separate evaluation env with the goal embedded for success measurement
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
        """Delegate to the agent's intrinsic-reward Q-learning update."""
        return self.agent.update(obs, action, reward, next_obs, terminated, truncated, info)

    def eval_step(self, obs) -> int:
        """Greedy evaluation: select the Q-table argmax action."""
        state = int(np.asarray(obs).flat[0])
        return int(np.argmax(self.agent.Q[state]))

    def run(self) -> list[EpisodeMetrics]:
        """Training loop with ε decay and optional interleaved evaluation.

        After each training episode, ``agent.finish_episode()`` is called to
        decay ε.  Every ``eval_every`` episodes, one greedy evaluation episode
        is run in ``_eval_env`` (which has the goal embedded) to measure
        goal-reaching performance.

        Q-table snapshots are saved at early / mid / late milestones.
        """
        all_metrics: list[EpisodeMetrics] = []
        last_eval_metrics: EpisodeMetrics | None = None

        _n = self.n_episodes
        _q_milestones = {
            max(1, _n // 3): "early",
            max(1, 2 * _n // 3): "mid",
            _n: "late",
        }

        print("ExDM — Pre-training with score-based intrinsic reward …")

        for episode in range(1, self.n_episodes + 1):
            metrics = self._run_episode(episode, training=True)
            all_metrics.append(metrics)

            # Decay ε at the end of each training episode
            self.agent.finish_episode()
            metrics.epsilon = self.agent.epsilon
            self.logger.log_episode(episode, metrics)

            if episode % max(1, self.n_episodes // 5) == 0:
                mean_r_int = metrics.mean_step_metric("intrinsic_reward")
                print(
                    f"  [train ep {episode:>4d}]  "
                    f"reward={metrics.total_reward:.2f}  "
                    f"length={metrics.length:>3d}  "
                    f"R_score={mean_r_int or 0.0:.4f}  "
                    f"ε={self.agent.epsilon:.3f}"
                )

            if self.eval_every > 0 and episode % self.eval_every == 0:
                eval_metrics = self._run_eval_episode(episode)
                all_metrics.append(eval_metrics)
                last_eval_metrics = eval_metrics
                self.logger.log_eval(episode, eval_metrics)
                print(
                    f"  [eval  ep {episode:>4d}]  "
                    f"reward={eval_metrics.total_reward:.2f}  "
                    f"length={eval_metrics.length:>3d}"
                )

            # Save Q-table snapshot at early / mid / late milestones
            if episode in _q_milestones:
                stage = _q_milestones[episode]
                np.save(
                    os.path.join(self.logger.log_dir, f"q_{stage}.npy"),
                    self.agent.Q.copy(),
                )

        # Save the full Q-table for transfer-goal evaluation in the notebook
        np.save(
            os.path.join(self.logger.log_dir, "q_table.npy"),
            self.agent.Q.copy(),
        )

        # Save trajectory of the last evaluation episode for visualisation
        if last_eval_metrics is not None and last_eval_metrics.trajectory:
            self.logger.log_trajectory(
                last_eval_metrics.episode, last_eval_metrics.trajectory
            )

        return all_metrics

    def _run_eval_episode(self, episode: int) -> EpisodeMetrics:
        """Run one greedy evaluation episode in the goal-embedded env."""
        obs, info = self._eval_env.reset(seed=int(self._rng.integers(0, 2**31)))
        self.agent.reset()

        total_reward = 0.0
        steps = 0
        trajectory: list[tuple[int, int, int, float]] = []

        while True:
            state = int(np.asarray(obs).flat[0])
            action = int(np.argmax(self.agent.Q[state]))
            next_obs, reward, terminated, truncated, info = self._eval_env.step(action)
            total_reward += float(reward)
            steps += 1
            trajectory.append((steps, state, action, float(reward)))
            obs = next_obs

            if terminated or truncated:
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


def parse_args(argv=None) -> argparse.Namespace:
    cfg = load_config(_CONFIG_PATH) or {}
    env_cfg    = cfg.get("env", {})
    agent_cfg  = cfg.get("agent", {})
    train_cfg  = cfg.get("training", {})
    log_cfg    = cfg.get("logging", {})

    parser = argparse.ArgumentParser(
        description="Run ExDM (Exploratory Diffusion Model) on GridWorld.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int,
                        default=train_cfg.get("n_episodes", 500))
    parser.add_argument("--height", type=int,
                        default=env_cfg.get("height", 10))
    parser.add_argument("--width", type=int,
                        default=env_cfg.get("width", 10))
    parser.add_argument("--max-steps", type=int,
                        default=env_cfg.get("max_steps", 800))
    parser.add_argument("--goal", type=int, default=None,
                        help="Flat goal state index (default: bottom-right cell).")
    parser.add_argument("--seed", type=int,
                        default=train_cfg.get("seed", 42))
    parser.add_argument("--alpha", type=float,
                        default=agent_cfg.get("alpha", 0.1))
    parser.add_argument("--model-lr", type=float,
                        default=agent_cfg.get("model_lr", 1e-2))
    parser.add_argument("--n-diffusion-steps", type=int,
                        default=agent_cfg.get("n_diffusion_steps", 10))
    parser.add_argument("--epsilon", type=float,
                        default=agent_cfg.get("epsilon", 1.0))
    parser.add_argument("--epsilon-min", type=float,
                        default=agent_cfg.get("epsilon_min", 0.05))
    parser.add_argument("--epsilon-decay", type=float,
                        default=agent_cfg.get("epsilon_decay", 0.995))
    parser.add_argument("--buffer-capacity", type=int,
                        default=agent_cfg.get("buffer_capacity", 10000))
    parser.add_argument("--batch-size", type=int,
                        default=agent_cfg.get("batch_size", 32))
    parser.add_argument("--n-model-updates", type=int,
                        default=agent_cfg.get("n_model_updates", 5))
    parser.add_argument("--reward-samples", type=int,
                        default=agent_cfg.get("reward_samples", 10))
    parser.add_argument("--eval-every", type=int,
                        default=train_cfg.get("eval_every", 50))
    parser.add_argument("--log-dir", type=str,
                        default=log_cfg.get("log_dir", "logs/exdm"))
    parser.add_argument("--render", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)

    n_states = args.height * args.width
    goal = args.goal if args.goal is not None else n_states - 1

    env = GridWorld(
        height=args.height,
        width=args.width,
        goal_pos=divmod(goal, args.width),
        max_steps=args.max_steps,
    )

    agent = ExDMAgent(
        n_states=n_states,
        n_actions=env.n_actions,
        gamma=0.99,
        alpha=args.alpha,
        model_lr=args.model_lr,
        n_diffusion_steps=args.n_diffusion_steps,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        n_model_updates=args.n_model_updates,
        reward_samples=args.reward_samples,
        seed=args.seed,
    )

    exp = ExDMExperiment(
        env=env,
        agent=agent,
        eval_goal=goal,
        n_episodes=args.episodes,
        eval_every=args.eval_every,
        seed=args.seed,
        log_dir=args.log_dir,
    )

    print(f"ExDM | {args.height}×{args.width} GridWorld | {args.episodes} episodes")
    print(f"  Diffusion steps : {args.n_diffusion_steps}")
    print(f"  Model LR        : {args.model_lr}")
    print(f"  Q-learning α    : {args.alpha}")
    print(f"  ε-greedy        : {args.epsilon} → {args.epsilon_min} (decay {args.epsilon_decay})")
    print(f"  Goal            : state {goal}")
    print(f"  Log dir         : {args.log_dir}")
    print()

    metrics = exp.run()

    train_metrics = [m for m in metrics if m.training]
    eval_metrics  = [m for m in metrics if not m.training]

    if train_metrics:
        rewards = [m.total_reward for m in train_metrics]
        print(f"Training  | mean reward = {sum(rewards)/len(rewards):.4f}"
              f" | last-10% = {sum(rewards[-max(1, len(rewards)//10):])/max(1, len(rewards)//10):.4f}")
    if eval_metrics:
        eval_rewards = [m.total_reward for m in eval_metrics]
        print(f"Eval      | mean reward = {sum(eval_rewards)/len(eval_rewards):.4f}")

    if args.render:
        env.reset()
        print("\nEnvironment layout:")
        print(env.render())


if __name__ == "__main__":
    main()
