"""Environment construction boundary. Trainers depend on contracts, not layouts."""

import json
from importlib.resources import files

import numpy as np
import torch

from . import maze, maze_discrete

_FACTORIES = {}


def register_environment(name, factory):
    """Register factory(goal=..., **options) -> Gymnasium environment."""
    if name in _FACTORIES:
        raise ValueError(f"Environment {name!r} is already registered.")
    _FACTORIES[name] = factory


def make_environment(name, goal, **options):
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown environment {name!r}; registered: {sorted(_FACTORIES)}") from exc
    return factory(goal=goal, **options)


def load_layout(name):
    return json.loads(files("tbtrl.environments").joinpath(f"{name}_layout.json").read_text())


class DiscreteGoalMaze(maze_discrete.MazeGoalWrapper):
    def relabel_transitions(self, batch, goal):
        """Recompute rewards and goal termination for shared task buffers.

        Time-limit truncation remains on the batch, independently of this goal.
        """
        states = batch.obs.detach().cpu().numpy()
        next_states = batch.next_obs.detach().cpu().numpy()
        goal = torch.as_tensor(goal).detach().cpu().numpy()
        goals = np.broadcast_to(goal, states.shape)
        reward_fn = (
            self.compute_simple_reward
            if self.reward_mode == "simple"
            else self.compute_shaped_reward
        )
        rewards = [
            reward_fn(s, int(a.item()), sp, g)
            for s, a, sp, g in zip(states, batch.actions, next_states, goals)
        ]
        terminated = np.all(next_states == goals, axis=-1)
        return (
            torch.as_tensor(rewards, dtype=batch.obs.dtype, device=batch.obs.device).view(-1, 1),
            torch.as_tensor(terminated, dtype=batch.obs.dtype, device=batch.obs.device).view(-1, 1),
        )

    def goal_distance(self, observation, goal):
        return float(np.linalg.norm(np.asarray(observation) - goal))


class ContinuousGoalMaze(maze.MazeGoalWrapper):
    def goal_distance(self, observation, goal):
        return float(np.linalg.norm(np.asarray(observation) - goal))


def _discrete(
    goal,
    *,
    layout="maze",
    max_horizon=100,
    start=None,
    slip_prob=0.0,
    reward_mode="simple",
    goal_reward=1.0,
    step_reward=0.0,
    wall_penalty=-0.1,
):
    base = maze_discrete.MazeGridWorld(
        load_layout(layout), start=start, max_episode_steps=max_horizon
    )
    return DiscreteGoalMaze(
        base,
        goal_position=goal,
        slip_prob=slip_prob,
        reward_mode=reward_mode,
        goal_reward=goal_reward,
        step_reward=step_reward,
        wall_penalty=wall_penalty,
    )


def _fourrooms(goal, **options):
    return _discrete(goal, layout="fourrooms", **options)


def _continuous(
    goal,
    *,
    layout="maze",
    max_horizon=200,
    start=None,
    slip_prob=0.0,
    reward_mode="simple",
    step_scale=0.35,
    substeps=5,
    start_noise=0.0,
    goal_radius=0.5,
    goal_reward=1.0,
    step_reward=0.0,
    wall_penalty=-0.1,
    movement_bonus_coef=0.01,
    movement_bonus_power=0.5,
    gamma=0.99,
):
    base = maze.MazeGridWorld(
        load_layout(layout),
        start=start,
        max_episode_steps=max_horizon,
        step_scale=step_scale,
        substeps=substeps,
        start_noise=start_noise,
        goal_noise=0.0,
    )
    return ContinuousGoalMaze(
        base,
        goal_position=goal,
        slip_prob=slip_prob,
        reward_mode=reward_mode,
        goal_radius=goal_radius,
        goal_reward=goal_reward,
        step_reward=step_reward,
        wall_penalty=wall_penalty,
        movement_bonus_coef=movement_bonus_coef,
        movement_bonus_power=movement_bonus_power,
        gamma=gamma,
    )


register_environment("maze-discrete", _discrete)
register_environment("fourrooms-discrete", _fourrooms)
register_environment("maze-continuous", _continuous)
