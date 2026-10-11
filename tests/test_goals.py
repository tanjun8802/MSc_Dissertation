"""Task-count experiments preserve settings and use achievable maze goals."""

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from tbtrl.environments.registry import make_environment
from tbtrl.experiments.config import load_config
from tbtrl.experiments.goals import extend_maze_goals
from tbtrl.experiments.runner import run_experiment
from tbtrl.experiments.transfer import configure_transfer

ROOT = Path(__file__).resolve().parents[1]


def base():
    return load_config(ROOT / "configs/tbtrl_gridworldmaze_sac_factorised_actor.json")


def test_goal_extension_is_stable_nested_and_does_not_change_config():
    config = base()
    before = config.to_dict()
    goals = extend_maze_goals(config, 20)
    assert config.to_dict() == before
    assert goals[:4] == config.goals == extend_maze_goals(config, 4)
    assert len({tuple(g) for g in goals}) == 20
    assert goals == extend_maze_goals(config, 20)
    assert goals == extend_maze_goals(config, 24)[:20]
    global_config = configure_transfer(config, "global_replay")
    control = configure_transfer(config, "temperature_carry")
    assert extend_maze_goals(global_config, 20) == extend_maze_goals(control, 20)
    goals[0][0] = -99
    assert config.to_dict() == before
    for count in (3, 1000, 20.5, True):
        with pytest.raises(ValueError):
            extend_maze_goals(config, count)


def test_all_twenty_goals_are_reached_by_valid_actions_within_original_horizon():
    config = base()
    # Find paths backwards from each goal; replay the longest path using actual
    # environment actions, testing the worst free-cell start for that goal.
    for goal in extend_maze_goals(config, 20):
        env = make_environment(config.environment, goal, **config.environment_options)
        try:
            grid = env.unwrapped.maze
            target = tuple(np.floor(goal).astype(int))
            paths = {target: [target]}
            queue = deque([target])
            while queue:
                x, y = queue.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    cell = (x + dx, y + dy)
                    if (
                        0 <= cell[1] < len(grid)
                        and 0 <= cell[0] < len(grid[0])
                        and grid[cell[1], cell[0]] == 0
                        and cell not in paths
                    ):
                        paths[cell] = [cell, *paths[x, y]]
                        queue.append(cell)
            assert len(paths) == int((grid == 0).sum())
            path = max(paths.values(), key=len)
            env.reset(seed=42)
            obs = np.asarray(path[0], dtype=np.float32) + 0.5
            env.unwrapped.agent_pos = obs.copy()
            terminated = truncated = False
            steps = 0
            for cell in path[1:]:
                waypoint = np.asarray(cell) + 0.5
                while np.linalg.norm(waypoint - obs) > 1e-5 and not (terminated or truncated):
                    action = np.clip((waypoint - obs) / env.unwrapped.step_scale, -1, 1)
                    obs, _, terminated, truncated, info = env.step(action)
                    steps += 1
                if terminated or truncated:
                    break
            assert terminated and info["success"], (goal, steps)
            assert steps <= config.environment_options["max_horizon"]
        finally:
            env.close()


@pytest.mark.parametrize("name", ["tbtrl_gridworldmaze_dqn", "tbtrl_fourrooms_dqn"])
def test_discrete_goal_extension_uses_integer_environment_coordinates(name):
    config = load_config(ROOT / f"configs/{name}.json")
    goals = extend_maze_goals(config, 20)
    assert goals[: len(config.goals)] == config.goals
    assert np.array_equal(goals, np.asarray(goals, dtype=int))
    for goal in goals:
        env = make_environment(config.environment, goal, **config.environment_options)
        env.close()


def test_twenty_goal_sequence_executes_under_a_fixed_replay_budget(tmp_path):
    full = configure_transfer(base(), "global_replay")
    goals = extend_maze_goals(full, 20)
    config = full.smoke()
    # Keep all 20 distinct goals while reducing only execution budgets/models.
    config.goals = goals
    config.training["transfer_probe_steps"] = None
    config.training["actor_gradient_diagnostics"] = False
    records = run_experiment(config, tmp_path / "run", modes=("sequential",), verbose=False)
    assert len(records) == 20
    assert all(len(row["evaluation"]) == 20 for row in records)
    for task, row in enumerate(records):
        stats = row["replay_memory"]
        assert stats["tasks"] == task + 1
        assert stats["capacity"] == config.global_replay_capacity
        assert stats["allocated_transition_bytes"] == config.global_replay_capacity * 36
        assert stats["archive_samples"] <= stats["archive_capacity"]
        assert stats["groups"] <= config.replay_max_groups
