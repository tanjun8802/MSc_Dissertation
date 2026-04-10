"""
shortest_path.py
================
BFS-based optimal policy solver for tabular GridWorld environments.

This module provides the **theoretical upper bound** (optimal baseline) for
any GridWorld comparison experiment.  It uses breadth-first search (BFS) to
find the shortest path from the start position to the goal, then derives the
optimal deterministic policy for every reachable state.

For stochastic environments (e.g. :class:`~environments.windy_gridworld.WindyGridWorld`),
BFS is run on the *deterministic approximation* (nominal dynamics with stochasticity
temporarily disabled).  The resulting policy is therefore optimal given the mean
dynamics; actual episode rewards in the stochastic env may be lower.

Usage
-----
    from mdp.shortest_path import ShortestPathSolver

    solver = ShortestPathSolver(env)          # env is a GridWorld instance
    result = solver.solve()                   # BFS; returns SolverResult
    print(result.distance)                    # steps on shortest path
    print(result.optimal_return)             # γ^(d-1) * R_goal
    rewards = solver.simulate(n_episodes=50)  # run episodes; always 1.0
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from environments.gridworld import GridWorld


@dataclass
class SolverResult:
    """Result from :class:`ShortestPathSolver`.

    Attributes
    ----------
    distance :
        Length of the shortest path from *start_pos* to *goal_pos* in
        number of environment steps.  ``-1`` if the goal is unreachable.
    optimal_policy :
        Dict mapping flat state index → optimal action (0-3).
        States with no outgoing path to the goal have no entry.
    optimal_return :
        Discounted return achieved by the optimal policy:
        ``γ^(distance-1) * 1.0``.  Returns ``0.0`` if the goal is
        unreachable (distance == -1).
    """

    distance: int
    optimal_policy: dict[int, int]
    optimal_return: float


class ShortestPathSolver:
    """BFS-based optimal solver for deterministic GridWorld environments.

    The solver builds the full transition graph by calling the environment's
    internal physics (``_DELTAS`` + wall/wind logic) in a *deterministic*
    mode.  For :class:`~environments.windy_gridworld.WindyGridWorld` this
    means using the nominal (mean) wind strength, not the stochastic
    perturbation.

    BFS is run **forward** from *start_pos*, discovering the shortest
    distance from the start to every reachable state.  The optimal action
    from each state is then the action that leads to a neighbour with the
    smallest distance-to-goal.

    Parameters
    ----------
    env :
        A :class:`~environments.gridworld.GridWorld` instance.
    gamma :
        Discount factor used to compute ``optimal_return``.  Defaults to 0.99.
    """

    def __init__(self, env: "GridWorld", gamma: float = 0.99) -> None:
        self.env = env
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> SolverResult:
        """Run BFS and return the optimal policy and shortest-path distance.

        Returns
        -------
        SolverResult
            Contains the shortest-path distance, the optimal policy dict,
            and the discounted optimal return at *start_pos*.
        """
        env = self.env
        if env.goal_pos is None:
            return SolverResult(distance=-1, optimal_policy={}, optimal_return=0.0)

        goal_state = env.state_index(env.goal_pos)
        start_state = env.state_index(env.start_pos)

        # Build the full transition table using deterministic dynamics.
        # Stochasticity (e.g. windy gridworld) is disabled during BFS.
        transitions = self._build_transition_table()

        # Forward BFS from start_state
        dist: dict[int, int] = {start_state: 0}
        # Also track which action was used to reach each state (for policy extraction)
        came_via: dict[int, tuple[int, int]] = {}  # state → (prev_state, action)
        queue: deque[int] = deque([start_state])

        while queue:
            s = queue.popleft()
            if s == goal_state:
                break
            for action, s_next in transitions.get(s, {}).items():
                if s_next not in dist:
                    dist[s_next] = dist[s] + 1
                    came_via[s_next] = (s, action)
                    queue.append(s_next)

        # Now do backward BFS from goal to compute dist-to-goal for all states.
        # This lets us extract the optimal policy for every reachable state,
        # not just those on the single path found above.
        dist_to_goal: dict[int, int] = {goal_state: 0}
        bq: deque[int] = deque([goal_state])
        # Build reverse adjacency: s_next → list of (s, action)
        reverse: dict[int, list[tuple[int, int]]] = {}
        for s, action_map in transitions.items():
            for action, s_next in action_map.items():
                reverse.setdefault(s_next, []).append((s, action))

        while bq:
            s_next = bq.popleft()
            for s, action in reverse.get(s_next, []):
                if s not in dist_to_goal:
                    dist_to_goal[s] = dist_to_goal[s_next] + 1
                    bq.append(s)

        # Build optimal policy: for each state, choose action minimising dist-to-goal
        optimal_policy: dict[int, int] = {}
        for s, action_map in transitions.items():
            if s == goal_state:
                continue
            best_action = None
            best_dist = dist_to_goal.get(s, float("inf"))
            for action, s_next in action_map.items():
                d = dist_to_goal.get(s_next, float("inf"))
                if d < best_dist:
                    best_dist = d
                    best_action = action
            if best_action is not None:
                optimal_policy[s] = best_action

        # Compute discounted optimal return from start
        if start_state in dist_to_goal:
            d_start = dist_to_goal[start_state]
            optimal_return = (self.gamma ** (d_start - 1)) if d_start > 0 else 1.0
        else:
            d_start = -1
            optimal_return = 0.0

        return SolverResult(
            distance=d_start,
            optimal_policy=optimal_policy,
            optimal_return=optimal_return,
        )

    def simulate(
        self,
        n_episodes: int = 100,
        seed: int | None = None,
    ) -> list[float]:
        """Run the optimal policy and return per-episode total rewards.

        For deterministic environments the agent will reach the goal every
        episode (``total_reward = 1.0``).  For stochastic environments
        (e.g. windy gridworld) some episodes may fail if the stochastic
        perturbation pushes the agent off the BFS-optimal path.

        Parameters
        ----------
        n_episodes :
            Number of episodes to simulate.
        seed :
            Random seed for environment resets.

        Returns
        -------
        list of float
            Per-episode undiscounted total reward.
        """
        result = self.solve()
        if result.distance == -1:
            return [0.0] * n_episodes

        rng = np.random.default_rng(seed)
        env = self.env
        rewards: list[float] = []

        for _ in range(n_episodes):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            total_reward = 0.0

            while True:
                state = int(np.asarray(obs).flat[0])
                action = result.optimal_policy.get(state)
                if action is None:
                    # At goal or no path — terminate
                    break
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(total_reward)

        return rewards

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_transition_table(self) -> dict[int, dict[int, int]]:
        """Build the full (state, action) → next_state table.

        Uses the environment's deterministic dynamics.  For stochastic
        environments (``env.stochastic is True``), stochasticity is
        temporarily disabled so that BFS finds the optimal policy under
        the *nominal* (mean) dynamics.

        Returns
        -------
        dict mapping state → {action: next_state}
        """
        env = self.env

        # Temporarily disable stochasticity if the env supports it
        was_stochastic = getattr(env, "stochastic", False)
        if was_stochastic:
            env.stochastic = False

        try:
            transitions: dict[int, dict[int, int]] = {}
            for state in range(env.n_states):
                row, col = divmod(state, env.width)
                if (row, col) in env.walls:
                    continue
                transitions[state] = {}
                for action, (dr, dc) in enumerate(env._DELTAS):
                    new_row, new_col = self._apply_dynamics(row, col, action)
                    next_state = new_row * env.width + new_col
                    transitions[state][action] = next_state
        finally:
            # Always restore stochasticity
            if was_stochastic:
                env.stochastic = True

        return transitions

    def _apply_dynamics(self, row: int, col: int, action: int) -> tuple[int, int]:
        """Compute the deterministic next (row, col) after taking *action*.

        Accounts for wind (WindyGridWorld) using the nominal strength, and
        wall collisions (FourRoomsGridWorld / standard GridWorld).
        """
        env = self.env
        dr, dc = env._DELTAS[action]

        # Apply nominal wind (upward displacement) if env has wind
        wind = 0
        if hasattr(env, "wind_strength"):
            wind = env.wind_strength.get(col, 0)  # nominal, not stochastic

        new_row = row + dr - wind
        new_col = col + dc

        # WindyGridWorld clamps to grid bounds (no walls)
        if hasattr(env, "wind_strength"):
            new_row = max(0, min(env.height - 1, new_row))
            new_col = max(0, min(env.width - 1, new_col))
        else:
            # Standard GridWorld: stay in place on boundary or wall
            if not env._is_valid(new_row, new_col):
                new_row, new_col = row, col

        return new_row, new_col
