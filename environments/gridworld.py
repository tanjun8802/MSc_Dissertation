"""
gridworld.py
============
A tabular GridWorld environment for reward-free / unsupervised RL research.

The GridWorld provides a simple, fully-observable discrete state-action space
that is convenient for benchmarking exploration and skill-discovery algorithms
(e.g. METRA, DIAYN, APS) before scaling to continuous-control domains.

Layout
------
Cells are indexed (row, col) starting from the top-left corner (0, 0).
The agent starts at `start_pos` and can move in four cardinal directions.
Walls block movement; the agent stays in place if it walks into a wall.

Actions
-------
0 : UP     (row - 1)
1 : DOWN   (row + 1)
2 : LEFT   (col - 1)
3 : RIGHT  (col + 1)
"""

from __future__ import annotations

from typing import Any, SupportsFloat, Tuple

import numpy as np

from environments.base_env import BaseEnv


class GridWorld(BaseEnv):
    """Simple tabular GridWorld with optional wall cells.

    Parameters
    ----------
    height, width :
        Grid dimensions.
    start_pos :
        (row, col) starting position of the agent.
    goal_pos :
        (row, col) goal position. If ``None`` no terminal goal is used
        (reward-free mode).
    walls :
        Set of (row, col) tuples that are impassable.
    max_steps :
        Episode time limit (``None`` → unlimited).
    """

    # Cardinal direction deltas: UP, DOWN, LEFT, RIGHT
    _DELTAS: list[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    _ACTION_NAMES: list[str] = ["UP", "DOWN", "LEFT", "RIGHT"]

    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] | None = None,
        walls: set[Tuple[int, int]] | None = None,
        max_steps: int | None = 200,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.walls: set[Tuple[int, int]] = walls or set()
        self.max_steps = max_steps

        self.n_states = height * width
        self.n_actions = 4

        self._agent_pos: Tuple[int, int] = start_pos
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium-style API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._agent_pos = self.start_pos
        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        if not (0 <= action < self.n_actions):
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.n_actions - 1}].")

        dr, dc = self._DELTAS[action]
        new_row = self._agent_pos[0] + dr
        new_col = self._agent_pos[1] + dc

        # Stay in place if out-of-bounds or wall
        if self._is_valid(new_row, new_col):
            self._agent_pos = (new_row, new_col)

        self._step_count += 1

        terminated = (self.goal_pos is not None) and (self._agent_pos == self.goal_pos)
        truncated = (self.max_steps is not None) and (self._step_count >= self.max_steps)

        reward = 1.0 if terminated else 0.0  # 0 reward in reward-free mode (no goal)

        info = {
            "position": self._agent_pos,
            "step": self._step_count,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> str:
        """Return an ASCII string representation of the grid."""
        rows = []
        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                pos = (r, c)
                if pos == self._agent_pos:
                    row_str += "A"
                elif pos == self.goal_pos:
                    row_str += "G"
                elif pos in self.walls:
                    row_str += "#"
                else:
                    row_str += "."
            rows.append(row_str)
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # State / observation helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return the agent position as a flat integer index."""
        idx = self._agent_pos[0] * self.width + self._agent_pos[1]
        return np.array([idx], dtype=np.int64)

    def state_index(self, pos: Tuple[int, int] | None = None) -> int:
        """Convert a (row, col) position to a flat state index."""
        if pos is None:
            pos = self._agent_pos
        return pos[0] * self.width + pos[1]

    def index_to_pos(self, index: int) -> Tuple[int, int]:
        """Convert a flat state index back to (row, col)."""
        return divmod(index, self.width)

    def get_all_states(self) -> list[int]:
        """Return all non-wall state indices."""
        return [
            r * self.width + c
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in self.walls
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_valid(self, row: int, col: int) -> bool:
        """Check whether (row, col) is inside the grid and not a wall."""
        in_bounds = (0 <= row < self.height) and (0 <= col < self.width)
        return in_bounds and (row, col) not in self.walls

    def __repr__(self) -> str:
        return (
            f"GridWorld(height={self.height}, width={self.width}, "
            f"start={self.start_pos}, goal={self.goal_pos})"
        )
