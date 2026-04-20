"""
windy_gridworld.py
==================
Windy GridWorld — a stochastic navigation environment from Sutton & Barto.

The agent moves on a rectangular grid, but wind in certain columns
applies an upward displacement *after* each action.  In stochastic mode
the actual wind strength is ±1 from the nominal value, so the agent can
never be completely certain of the outcome of its actions.

Reference
---------
Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.,
Example 6.5 — Windy Gridworld (p. 130).  We extend the classic with a
stochastic option and a terminal reward.
"""

from __future__ import annotations

from typing import SupportsFloat, Tuple

import numpy as np

from environments.gridworld import GridWorld


# Default wind-strength pattern (column → upward displacement in cells).
# Matches the Sutton & Barto Example 6.5 layout for a 7×10 grid.
_DEFAULT_WIND: dict[int, int] = {
    3: 1, 4: 1, 5: 1,
    6: 2, 7: 2,
    8: 1,
}


class WindyGridWorld(GridWorld):
    """GridWorld with column-specific upward wind and optional stochasticity.

    After the agent moves according to its chosen action, the wind in the
    agent's *pre-move* column displaces it upward (i.e., decreasing the
    row index) by the column's wind strength.  In stochastic mode the
    displacement is drawn uniformly from ``{strength-1, strength, strength+1}``
    (clamped at 0), so a "strength-0" column never produces wind.

    Parameters
    ----------
    height, width :
        Grid dimensions.  The classic Sutton & Barto layout is 7×10.
    wind_strength :
        Dict mapping column index to upward displacement (cells).
        Columns absent from the dict have no wind.
        If ``None`` the Sutton & Barto default pattern is used.
    stochastic :
        If ``True``, the wind in each step is drawn uniformly from
        ``{nominal-1, nominal, nominal+1}`` (all ≥ 0).
    start_pos :
        (row, col) starting position.
    goal_pos :
        (row, col) goal position.  Defaults to the right-hand end of the
        same row as the start (matching the Sutton & Barto layout).
    max_steps :
        Episode time limit.
    """

    def __init__(
        self,
        height: int = 7,
        width: int = 10,
        wind_strength: dict[int, int] | None = None,
        stochastic: bool = True,
        start_pos: Tuple[int, int] = (3, 0),
        goal_pos: Tuple[int, int] | None = None,
        max_steps: int | None = 500,
    ) -> None:
        if goal_pos is None:
            goal_pos = (3, width - 1)

        self.wind_strength: dict[int, int] = (
            dict(wind_strength) if wind_strength is not None else dict(_DEFAULT_WIND)
        )
        self.stochastic = stochastic

        # Windy GridWorld has no internal walls (walls handled by GridWorld bounds)
        super().__init__(
            height=height,
            width=width,
            start_pos=start_pos,
            goal_pos=goal_pos,
            walls=set(),
            max_steps=max_steps,
        )

    # ------------------------------------------------------------------
    # Override step to apply wind
    # ------------------------------------------------------------------

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """Apply action then apply column wind displacement.

        Wind acts on the agent's column *before* the move, pushing it
        upward (decreasing row index) by the column's nominal strength
        (or a stochastic variant thereof).
        """
        if not (0 <= action < self.n_actions):
            raise ValueError(
                f"Invalid action {action}. Must be in [0, {self.n_actions - 1}]."
            )

        # Determine wind *before* moving (uses current column)
        nominal_wind = self.wind_strength.get(self._agent_pos[1], 0)
        if self.stochastic and nominal_wind > 0:
            # np.random.integers(lo, hi) is exclusive of hi, so integers(-1, 2)
            # draws uniformly from {-1, 0, 1}.  We clip at 0 so that a
            # "strength-0" column never produces negative (downward) wind.
            wind = max(0, nominal_wind + int(self.np_random.integers(-1, 2)))
        else:
            wind = nominal_wind

        # Move in the chosen direction
        dr, dc = self._DELTAS[action]
        new_row = self._agent_pos[0] + dr - wind   # wind = upward offset
        new_col = self._agent_pos[1] + dc

        # Clamp to grid bounds (no walls in Windy GridWorld)
        new_row = max(0, min(self.height - 1, new_row))
        new_col = max(0, min(self.width - 1, new_col))
        self._agent_pos = (new_row, new_col)

        self._step_count += 1

        terminated = (self.goal_pos is not None) and (self._agent_pos == self.goal_pos)
        truncated = (self.max_steps is not None) and (self._step_count >= self.max_steps)
        reward = 1.0 if terminated else 0.0

        info = {
            "position": self._agent_pos,
            "step": self._step_count,
            "wind": wind,
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def render(self) -> str:
        """ASCII render with wind strength shown along the bottom."""
        rows = []
        for r in range(self.height):
            row_str = ""
            for c in range(self.width):
                pos = (r, c)
                if pos == self._agent_pos:
                    row_str += "A"
                elif pos == self.goal_pos:
                    row_str += "G"
                else:
                    row_str += "."
            rows.append(row_str)
        # Wind-strength legend
        legend = "".join(
            str(self.wind_strength.get(c, 0)) for c in range(self.width)
        )
        rows.append(legend)
        return "\n".join(rows)

    def __repr__(self) -> str:
        return (
            f"WindyGridWorld(height={self.height}, width={self.width}, "
            f"stochastic={self.stochastic}, "
            f"start={self.start_pos}, goal={self.goal_pos})"
        )
