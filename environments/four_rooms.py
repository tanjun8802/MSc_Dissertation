"""
four_rooms.py
=============
Four-Rooms GridWorld — a classic benchmark for hierarchical and
goal-conditioned RL (Sutton, Precup & Singh, 1999).

Layout (11×11 default)
----------------------
The grid is partitioned into four rooms by two perpendicular internal walls,
each with a single narrow doorway.  The agent starts in the top-left room
and must navigate to a goal that is, by default, in the bottom-right room.
This requires it to discover and pass through two doorways, making it
significantly harder than an open GridWorld.

::

    ###########
    #....#....#
    #....D....#     D = doorway (free cell in the internal wall)
    #....#....#
    #....#....#
    ##D########
    #....#....#
    #....#....#
    #....D....#
    #....#....#
    ###########

(actual doorway positions depend on height/width)
"""

from __future__ import annotations

from typing import Tuple

from environments.gridworld import GridWorld


def _build_four_rooms_walls(height: int, width: int) -> set[Tuple[int, int]]:
    """Return the wall set for a four-rooms layout.

    The horizontal wall spans the entire middle row, and the vertical wall
    spans the entire middle column.  One doorway is cut into each of the
    four resulting wall segments so that all rooms are reachable.

    Parameters
    ----------
    height, width :
        Dimensions of the grid (should be >= 7 for meaningful rooms).

    Returns
    -------
    set of (row, col) wall positions.
    """
    walls: set[Tuple[int, int]] = set()

    mid_row = height // 2
    mid_col = width // 2

    # ------------------------------------------------------------------
    # Horizontal wall (full middle row)
    # Doorways: one in the left half, one in the right half
    # ------------------------------------------------------------------
    door_left_col = mid_col // 2           # somewhere in the left half
    door_right_col = mid_col + (width - mid_col) // 2  # somewhere in the right half

    for c in range(width):
        if c != door_left_col and c != door_right_col:
            walls.add((mid_row, c))

    # ------------------------------------------------------------------
    # Vertical wall (full middle column, excluding the mid_row intersection
    # which is already occupied by the horizontal wall)
    # Doorways: one in the top half, one in the bottom half
    # ------------------------------------------------------------------
    door_top_row = mid_row // 2            # somewhere in the top half
    door_bot_row = mid_row + (height - mid_row) // 2  # somewhere in the bottom half

    for r in range(height):
        if r == mid_row:
            continue  # skip: already set (or gap) by horizontal wall
        if r != door_top_row and r != door_bot_row:
            walls.add((r, mid_col))

    return walls


class FourRoomsGridWorld(GridWorld):
    """GridWorld divided into four rooms by internal walls with narrow doorways.

    This is a classic benchmark used to evaluate skill discovery and
    hierarchical RL.  Unlike the open GridWorld, the agent must learn to
    navigate through bottleneck doorways, which typically requires more
    exploration and many more steps to first reach the goal.

    Parameters
    ----------
    height, width :
        Grid dimensions (minimum 7×7 recommended for sensible rooms).
    start_pos :
        (row, col) starting position. Defaults to top-left corner.
    goal_pos :
        (row, col) goal position. Defaults to bottom-right corner (opposite
        room from the default start). Any position inside a wall is shifted
        to the nearest free cell.
    max_steps :
        Episode time limit.
    """

    def __init__(
        self,
        height: int = 11,
        width: int = 11,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] | None = None,
        max_steps: int | None = 500,
    ) -> None:
        walls = _build_four_rooms_walls(height, width)

        # Default goal: bottom-right corner; shift up if it falls on a wall
        if goal_pos is None:
            goal_pos = (height - 1, width - 1)
            # Shift the goal to a free cell if it falls on a wall
            r, c = goal_pos
            while (r, c) in walls and r >= 0:
                r -= 1
            if (r, c) in walls:
                # Fall back to top-right corner if the whole column is walled
                r, c = 0, width - 1
                while (r, c) in walls and c >= 0:
                    c -= 1
            goal_pos = (r, c)

        super().__init__(
            height=height,
            width=width,
            start_pos=start_pos,
            goal_pos=goal_pos,
            walls=walls,
            max_steps=max_steps,
        )

    def __repr__(self) -> str:
        return (
            f"FourRoomsGridWorld(height={self.height}, width={self.width}, "
            f"start={self.start_pos}, goal={self.goal_pos}, "
            f"n_walls={len(self.walls)})"
        )
