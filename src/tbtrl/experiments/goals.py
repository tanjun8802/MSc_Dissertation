"""Deterministic maze goal sequences for experiments at different task counts."""

from collections import deque
from copy import deepcopy

from tbtrl.environments.registry import load_layout


def extend_maze_goals(config, count):
    """Keep the original prefix and add farthest-first reachable cell centres.

    Distance is shortest cardinal path length through free cells, not Euclidean
    distance through walls. Ties use row-major cell order. No RNG, training
    settings or input objects are changed. Supports the registered maze adapters;
    a disconnected layout is rejected because reset can sample any free cell.
    """
    supported = {"maze-continuous", "maze-discrete", "fourrooms-discrete"}
    if config.environment not in supported:
        raise ValueError("Goal extension requires a registered maze environment.")
    if isinstance(count, bool) or not isinstance(count, int) or count < len(config.goals):
        raise ValueError("Goal count must be an integer at least as large as the original prefix.")
    layout = load_layout(
        config.environment_options.get(
            "layout", "fourrooms" if config.environment == "fourrooms-discrete" else "maze"
        )
    )
    cells = [(x, y) for y, row in enumerate(layout) for x, value in enumerate(row) if value == 0]
    if count > len(cells):
        raise ValueError(
            f"Requested {count} goals, but the layout has only {len(cells)} free cells."
        )
    offset = 0.5 if config.environment == "maze-continuous" else 0
    coordinates = {(x + offset, y + offset): (x, y) for x, y in cells}
    goals = deepcopy(config.goals)
    if not goals or any(tuple(goal) not in coordinates for goal in goals):
        raise ValueError("Original goals must be free-cell centres in the environment coordinates.")
    selected = {coordinates[tuple(goal)] for goal in goals}
    if len(selected) != len(goals):
        raise ValueError("Original goals must be distinct.")
    free = set(cells)

    def distances(start):
        result = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbour in free and neighbour not in result:
                    result[neighbour] = result[x, y] + 1
                    queue.append(neighbour)
        return result

    nearest = distances(coordinates[tuple(goals[0])])
    if len(nearest) != len(cells):
        raise ValueError("All free cells must be connected so goals are reachable from any reset.")
    for goal in goals[1:]:
        distance = distances(coordinates[tuple(goal)])
        nearest = {cell: min(nearest[cell], distance[cell]) for cell in cells}
    while len(goals) < count:
        cell = max((cell for cell in cells if cell not in selected), key=nearest.__getitem__)
        selected.add(cell)
        goals.append([cell[0] + offset, cell[1] + offset])
        distance = distances(cell)
        nearest = {cell: min(nearest[cell], distance[cell]) for cell in cells}
    return goals
