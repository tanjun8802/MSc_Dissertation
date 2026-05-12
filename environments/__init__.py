"""
environments/
=============
Custom environments and environment wrappers for RL experiments.

Modules
-------
base_env        : Abstract base class for all environments.
wrappers        : Common observation / reward wrappers.
gridworld       : A simple tabular GridWorld useful for reward-free exploration.
four_rooms      : Four-Rooms GridWorld — classic hierarchical RL benchmark.
windy_gridworld : Windy GridWorld — stochastic column-wind navigation.

Factory
-------
Use :func:`make_env` to instantiate any environment by name string:

    >>> env = make_env("four_rooms", height=11, width=11)
    >>> env = make_env("windy", stochastic=True)
    >>> env = make_env("gridworld", height=5, width=5, goal_pos=(4, 4))
"""

from environments.base_env import BaseEnv
from environments.gridworld import GridWorld
from environments.four_rooms import FourRoomsGridWorld
from environments.windy_gridworld import WindyGridWorld
from environments.wrappers import TimeLimit, NormaliseObservation


# ---------------------------------------------------------------------------
# Environment registry: maps short name → class
# ---------------------------------------------------------------------------

_ENV_REGISTRY: dict[str, type] = {
    "gridworld": GridWorld,
    "grid": GridWorld,
    "four_rooms": FourRoomsGridWorld,
    "fourrooms": FourRoomsGridWorld,
    "windy": WindyGridWorld,
    "windy_gridworld": WindyGridWorld,
}


def make_env(name: str, **kwargs) -> BaseEnv:
    """Instantiate an environment by name.

    Parameters
    ----------
    name :
        One of: ``"gridworld"``, ``"grid"``, ``"four_rooms"``,
        ``"fourrooms"``, ``"windy"``, ``"windy_gridworld"``.
        Case-insensitive.
    **kwargs :
        Forwarded to the environment constructor.

    Returns
    -------
    BaseEnv
        An instance of the requested environment.

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    """
    key = name.lower().strip()
    if key not in _ENV_REGISTRY:
        registered = ", ".join(sorted(_ENV_REGISTRY))
        raise ValueError(
            f"Unknown environment '{name}'. "
            f"Registered names: {registered}."
        )
    return _ENV_REGISTRY[key](**kwargs)


__all__ = [
    "BaseEnv",
    "GridWorld",
    "FourRoomsGridWorld",
    "WindyGridWorld",
    "TimeLimit",
    "NormaliseObservation",
    "make_env",
]
