"""
metrics.py
==========
Data classes for storing and aggregating per-episode statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeMetrics:
    """Metrics collected over a single episode.

    Parameters
    ----------
    episode :
        Global episode index (1-based).
    total_reward :
        Sum of rewards received during the episode.
    length :
        Number of environment steps taken.
    training :
        ``True`` if this was a training episode, ``False`` if evaluation.
    step_metrics :
        List of per-step metric dicts returned by the agent's update method.
    """

    episode: int
    total_reward: float
    length: int
    training: bool = True
    step_metrics: list[dict[str, Any]] = field(default_factory=list)

    def mean_step_metric(self, key: str) -> float | None:
        """Return the mean of a per-step metric across the episode.

        Returns ``None`` if the key is absent from all step records.
        """
        values = [m[key] for m in self.step_metrics if key in m]
        if not values:
            return None
        return sum(values) / len(values)

    def __str__(self) -> str:
        mode = "train" if self.training else "eval"
        return (
            f"Episode {self.episode:>6d} [{mode}] | "
            f"reward={self.total_reward:>8.2f} | "
            f"steps={self.length:>5d}"
        )
