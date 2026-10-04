"""Named training outputs; an unsolved task has steps_to_threshold=None."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DQNResult:
    network: Any
    target: Any
    evaluations: list
    steps: int
    steps_to_threshold: int | None
    elapsed_seconds: float
    buffer: Any
    task_embedding: Any
    goal_cosines: Any
    losses: list = field(default_factory=list)


@dataclass
class SACResult:
    actor: Any
    critic: Any
    target: Any
    evaluations: list
    success_rates: list
    final_distances: list
    steps: int
    steps_to_threshold: int | None
    elapsed_seconds: float
    buffer: Any
    state_normalizer: Any
    goal_normalizer: Any
    entropy_coefficient: float
    losses: list = field(default_factory=list)
