"""Bounded task-direction selection, independent of exploration and replay RNGs."""

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class KReplayOptions:
    max_tasks: int = 4
    sample_budget: int = 256
    selector: str = "basis"
    candidate_capacity: int = 32
    refresh_updates: int = 100
    seed: int = 0
    rank_tolerance: float = 1e-6
    ridge: float = 1e-6
    weighting: str = "uniform"
    weight_floor: float = 0.2
    weight_temperature: float = 1.0
    weight_smoothing: float = 0.8

    def __post_init__(self):
        for name in ("max_tasks", "sample_budget", "candidate_capacity", "refresh_updates", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"k_replay.{name} must be a nonnegative integer.")
        if self.refresh_updates < 1 or self.candidate_capacity < max(1, self.max_tasks):
            raise ValueError("k_replay needs refresh_updates >= 1 and candidate_capacity >= K, 1.")
        if (self.max_tasks == 0) != (self.sample_budget == 0):
            raise ValueError("K and sample_budget must both be zero or both positive.")
        if self.selector not in ("basis", "random", "rotating"):
            raise ValueError("Unknown k_replay selector.")
        if self.weighting not in ("uniform", "interference"):
            raise ValueError("Unknown k_replay weighting.")
        for name in ("rank_tolerance", "ridge", "weight_temperature"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"k_replay.{name} must be finite and positive.")
        if not 0 < self.weight_floor <= 1 or not 0 <= self.weight_smoothing < 1:
            raise ValueError("Require 0 < weight_floor <= 1 and 0 <= weight_smoothing < 1.")

    @classmethod
    def from_dict(cls, values):
        if isinstance(values, cls):
            return values
        if not isinstance(values, dict):
            raise ValueError("k_replay options must be a dictionary.")
        return cls(**values)


class TaskSelector:
    """Newest task plus a uniform reservoir, shared by every selection strategy.

    Admission RNG and selection RNG are separate: changing the selector does not
    change the candidate catalogue. All per-refresh geometry is bounded by M.
    """

    def __init__(self, options, state=None):
        self.options = KReplayOptions.from_dict(options)
        self.admission_rng = np.random.default_rng(self.options.seed)
        self.selection_rng = np.random.default_rng(self.options.seed + 1)
        self.reservoir = []
        self.newest = None
        self.completed = 0
        self.refreshes = 0
        self.updates = 0
        self.selected = []
        self.weights = []
        self.scores = {}
        if state:
            if state.get("version") != 1 or state["options"] != asdict(self.options):
                raise ValueError("Incompatible bounded replay selector state.")
            for name in (
                "reservoir",
                "newest",
                "completed",
                "refreshes",
                "updates",
                "selected",
                "weights",
                "scores",
            ):
                setattr(self, name, deepcopy(state[name]))
            self.admission_rng.bit_generator.state = deepcopy(state["admission_rng"])
            self.selection_rng.bit_generator.state = deepcopy(state["selection_rng"])

    @property
    def candidate_ids(self):
        return sorted(self.reservoir + ([] if self.newest is None else [self.newest]))

    def observe_completed_task(self, task_id):
        if self.newest == task_id:
            return
        if self.newest is not None:
            slots = self.options.candidate_capacity - 1
            if len(self.reservoir) < slots:
                self.reservoir.append(self.newest)
            else:
                index = int(self.admission_rng.integers(self.completed))
                if index < slots:
                    self.reservoir[index] = self.newest
        self.newest = int(task_id)
        self.completed += 1

    def select(self, vectors):
        ids = sorted(vectors)
        count = min(self.options.max_tasks, self.options.sample_budget, len(ids))
        if not ids or not count:
            self.selected, self.weights, self.scores = [], [], {}
            return [], dict(rank=0, max_residual=0.0, max_coefficient=0.0, condition=1.0)
        values = np.asarray([vectors[i] for i in ids], dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Task vectors must be a finite matrix.")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        unit = values / np.maximum(norms, self.options.rank_tolerance)
        start = self.refreshes % len(ids)
        order = list(range(start, len(ids))) + list(range(start))
        if self.options.selector == "random":
            indices = self.selection_rng.choice(len(ids), count, replace=False).tolist()
        elif self.options.selector == "rotating":
            indices = order[:count]
        else:
            residual = unit.copy()
            indices = []
            for _ in range(count):
                lengths = np.linalg.norm(residual, axis=1)
                lengths[indices] = -1
                pivot = int(np.argmax(lengths))
                if lengths[pivot] <= self.options.rank_tolerance:
                    break
                indices.append(pivot)
                direction = residual[pivot] / lengths[pivot]
                # Reorthogonalisation keeps nearly parallel task vectors finite.
                for _ in range(2):
                    residual -= (residual @ direction)[:, None] * direction
            indices += [i for i in order if i not in indices][: count - len(indices)]
        basis = unit[indices]
        coefficients = np.linalg.lstsq(basis.T, unit.T, rcond=self.options.rank_tolerance)[0]
        residual = unit - coefficients.T @ basis
        gram = basis @ basis.T + self.options.ridge * np.eye(count)
        metrics = dict(
            rank=int(np.linalg.matrix_rank(basis, tol=self.options.rank_tolerance)),
            max_residual=float(np.linalg.norm(residual, axis=1).max()),
            max_coefficient=float(np.abs(coefficients).max()),
            condition=float(np.linalg.cond(gram)),
            min_raw_norm=float(norms.min()),
            max_raw_norm=float(norms.max()),
        )
        self.selected = [ids[i] for i in indices]
        self.scores = {i: self.scores.get(i, 0.0) for i in self.selected}
        self.weights = [1.0 / count] * count
        self.refreshes += 1
        return self.selected.copy(), metrics

    def sample_counts(self, ids):
        if not ids:
            return []
        quotient, remainder = divmod(self.options.sample_budget, len(ids))
        return [quotient + (index < remainder) for index in range(len(ids))]

    def update_weights(self, current_gradient, old_gradients):
        """Detached negative-cosine scores, with a positive uniform floor."""
        if self.options.weighting == "uniform" or not old_gradients:
            return self.weights.copy()
        current = current_gradient.detach().to(device="cpu", dtype=torch.float64)
        current_norm = torch.linalg.vector_norm(current)
        if not torch.isfinite(current).all() or current_norm <= 1e-12:
            self.weights = [1.0 / len(self.selected)] * len(self.selected)
            return self.weights.copy()
        for task, gradient in zip(self.selected, old_gradients):
            gradient = gradient.detach().to(device="cpu", dtype=torch.float64)
            norm = torch.linalg.vector_norm(gradient)
            if not torch.isfinite(gradient).all() or norm <= 1e-12:
                score = 0.0
            else:
                score = max(0.0, float(-(current @ gradient) / (current_norm * norm)))
            previous = self.scores.get(task, score)
            self.scores[task] = (
                self.options.weight_smoothing * previous
                + (1 - self.options.weight_smoothing) * score
            )
        scores = np.asarray([self.scores[i] for i in self.selected])
        scores = scores / self.options.weight_temperature
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        rho = self.options.weight_floor
        self.weights = (rho / len(scores) + (1 - rho) * probabilities).tolist()
        return self.weights.copy()

    def state_dict(self):
        return deepcopy(
            dict(
                version=1,
                options=asdict(self.options),
                reservoir=self.reservoir,
                newest=self.newest,
                completed=self.completed,
                refreshes=self.refreshes,
                updates=self.updates,
                selected=self.selected,
                weights=self.weights,
                scores=self.scores,
                admission_rng=self.admission_rng.bit_generator.state,
                selection_rng=self.selection_rng.bit_generator.state,
            )
        )
