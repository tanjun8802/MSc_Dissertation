"""A fixed transition budget shared by recent experience and similarity groups.

All buffer views alias one allocation. Task-to-group labels grow with the goal
catalogue; transition storage and the number of groups do not grow with tasks.
"""

import numpy as np

from .replay import ReplayBuffer


class GlobalReplayMemory:
    def __init__(
        self,
        capacity,
        recent_capacity,
        obs_dim,
        action_dim,
        device="cpu",
        *,
        discrete=False,
        keep_fraction=0.4,
        similarity_threshold=0.8,
        max_groups=16,
        compression="similarity",
    ):
        if not 1 <= recent_capacity < capacity:
            raise ValueError("Recent capacity must leave space for the shared archive.")
        if not 0 < keep_fraction <= 1 or not -1 <= similarity_threshold < 1:
            raise ValueError("Invalid retention fraction or similarity threshold.")
        if not 2 <= max_groups <= capacity - recent_capacity:
            raise ValueError("Use at least two groups and at least one archive slot per group.")
        self.storage = ReplayBuffer(capacity, obs_dim, action_dim, device, discrete=discrete)
        self.recent_capacity = recent_capacity
        self.archive_capacity = capacity - recent_capacity
        self.recent = ReplayBuffer.view(self.storage, 0, recent_capacity)
        self.keep_fraction, self.similarity_threshold = keep_fraction, similarity_threshold
        self.max_groups = max_groups
        if compression not in ("similarity", "capacity"):
            raise ValueError("compression must be similarity or capacity.")
        self.compression = compression
        self.groups = {}  # representative task ID -> (start, stop) in storage
        self.task_groups = {}  # task ID -> representative task ID; no transition copies
        self.buffers = {}
        self.last_admission = {}

    @property
    def representatives(self):
        return list(self.groups)

    def begin_task(self):
        self.recent.pos = self.recent.size = 0
        return self.recent

    def _refresh_views(self):
        views = {
            key: ReplayBuffer.view(self.storage, start, stop, size=stop - start)
            for key, (start, stop) in self.groups.items()
        }
        self.buffers = {task: views[group] for task, group in self.task_groups.items()}

    @staticmethod
    def _quotas(sizes, budget):
        """Equal group shares, redistributing space unused by small groups."""
        quotas = np.zeros(len(sizes), dtype=np.int64)
        sizes = np.asarray(sizes)
        remaining = min(budget, int(sizes.sum()))
        while remaining:
            active = np.flatnonzero(quotas < sizes)
            share = max(1, remaining // len(active))
            for i in active:
                count = min(share, int(sizes[i] - quotas[i]), remaining)
                quotas[i] += count
                remaining -= count
                if not remaining:
                    break
        return quotas

    def finish_task(self, task_id, similarities):
        """Admit the recent tail and rebalance within the fixed archive budget.

        similarities is the cosine matrix for representatives followed by this
        task, recomputed with the current critic. When the group limit is reached,
        a novel task gets a group by merging the most similar existing pair.
        """
        if task_id in self.task_groups:
            raise ValueError("Each sequential task must have a unique ID.")
        if not len(self.recent):
            raise ValueError("Cannot archive an empty task.")
        reps = self.representatives
        similarities = np.asarray(similarities)
        if (
            similarities.shape != (len(reps) + 1, len(reps) + 1)
            or not np.isfinite(similarities).all()
        ):
            raise ValueError("Supply finite similarities for representatives plus current task.")
        similarities = np.clip(similarities, -1, 1)
        candidates = {group: np.arange(start, stop) for group, (start, stop) in self.groups.items()}
        count = max(1, int(len(self.recent) * self.keep_fraction))
        incoming = self.recent.chronological_indices()[-count:]
        group, merged_pair, similarity = task_id, None, None
        if reps:
            best = int(np.argmax(similarities[-1, :-1]))
            similarity = float(similarities[-1, best])
            if similarity > self.similarity_threshold:
                group = reps[best]
                fraction = (1 - similarity) / (1 - self.similarity_threshold)
                count = 0 if abs(similarity - 1) < 1e-6 else max(1, int(count * fraction))
                if self.compression == "similarity":
                    incoming = incoming[-count:] if count else incoming[:0]
            elif len(reps) == self.max_groups:
                # Consolidate redundant groups rather than dropping a whole old task.
                left, right = np.triu_indices(len(reps), k=1)
                pair = int(np.argmax(similarities[left, right]))
                keep, remove = reps[left[pair]], reps[right[pair]]
                candidates[keep] = np.concatenate((candidates[keep], candidates.pop(remove)))
                self.task_groups = {
                    task: keep if old == remove else old for task, old in self.task_groups.items()
                }
                merged_pair = [keep, remove]
        candidates[group] = np.concatenate((candidates.get(group, incoming[:0]), incoming))
        self.task_groups[task_id] = group
        quotas = self._quotas(
            [len(indices) for indices in candidates.values()], self.archive_capacity
        )
        selected, bounds, offset = [], {}, self.recent_capacity
        for (key, indices), quota in zip(candidates.items(), quotas):
            # Deterministic coverage-preserving thinning, not just oldest-first eviction.
            chosen = indices[np.linspace(0, len(indices) - 1, int(quota), dtype=int)]
            selected.append(chosen)
            bounds[key] = (offset, offset + int(quota))
            offset += int(quota)
        indices = np.concatenate(selected)
        # One field-sized temporary avoids overwriting sources during compaction.
        for name in ReplayBuffer.fields:
            data = getattr(self.storage, name)
            data[self.recent_capacity : offset] = data[indices]
        self.groups = bounds
        self._refresh_views()
        self.last_admission = dict(
            task_id=int(task_id),
            group=int(group),
            nearest_similarity=similarity,
            tail_candidates=max(1, int(len(self.recent) * self.keep_fraction)),
            admitted_candidates=len(incoming),
            merged_groups=merged_pair,
        )

    def fork(self):
        """Independent recovery archive; all views in the copy still share one allocation."""
        result = GlobalReplayMemory(
            self.storage.capacity,
            self.recent_capacity,
            self.storage.obs_dim,
            self.storage.action_dim,
            self.storage.device,
            discrete=self.storage.discrete,
            keep_fraction=self.keep_fraction,
            similarity_threshold=self.similarity_threshold,
            max_groups=self.max_groups,
            compression=self.compression,
        )
        for name in ReplayBuffer.fields:
            getattr(result.storage, name)[self.recent_capacity :] = getattr(self.storage, name)[
                self.recent_capacity :
            ]
        result.groups = self.groups.copy()
        result.task_groups = self.task_groups.copy()
        result._refresh_views()
        return result

    def stats(self):
        archive_size = sum(stop - start for start, stop in self.groups.values())
        return dict(
            capacity=self.storage.capacity,
            compression=self.compression,
            recent_capacity=self.recent_capacity,
            archive_capacity=self.archive_capacity,
            recent_samples=len(self.recent),
            archive_samples=archive_size,
            allocated_transition_bytes=sum(
                getattr(self.storage, name).nbytes for name in ReplayBuffer.fields
            ),
            groups=len(self.groups),
            tasks=len(self.task_groups),
            group_sizes={str(key): stop - start for key, (start, stop) in self.groups.items()},
            admission=self.last_admission.copy(),
        )
