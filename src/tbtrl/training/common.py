"""Validation and bounded diagnostics shared by both training algorithms."""

import math


def validate_training_options(
    *,
    total_steps,
    warmup_steps,
    buffer_capacity,
    batch_size,
    eval_freq,
    train_freq,
    tau,
    gamma,
    replay_ratio,
    replay_tasks_per_batch,
    early_stop_patience,
    eval_episodes,
):
    for name, value in dict(
        total_steps=total_steps,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        eval_freq=eval_freq,
        train_freq=train_freq,
        early_stop_patience=early_stop_patience,
        eval_episodes=eval_episodes,
    ).items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(warmup_steps, int) or not 0 <= warmup_steps <= buffer_capacity:
        raise ValueError("warmup_steps must be between zero and buffer_capacity.")
    if not 0 < tau <= 1 or not 0 <= gamma <= 1:
        raise ValueError("Require 0 < tau <= 1 and 0 <= gamma <= 1.")
    # This is additional replay relative to the current batch, not a probability.
    if not math.isfinite(replay_ratio) or replay_ratio < 0:
        raise ValueError("replay_ratio must be finite and nonnegative.")
    if replay_tasks_per_batch is not None and (
        not isinstance(replay_tasks_per_batch, int) or replay_tasks_per_batch < 1
    ):
        raise ValueError("replay_tasks_per_batch must be positive or None.")


class LossHistory(list):
    """Keep evaluation-interval diagnostics and the latest update, not every update."""

    def __init__(self, interval):
        super().__init__()
        self.interval = interval

    def append(self, row):
        if self and self[-1]["step"] % self.interval != 0:
            self[-1] = row
        else:
            super().append(row)
