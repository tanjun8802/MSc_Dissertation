"""Serializable experiment settings with validation before expensive training."""

import inspect
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass
class ExperimentConfig:
    name: str
    algorithm: str
    environment: str
    goals: list[list[float]]
    seeds: list[int] = field(default_factory=lambda: [42])
    environment_options: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    replay_keep_fraction: float = 0.4
    replay_schedule: str = "constant"
    similarity_threshold: float | None = None
    recovery: dict = field(default_factory=dict)
    recovery_threshold: float | None = None
    eval_episodes: int = 8

    def validate(self):
        from tbtrl.training.dqn import train_dqn
        from tbtrl.training.sac import train_sac

        if self.algorithm not in ("dqn", "sac"):
            raise ValueError("algorithm must be dqn or sac.")
        if not self.goals or not self.seeds or self.eval_episodes < 1:
            raise ValueError("Supply goals, seeds, and positive eval_episodes.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique.")
        if not 0 < self.replay_keep_fraction <= 1:
            raise ValueError("replay_keep_fraction must be in (0, 1].")
        if self.replay_schedule not in ("constant", "task_count"):
            raise ValueError("Unknown replay_schedule.")
        if self.similarity_threshold is not None and self.algorithm != "dqn":
            raise ValueError("Shared task buffers require DQN reward relabelling.")
        if self.recovery_threshold is not None and not 0 <= self.recovery_threshold <= 1:
            raise ValueError("recovery_threshold must be a success rate in [0, 1].")
        trainer = train_dqn if self.algorithm == "dqn" else train_sac
        owned = {
            "seed",
            "env",
            "goal",
            "task_id",
            "make_env",
            "task_goals",
            "replay_task_buffers",
            "q_network",
            "q_target_network",
            "actor",
            "critic",
            "critic_target",
            "device",
            "obs_dim",
            "action_dim",
        }
        allowed = set(inspect.signature(trainer).parameters) - owned
        for settings in (self.training, self.recovery):
            unknown = set(settings) - allowed
            if unknown:
                raise ValueError(f"Unknown or runner-owned training settings: {sorted(unknown)}")
        return self

    def to_dict(self):
        return asdict(self)

    def smoke(self):
        """Exercise real updates, replay, evaluation and recovery on a small CPU budget."""
        training = dict(
            self.training,
            total_steps=24,
            warmup_steps=4,
            batch_size=4,
            buffer_capacity=64,
            eval_freq=8,
            train_freq=1,
            enable_early_stop=False,
            sketch_dim=4,
        )
        model = dict(self.model, hidden_dim=8)
        model["rep_dim" if self.algorithm == "dqn" else "latent_dim"] = 4
        return replace(
            self,
            goals=self.goals[:2],
            seeds=self.seeds[:1],
            model=model,
            training=training,
            environment_options=dict(self.environment_options, max_horizon=8),
            recovery={"total_steps": 16, "warmup_steps": 4, "enable_early_stop": False},
            eval_episodes=1,
            recovery_threshold=None,
        )


def load_config(path):
    return ExperimentConfig(**json.loads(Path(path).read_text())).validate()
