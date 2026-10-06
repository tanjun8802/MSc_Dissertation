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
    actor_type: str = "mlp"
    actor_model: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    replay_keep_fraction: float = 0.4
    replay_schedule: str = "constant"
    similarity_threshold: float | None = None
    scratch: dict = field(default_factory=dict)
    first_task: dict = field(default_factory=dict)
    task_seed_stride: int = 1
    recovery_seed_offset: int = 0
    scratch_seen_goals: bool = False
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
        from tbtrl.models.sac import FactorisedGaussianActor
        from tbtrl.training.actor import ActorTBTRLOptions

        if self.actor_type not in ("mlp", "factorised"):
            raise ValueError("actor_type must be mlp or factorised.")
        if self.actor_type == "factorised" and self.algorithm != "sac":
            raise ValueError("A factorised actor requires SAC.")
        if self.actor_type == "mlp" and self.actor_model:
            raise ValueError("actor_model options are only supported for the factorised actor.")
        actor_keys = set(inspect.signature(FactorisedGaussianActor).parameters) - {
            "state_dim",
            "action_dim",
            "goal_dim",
        }
        if set(self.actor_model) - actor_keys:
            raise ValueError(
                f"Unknown actor model settings: {sorted(set(self.actor_model) - actor_keys)}"
            )
        for settings in (self.training, self.scratch, self.first_task, self.recovery):
            if settings.get("actor_tbtrl") is not None:
                actor_options = ActorTBTRLOptions.from_dict(settings["actor_tbtrl"])
                if actor_options.needs_factors and self.actor_type != "factorised":
                    raise ValueError(
                        "Actor representation penalties require actor_type='factorised'."
                    )
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
        for settings in (self.training, self.scratch, self.first_task, self.recovery):
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
            actor_model=dict(self.actor_model, hidden_dim=8, latent_dim=4)
            if self.actor_type == "factorised"
            else {},
            training=training,
            environment_options=dict(self.environment_options, max_horizon=8),
            recovery=dict(self.recovery, total_steps=16, warmup_steps=4, enable_early_stop=False),
            eval_episodes=1,
            recovery_threshold=None,
        )


def load_config(path):
    return ExperimentConfig(**json.loads(Path(path).read_text())).validate()


def training_settings(config, mode, task_id):
    """Resolve the original notebook's phase/task overrides in one audited place."""
    settings = dict(config.training, eval_episodes=config.eval_episodes)
    if mode == "scratch":
        settings.update(config.scratch)
    else:
        if config.replay_schedule == "task_count":
            settings["replay_ratio"] = float(task_id + 1)
        if mode == "sequential" and task_id == 0:
            settings.update(config.first_task)
        if mode == "recovery":
            settings.update(config.recovery)
    return settings


def training_seed(config, seed, mode, task_id):
    return (
        seed
        + config.task_seed_stride * task_id
        + (config.recovery_seed_offset if mode == "recovery" else 0)
    )
