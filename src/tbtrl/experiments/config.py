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
    entropy_transfer: str = "reset"
    actor_type: str = "mlp"
    critic_type: str = "factorised"
    archive_compression: str = "similarity"
    carry_optimizer_state: bool = False
    independent_model_initialization: bool = False
    actor_model: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    replay_keep_fraction: float = 0.4
    replay_schedule: str = "constant"
    similarity_threshold: float | None = None
    replay_mode: str = "per_task"
    global_replay_capacity: int = 1000000
    recent_replay_capacity: int = 250000
    replay_max_groups: int = 16
    scratch: dict = field(default_factory=dict)
    first_task: dict = field(default_factory=dict)
    task_seed_stride: int = 1
    recovery_seed_offset: int = 0
    scratch_seen_goals: bool = False
    recovery: dict = field(default_factory=dict)
    recovery_threshold: float | None = None
    eval_episodes: int = 8
    post_eval_episodes: int | None = None
    study_overrides: dict = field(default_factory=dict)

    def validate(self):
        from tbtrl.training.dqn import train_dqn
        from tbtrl.training.sac import train_sac

        if not isinstance(self.study_overrides, dict) or set(self.study_overrides) - {
            "exploration",
            "architecture",
            "retention",
        }:
            raise ValueError("Unknown study_overrides section.")
        for cases in self.study_overrides.values():
            if not isinstance(cases, dict):
                raise ValueError("Study sections must map case names to config overrides.")
            for overrides in cases.values():
                if not isinstance(overrides, dict) or set(overrides) - (
                    self.__dataclass_fields__.keys() - {"study_overrides"}
                ):
                    raise ValueError("Unknown or recursive study config override.")

        if self.critic_type not in ("factorised", "mlp"):
            raise ValueError("critic_type must be factorised or mlp.")
        if self.archive_compression not in ("similarity", "capacity"):
            raise ValueError("archive_compression must be similarity or capacity.")
        if self.critic_type == "mlp":
            if self.algorithm != "sac" or self.replay_mode == "global":
                raise ValueError("MLP critic controls require SAC with per_task replay.")
            for phase in ({}, self.scratch, self.first_task, self.recovery):
                settings = dict(self.training, **phase)
                if any(
                    settings.get(k, 0)
                    for k in (
                        "sigreg_coef",
                        "goal_separation_coef",
                        "phi_raw_norm_coef",
                        "psi_raw_norm_coef",
                    )
                ):
                    raise ValueError("MLP critics cannot use factor regularisers.")

        if self.post_eval_episodes is not None and self.post_eval_episodes < 1:
            raise ValueError("post_eval_episodes must be positive.")
        if self.carry_optimizer_state and self.algorithm != "sac":
            raise ValueError("Optimiser carry is supported for SAC only.")
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
        if self.replay_mode not in ("per_task", "global"):
            raise ValueError("replay_mode must be per_task or global.")
        if self.replay_mode == "global":
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (
                    self.global_replay_capacity,
                    self.recent_replay_capacity,
                    self.replay_max_groups,
                )
            ):
                raise ValueError("Global replay capacities and group limit must be integers.")
            if not 1 <= self.recent_replay_capacity < self.global_replay_capacity:
                raise ValueError("Recent replay capacity must leave space in the global budget.")
            if (
                not 2
                <= self.replay_max_groups
                <= (self.global_replay_capacity - self.recent_replay_capacity)
            ):
                raise ValueError("Global replay requires at least two groups and a slot per group.")
            if self.similarity_threshold is None or not -1 <= self.similarity_threshold < 1:
                raise ValueError("Global replay requires a similarity_threshold in [-1, 1).")
            for phase in ({}, self.scratch, self.first_task, self.recovery):
                if (
                    dict(self.training, **phase).get(
                        "warmup_steps", 25000 if self.algorithm == "sac" else 5000
                    )
                    > self.recent_replay_capacity
                ):
                    raise ValueError("Recent replay capacity must cover every phase's warmup.")
        if (
            self.similarity_threshold is not None
            and self.algorithm != "dqn"
            and self.replay_mode != "global"
        ):
            raise ValueError("Shared task buffers require DQN reward relabelling.")
        if self.recovery_threshold is not None and not 0 <= self.recovery_threshold <= 1:
            raise ValueError("recovery_threshold must be a success rate in [0, 1].")
        from tbtrl.models.sac import FactorisedGaussianActor
        from tbtrl.training.actor import ActorTBTRLOptions

        if self.entropy_transfer not in ("reset", "carry"):
            raise ValueError("entropy_transfer must be reset or carry.")
        if self.entropy_transfer == "carry":
            if self.algorithm != "sac":
                raise ValueError("Entropy transfer requires SAC.")
            for phase in ({}, self.scratch, self.first_task, self.recovery):
                if dict(self.training, **phase).get("ent_coef", "auto") != "auto":
                    raise ValueError("Entropy transfer requires automatic entropy tuning.")
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
            "optimizer_state",
            "replay_task_buffers",
            "current_buffer",
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
        if training.get("transfer_probe_steps") is not None:
            training["transfer_probe_steps"] = [0, 3, 4, 5, 8]
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
            post_eval_episodes=1 if self.post_eval_episodes is not None else None,
            recovery_threshold=None,
            global_replay_capacity=64
            if self.replay_mode == "global"
            else self.global_replay_capacity,
            recent_replay_capacity=32
            if self.replay_mode == "global"
            else self.recent_replay_capacity,
            replay_max_groups=min(self.replay_max_groups, 8)
            if self.replay_mode == "global"
            else self.replay_max_groups,
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
    if config.replay_mode == "global":
        settings["buffer_capacity"] = config.recent_replay_capacity
        if config.algorithm == "sac":
            settings["relabel_rewards"] = True
    return settings


def training_seed(config, seed, mode, task_id):
    return (
        seed
        + config.task_seed_stride * task_id
        + (config.recovery_seed_offset if mode == "recovery" else 0)
    )
