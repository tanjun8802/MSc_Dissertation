"""Restore a policy for evaluation from a trusted experiment checkpoint."""

from functools import partial
from types import SimpleNamespace

import torch

from tbtrl.environments.registry import make_environment
from tbtrl.evaluation import make_policy
from tbtrl.normalization import RunningMeanStd

from .config import ExperimentConfig
from .runner import _models


def load_policy(path, *, goal=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    config = ExperimentConfig(**checkpoint["config"]).validate()
    factory = partial(make_environment, config.environment, **config.environment_options)
    models = _models(config, factory, device)
    if config.algorithm == "dqn":
        network, target = models
        network.load_state_dict(checkpoint["network"])
        target.load_state_dict(checkpoint["target"])
        network.eval()
        result = SimpleNamespace(network=network)
    else:
        actor, critic, target = models
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        normalizers = []
        for key in ("state_normalizer", "goal_normalizer"):
            state = checkpoint[key]
            normalizer = RunningMeanStd(tuple(state["mean"].shape), device=device)
            for name, value in state.items():
                getattr(normalizer, name).copy_(value)
            normalizers.append(normalizer)
        result = SimpleNamespace(
            actor=actor, state_normalizer=normalizers[0], goal_normalizer=normalizers[1]
        )
    selected_goal = config.goals[checkpoint["task_id"]] if goal is None else goal
    return make_policy(
        result,
        selected_goal,
        config.algorithm,
        normalize_state=config.training.get("normalize_state_inputs", False),
        normalize_goal=config.training.get("normalize_goal_inputs", False),
        clip=config.training.get("obs_norm_clip", 10.0),
    )
