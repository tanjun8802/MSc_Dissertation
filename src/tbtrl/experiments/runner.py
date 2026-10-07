"""Scratch, sequential transfer, retention, and independent recovery experiments."""

import json
import logging
import platform
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import gymnasium
import numpy as np
import torch

from tbtrl.environments.registry import make_environment
from tbtrl.evaluation import evaluate_policy_with_success, make_policy
from tbtrl.models.dqn import FactorisedQNetwork
from tbtrl.models.sac import FactorisedGaussianActor, FactorisedTwinCritic, GaussianActor
from tbtrl.random import set_seed
from tbtrl.replay import TaskReplayMemory
from tbtrl.training.dqn import train_dqn
from tbtrl.training.sac import train_sac

from .config import training_seed, training_settings

logger = logging.getLogger(__name__)


def _models(config, factory, device):
    env = factory(goal=config.goals[0])
    try:
        if len(env.observation_space.shape) != 1:
            raise ValueError("Adapt observations to a flat vector before using TBTRL.")
        obs_dim, goal_dim = env.observation_space.shape[0], len(config.goals[0])
        if config.algorithm == "dqn":
            if not isinstance(env.action_space, gymnasium.spaces.Discrete):
                raise ValueError("DQN requires Discrete actions.")
            model = FactorisedQNetwork(obs_dim, env.action_space.n, goal_dim, **config.model).to(
                device
            )
            target = FactorisedQNetwork(obs_dim, env.action_space.n, goal_dim, **config.model).to(
                device
            )
            target.load_state_dict(model.state_dict())
            return model, target
        if not isinstance(env.action_space, gymnasium.spaces.Box):
            raise ValueError("SAC requires Box actions.")
        act_dim = env.action_space.shape[0]
        if config.actor_type == "factorised":
            actor = FactorisedGaussianActor(obs_dim, act_dim, goal_dim, **config.actor_model).to(
                device
            )
        else:
            actor = GaussianActor(
                obs_dim, act_dim, goal_dim, net_arch=(config.model.get("hidden_dim", 64),) * 2
            ).to(device)
        critic = FactorisedTwinCritic(obs_dim, act_dim, goal_dim, **config.model).to(device)
        target = FactorisedTwinCritic(obs_dim, act_dim, goal_dim, **config.model).to(device)
        target.load_state_dict(critic.state_dict())
        return actor, critic, target
    finally:
        env.close()


def _train(config, models, factory, task_id, seed, device, goals, buffers, settings):
    env = factory(goal=config.goals[task_id])
    common = dict(
        seed=seed,
        env=env,
        goal=config.goals[task_id],
        task_id=task_id,
        device=device,
        make_env=factory,
        task_goals=goals,
        replay_task_buffers=buffers,
        **settings,
    )
    logger.info(
        "Training task=%s goal=%s seed=%s settings=%s",
        task_id,
        config.goals[task_id],
        seed,
        settings,
    )
    try:
        if config.algorithm == "dqn":
            return train_dqn(q_network=models[0], q_target_network=models[1], **common)
        return train_sac(actor=models[0], critic=models[1], critic_target=models[2], **common)
    finally:
        env.close()


def _evaluate(config, result, factory, seed):
    rows = []
    for task_id, goal in enumerate(config.goals):
        env = factory(goal=goal)
        try:
            policy = make_policy(
                result,
                goal,
                config.algorithm,
                normalize_state=config.training.get("normalize_state_inputs", False),
                normalize_goal=config.training.get("normalize_goal_inputs", False),
                clip=config.training.get("obs_norm_clip", 10.0),
            )
            ret, length, success, distance = evaluate_policy_with_success(
                env, policy, goal, config.eval_episodes, seed=seed + 1_000_000 + task_id * 100
            )
            rows.append(
                dict(
                    task_id=task_id,
                    mean_return=ret,
                    mean_length=length,
                    success_rate=success,
                    final_distance=distance if np.isfinite(distance) else None,
                )
            )
        finally:
            env.close()
    return rows


def _normalizer_state(normalizer):
    return {name: getattr(normalizer, name).detach().cpu() for name in ("mean", "var", "count")}


def _save_task(directory, result, config, seed, task_id, mode, evaluations):
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(
        seed=seed,
        task_id=task_id,
        goal=config.goals[task_id],
        mode=mode,
        steps=result.steps,
        steps_to_threshold=result.steps_to_threshold,
        solved=result.steps_to_threshold is not None,
        elapsed_seconds=result.elapsed_seconds,
        learning_curve=result.evaluations,
        evaluation=evaluations,
    )
    if config.algorithm == "dqn":
        checkpoint = {"network": result.network.state_dict(), "target": result.target.state_dict()}
    else:
        checkpoint = {
            "actor": result.actor.state_dict(),
            "critic": result.critic.state_dict(),
            "target": result.target.state_dict(),
            "state_normalizer": _normalizer_state(result.state_normalizer),
            "goal_normalizer": _normalizer_state(result.goal_normalizer),
            "entropy_coefficient": result.entropy_coefficient,
        }
        record["success_curve"] = result.success_rates
        record["initial_alpha"] = (
            config.training.get("initial_alpha", 1.0)
            if config.training.get("ent_coef", "auto") == "auto"
            else float(config.training["ent_coef"])
        )
        record["final_alpha"] = result.entropy_coefficient
        if result.transfer_probes:
            record["transfer_probes"] = result.transfer_probes
            (directory / "transfer-probes.json").write_text(
                json.dumps(result.transfer_probes, indent=2, allow_nan=False) + "\n"
            )
    checkpoint.update(
        config=asdict(config), seed=seed, task_id=task_id, mode=mode, format_version=1
    )
    torch.save(checkpoint, directory / "checkpoint.pt")
    (directory / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    with (directory / "losses.jsonl").open("w") as stream:
        for row in result.losses:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return record


def _run_experiment(
    config,
    output_dir,
    *,
    device="cpu",
    modes=("scratch", "sequential", "recovery"),
    on_task_end=None,
):
    """Run explicit task sequences; recovery starts from the same final model each time.

    Output paths must be new. Checkpoints support evaluation, not exact optimiser
    resume. The public trainers return live models for programmatic continuation.
    """
    config.validate()
    if not modes or set(modes) - {"scratch", "sequential", "recovery"}:
        raise ValueError("Select scratch, sequential and/or recovery modes.")
    if "recovery" in modes and "sequential" not in modes:
        raise ValueError("Recovery requires sequential training in the same run.")
    output = Path(output_dir)
    factory = partial(make_environment, config.environment, **config.environment_options)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = dict(
        config=asdict(config),
        git_commit=commit,
        git_dirty=dirty,
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        gymnasium=gymnasium.__version__,
        device=str(device),
        modes=list(modes),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    for seed in config.seeds:
        final_result = None
        sequential_memory = None
        sequential_goals = None
        for mode in ("scratch", "sequential"):
            if mode not in modes:
                continue
            set_seed(seed)
            models = _models(config, factory, device) if mode == "sequential" else None
            memory = TaskReplayMemory(config.replay_keep_fraction, config.similarity_threshold)
            task_goals = {}
            for task_id, goal in enumerate(config.goals):
                if mode == "scratch":
                    models = _models(config, factory, device)
                    if not config.scratch_seen_goals:
                        task_goals = {}
                task_goals[task_id] = np.asarray(goal, dtype=np.float32)
                settings = training_settings(config, mode, task_id)
                if (
                    mode == "sequential"
                    and final_result is not None
                    and config.entropy_transfer == "carry"
                ):
                    settings["initial_alpha"] = final_result.entropy_coefficient
                    logger.info(
                        "Carrying temperature into task=%s: alpha=%.8g",
                        task_id,
                        settings["initial_alpha"],
                    )
                logger.info("=== %s | seed=%s | task=%s | goal=%s ===", mode, seed, task_id, goal)
                before = _snapshot(models, config.algorithm) if on_task_end else None
                result = _train(
                    config,
                    models,
                    factory,
                    task_id,
                    training_seed(config, seed, mode, task_id),
                    device,
                    task_goals,
                    memory.buffers if mode == "sequential" else {},
                    settings,
                )
                task_config = replace(config, training=settings)
                evaluation = _evaluate(task_config, result, factory, seed)
                records.append(
                    _save_task(
                        output / f"seed_{seed}" / mode / f"task_{task_id}",
                        result,
                        task_config,
                        seed,
                        task_id,
                        mode,
                        evaluation,
                    )
                )
                if on_task_end:
                    on_task_end(
                        result=result,
                        config=task_config,
                        factory=factory,
                        record=records[-1],
                        task_goals=deepcopy(task_goals),
                        before=before,
                        directory=output / f"seed_{seed}" / mode / f"task_{task_id}",
                    )
                logger.info(
                    "Completed %s task=%s steps=%s threshold_steps=%s elapsed=%.2fs",
                    mode,
                    task_id,
                    result.steps,
                    result.steps_to_threshold,
                    result.elapsed_seconds,
                )
                if mode == "sequential":
                    similarities = None
                    if (
                        config.algorithm == "dqn"
                        and task_id > 0
                        and config.similarity_threshold is not None
                    ):
                        with torch.no_grad():
                            goals_t = torch.as_tensor(
                                np.asarray(list(task_goals.values())),
                                dtype=torch.float32,
                                device=device,
                            )
                            embeddings = torch.nn.functional.normalize(
                                result.network.encode_goal(goals_t), dim=-1
                            )
                            similarities = (embeddings[-1] @ embeddings[:-1].T).cpu().numpy()
                    memory.add(task_id, result.buffer, similarities)
                    final_result, sequential_memory, sequential_goals = result, memory, task_goals
        if "recovery" in modes:
            retention = _evaluate(config, final_result, factory, seed)
            for task_id, goal in enumerate(config.goals):
                if (
                    config.recovery_threshold is not None
                    and retention[task_id]["success_rate"] >= config.recovery_threshold
                ):
                    record = dict(
                        seed=seed,
                        task_id=task_id,
                        goal=goal,
                        mode="recovery",
                        recovery_performed=False,
                        steps=0,
                        steps_to_threshold=0,
                        solved=True,
                        elapsed_seconds=0.0,
                        learning_curve=[],
                        evaluation=retention,
                    )
                    directory = output / f"seed_{seed}" / "recovery" / f"task_{task_id}"
                    directory.mkdir(parents=True)
                    (directory / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
                    records.append(record)
                    logger.info(
                        "Recovery skipped: seed=%s task=%s success=%.3f threshold=%.3f",
                        seed,
                        task_id,
                        retention[task_id]["success_rate"],
                        config.recovery_threshold,
                    )
                    continue
                if config.algorithm == "dqn":
                    models = deepcopy((final_result.network, final_result.target))
                    buffers = {}
                else:
                    models = deepcopy(
                        (final_result.actor, final_result.critic, final_result.target)
                    )
                    buffers = {
                        key: value
                        for key, value in sequential_memory.buffers.items()
                        if key != task_id
                    }
                settings = training_settings(config, "recovery", task_id)
                if config.entropy_transfer == "carry":
                    settings["initial_alpha"] = final_result.entropy_coefficient
                    logger.info(
                        "Recovery task=%s starts from final sequential alpha=%.8g",
                        task_id,
                        settings["initial_alpha"],
                    )
                before = _snapshot(models, config.algorithm) if on_task_end else None
                logger.info("=== recovery | seed=%s | task=%s | goal=%s ===", seed, task_id, goal)
                result = _train(
                    config,
                    models,
                    factory,
                    task_id,
                    training_seed(config, seed, "recovery", task_id),
                    device,
                    deepcopy(sequential_goals),
                    buffers,
                    settings,
                )
                task_config = replace(config, training=settings)
                evaluation = _evaluate(task_config, result, factory, seed)
                records.append(
                    _save_task(
                        output / f"seed_{seed}" / "recovery" / f"task_{task_id}",
                        result,
                        task_config,
                        seed,
                        task_id,
                        "recovery",
                        evaluation,
                    )
                )
                if on_task_end:
                    on_task_end(
                        result=result,
                        config=task_config,
                        factory=factory,
                        record=records[-1],
                        task_goals=deepcopy(sequential_goals),
                        before=before,
                        directory=output / f"seed_{seed}" / "recovery" / f"task_{task_id}",
                    )
                logger.info(
                    "Completed recovery task=%s steps=%s threshold_steps=%s elapsed=%.2fs",
                    task_id,
                    result.steps,
                    result.steps_to_threshold,
                    result.elapsed_seconds,
                )
    (output / "summary.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    return records


def _snapshot(models, algorithm):
    names = ("q", "target") if algorithm == "dqn" else ("actor", "critic", "target")
    return {
        f"{group}.{name}": p.detach().cpu().clone()
        for group, model in zip(names, models)
        for name, p in model.named_parameters()
    }


@contextmanager
def _training_log(output, verbose):
    """Scope notebook stdout and file handlers; re-execution never duplicates logs."""
    package_logger = logging.getLogger("tbtrl")
    old_level, old_propagate = package_logger.level, package_logger.propagate
    handlers = [logging.FileHandler(output / "training.log")]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False
    try:
        yield
    finally:
        for handler in handlers:
            package_logger.removeHandler(handler)
            handler.close()
        package_logger.setLevel(old_level)
        package_logger.propagate = old_propagate


def run_experiment(
    config,
    output_dir,
    *,
    device="cpu",
    modes=("scratch", "sequential", "recovery"),
    on_task_end=None,
    verbose=True,
):
    """Run experiments with live/file logging and an optional per-goal diagnostic callback.

    The callback runs immediately after each trained goal, before the next goal.
    Use NotebookDiagnostics for inline plots and saved numeric diagnostics.
    """
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with _training_log(output, verbose):
        return _run_experiment(config, output, device=device, modes=modes, on_task_end=on_task_end)
