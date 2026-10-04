"""Scratch, sequential transfer, retention, and independent recovery experiments."""

import json
import platform
import subprocess
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
from tbtrl.models.sac import FactorisedTwinCritic, GaussianActor
from tbtrl.random import set_seed
from tbtrl.replay import TaskReplayMemory
from tbtrl.training.dqn import train_dqn
from tbtrl.training.sac import train_sac


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
            return model, deepcopy(model)
        if not isinstance(env.action_space, gymnasium.spaces.Box):
            raise ValueError("SAC requires Box actions.")
        act_dim = env.action_space.shape[0]
        actor = GaussianActor(
            obs_dim, act_dim, goal_dim, net_arch=(config.model.get("hidden_dim", 64),) * 2
        ).to(device)
        critic = FactorisedTwinCritic(obs_dim, act_dim, goal_dim, **config.model).to(device)
        return actor, critic, deepcopy(critic)
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
    checkpoint.update(
        config=asdict(config), seed=seed, task_id=task_id, mode=mode, format_version=1
    )
    torch.save(checkpoint, directory / "checkpoint.pt")
    (directory / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    with (directory / "losses.jsonl").open("w") as stream:
        for row in result.losses:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return record


def run_experiment(
    config, output_dir, *, device="cpu", modes=("scratch", "sequential", "recovery")
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
    output.mkdir(parents=True, exist_ok=False)
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
            models = _models(config, factory, device)
            memory = TaskReplayMemory(config.replay_keep_fraction, config.similarity_threshold)
            task_goals = {}
            for task_id, goal in enumerate(config.goals):
                if mode == "scratch":
                    set_seed(seed + task_id)
                    models = _models(config, factory, device)
                    task_goals = {}
                task_goals[task_id] = np.asarray(goal, dtype=np.float32)
                settings = dict(config.training, eval_episodes=config.eval_episodes)
                if mode == "scratch":
                    settings["replay_ratio"] = 0.0
                    if config.algorithm == "sac":
                        settings["initial_alpha"] = 0.1
                elif config.replay_schedule == "task_count":
                    settings["replay_ratio"] = float(task_id + 1)
                result = _train(
                    config,
                    models,
                    factory,
                    task_id,
                    seed + task_id,
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
                settings = dict(config.training, **config.recovery)
                settings["eval_episodes"] = config.eval_episodes
                result = _train(
                    config,
                    models,
                    factory,
                    task_id,
                    seed + task_id,
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
    (output / "summary.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    return records
