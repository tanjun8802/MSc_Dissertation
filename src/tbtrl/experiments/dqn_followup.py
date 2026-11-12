"""Fixed-budget DQN comparisons and independent, auditable recovery probes."""

import hashlib
import json
import logging
import shutil
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

from tbtrl.continuation import atomic_json, prepare_run, run_lock, save_progress
from tbtrl.environments.registry import make_environment

from .config import ExperimentConfig, training_seed, training_settings
from .continuation import load_sequence
from .dqn_study import configure_k_replay
from .runner import _evaluate, _memory, _record_memory, _save_task, _train, _training_log

logger = logging.getLogger(__name__)
RECOVERY_PROTOCOLS = ("no-replay", "method-replay")


def configure_followup(base, *, k_values=(1, 2, 4, 8, 11), **replay_options):
    """Retain the supplied model/penalties; compare selectors at fixed interactions.

    Each sequential condition trains for the full per-task budget. Threshold
    crossings are still measured, but never shorten historical rehearsal.
    """
    values = tuple(k_values)
    if not values or len(values) != len(set(values)) or any(k < 1 for k in values):
        raise ValueError("k_values must contain distinct positive task caps")
    cases = {}
    for k in values:
        for selector in ("random", "basis"):
            name = f"{selector}-k{k}"
            config = configure_k_replay(
                base, max_tasks=k, selector=selector, weighting="uniform", **replay_options
            )
            config.model = deepcopy(base.model)
            config.model.setdefault("embedding_constraint", "projected")
            for key in ("sigreg_coef", "goal_separation_coef"):
                config.training[key] = base.training.get(key, 0)
            config.training["enable_early_stop"] = False
            config.recovery = dict(total_steps=base.training["total_steps"], enable_early_stop=True)
            config.name = f"{base.name}-fixed-budget-{name}"
            cases[name] = config.validate()
    control = deepcopy(next(iter(cases.values())))
    control.training["k_replay"].update(max_tasks=0, sample_budget=0, selector="basis")
    for name in ("fine-tuning", "scratch"):
        cases[name] = deepcopy(control)
        cases[name].name = f"{base.name}-fixed-budget-{name}"
        cases[name].validate()
    return cases


def _fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _probe_config(source, protocol):
    config = deepcopy(source)
    if protocol == "no-replay":
        # All starting models face the same current-task-only adaptation rule.
        options = dict(config.training["k_replay"])
        options.update(max_tasks=0, sample_budget=0, selector="basis", weighting="uniform")
        config.recovery = dict(config.recovery, k_replay=options, replay_loss_coef=0)
    return config.validate()


def run_recovery_probe(
    source_dir, output_dir, *, protocol, device="cpu", resume=True, verbose=True
):
    """Recover each task from the same final boundary, never updating the source.

    no-replay uses fresh current-task storage and a fresh disabled selector.
    method-replay forks the source archive and selector for each trial. Both
    modes reset optimisers/targets in the DQN trainer and keep both encoders live.
    A complete receipt is written last, allowing interrupted trials to restart.
    """
    if protocol not in RECOVERY_PROTOCOLS:
        raise ValueError(f"Unknown recovery protocol: {protocol}")
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output or output in source.parents:
        raise ValueError("Recovery needs a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(source), run_lock(output):
        manifest = json.loads((source / "manifest.json").read_text())
        original = ExperimentConfig(**manifest["config"]).validate()
        if original.algorithm != "dqn" or original.training.get("k_replay") is None:
            raise ValueError("Recovery probes require a bounded DQN sequential run")
        if str(device) != manifest["device"]:
            raise ValueError("Use the acquisition device for matched recovery probes")
        if original.recovery_return_threshold is None:
            raise ValueError("Recovery probes require an explicit mean-return threshold")
        source_rows = json.loads((source / "summary.json").read_text())
        count = len(original.goals)
        boundaries = {}
        for seed in original.seeds:
            rows = sorted(
                (r for r in source_rows if r["mode"] == "sequential" and r["seed"] == seed),
                key=lambda r: r["task_id"],
            )
            if [r["task_id"] for r in rows] != list(range(count)):
                raise ValueError(f"Incomplete acquisition for seed {seed}")
            if [r["goal"] for r in rows] != original.goals:
                raise ValueError("Acquisition task definitions differ from the manifest")
            boundary = source / f"seed_{seed}/sequential/boundary.pt"
            boundaries[str(seed)] = _fingerprint(boundary)
        config = _probe_config(original, protocol)
        provenance = dict(
            source=str(source),
            boundary_sha256=boundaries,
            manifest_sha256=_fingerprint(source / "manifest.json"),
        )
        prepare_run(
            output,
            dict(
                kind="dqn-recovery-probe",
                config=asdict(config),
                protocol_name=protocol,
                device=str(device),
                provenance=provenance,
            ),
            config.goals,
            resume=resume,
        )
        atomic_json(
            output / "manifest.json",
            dict(
                config=asdict(config),
                device=str(device),
                modes=["recovery"],
                recovery_protocol=protocol,
                provenance=provenance,
            ),
        )
        with _training_log(output, verbose):
            records = []
            factory = partial(make_environment, config.environment, **config.environment_options)
            for seed in config.seeds:
                saved = load_sequence(source / f"seed_{seed}/sequential/boundary.pt", device)
                if saved["completed"] != count:
                    raise ValueError("Source boundary is not the requested final task boundary")
                final = saved["result"]
                before = _evaluate(config, final, factory, seed)
                for task_id, goal in enumerate(config.goals):
                    directory = output / f"seed_{seed}/recovery/after_{count}/task_{task_id}"
                    receipt = directory / "complete.json"
                    if receipt.exists():
                        record = json.loads((directory / "metrics.json").read_text())
                        records.append(record)
                        continue
                    if directory.exists():
                        shutil.rmtree(directory)
                    retained = before[task_id]["mean_return"] >= config.recovery_return_threshold
                    if retained:
                        directory.mkdir(parents=True)
                        record = dict(
                            seed=seed,
                            task_id=task_id,
                            goal=goal,
                            mode="recovery",
                            after_tasks=count,
                            recovery_performed=False,
                            steps=0,
                            steps_to_threshold=0,
                            solved=True,
                            elapsed_seconds=0.0,
                            learning_curve=[],
                            evaluation=before,
                        )
                    else:
                        memory = (
                            saved["memory"].fork()
                            if protocol == "method-replay"
                            else _memory(config, factory, device)
                        )
                        buffers = (
                            {k: v for k, v in memory.buffers.items() if k != task_id}
                            if protocol == "method-replay"
                            else {}
                        )
                        state = (
                            deepcopy(final.selector_state) if protocol == "method-replay" else None
                        )
                        settings = training_settings(config, "recovery", task_id)
                        result = _train(
                            config,
                            deepcopy((final.network, final.target)),
                            factory,
                            task_id,
                            training_seed(config, seed, "recovery", task_id),
                            device,
                            deepcopy(saved["goals"]),
                            buffers,
                            settings,
                            memory,
                            selector_state=state,
                        )
                        record = _save_task(
                            directory,
                            result,
                            replace(config, training=settings),
                            seed,
                            task_id,
                            "recovery",
                            _evaluate(config, result, factory, seed),
                        )
                        record["recovery_performed"] = True
                        _record_memory(directory, record, memory)
                    record.update(
                        recovery_protocol=protocol,
                        initial_evaluation=before,
                        recovery_target=dict(
                            metric="mean_return", value=config.recovery_return_threshold
                        ),
                        source_boundary_sha256=boundaries[str(seed)],
                    )
                    atomic_json(directory / "metrics.json", record)
                    atomic_json(receipt, dict(after_tasks=count, protocol=protocol))
                    records.append(record)
                    save_progress(output / "summary.json", records, endpoint=count)
                    logger.info(
                        "Recovery %s seed=%s task=%s steps=%s solved=%s",
                        protocol,
                        seed,
                        task_id,
                        record["steps"],
                        record["solved"],
                    )
            atomic_json(output / "summary.json", records)
            return json.loads((output / "summary.json").read_text())
