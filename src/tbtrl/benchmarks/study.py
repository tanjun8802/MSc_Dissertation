"""Controlled candidate-learning studies with campaign-boundary checkpoints.

Checkpoints are trusted local torch/pickle files, not exchange-format models.
A checkpoint commits a whole campaign, including the learner and native MADE
state. Incomplete campaigns restart; completed campaigns spend no further oracle
calls. Evaluation and recovery are separate, explicit jobs.
"""

import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from tbtrl.benchmarks.candidates import CandidateConfig, CandidateLearner, Transition

PROTOCOL = "made-study-v1"
METHODS = {
    "scratch": "From scratch",
    "sequential": "TBTRL",
    "fine_tuning": "Sequential fine-tuning",
    "uniform_replay": "Uniform replay",
    "no_covariance": "TBTRL without covariance",
    "no_separation": "TBTRL without separation",
    "small_archive": "TBTRL · 100-slot archive",
}


def made_study_config():
    """Freeze the successful v2 pilot settings; opt in to independent RNG streams."""
    return CandidateConfig(
        batch_size=32,
        warmup=32,
        train_frequency=1,
        log_frequency=10,
        capacity=2500,
        recent_capacity=500,
        byte_budget=1024**3,
        archive_strategy="capacity_only",
        balanced_replay=True,
        sample_without_replacement=True,
        replay_coef=0.5,
        covariance_coef=0.1,
        separation_coef=1.0,
        covariance_target_variance=1 / 64,
        regularizer_batch_size=128,
        gradient_diagnostics=True,
        independent_rng=True,
    )


def method_config(base, method):
    if method not in METHODS:
        raise ValueError(f"Unknown study method: {method}")
    base = replace(base, independent_rng=True)
    if method == "fine_tuning":
        return replace(base, replay_coef=0, covariance_coef=0, separation_coef=0)
    if method == "uniform_replay":
        # Same tail/archive allocation; only uniform historical sampling and no
        # covariance/separation. Norm safety bounds remain matched in every method.
        return replace(base, balanced_replay=False, covariance_coef=0, separation_coef=0)
    if method == "no_covariance":
        return replace(base, covariance_coef=0)
    if method == "no_separation":
        return replace(base, separation_coef=0)
    if method == "small_archive":
        return replace(
            base,
            capacity=base.recent_capacity
            + min(100, max(1, (base.capacity - base.recent_capacity) // 5)),
        )
    return base


def _atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        torch.save(value, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def _load(path):
    # Native phase diagrams and replay arrays require pickle. Paths are generated
    # by this runner; never use untrusted downloaded checkpoints here.
    return torch.load(path, map_location="cpu", weights_only=False)


def _rng_state():
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def pack_learner(learner):
    return dict(
        config=asdict(learner.config),
        goals=learner.goals,
        seen=learner.seen,
        state_dim=learner.network.obs_dim,
        action_dim=learner.network.action_dim,
        network=learner.network.state_dict(),
        target=learner.target.state_dict(),
        optimizer=learner.optimizer.state_dict(),
        regularizer=learner.regularizer.state_dict(),
        recent=learner.memory.recent,
        groups=learner.memory.groups,
        admission=learner.memory.admission,
        rng=learner.rng.bit_generator.state,
        exploration_rng=learner.exploration_rng.bit_generator.state,
        regularizer_rng=learner.regularizer_rng.bit_generator.state,
    )


def unpack_learner(state, device, *, evaluation=False):
    learner = CandidateLearner(
        state["state_dim"],
        state["action_dim"],
        state["goals"],
        CandidateConfig(**state["config"]),
        0,
        device,
    )
    learner.network.load_state_dict(state["network"])
    learner.target.load_state_dict(state["target"])
    if not evaluation:
        learner.optimizer.load_state_dict(state["optimizer"])
    learner.regularizer.projection = state["regularizer"]["projection"].to(device)
    learner.regularizer.load_state_dict(state["regularizer"])
    learner.seen = state["seen"]
    if not evaluation:
        learner.memory.recent, learner.memory.groups = state["recent"], state["groups"]
        learner.memory.admission = state["admission"]
    for name in ("rng", "exploration_rng", "regularizer_rng"):
        getattr(learner, name).bit_generator.state = state[name]
    return learner


@contextmanager
def _lock(output):
    """Prevent two kernels writing the same study. OS releases the lock after a crash."""
    import fcntl

    with (output / ".run.lock").open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is running this study directory") from error
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def _manifest(output, specification, resume):
    packages = {}
    for name in ("torch", "numpy", "made", "orb-models", "pymatgen", "smact", "ase", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    package = Path(__file__).resolve().parents[1]
    sources = [
        "benchmarks/study.py",
        "benchmarks/candidates.py",
        "benchmarks/made.py",
        "models/dqn.py",
        "losses.py",
    ]
    source_hashes = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sources
    }
    specification["source_sha256"] = source_hashes
    specification = json.loads(
        json.dumps(dict(protocol=PROTOCOL, packages=packages, **specification))
    )
    path = output / "study.json"
    if path.exists():
        if not resume:
            raise FileExistsError("Study exists; use resume=True or choose a new run directory")
        if json.loads(path.read_text()) != specification:
            raise ValueError(
                "Study configuration differs from saved manifest; use a new run directory"
            )
    else:
        if any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("Refusing to use a nonempty directory without a study manifest")
        _json(path, specification)
    return specification


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _representation(learner):
    # A separate deterministic diagnostic sample never advances training RNGs.
    pool = learner.memory.recent + learner.memory.archive
    if not pool:
        return {}
    indices = np.random.default_rng(0).choice(len(pool), min(128, len(pool)), replace=False)
    rows = [pool[i] for i in indices]
    with torch.no_grad():
        phi = (
            learner.network.encode_state_action(
                learner.tensor([r.state for r in rows]), learner.tensor([r.action for r in rows])
            )
            .cpu()
            .numpy()
        )
    centered = phi - phi.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / max(1, len(phi) - 1))
    return dict(
        covariance_eigenvalues=np.maximum(eigenvalues, 0)[::-1].tolist(),
        covariance_target_variance=learner.config.covariance_target_variance,
        sample_count=len(rows),
    )


def _unit(
    tasks,
    task_id,
    factory,
    path,
    *,
    config,
    seed,
    method,
    device,
    learner=None,
    learn=True,
    phase="training",
    campaigns=None,
    on_task_end=None,
    keep_campaign_checkpoints=False,
):
    """One task, resumable at native campaign boundaries."""
    path.mkdir(parents=True, exist_ok=True)
    complete, checkpoint = path / "complete.pt", path / "resume.pt"
    if complete.exists():
        saved = _load(complete)
        record = saved["record"]
        record["path"] = str(path)
        print(f"Reusing completed {phase}: {method} {record['task']} seed={seed}", flush=True)
        if on_task_end is not None and hasattr(on_task_end, "show_cached"):
            on_task_end.show_cached(record)
        return record, unpack_learner(saved["learner"], device)
    task = tasks[task_id]
    task_seed = seed + 1009 * task_id
    _seed(task_seed)
    started = time.perf_counter()
    elapsed, losses, history = 0.0, [], []
    saved = _load(checkpoint) if checkpoint.exists() else None
    env = factory(
        task, task_seed, checkpoint=saved["environment"] if saved else None, campaigns=campaigns
    )
    if saved:
        learner = unpack_learner(saved["learner"], device)
        history, losses, elapsed = saved["history"], saved["losses"], saved["seconds"]
        _restore_rng(saved["global_rng"])
        print(f"Resuming {method} {task['name']} after {len(history)} proposals", flush=True)
        if not env.finished:
            state, candidates = env.reset()
    else:
        state, candidates = env.reset()
        if learner is None:
            goals = {i: np.asarray(t["goal"], np.float32) for i, t in enumerate(tasks)}
            learner = CandidateLearner(len(state), candidates.shape[1], goals, config, seed, device)
        learner.reset_rng(task_seed)
        if learn:
            learner.memory.begin_task()
            learner.seen[task_id] = learner.goals[task_id]
    print(f"{phase}: {METHODS[method]} · {task['name']} · seed={seed}", flush=True)
    step = len(history)
    campaign_start = step
    transfer = method not in ("scratch", "fine_tuning")
    # The committed checkpoint is authoritative. Remove partial text-log rows
    # from an interrupted campaign before replaying that campaign.
    with (path / "training.jsonl").open("w") as log:
        for row in history:
            log.write(json.dumps(row) + "\n")
        while not env.finished:
            epsilon = (
                (
                    config.epsilon_start
                    + min(env.progress * 3, 1) * (config.epsilon_end - config.epsilon_start)
                )
                if learn
                else 0.0
            )
            index = learner.choose(state, candidates, task_id, epsilon)
            with torch.no_grad():
                predicted = float(
                    learner.values(learner.network, state, candidates[index : index + 1], task_id)[
                        0
                    ]
                )
            next_state, next_actions, reward, done = env.step(index)
            step += 1
            if learn:
                learner.memory.add(
                    Transition(
                        task_id,
                        state.copy(),
                        candidates[index].copy(),
                        float(reward),
                        next_state.copy(),
                        next_actions.copy(),
                        bool(done),
                    )
                )
                if step >= config.warmup and step % config.train_frequency == 0:
                    losses.append(dict(step=step, **learner.update(transfer)))
            row = dict(
                step=step,
                reward=float(reward),
                epsilon=epsilon,
                predicted_q=predicted,
                **env.metrics(),
            )
            history.append(row)
            log.write(json.dumps(row) + "\n")
            if step % config.log_frequency == 0 or done:
                log.flush()
                print(
                    f"  step={step:,} campaign={row['campaign']} discoveries={row['discoveries']} "
                    f"oracle={row['oracle_evaluations']}",
                    flush=True,
                )
            state, candidates = next_state, next_actions
            if done:
                value = 0.0
                for entry in reversed(history[campaign_start:]):
                    value = entry["reward"] + config.gamma * value
                    entry["realised_return"] = value
                campaign_start = step
                # Finalise realised returns in the per-step log as well as the
                # checkpoint; on restart the checkpoint remains authoritative.
                log.seek(0)
                log.truncate()
                for entry in history:
                    log.write(json.dumps(entry) + "\n")
                log.flush()
                payload = dict(
                    learner=pack_learner(learner),
                    environment=env.checkpoint(),
                    history=history,
                    losses=losses,
                    global_rng=_rng_state(),
                    seconds=elapsed + time.perf_counter() - started,
                )
                if keep_campaign_checkpoints:
                    _atomic_torch(path / f"campaign_{env.campaign}.pt", payload)
                _atomic_torch(checkpoint, payload)
                print(f"  Saved campaign {env.campaign}; safe to resume here", flush=True)
                if not env.finished:
                    state, candidates = env.reset()
    representation = _representation(learner) if learn else {}
    if learn and transfer and phase == "training":
        learner.memory.finish_task(task_id, learner.network, learner.goals, device)
    env.save(path)
    record = dict(
        seed=seed,
        mode="scratch" if method == "scratch" else "sequential",
        method=method,
        method_label=METHODS[method],
        phase=phase,
        task_id=task_id,
        task=task["name"],
        steps=step,
        seconds=elapsed + time.perf_counter() - started,
        replay_memory=learner.memory.stats(),
        final=env.metrics(),
        history=history,
        losses=losses,
        representation=representation,
        path=str(path),
    )
    # Retain a lightweight policy artifact compatible with existing analysis.
    torch.save(
        dict(
            network=learner.network.state_dict(),
            target=learner.target.state_dict(),
            task_goals={k: np.asarray(v).tolist() for k, v in learner.goals.items()},
            state_dim=learner.network.obs_dim,
            action_dim=learner.network.action_dim,
            config=asdict(config),
        ),
        path / "policy.pt",
    )
    _json(path / "results.json", record)
    _atomic_torch(complete, dict(record=record, learner=pack_learner(learner)))
    checkpoint.unlink(missing_ok=True)
    if on_task_end:
        on_task_end(record=record, environment=env, learner=learner)
    return record, learner


def run_study(
    tasks,
    make_environment,
    output_dir,
    *,
    config=None,
    seeds=(42, 123, 456),
    methods=("scratch", "sequential"),
    device="cpu",
    resume=True,
    metadata=None,
    on_task_end=None,
):
    """Run/continue a fixed manifest. Factory accepts checkpoint= and campaigns=."""
    if (
        not tasks
        or not seeds
        or not methods
        or len(set(methods)) != len(methods)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Use nonempty tasks and unique methods/seeds")
    config = config or made_study_config()
    configs = {m: method_config(config, m) for m in methods}
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _lock(output):
        _manifest(
            output,
            dict(
                tasks=tasks,
                seeds=list(seeds),
                methods=list(methods),
                device=str(device),
                configs={m: asdict(c) for m, c in configs.items()},
                metadata=metadata or {},
            ),
            resume,
        )
        records = []
        for seed in seeds:
            for method in methods:
                learner = None
                for task_id in range(len(tasks)):
                    path = output / f"seed_{seed}" / method / f"task_{task_id}"
                    record, learner = _unit(
                        tasks,
                        task_id,
                        make_environment,
                        path,
                        config=configs[method],
                        seed=seed,
                        method=method,
                        device=device,
                        learner=None if method == "scratch" else learner,
                        on_task_end=on_task_end,
                    )
                    records.append(record)
                    _json(output / "summary.json", records)
    return records


def _stage_path(output, seed, method, task):
    return output / f"seed_{seed}" / method / f"task_{task}" / "complete.pt"


def _evaluation_manifest(path, parent, *, campaigns, recovery_campaigns=None):
    # Bind evaluations to the exact parent protocol. No accidental mixing after edits.
    parent_hash = hashlib.sha256((parent / "study.json").read_bytes()).hexdigest()
    _manifest(
        path,
        dict(
            parent_manifest_sha256=parent_hash,
            evaluation_campaigns=campaigns,
            recovery_campaigns=recovery_campaigns,
            evaluation_seed_offset=10_000_000,
            recovery_seed_offset=20_000_000,
        ),
        True,
    )


def run_retention(output_dir, make_environment, *, campaigns=2, on_task_end=None):
    """Frozen greedy held-out evaluation after every acquired task, never training."""
    if campaigns < 1:
        raise ValueError("campaigns must be positive")
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "study.json").read_text())
    path = output / "retention"
    path.mkdir(exist_ok=True)
    records = []
    with _lock(output), _lock(path):
        _evaluation_manifest(path, output, campaigns=campaigns)
        for seed in manifest["seeds"]:
            for method in manifest["methods"]:
                for stage in range(len(manifest["tasks"])):
                    source = _stage_path(output, seed, method, stage)
                    if not source.exists():
                        continue
                    for task_id in [stage] if method == "scratch" else range(stage + 1):
                        learner = unpack_learner(
                            _load(source)["learner"], manifest["device"], evaluation=True
                        )
                        record, _ = _unit(
                            manifest["tasks"],
                            task_id,
                            make_environment,
                            path / f"seed_{seed}" / method / f"after_{stage}" / f"task_{task_id}",
                            config=learner.config,
                            seed=seed + 10_000_000,
                            method=method,
                            device=manifest["device"],
                            learner=learner,
                            learn=False,
                            phase="retention",
                            campaigns=campaigns,
                            on_task_end=on_task_end,
                        )
                        records.append(dict(record, seed=seed, after_task=stage))
                        _json(path / "summary.json", records)
    return records


def run_recovery(
    output_dir, make_environment, *, campaigns=2, evaluation_campaigns=2, on_task_end=None
):
    """Independent revisits from final checkpoints; frozen all-goal probes at each boundary.

    Training a recovery branch never overwrites the main sequence or another branch.
    Every probe uses the retention seeds, so comparisons against post-acquisition
    performance have matched evaluation campaigns.
    """
    if min(campaigns, evaluation_campaigns) < 1:
        raise ValueError("campaigns must be positive")
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "study.json").read_text())
    path = output / "recovery"
    path.mkdir(exist_ok=True)
    tasks = manifest["tasks"]
    retention_manifest = output / "retention" / "study.json"
    if retention_manifest.exists():
        retention_settings = json.loads(retention_manifest.read_text())
        if retention_settings["evaluation_campaigns"] != evaluation_campaigns:
            raise ValueError("Recovery probes must use the same campaign count as retention")
    records, probes = [], []
    with _lock(output), _lock(path):
        _evaluation_manifest(
            path, output, campaigns=evaluation_campaigns, recovery_campaigns=campaigns
        )
        for seed in manifest["seeds"]:
            for method in manifest["methods"]:
                if method == "scratch":
                    continue
                source = _stage_path(output, seed, method, len(tasks) - 1)
                if not source.exists():
                    continue
                for recovered in range(len(tasks) - 1):
                    branch = path / f"seed_{seed}" / method / f"recover_{recovered}"
                    learner = unpack_learner(_load(source)["learner"], manifest["device"])
                    record, _ = _unit(
                        tasks,
                        recovered,
                        make_environment,
                        branch / "training",
                        config=learner.config,
                        seed=seed + 20_000_000,
                        method=method,
                        device=manifest["device"],
                        learner=learner,
                        phase="recovery",
                        campaigns=campaigns,
                        on_task_end=on_task_end,
                        keep_campaign_checkpoints=True,
                    )
                    records.append(dict(record, seed=seed, recovered_task=recovered))
                    _json(path / "summary.json", records)
                    for boundary in range(campaigns + 1):
                        checkpoint = (
                            source
                            if boundary == 0
                            else branch / "training" / f"campaign_{boundary}.pt"
                        )
                        saved = _load(checkpoint)
                        steps = 0 if boundary == 0 else len(saved["history"])
                        for task_id in range(len(tasks)):
                            learner = unpack_learner(
                                saved["learner"], manifest["device"], evaluation=True
                            )
                            probe, _ = _unit(
                                tasks,
                                task_id,
                                make_environment,
                                branch / f"probe_{boundary}" / f"task_{task_id}",
                                config=learner.config,
                                seed=seed + 10_000_000,
                                method=method,
                                device=manifest["device"],
                                learner=learner,
                                learn=False,
                                phase="recovery_probe",
                                campaigns=evaluation_campaigns,
                            )
                            probes.append(
                                dict(
                                    probe, seed=seed, recovered_task=recovered, recovery_steps=steps
                                )
                            )
                            _json(path / "probes.json", probes)
    return records, probes
