"""TBTRL for variable candidate sets, without querying an oracle during replay.

The fixed-action maze trainer is intentionally separate. This implementation
reuses its factorisation, covariance/separation/norm losses and moving TD targets,
but maximises only over the candidate set observed at the next state.
"""

import json
import math
import random
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tbtrl.losses import CovarianceRegularizer, norm_penalty_loss_l1, online_goal_separation_loss
from tbtrl.models.dqn import FactorisedQNetwork


@dataclass(frozen=True)
class CandidateConfig:
    hidden_dim: int = 128
    rep_dim: int = 64
    phi_max_norm: float = 8.0
    psi_max_norm: float = 8.0
    gamma: float = 1.0
    lr: float = 1e-4
    batch_size: int = 32
    warmup: int = 32
    train_frequency: int = 4
    tau: float = 0.01
    replay_coef: float = 1.0
    covariance_coef: float = 0.01
    separation_coef: float = 0.01
    norm_coef: float = 0.001
    covariance_target_variance: float = 1.0
    separation_target_cosine: float = 0.85
    regularizer_batch_size: int | None = None
    gradient_diagnostics: bool = False
    independent_rng: bool = False
    balanced_replay: bool = False
    sample_without_replacement: bool = False
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    capacity: int = 500
    recent_capacity: int = 250
    byte_budget: int = 64 * 1024**2
    keep_fraction: float = 0.4
    similarity_threshold: float = 0.8
    max_groups: int = 16
    archive_strategy: str = "similarity_admission"
    log_frequency: int = 100

    def __post_init__(self):
        for key in (
            "hidden_dim",
            "rep_dim",
            "batch_size",
            "warmup",
            "train_frequency",
            "capacity",
            "recent_capacity",
            "byte_budget",
            "max_groups",
            "log_frequency",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if not self.warmup <= self.recent_capacity < self.capacity:
            raise ValueError("Require warmup <= recent_capacity < capacity")
        if not 0 < self.keep_fraction <= 1 or not -1 <= self.similarity_threshold < 1:
            raise ValueError("Invalid archive admission settings")
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1 or self.lr <= 0:
            raise ValueError("Invalid learning settings")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Invalid exploration schedule")
        for key in ("replay_coef", "covariance_coef", "separation_coef", "norm_coef"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if (
            not math.isfinite(self.covariance_target_variance)
            or self.covariance_target_variance <= 0
        ):
            raise ValueError("covariance_target_variance must be finite and positive")
        if not -1 <= self.separation_target_cosine < 1:
            raise ValueError("Invalid separation_target_cosine")
        if self.regularizer_batch_size is not None and self.regularizer_batch_size < 2:
            raise ValueError("regularizer_batch_size must be at least two")
        if self.archive_strategy not in ("similarity_admission", "capacity_only"):
            raise ValueError("Unknown archive_strategy")


@dataclass(frozen=True)
class Transition:
    task: int
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    next_actions: np.ndarray
    done: bool

    @property
    def nbytes(self):
        return sum(x.nbytes for x in (self.state, self.action, self.next_state, self.next_actions))


class CandidateMemory:
    """Bounded recent + archive pool retaining ORIGINAL task labels and rewards.

    Candidate sets are ragged, so both sample count and numpy payload bytes are
    capped. Python bookkeeping is bounded by sample/group counts; task metadata
    and benchmark caches are reported separately. Similarity affects allocation,
    never reward relabelling across molecular objectives or chemical systems.
    """

    def __init__(self, config):
        self.config = config
        self.recent = []
        self.groups = {}
        self.admission = {}

    def begin_task(self):
        self.recent.clear()

    def add(self, transition):
        # Reserve half the payload budget for recent experience and half for archive.
        limit = self.config.byte_budget // 2
        if transition.nbytes > limit:
            raise ValueError(
                "A candidate set exceeds the replay byte budget; increase byte_budget."
            )
        self.recent.append(transition)
        while (
            len(self.recent) > self.config.recent_capacity
            or sum(t.nbytes for t in self.recent) > limit
        ):
            self.recent.pop(0)

    @property
    def archive(self):
        return [t for group in self.groups.values() for t in group]

    def finish_task(self, task, network, goals, device):
        if not self.recent:
            raise ValueError("Cannot archive an empty task")
        count = max(1, int(len(self.recent) * self.config.keep_fraction))
        incoming = self.recent[-count:]
        reps = list(self.groups)
        with torch.no_grad():
            embeddings = network.encode_goal(
                torch.as_tensor(
                    np.asarray([goals[t] for t in reps + [task]]),
                    device=device,
                    dtype=torch.float32,
                )
            )
            embeddings = F.normalize(embeddings, dim=-1)
            cosine = (embeddings @ embeddings.T).cpu().numpy()
        group, similarity, merged = task, None, None
        if reps:
            nearest = int(np.argmax(cosine[-1, :-1]))
            similarity = float(cosine[-1, nearest])
            if similarity > self.config.similarity_threshold:
                group = reps[nearest]
                if self.config.archive_strategy == "similarity_admission":
                    # Historical PMO setting. MADE now groups similar tasks without
                    # dropping their samples until the archive actually exceeds its budget.
                    fraction = (1 - min(similarity, 1)) / (1 - self.config.similarity_threshold)
                    incoming = incoming[-max(1, int(count * fraction)) :]
            elif len(reps) >= self.config.max_groups and len(reps) > 1:
                left, right = np.triu_indices(len(reps), 1)
                index = int(np.argmax(cosine[left, right]))
                keep, remove = reps[left[index]], reps[right[index]]
                self.groups[keep].extend(self.groups.pop(remove))
                merged = [keep, remove]
            elif len(reps) >= self.config.max_groups:
                group = reps[nearest]
        self.groups.setdefault(group, []).extend(incoming)
        archive_capacity = self.config.capacity - self.config.recent_capacity
        evicted = 0
        while (
            sum(map(len, self.groups.values())) > archive_capacity
            or sum(t.nbytes for t in self.archive) > self.config.byte_budget // 2
        ):
            # Thin the largest group; keep original task IDs even after a group merge.
            largest = max(self.groups, key=lambda g: sum(t.nbytes for t in self.groups[g]))
            index = 0
            if self.config.archive_strategy == "capacity_only":
                # A merged group must not let its first task monopolise its allocation.
                counts = Counter(t.task for t in self.groups[largest])
                overrepresented = max(counts, key=counts.get)
                index = next(
                    i for i, t in enumerate(self.groups[largest]) if t.task == overrepresented
                )
            self.groups[largest].pop(index)
            evicted += 1
            if not self.groups[largest]:
                del self.groups[largest]
        self.admission = dict(
            task=task,
            group=group,
            nearest_similarity=similarity,
            tail_candidates=count,
            admitted_candidates=len(incoming),
            merged_groups=merged,
            evicted_for_capacity=evicted,
            retained_incoming=sum(t.task == task for t in self.archive),
        )
        self.recent.clear()  # completed individual buffer is discarded

    def stats(self):
        archive = self.archive
        return dict(
            capacity=self.config.capacity,
            byte_budget=self.config.byte_budget,
            recent_byte_budget=self.config.byte_budget // 2,
            archive_byte_budget=self.config.byte_budget // 2,
            archive_capacity=self.config.capacity - self.config.recent_capacity,
            archive_strategy=self.config.archive_strategy,
            recent_samples=len(self.recent),
            archive_samples=len(archive),
            payload_bytes=sum(t.nbytes for t in self.recent + archive),
            group_sizes={str(k): len(v) for k, v in self.groups.items()},
            original_task_counts=dict(Counter(t.task for t in archive)),
            admission=self.admission,
        )


class CandidateLearner:
    def __init__(self, state_dim, action_dim, goals, config, seed, device="cpu"):
        self.config, self.goals, self.device = config, goals, device
        torch.manual_seed(seed)
        self.reset_rng(seed)
        self.network = FactorisedQNetwork(
            state_dim,
            action_dim,
            len(goals[0]),
            config.hidden_dim,
            config.rep_dim,
            config.phi_max_norm,
            config.psi_max_norm,
        ).to(device)
        self.target = deepcopy(self.network).eval()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr)
        self.regularizer = CovarianceRegularizer().to(device)
        self.memory = CandidateMemory(config)
        self.seen = {}

    def reset_rng(self, seed):
        """Opt-in independent exploration, replay and regularisation streams."""
        self.rng = np.random.default_rng(seed)
        streams = np.random.SeedSequence(seed).spawn(2)
        self.exploration_rng = np.random.default_rng(streams[0])
        self.regularizer_rng = np.random.default_rng(streams[1])

    def tensor(self, value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)

    def values(self, network, state, candidates, task):
        actions = self.tensor(candidates)
        states = self.tensor(state).unsqueeze(0).expand(len(actions), -1)
        return network(states, actions, self.tensor(self.goals[task])).flatten()

    def choose(self, state, candidates, task, epsilon):
        if len(candidates) == 0:
            raise ValueError("No valid candidates returned by benchmark")
        rng = self.exploration_rng if self.config.independent_rng else self.rng
        if rng.random() < epsilon:
            return int(rng.integers(len(candidates)))
        with torch.no_grad():
            return int(self.values(self.network, state, candidates, task).argmax())

    def td_loss(self, transitions):
        states = self.tensor([t.state for t in transitions])
        actions = self.tensor([t.action for t in transitions])
        goals = self.tensor([self.goals[t.task] for t in transitions])
        targets = self.tensor([t.reward for t in transitions])
        with torch.no_grad():
            for i, transition in enumerate(transitions):
                if not transition.done:
                    if not len(transition.next_actions):
                        raise ValueError("Nonterminal transition has no next candidates")
                    index = self.values(
                        self.network,
                        transition.next_state,
                        transition.next_actions,
                        transition.task,
                    ).argmax()
                    value = self.values(
                        self.target, transition.next_state, transition.next_actions, transition.task
                    )[index]
                    targets[i] += self.config.gamma * value
        q = self.network(states, actions, goals).flatten()
        return F.mse_loss(q, targets), states, actions

    def sample(self, pool, size, *, balanced=False, rng=None):
        """Balance original task labels, regardless of similarity-group membership."""
        rng = self.rng if rng is None else rng
        if not pool:
            raise ValueError("Cannot sample an empty replay pool")
        if balanced:
            by_task = {}
            for transition in pool:
                by_task.setdefault(transition.task, []).append(transition)
            ids = list(by_task)
            chosen = []
            while len(chosen) < size:
                chosen.extend(rng.permutation(ids).tolist())
            counts = Counter(chosen[:size])
            batch = [
                t
                for task, count in counts.items()
                for t in self.sample(by_task[task], count, rng=rng)
            ]
            rng.shuffle(batch)
            return batch
        if self.config.sample_without_replacement:
            indices = rng.choice(len(pool), size=size, replace=len(pool) < size)
        else:
            # Preserve the original PMO random stream unless explicitly opted in.
            indices = rng.integers(len(pool), size=size)
        return [pool[i] for i in indices]

    def update(self, transfer):
        config = self.config
        td, states, actions = self.td_loss(self.sample(self.memory.recent, config.batch_size))
        old = self.memory.archive
        replay = (
            self.td_loss(self.sample(old, config.batch_size, balanced=config.balanced_replay))[0]
            if transfer and old
            else td * 0
        )
        if config.regularizer_batch_size is not None:
            size = config.regularizer_batch_size
            historical_size = size // 2 if transfer and old else 0
            rng = self.regularizer_rng if config.independent_rng else self.rng
            samples = self.sample(self.memory.recent, size - historical_size, rng=rng)
            if historical_size:
                samples += self.sample(
                    old, historical_size, balanced=config.balanced_replay, rng=rng
                )
            states = self.tensor([t.state for t in samples])
            actions = self.tensor([t.action for t in samples])
        phi = self.network.encode_state_action(states, actions)
        # Normalise by a stated variance target rather than demanding 64 units
        # of centred squared norm from representations that initially have little spread.
        covariance = self.regularizer(phi / math.sqrt(config.covariance_target_variance))
        separation, cosines = online_goal_separation_loss(
            self.network, self.seen, self.device, target_cosine=config.separation_target_cosine
        )
        raw_phi = self.network.sa_encoder(torch.cat([states, actions], dim=-1))
        raw_psi = self.network.goal_encoder(self.tensor(list(self.seen.values())))
        norm = norm_penalty_loss_l1(raw_phi, config.phi_max_norm - 0.1)
        norm = norm + norm_penalty_loss_l1(raw_psi, config.psi_max_norm - 0.1)
        components = dict(
            td=td,
            replay=config.replay_coef * replay,
            covariance=config.covariance_coef * covariance,
            separation=config.separation_coef * separation,
            norm=config.norm_coef * norm,
        )
        loss = sum(components.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite candidate TBTRL loss")
        diagnostics = {}
        if config.gradient_diagnostics:
            parameters = tuple(self.network.parameters())
            gradients = {}
            for name, component in components.items():
                grads = (
                    torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True)
                    if component.requires_grad
                    else (None,) * len(parameters)
                )
                vector = torch.cat(
                    [
                        (torch.zeros_like(p) if g is None else g).detach().flatten()
                        for p, g in zip(parameters, grads)
                    ]
                )
                gradients[name] = vector
                diagnostics[f"grad_{name}"] = vector.norm().item()
            diagnostics["td_replay_grad_cosine"] = F.cosine_similarity(
                gradients["td"], gradients["replay"], dim=0
            ).item()
        with torch.no_grad():
            diagnostics.update(
                phi_std_mean=phi.std(dim=0, unbiased=False).mean().item(),
                phi_norm_mean=phi.norm(dim=-1).mean().item(),
                psi_norm_mean=raw_psi.norm(dim=-1).mean().item(),
                task_cosine_max=(
                    cosines[torch.triu(torch.ones_like(cosines), diagonal=1).bool()].max().item()
                    if len(self.seen) > 1
                    else 0.0
                ),
            )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.network.parameters()):
                target.lerp_(online, config.tau)
        return diagnostics | {
            k: float(v.detach())
            for k, v in dict(
                td=td,
                replay=replay,
                covariance=covariance,
                separation=separation,
                norm=norm,
                total=loss,
                **{f"weighted_{k}": v for k, v in components.items()},
            ).items()
        }


def run_experiment(
    tasks,
    make_environment,
    output_dir,
    *,
    config=CandidateConfig(),
    seeds=(42, 123, 456),
    device="cpu",
    on_task_end=None,
    metadata=None,
):
    """Paired fresh vs sequential runs. Each factory call MUST create fresh oracle state.

    An environment exposes reset()->(state,candidate_features), step(index)->
    (state,candidates,reward,episode_done), finished, progress, metrics(), save(path).
    Candidate order must match the environment's stored proposals. Full-budget
    campaigns are online learning curves, not additional free evaluation runs.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    goals = {i: np.asarray(task["goal"], dtype=np.float32) for i, task in enumerate(tasks)}
    manifest = dict(
        tasks=tasks,
        config=asdict(config),
        seeds=list(seeds),
        device=str(device),
        modes=["scratch", "sequential"],
        metadata=metadata or {},
    )
    manifest["package_versions"] = {}
    for package in ("numpy", "torch", "rdkit", "PyTDC", "made", "orb-models", "pymatgen", "smact"):
        try:
            manifest["package_versions"][package] = version(package)
        except PackageNotFoundError:
            pass
    (output / "config.json").write_text(json.dumps(manifest, indent=2))
    records = []
    for seed in seeds:
        for mode in ("scratch", "sequential"):
            learner = None
            for task_id, task in enumerate(tasks):
                # Matched task-local RNGs; source tasks cannot advance target generators' RNGs.
                task_seed = seed + 1009 * task_id
                random.seed(task_seed)
                np.random.seed(task_seed)
                torch.manual_seed(task_seed)
                started = time.perf_counter()
                print(f"Initialising {task['name']} mode={mode} seed={seed}", flush=True)
                env = make_environment(task, task_seed)
                state, candidates = env.reset()
                if learner is None or mode == "scratch":
                    learner = CandidateLearner(
                        len(state), candidates.shape[1], goals, config, seed, device
                    )
                learner.reset_rng(task_seed)
                learner.memory.begin_task()
                learner.seen[task_id] = goals[task_id]
                path = output / f"seed_{seed}" / mode / f"task_{task_id}"
                path.mkdir(parents=True)
                step, losses, history = 0, [], []
                print(f"Training task={task_id} {task['name']} mode={mode} seed={seed}", flush=True)
                with (path / "training.jsonl").open("w") as log:
                    while not env.finished:
                        epsilon = config.epsilon_start + min(env.progress * 3, 1) * (
                            config.epsilon_end - config.epsilon_start
                        )
                        index = learner.choose(state, candidates, task_id, epsilon)
                        next_state, next_actions, reward, done = env.step(index)
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
                        step += 1
                        if step >= config.warmup and step % config.train_frequency == 0:
                            losses.append(dict(step=step, **learner.update(mode == "sequential")))
                        metrics = dict(
                            step=step, reward=float(reward), epsilon=epsilon, **env.metrics()
                        )
                        history.append(metrics)
                        log.write(json.dumps(metrics) + "\n")
                        if step % config.log_frequency == 0 or done or env.finished:
                            log.flush()
                            print(
                                f"  step={step:,} {env.progress:.1%} {env.metrics()} "
                                f"loss={losses[-1]['total'] if losses else 'warming up'}",
                                flush=True,
                            )
                        state, candidates = next_state, next_actions
                        if done and not env.finished:
                            state, candidates = env.reset()
                if mode == "sequential":
                    learner.memory.finish_task(task_id, learner.network, goals, device)
                env.save(path)
                torch.save(
                    dict(
                        network=learner.network.state_dict(),
                        target=learner.target.state_dict(),
                        task_goals={k: v.tolist() for k, v in goals.items()},
                        state_dim=len(state),
                        action_dim=candidates.shape[1],
                        config=asdict(config),
                    ),
                    path / "policy.pt",
                )
                record = dict(
                    seed=seed,
                    mode=mode,
                    task_id=task_id,
                    task=task["name"],
                    steps=step,
                    seconds=time.perf_counter() - started,
                    replay_memory=learner.memory.stats(),
                    final=env.metrics(),
                    history=history,
                    losses=losses,
                    path=str(path),
                )
                records.append(record)
                (path / "results.json").write_text(json.dumps(record, indent=2))
                (output / "summary.json").write_text(json.dumps(records, indent=2))
                if on_task_end:
                    on_task_end(record=record, environment=env, learner=learner)
                del env
    return records
