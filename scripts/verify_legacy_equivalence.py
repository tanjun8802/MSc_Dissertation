"""Compare original models/trainers against their refactored implementations.

Includes an all-penalty fixture and all three notebook network/loss settings,
with shortened budgets and evaluation interleaved with optimizer updates.

Run from a full Git checkout: python scripts/verify_legacy_equivalence.py.
The original source is read from the pinned pre-refactor commit, never imported
from the working tree. This validates update equivalence, not learning curves.
"""

import ast
import copy
import json
import random
import subprocess
import time
import typing
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from tbtrl.environments.registry import make_environment
from tbtrl.evaluation import evaluate_policy, evaluate_policy_with_success
from tbtrl.experiments.config import load_config, training_settings
from tbtrl.models.dqn import FactorisedQNetwork
from tbtrl.models.sac import FactorisedTwinCritic, GaussianActor
from tbtrl.normalization import RunningMeanStd
from tbtrl.random import set_seed
from tbtrl.replay import ReplayBuffer
from tbtrl.training.dqn import train_dqn
from tbtrl.training.sac import train_sac

revision = "0606a8f664974ecd8af2785fc575d20785f17b24"
sources = {
    name: subprocess.check_output(["git", "show", f"{revision}:src/{name}"], text=True)
    for name in [
        "utils.py",
        "trainer.py",
        "loss_functions.py",
        "networks.py",
        "gym_robotics_networks.py",
    ]
}
ns = dict(
    np=np,
    torch=torch,
    nn=nn,
    optim=optim,
    F=F,
    time=time,
    random=random,
    Optional=typing.Optional,
    Dict=typing.Dict,
    Any=typing.Any,
    RunningMeanStd=RunningMeanStd,
    dataclass=dataclass,
    chain=chain,
    Tuple=typing.Tuple,
    List=typing.List,
    Sequence=typing.Sequence,
)
ns["TrajectoryReplayBufferDiscrete"] = lambda *a, **k: ReplayBuffer(*a, **k, discrete=True)
ns["TrajectoryReplayBufferContinuous"] = ReplayBuffer


def definitions(file, names):
    tree = ast.parse(sources[file])
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{revision}:{file}", "exec"), ns)


definitions(
    "loss_functions.py", ["sigreg_loss", "online_goal_separation_loss", "norm_penalty_loss_l1"]
)
definitions(
    "utils.py",
    [
        "ReplayBatch",
        "TrajectoryReplayBuffer",
        "TrajectoryReplayBufferDiscrete",
        "TrajectoryReplayBufferContinuous",
        "set_seed",
        "extract_mean_sa_embedding",
        "extract_fixed_probe_sa_embedding",
        "extract_sa_batch_for_isotropy",
        "inspect_raw_phi_norms",
    ],
)
definitions("trainer.py", ["dqn_train_phi_psi_adjustment", "sac_train_tbtrl_maze"])
definitions("networks.py", ["project_to_l2_ball", "FactorisedDQN_QNetwork_BallNorm"])
definitions("gym_robotics_networks.py", ["FactorisedTwinCriticFetch", "GaussianPolicyActorSAC"])
report = {}
cases = [(algo + "_all_penalties", algo, None) for algo in ("dqn", "sac")]
for path in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.json")):
    config = load_config(path)
    if config.actor_type != "mlp":
        continue  # Preserve parity checks for the original algorithms.
    cases.append((config.name, config.algorithm, config))
for label, algo, config in cases:

    def env_factory(goal):
        env = make_environment(
            config.environment
            if config
            else ("maze-discrete" if algo == "dqn" else "maze-continuous"),
            goal,
            max_horizon=100,
        )
        reset = env.reset
        first = True

        def first_seed(*, seed=None, options=None):
            nonlocal first
            if first:
                seed = 9
                first = False
            return reset(seed=seed, options=options)

        env.reset = first_seed
        env.action_space.seed(9)
        return env

    def components(legacy):
        set_seed(44)
        if algo == "dqn":
            cls = ns["FactorisedDQN_QNetwork_BallNorm"] if legacy else FactorisedQNetwork
            return cls(
                2,
                4,
                **(
                    config.model
                    if config
                    else dict(hidden_dim=8, rep_dim=4, phi_max_norm=1.0, psi_max_norm=1.0)
                ),
            ), None
        cls = ns["FactorisedTwinCriticFetch"] if legacy else FactorisedTwinCritic
        actor_cls = ns["GaussianPolicyActorSAC"] if legacy else GaussianActor
        options = config.model if config else dict(hidden_dim=8, latent_dim=4)
        return cls(2, 2, 2, **options), actor_cls(2, 2, 2, net_arch=(options["hidden_dim"],) * 2)

    model, actor = components(False)
    legacy_model, legacy_actor = components(True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, legacy_model.state_dict()[key], rtol=0, atol=0)
    # Keep old-task batches eligible even at the original notebook batch sizes.
    old_capacity = 512
    oldbuf = ReplayBuffer(old_capacity, 2, 1 if algo == "dqn" else 2, discrete=algo == "dqn")
    for _ in range(old_capacity):
        oldbuf.add_transition([1, 1], 3 if algo == "dqn" else [0.1, 0.2], 0, [2, 1], False)
    outcomes = []
    for legacy in [True, False]:
        replay = (
            ns[
                "TrajectoryReplayBufferDiscrete"
                if algo == "dqn"
                else "TrajectoryReplayBufferContinuous"
            ](old_capacity, 2, 1 if algo == "dqn" else 2)
            if legacy
            else oldbuf
        )
        if legacy:
            for i in range(old_capacity):
                replay.add_transition(
                    oldbuf.obs[i],
                    int(oldbuf.actions[i].item()) if algo == "dqn" else oldbuf.actions[i],
                    0.0,
                    oldbuf.next_obs[i],
                    False,
                )
        q = copy.deepcopy(legacy_model if legacy else model)
        t = copy.deepcopy(q)
        a = copy.deepcopy(legacy_actor if legacy else actor)
        goal = [3, 1] if algo == "dqn" else [3.5, 1.5]
        kw = dict(
            seed=9,
            env=env_factory(goal),
            buffer_capacity=32,
            total_steps=8,
            warmup_steps=2,
            batch_size=4,
            eval_freq=100,
            train_freq=1,
            goal=goal,
            task_id=1,
            make_env=env_factory,
            task_goals={0: np.array([1.0, 1.0])},
            replay_task_buffers={0: replay},
            replay_ratio=1.0,
            replay_loss_coef=0.3,
            sigreg_coef=0.01,
            sketch_dim=4,
            goal_separation_coef=0.04,
            goal_separation_target_cosine=-1.0,
            phi_raw_norm_coef=0.1,
            psi_raw_norm_coef=0.1,
            enable_early_stop=False,
        )
        if config:
            settings = training_settings(config, "sequential", 1)
            settings.pop("eval_episodes")
            kw.update(settings)
            kw.update(
                total_steps=36,
                warmup_steps=2,
                buffer_capacity=64,
                eval_freq=12,
                enable_early_stop=False,
                goal=np.asarray(config.goals[1], dtype=np.float32),
                task_goals={0: np.asarray(config.goals[0], dtype=np.float32)},
            )
        eval_seeds = iter([9 + 100000 + step for step in (12, 24, 36)])
        ns["evaluate_policy"] = lambda env, policy, episodes: evaluate_policy(
            env, policy, episodes, seed=next(eval_seeds)
        )
        ns["evaluate_policy_with_success"] = evaluate_policy_with_success
        if algo == "dqn":
            fn = ns["dqn_train_phi_psi_adjustment"] if legacy else train_dqn
            result = fn(q_network=q, q_target_network=t, **kw)
        else:
            fn = ns["sac_train_tbtrl_maze"] if legacy else train_sac
            if not config:
                kw.update(phi_norm_target=0.01, psi_norm_target=0.01)
            result = fn(actor=a, critic=q, critic_target=t, **kw)
        curves = result[2 if algo == "dqn" else 3] if legacy else result.evaluations
        if legacy:
            legacy_curves = curves
        else:
            assert curves == legacy_curves, (label, "evaluation mismatch")
            assert result.losses and all(row["replay_tasks"] > 0 for row in result.losses)
        outcomes.append([copy.deepcopy(m.state_dict()) for m in [q, t] + ([a] if a else [])])
    deltas = []
    for left, right in zip(*outcomes):
        for key in left:
            deltas.append((left[key] - right[key]).abs().max().item())
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    report[label] = dict(
        environment_steps=kw["total_steps"],
        evaluation_points=len(curves),
        model_source="original vs refactored",
        old_replay=True,
        regularizers=True,
        max_parameter_difference=max(deltas),
    )
print(json.dumps(report, indent=2))
