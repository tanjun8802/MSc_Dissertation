"""Compare seven real TBTRL updates against the original working trainers.

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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from tbtrl.environments.registry import make_environment
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
    for name in ["utils.py", "trainer.py", "loss_functions.py"]
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
report = {}
for algo in ["dqn", "sac"]:

    def env_factory(goal):
        env = make_environment(
            "maze-discrete" if algo == "dqn" else "maze-continuous", goal, max_horizon=100
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

    set_seed(44)
    if algo == "dqn":
        model = FactorisedQNetwork(
            2, 4, hidden_dim=8, rep_dim=4, phi_max_norm=1.0, psi_max_norm=1.0
        )
        actor = None
    else:
        model = FactorisedTwinCritic(2, 2, 2, hidden_dim=8, latent_dim=4)
        actor = GaussianActor(2, 2, 2, net_arch=(8, 8))
    oldbuf = ReplayBuffer(16, 2, 1 if algo == "dqn" else 2, discrete=algo == "dqn")
    for _ in range(16):
        oldbuf.add_transition([1, 1], 3 if algo == "dqn" else [0.1, 0.2], 0, [2, 1], False)
    outcomes = []
    for legacy in [True, False]:
        replay = (
            ns[
                "TrajectoryReplayBufferDiscrete"
                if algo == "dqn"
                else "TrajectoryReplayBufferContinuous"
            ](16, 2, 1 if algo == "dqn" else 2)
            if legacy
            else oldbuf
        )
        if legacy:
            for i in range(16):
                replay.add_transition(
                    oldbuf.obs[i],
                    int(oldbuf.actions[i].item()) if algo == "dqn" else oldbuf.actions[i],
                    0.0,
                    oldbuf.next_obs[i],
                    False,
                )
        q = copy.deepcopy(model)
        t = copy.deepcopy(model)
        a = copy.deepcopy(actor)
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
        if algo == "dqn":
            fn = ns["dqn_train_phi_psi_adjustment"] if legacy else train_dqn
            fn(q_network=q, q_target_network=t, **kw)
        else:
            fn = ns["sac_train_tbtrl_maze"] if legacy else train_sac
            fn(actor=a, critic=q, critic_target=t, phi_norm_target=0.01, psi_norm_target=0.01, **kw)
        outcomes.append([copy.deepcopy(m.state_dict()) for m in [q, t] + ([a] if a else [])])
    deltas = []
    for left, right in zip(*outcomes):
        for key in left:
            deltas.append((left[key] - right[key]).abs().max().item())
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    report[algo] = dict(
        environment_steps=8,
        gradient_updates=7,
        old_replay=True,
        regularizers=True,
        max_parameter_difference=max(deltas),
    )
print(json.dumps(report, indent=2))
