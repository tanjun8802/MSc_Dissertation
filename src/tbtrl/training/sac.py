import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from tbtrl.evaluation import evaluate_policy_with_success
from tbtrl.losses import CovarianceRegularizer
from tbtrl.normalization import RunningMeanStd
from tbtrl.random import set_seed
from tbtrl.replay import ReplayBuffer
from tbtrl.training.common import LossHistory, validate_training_options
from tbtrl.training.results import SACResult

logger = logging.getLogger(__name__)


def train_sac(
    seed: int = 42,
    actor: nn.Module = None,
    critic: nn.Module = None,
    critic_target: nn.Module = None,
    env=None,
    buffer_capacity: int = None,
    lr_actor: float = 0.0003,
    lr_critic: float = 0.0003,
    lr_ent_coef: float = 0.0003,
    obs_dim: int = None,
    action_dim: int = None,
    device: torch.device = None,
    total_steps: int = 300000,
    warmup_steps: int = 25000,
    batch_size: int = 256,
    eval_freq: int = 5000,
    gamma: float = 0.99,
    tau: float = 0.005,
    train_freq: int = 1,
    gradient_steps: int = 1,
    goal: np.ndarray = None,
    task_id: int = None,
    make_env=None,
    replay_task_buffers: Optional[Dict[int, Any]] = None,
    task_goals: Optional[Dict[int, np.ndarray]] = None,
    replay_ratio: float = 0.0,
    replay_tasks_per_batch: Optional[int] = None,
    replay_loss_coef: float = 1.0,
    bootstrap_on_truncation: bool = True,
    ent_coef: str | float = "auto",
    target_entropy: float | None = None,
    normalize_state_inputs: bool = False,
    normalize_goal_inputs: bool = False,
    obs_norm_clip: float = 10.0,
    early_stop_reward: float = -0.05,
    early_stop_success_rate: float | None = None,
    early_stop_patience: int = 5,
    eval_episodes=8,
    enable_early_stop: bool = True,
    sigreg_coef: float = 0.0,
    sketch_dim: int = 64,
    goal_separation_coef: float = 0.0,
    goal_separation_target_cosine: float = 0.85,
    phi_raw_norm_coef: float = 0.0,
    psi_raw_norm_coef: float = 0.0,
    phi_norm_target: float | None = None,
    psi_norm_target: float | None = None,
    initial_alpha: float = 1.0,
    reset_target_from_critic: bool = True,
    q_target_min: float | None = None,
    q_target_max: float | None = 5.0,
    critic_grad_clip_norm: float = 10.0,
    actor_grad_clip_norm: float = 10.0,
):
    """
    Factorised SAC-TBTRL trainer with independent critic branches.

    SAC bootstrap target:
        y = r + gamma * mask *
            [min(Q1_target(s', a', g), Q2_target(s', a', g))
             - alpha * log pi(a' | s', g)]

    Critic 1:
        L1 = TD1_current
           + replay_loss_coef * TD1_replay
           + sigreg_coef * SIGReg(phi1)
           + goal_separation_coef * GoalSeparation(psi1)
           + phi_raw_norm_coef * Norm(phi1)
           + psi_raw_norm_coef * Norm(psi1)

    Critic 2:
        L2 = TD2_current
           + replay_loss_coef * TD2_replay
           + sigreg_coef * SIGReg(phi2)
           + goal_separation_coef * GoalSeparation(psi2)
           + phi_raw_norm_coef * Norm(phi2)
           + psi_raw_norm_coef * Norm(psi2)

    Required factorised critic API:
        critic.q1_forward(state, action, goal)
        critic.q2_forward(state, action, goal)

        critic.phi1_forward(state, action)
        critic.psi1_forward(goal)

        critic.phi2_forward(state, action)
        critic.psi2_forward(goal)

    Strongly recommended extra API:
        critic.critic1_parameters()
        critic.critic2_parameters()

    Those methods must return disjoint parameter iterables.
    """
    if actor is None:
        raise ValueError("actor must be provided.")
    if critic is None:
        raise ValueError("critic must be provided.")
    if critic_target is None:
        raise ValueError("critic_target must be provided.")
    if env is None:
        raise ValueError("env must be provided.")
    if buffer_capacity is None:
        raise ValueError("buffer_capacity must be provided.")
    if goal is None:
        raise ValueError("goal must be provided.")
    if task_id is None:
        raise ValueError("task_id must be provided.")
    if not isinstance(task_id, (int, np.integer)):
        raise TypeError("task_id must be an integer.")
    if make_env is None:
        raise ValueError("make_env must be provided.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if train_freq < 1:
        raise ValueError("train_freq must be >= 1.")
    if gradient_steps < 1:
        raise ValueError("gradient_steps must be >= 1.")
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1].")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if obs_norm_clip <= 0.0:
        raise ValueError("obs_norm_clip must be positive.")
    if isinstance(ent_coef, str) and ent_coef != "auto":
        raise ValueError("ent_coef must be a positive float or 'auto'.")
    if isinstance(ent_coef, (float, int)) and ent_coef <= 0.0:
        raise ValueError("Fixed ent_coef must be positive.")
    if sigreg_coef < 0.0:
        raise ValueError("sigreg_coef must be >= 0.")
    if goal_separation_coef < 0.0:
        raise ValueError("goal_separation_coef must be >= 0.")
    if phi_raw_norm_coef < 0.0:
        raise ValueError("phi_raw_norm_coef must be >= 0.")
    if psi_raw_norm_coef < 0.0:
        raise ValueError("psi_raw_norm_coef must be >= 0.")
    if not -1.0 <= goal_separation_target_cosine <= 1.0:
        raise ValueError("goal_separation_target_cosine must be within [-1, 1].")
    if device is None:
        device = next(critic.parameters()).device
    device = torch.device(device)
    if obs_dim is None:
        obs_dim = env.observation_space.shape[0]
    if action_dim is None:
        action_dim = int(env.action_space.shape[0])
    goal_dim = int(np.asarray(goal).size)
    validate_training_options(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        eval_freq=eval_freq,
        train_freq=train_freq,
        tau=tau,
        gamma=gamma,
        replay_ratio=replay_ratio,
        replay_tasks_per_batch=replay_tasks_per_batch,
        early_stop_patience=early_stop_patience,
        eval_episodes=eval_episodes,
    )
    set_seed(seed)
    env.action_space.seed(seed)
    sigreg_loss = CovarianceRegularizer(sketch_dim=sketch_dim)
    loss_history = LossHistory(eval_freq)
    if replay_task_buffers is None:
        replay_task_buffers = {}
    if task_goals is None:
        task_goals = {}
    training_goal = np.asarray(goal, dtype=np.float32).copy()
    if training_goal.shape != (goal_dim,):
        raise ValueError(f"goal has shape {training_goal.shape}; expected ({goal_dim},).")
    task_goals[task_id] = training_goal.copy()
    actor = actor.to(device)
    critic = critic.to(device)
    critic_target = critic_target.to(device)
    if reset_target_from_critic:
        critic_target.load_state_dict(critic.state_dict())
    actor.train()
    critic.train()
    critic_target.eval()
    for parameter in critic_target.parameters():
        parameter.requires_grad_(False)
    required_methods = [
        "q1_forward",
        "q2_forward",
        "phi1_forward",
        "psi1_forward",
        "phi2_forward",
        "psi2_forward",
    ]
    missing_methods = [
        method_name for method_name in required_methods if not hasattr(critic, method_name)
    ]
    if len(missing_methods) > 0:
        raise AttributeError(f"Factorised critic is missing required methods: {missing_methods}.")
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device).view(
        1, -1
    )
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=device).view(
        1, -1
    )
    if action_low.shape[-1] != action_dim:
        raise RuntimeError("Action-space bounds do not match action_dim.")
    if not (
        torch.allclose(action_low, -torch.ones_like(action_low))
        and torch.allclose(action_high, torch.ones_like(action_high))
    ):
        raise ValueError("This SAC actor assumes action bounds [-1, 1].")

    def unique_parameters(parameters):
        seen_parameter_ids = set()
        unique = []
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id not in seen_parameter_ids:
                unique.append(parameter)
                seen_parameter_ids.add(parameter_id)
        return unique

    def critic_parameter_groups():
        """
        Preferred design: define these in the critic class:

            def critic1_parameters(self):
                return chain(
                    self.phi1_encoder.parameters(),
                    self.psi1_encoder.parameters(),
                    self.q1_head.parameters(),
                )

            def critic2_parameters(self):
                return chain(
                    self.phi2_encoder.parameters(),
                    self.psi2_encoder.parameters(),
                    self.q2_head.parameters(),
                )
        """
        if hasattr(critic, "critic1_parameters") and hasattr(critic, "critic2_parameters"):
            critic1_params = unique_parameters(list(critic.critic1_parameters()))
            critic2_params = unique_parameters(list(critic.critic2_parameters()))
        else:
            raise AttributeError(
                "The critic must expose critic1_parameters() and critic2_parameters() returning the two disjoint parameter groups. This is required to guarantee branch-specific optimisation."
            )
        if len(critic1_params) == 0:
            raise RuntimeError("critic1_parameters() returned no trainable parameters.")
        if len(critic2_params) == 0:
            raise RuntimeError("critic2_parameters() returned no trainable parameters.")
        critic1_ids = {id(parameter) for parameter in critic1_params}
        critic2_ids = {id(parameter) for parameter in critic2_params}
        shared_ids = critic1_ids.intersection(critic2_ids)
        if len(shared_ids) > 0:
            raise RuntimeError(
                "Critic 1 and critic 2 share trainable parameters. Fully separate phi1/psi1/q1 from phi2/psi2/q2 before using independent TBTRL losses."
            )
        all_critic_ids = {
            id(parameter) for parameter in critic.parameters() if parameter.requires_grad
        }
        grouped_ids = critic1_ids.union(critic2_ids)
        missing_ids = all_critic_ids.difference(grouped_ids)
        if len(missing_ids) > 0:
            raise RuntimeError(
                "Some trainable critic parameters are absent from both critic parameter groups. Put every critic parameter in exactly one branch, or make it non-trainable."
            )
        return (critic1_params, critic2_params)

    (critic1_params, critic2_params) = critic_parameter_groups()
    opt_actor = optim.AdamW(actor.parameters(), lr=lr_actor, weight_decay=0.0)
    opt_critic1 = optim.AdamW(critic1_params, lr=lr_critic, weight_decay=0.0)
    opt_critic2 = optim.AdamW(critic2_params, lr=lr_critic, weight_decay=0.0)
    buffer = ReplayBuffer(buffer_capacity, obs_dim, action_dim, device=device)
    if initial_alpha <= 0 or not np.isfinite(initial_alpha):
        raise ValueError("initial_alpha must be finite and positive.")
    if target_entropy is None:
        target_entropy = -float(action_dim)
    if ent_coef == "auto":
        log_ent_coef = torch.tensor(
            np.log(initial_alpha), dtype=torch.float32, device=device, requires_grad=True
        )
        opt_ent_coef = optim.Adam([log_ent_coef], lr=lr_ent_coef)
        fixed_ent_coef = None
    else:
        fixed_ent_coef = torch.as_tensor(float(ent_coef), dtype=torch.float32, device=device)
        log_ent_coef = None
        opt_ent_coef = None
    state_rms = RunningMeanStd(shape=(obs_dim,), device=device)
    goal_rms = RunningMeanStd(shape=(goal_dim,), device=device)

    def goal_batch_for(
        goal_value, requested_batch_size: int, target_device: torch.device
    ) -> torch.Tensor:
        if isinstance(goal_value, torch.Tensor):
            goal_batch = goal_value.to(device=target_device, dtype=torch.float32)
        else:
            goal_batch = torch.as_tensor(
                np.asarray(goal_value, dtype=np.float32), dtype=torch.float32, device=target_device
            )
        if goal_batch.ndim == 1:
            goal_batch = goal_batch.unsqueeze(0)
        if goal_batch.ndim != 2:
            raise ValueError(
                f"Goal must be [goal_dim] or [B, goal_dim]. Got {tuple(goal_batch.shape)}."
            )
        if goal_batch.shape[0] == 1:
            goal_batch = goal_batch.expand(requested_batch_size, -1)
        elif goal_batch.shape[0] != requested_batch_size:
            raise ValueError("Goal batch size does not match transition batch size.")
        return goal_batch

    def normalize_state(state_tensor: torch.Tensor) -> torch.Tensor:
        if not normalize_state_inputs:
            return state_tensor
        return state_rms.normalize(state_tensor, clip=obs_norm_clip)

    def normalize_goal(goal_tensor: torch.Tensor) -> torch.Tensor:
        if not normalize_goal_inputs:
            return goal_tensor
        return goal_rms.normalize(goal_tensor, clip=obs_norm_clip)

    def current_ent_coef() -> torch.Tensor:
        if log_ent_coef is not None:
            return log_ent_coef.exp()
        return fixed_ent_coef

    def zero_scalar() -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=device)

    def set_requires_grad(parameters, requires_grad: bool) -> None:
        for parameter in parameters:
            parameter.requires_grad_(requires_grad)

    def polyak_update_critic() -> None:
        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                critic.parameters(), critic_target.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)

    def td_target(
        batch, raw_goal_batch: torch.Tensor, ent_coef_tensor: torch.Tensor
    ) -> torch.Tensor:
        rewards = batch.rewards
        terminated = batch.terminated
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        if terminated.ndim == 1:
            terminated = terminated.unsqueeze(-1)
        next_state = normalize_state(batch.next_obs)
        next_goal = normalize_goal(
            goal_batch_for(raw_goal_batch, batch.next_obs.shape[0], batch.next_obs.device)
        )
        with torch.no_grad():
            (next_action, next_log_prob, _) = actor.sample(next_state, next_goal)
            next_q1 = critic_target.q1_forward(next_state, next_action, next_goal)
            next_q2 = critic_target.q2_forward(next_state, next_action, next_goal)
            next_q = torch.minimum(next_q1, next_q2)
            next_soft_value = next_q - ent_coef_tensor * next_log_prob
            if bootstrap_on_truncation:
                bootstrap_mask = 1.0 - terminated.float()
            else:
                truncated = batch.truncated
                if truncated.ndim == 1:
                    truncated = truncated.unsqueeze(-1)
                done = torch.logical_or(terminated.bool(), truncated.bool()).float()
                bootstrap_mask = 1.0 - done
            target = rewards + gamma * bootstrap_mask * next_soft_value
            if q_target_min is not None or q_target_max is not None:
                target = torch.clamp(target, min=q_target_min, max=q_target_max)
        return target

    def critic_td_losses(
        batch, raw_goal_batch: torch.Tensor, ent_coef_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = td_target(batch, raw_goal_batch, ent_coef_tensor)
        state = normalize_state(batch.obs)
        goal_batch = normalize_goal(raw_goal_batch)
        q1 = critic.q1_forward(state, batch.actions, goal_batch)
        q2 = critic.q2_forward(state, batch.actions, goal_batch)
        if q1.shape != target.shape:
            raise RuntimeError(f"Q1 shape {q1.shape} does not match target shape {target.shape}.")
        if q2.shape != target.shape:
            raise RuntimeError(f"Q2 shape {q2.shape} does not match target shape {target.shape}.")
        td_loss1 = F.mse_loss(q1, target)
        td_loss2 = F.mse_loss(q2, target)
        return (td_loss1, td_loss2)

    def seen_goals_tensor() -> torch.Tensor:
        seen_task_ids = sorted(task_goals.keys())
        return torch.as_tensor(
            np.asarray(
                [task_goals[seen_task_id] for seen_task_id in seen_task_ids], dtype=np.float32
            ),
            dtype=torch.float32,
            device=device,
        )

    def resolve_norm_target(name: str, supplied_target: float | None) -> float:
        if supplied_target is not None:
            target = float(supplied_target)
        elif name == "phi":
            target = float(getattr(critic, "phi_max_norm", 10.1)) - 0.1
        elif name == "psi":
            target = float(getattr(critic, "psi_max_norm", 10.1)) - 0.1
        else:
            raise ValueError(f"Unknown embedding name: {name}.")
        if target <= 0.0:
            raise ValueError(f"{name}_norm_target must be positive, got {target}.")
        return target

    def soft_excess_norm_loss(embeddings: torch.Tensor, target_norm: float) -> torch.Tensor:
        norms = embeddings.norm(p=2, dim=-1)
        return F.relu(norms - target_norm).mean()

    def head_goal_separation_loss(psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_goals = psi.shape[0]
        if n_goals < 2:
            return (zero_scalar(), zero_scalar())
        normalized_psi = F.normalize(psi, p=2, dim=-1, eps=1e-08)
        cosine_matrix = normalized_psi @ normalized_psi.T
        off_diagonal_mask = ~torch.eye(n_goals, dtype=torch.bool, device=device)
        off_diagonal_cosines = cosine_matrix[off_diagonal_mask]
        separation_loss = F.relu(off_diagonal_cosines - goal_separation_target_cosine).mean()
        return (separation_loss, off_diagonal_cosines.max().detach())

    def tbtrl_sigreg_losses(
        state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sigreg_coef <= 0.0:
            return (zero_scalar(), zero_scalar())
        phi1 = critic.phi1_forward(state, action)
        phi2 = critic.phi2_forward(state, action)
        sigreg1 = sigreg_loss(phi1)
        sigreg2 = sigreg_loss(phi2)
        return (sigreg1, sigreg2)

    def mixed_phi_regularisation_batch(
        current_batch, replay_batches
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the union of all (state, action) samples participating
        in TD learning at this update: the current task batch plus each
        selected old-task replay batch.

        The order does not matter for the phi-norm loss. It does matter
        only as ordinary batch ordering for SIGReg, not task identity.
        """
        state_chunks = [current_batch.obs]
        action_chunks = [current_batch.actions]
        for _, old_batch in replay_batches:
            state_chunks.append(old_batch.obs)
            action_chunks.append(old_batch.actions)
        mixed_states = torch.cat(state_chunks, dim=0)
        mixed_actions = torch.cat(action_chunks, dim=0)
        return (normalize_state(mixed_states), mixed_actions)

    def tbtrl_goal_separation_losses() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if goal_separation_coef <= 0.0 or len(task_goals) <= 1:
            return (zero_scalar(), zero_scalar(), zero_scalar(), zero_scalar())
        all_seen_goals = normalize_goal(seen_goals_tensor())
        psi1_all = critic.psi1_forward(all_seen_goals)
        psi2_all = critic.psi2_forward(all_seen_goals)
        (goal_sep1, max_cosine1) = head_goal_separation_loss(psi1_all)
        (goal_sep2, max_cosine2) = head_goal_separation_loss(psi2_all)
        return (goal_sep1, goal_sep2, max_cosine1, max_cosine2)

    def tbtrl_norm_losses(
        state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        phi1 = critic.phi1_forward(state, action)
        phi2 = critic.phi2_forward(state, action)
        all_seen_goals = normalize_goal(seen_goals_tensor())
        psi1_all = critic.psi1_forward(all_seen_goals)
        psi2_all = critic.psi2_forward(all_seen_goals)
        phi_target = resolve_norm_target("phi", phi_norm_target)
        psi_target = resolve_norm_target("psi", psi_norm_target)
        phi_norm1 = soft_excess_norm_loss(phi1, phi_target)
        psi_norm1 = soft_excess_norm_loss(psi1_all, psi_target)
        phi_norm2 = soft_excess_norm_loss(phi2, phi_target)
        psi_norm2 = soft_excess_norm_loss(psi2_all, psi_target)
        statistics = {
            "phi1_norm": phi1.norm(p=2, dim=-1).mean().detach(),
            "phi2_norm": phi2.norm(p=2, dim=-1).mean().detach(),
            "psi1_norm": psi1_all.norm(p=2, dim=-1).mean().detach(),
            "psi2_norm": psi2_all.norm(p=2, dim=-1).mean().detach(),
        }
        return (phi_norm1, psi_norm1, phi_norm2, psi_norm2, statistics)

    (obs_dict, _) = env.reset(seed=seed)
    global_step = 0
    update_count = 0
    success_streak = 0
    start_time = time.perf_counter()
    eval_returns = []
    eval_success_rates = []
    eval_final_distances = []
    solved = False
    min_steps = None
    min_time = None
    while global_step < total_steps:
        state = np.asarray(obs_dict, dtype=np.float32)
        current_goal = training_goal.copy()
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        goal_t = torch.as_tensor(current_goal, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            state_rms.update(state_t)
            if normalize_goal_inputs:
                goal_rms.update(goal_t)
        if global_step < warmup_steps:
            action = env.action_space.sample().astype(np.float32)
        else:
            with torch.no_grad():
                (action_t, _, _) = actor.sample(normalize_state(state_t), normalize_goal(goal_t))
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        (next_obs_dict, reward, terminated, truncated, _) = env.step(action)
        next_state = np.asarray(next_obs_dict, dtype=np.float32)
        next_goal = training_goal.copy()
        with torch.no_grad():
            next_state_t = torch.as_tensor(
                next_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            state_rms.update(next_state_t)
            if normalize_goal_inputs:
                next_goal_t = torch.as_tensor(
                    next_goal, dtype=torch.float32, device=device
                ).unsqueeze(0)
                goal_rms.update(next_goal_t)
        buffer.add_transition(
            obs=state,
            action=action,
            reward=reward,
            next_obs=next_state,
            terminated=terminated,
            truncated=truncated,
        )
        obs_dict = next_obs_dict
        global_step += 1
        if terminated or truncated:
            (obs_dict, _) = env.reset()
        if len(buffer) < warmup_steps:
            continue
        if global_step % train_freq != 0:
            continue
        for _ in range(gradient_steps):
            current_batch = buffer.sample(batch_size)
            current_goal_tensor = goal_batch_for(training_goal, current_batch.obs.shape[0], device)
            eligible_old_task_ids = []
            for old_task_id, old_buffer in replay_task_buffers.items():
                if old_buffer is None:
                    continue
                if len(old_buffer) < 1:
                    continue
                if old_task_id not in task_goals:
                    raise KeyError(f"Missing goal for replay task {old_task_id}.")
                eligible_old_task_ids.append(old_task_id)
            replay_batches = []
            if len(eligible_old_task_ids) > 0 and replay_ratio > 0.0:
                if replay_tasks_per_batch is None:
                    n_old_tasks = len(eligible_old_task_ids)
                else:
                    n_old_tasks = min(int(replay_tasks_per_batch), len(eligible_old_task_ids))
                cycle_index = update_count % len(eligible_old_task_ids)
                ordered_old_task_ids = (
                    eligible_old_task_ids[cycle_index:] + eligible_old_task_ids[:cycle_index]
                )
                selected_old_task_ids = ordered_old_task_ids[:n_old_tasks]
                replay_batch_size = max(1, int(batch_size * replay_ratio / n_old_tasks))
                for old_task_id in selected_old_task_ids:
                    old_buffer = replay_task_buffers[old_task_id]
                    if len(old_buffer) < replay_batch_size:
                        continue
                    old_batch = old_buffer.sample(replay_batch_size)
                    old_goal = task_goals[old_task_id]
                    replay_batches.append((old_goal, old_batch))
            ent_coef_tensor = current_ent_coef().detach()
            (current_td_loss1, current_td_loss2) = critic_td_losses(
                current_batch, current_goal_tensor, ent_coef_tensor
            )
            old_td_losses1 = []
            old_td_losses2 = []
            for old_goal, old_batch in replay_batches:
                old_goal_tensor = goal_batch_for(old_goal, old_batch.obs.shape[0], device)
                (old_td_loss1, old_td_loss2) = critic_td_losses(
                    old_batch, old_goal_tensor, ent_coef_tensor
                )
                old_td_losses1.append(old_td_loss1)
                old_td_losses2.append(old_td_loss2)
            if len(old_td_losses1) > 0:
                old_replay_td_loss1 = torch.stack(old_td_losses1).mean()
                old_replay_td_loss2 = torch.stack(old_td_losses2).mean()
            else:
                old_replay_td_loss1 = zero_scalar()
                old_replay_td_loss2 = zero_scalar()
            (state_for_phi_reg, action_for_phi_reg) = mixed_phi_regularisation_batch(
                current_batch, replay_batches
            )
            (current_sigreg1, current_sigreg2) = tbtrl_sigreg_losses(
                state_for_phi_reg, action_for_phi_reg
            )
            (
                current_goal_separation1,
                current_goal_separation2,
                psi1_max_cosine,
                psi2_max_cosine,
            ) = tbtrl_goal_separation_losses()
            (phi_norm1, psi_norm1, phi_norm2, psi_norm2, norm_statistics) = tbtrl_norm_losses(
                state_for_phi_reg, action_for_phi_reg
            )
            critic1_total_loss = (
                current_td_loss1
                + replay_loss_coef * old_replay_td_loss1
                + sigreg_coef * current_sigreg1
                + goal_separation_coef * current_goal_separation1
                + phi_raw_norm_coef * phi_norm1
                + psi_raw_norm_coef * psi_norm1
            )
            critic2_total_loss = (
                current_td_loss2
                + replay_loss_coef * old_replay_td_loss2
                + sigreg_coef * current_sigreg2
                + goal_separation_coef * current_goal_separation2
                + phi_raw_norm_coef * phi_norm2
                + psi_raw_norm_coef * psi_norm2
            )
            opt_critic1.zero_grad(set_to_none=True)
            critic1_total_loss.backward()
            critic1_grad_norm = nn.utils.clip_grad_norm_(
                critic1_params, max_norm=critic_grad_clip_norm
            )
            opt_critic1.step()
            opt_critic2.zero_grad(set_to_none=True)
            critic2_total_loss.backward()
            critic2_grad_norm = nn.utils.clip_grad_norm_(
                critic2_params, max_norm=critic_grad_clip_norm
            )
            opt_critic2.step()
            state_batch = normalize_state(current_batch.obs)
            goal_batch = normalize_goal(current_goal_tensor)
            set_requires_grad(critic1_params, requires_grad=False)
            set_requires_grad(critic2_params, requires_grad=False)
            (sampled_actions, log_prob, _) = actor.sample(state_batch, goal_batch)
            q1_pi = critic.q1_forward(state_batch, sampled_actions, goal_batch)
            q2_pi = critic.q2_forward(state_batch, sampled_actions, goal_batch)
            min_q_pi = torch.minimum(q1_pi, q2_pi)
            ent_coef_tensor = current_ent_coef().detach()
            actor_loss = (ent_coef_tensor * log_prob - min_q_pi).mean()
            opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(
                actor.parameters(), max_norm=actor_grad_clip_norm
            )
            opt_actor.step()
            set_requires_grad(critic1_params, requires_grad=True)
            set_requires_grad(critic2_params, requires_grad=True)
            if log_ent_coef is not None:
                ent_coef_loss = -(log_ent_coef * (log_prob.detach() + target_entropy)).mean()
                opt_ent_coef.zero_grad(set_to_none=True)
                ent_coef_loss.backward()
                opt_ent_coef.step()
            else:
                ent_coef_loss = zero_scalar()
            polyak_update_critic()
            update_count += 1
            with torch.no_grad():
                q_data_1 = critic.q1_forward(state_batch, current_batch.actions, goal_batch)
                q_data_2 = critic.q2_forward(state_batch, current_batch.actions, goal_batch)
                q_data = torch.minimum(q_data_1, q_data_2)
                pi_abs = sampled_actions.abs().mean()
                pi_saturation = (sampled_actions.abs() > 0.95).float().mean()
            loss_history.append(
                {
                    "step": global_step,
                    "td1": current_td_loss1.detach().item(),
                    "td2": current_td_loss2.detach().item(),
                    "replay1": old_replay_td_loss1.detach().item(),
                    "replay2": old_replay_td_loss2.detach().item(),
                    "sigreg1": current_sigreg1.detach().item(),
                    "sigreg2": current_sigreg2.detach().item(),
                    "goal_separation1": current_goal_separation1.detach().item(),
                    "goal_separation2": current_goal_separation2.detach().item(),
                    "phi_norm1": phi_norm1.detach().item(),
                    "phi_norm2": phi_norm2.detach().item(),
                    "psi_norm1": psi_norm1.detach().item(),
                    "psi_norm2": psi_norm2.detach().item(),
                    "total1": critic1_total_loss.detach().item(),
                    "total2": critic2_total_loss.detach().item(),
                    "actor": actor_loss.detach().item(),
                    "alpha": current_ent_coef().detach().item(),
                    "alpha_loss": ent_coef_loss.detach().item(),
                    "critic1_grad_norm": float(critic1_grad_norm),
                    "critic2_grad_norm": float(critic2_grad_norm),
                    "actor_grad_norm": float(actor_grad_norm),
                    "q_data": q_data.mean().item(),
                    "q_policy": min_q_pi.mean().detach().item(),
                    "log_prob": log_prob.mean().detach().item(),
                    "entropy": -log_prob.mean().detach().item(),
                    "psi1_max_cosine": float(psi1_max_cosine),
                    "psi2_max_cosine": float(psi2_max_cosine),
                    "policy_abs": pi_abs.item(),
                    "policy_saturation": pi_saturation.item(),
                    **{key: value.item() for key, value in norm_statistics.items()},
                    "replay_tasks": len(replay_batches),
                }
            )
        if global_step % eval_freq != 0:
            continue
        eval_env = make_env(goal=training_goal)

        def eval_policy(observation):
            eval_state = np.asarray(observation, dtype=np.float32)
            eval_goal = training_goal.copy()
            eval_state_t = torch.as_tensor(
                eval_state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            eval_goal_t = torch.as_tensor(eval_goal, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            with torch.no_grad():
                eval_action = actor.deterministic(
                    normalize_state(eval_state_t), normalize_goal(eval_goal_t)
                )
            return eval_action.squeeze(0).cpu().numpy()

        (mean_return, mean_length, success_rate, mean_final_distance) = (
            evaluate_policy_with_success(
                eval_env,
                eval_policy,
                goal=training_goal,
                episodes=eval_episodes,
                seed=seed + 100000 + global_step,
            )
        )
        eval_returns.append((global_step, mean_return))
        eval_success_rates.append((global_step, success_rate))
        eval_final_distances.append((global_step, mean_final_distance))
        logger.info(
            "SAC task=%s step=%s return=%.3f length=%.1f success=%.3f final_distance=%.4f losses=%s",
            task_id,
            global_step,
            mean_return,
            mean_length,
            success_rate,
            mean_final_distance,
            loss_history[-1] if loss_history else {},
        )
        eval_env.close()
        if early_stop_success_rate is not None:
            criterion_met = success_rate >= early_stop_success_rate
        else:
            criterion_met = mean_return >= early_stop_reward
        if criterion_met:
            success_streak += 1
        else:
            success_streak = 0
        if enable_early_stop and success_streak >= early_stop_patience:
            solved = True
            min_steps = global_step
            min_time = time.perf_counter() - start_time
            logger.info(
                f"Early stopping at step={global_step}, return={mean_return:.3f}, success={success_rate:.3f}, streak={success_streak}"
            )
            break
    if min_steps is None:
        min_steps = global_step
        min_time = time.perf_counter() - start_time
    return SACResult(
        actor,
        critic,
        critic_target,
        eval_returns,
        eval_success_rates,
        eval_final_distances,
        global_step,
        min_steps if solved else None,
        min_time,
        buffer,
        state_rms,
        goal_rms,
        current_ent_coef().detach().item(),
        loss_history,
    )
