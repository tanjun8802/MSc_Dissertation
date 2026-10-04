import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from tbtrl.evaluation import evaluate_policy
from tbtrl.losses import CovarianceRegularizer, norm_penalty_loss_l1, online_goal_separation_loss
from tbtrl.random import set_seed
from tbtrl.replay import ReplayBuffer
from tbtrl.training.common import LossHistory, validate_training_options
from tbtrl.training.results import DQNResult

logger = logging.getLogger(__name__)


def train_dqn(
    seed=42,
    q_network=None,
    q_target_network=None,
    env=None,
    buffer_capacity=None,
    lr_sa=0.001,
    lr_goal_init=0.001,
    obs_dim=None,
    device=None,
    total_steps=100000,
    warmup_steps=5000,
    batch_size=256,
    eval_freq=1000,
    gamma=0.99,
    tau=0.005,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50000,
    train_freq=4,
    goal=None,
    task_id=None,
    make_env=None,
    replay_task_buffers=None,
    task_goals=None,
    replay_ratio=0.5,
    replay_tasks_per_batch=None,
    replay_loss_coef=1.0,
    bootstrap_on_truncation=True,
    early_stop_reward=0.99,
    early_stop_patience=5,
    eval_episodes=8,
    enable_early_stop=True,
    sigreg_coef=0.001,
    sketch_dim=64,
    goal_separation_coef=0.001,
    goal_separation_target_cosine=0.85,
    psi_raw_norm_coef=0.0001,
    phi_raw_norm_coef=0.0001,
):
    """
    Functional factorised critic:

        Q(s, a, g)
        =
        phi_theta(s, a)^T psi_eta(g)

    Replay rewards and goal termination are recomputed by env.relabel_transitions.

    A replay buffer may be shared across tasks. Each task is relabelled with
    its own goal from task_goals[task_id].

    No fixed targets are used or stored.

    Required replay batch fields:

        batch.obs
        batch.actions
        batch.rewards
        batch.next_obs
        batch.terminated
        batch.truncated

    Required network methods:

        q_network.encode_state_action(
            obs,
            action_onehot,
        )

        q_network.encode_goal(
            goal,
        )

        q_network.q_val_for_argmax_action_from_embedding(
            obs,
            task_embedding,
            normalize_embedding=False,
        )
    """
    if q_network is None:
        raise ValueError("q_network must be provided.")
    if q_target_network is None:
        raise ValueError("q_target_network must be provided.")
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
    if train_freq < 1:
        raise ValueError("train_freq must be >= 1.")
    if device is None:
        device = next(q_network.parameters()).device
    device = torch.device(device)
    if obs_dim is None:
        obs_dim = int(env.observation_space.shape[0])
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
    if isinstance(goal, torch.Tensor):
        task_goals[task_id] = goal.detach().cpu().numpy().astype(np.float32, copy=True)
    else:
        task_goals[task_id] = np.asarray(goal, dtype=np.float32).copy()
    num_actions = env.action_space.n
    q_network = q_network.to(device)
    q_target_network = q_target_network.to(device)
    q_target_network.load_state_dict(q_network.state_dict())
    q_target_network.eval()
    for parameter in q_target_network.parameters():
        parameter.requires_grad_(False)
    goal_tensor = torch.as_tensor(
        np.asarray(goal, dtype=np.float32), dtype=torch.float32, device=device
    ).view(1, -1)
    opt_sa = optim.Adam(q_network.sa_encoder.parameters(), lr=lr_sa)
    opt_goal = optim.Adam(q_network.goal_encoder.parameters(), lr=lr_goal_init)
    buffer = ReplayBuffer(buffer_capacity, obs_dim, 1, device=device, discrete=True)

    def goal_batch_for(goal_value, batch_size, target_device):
        if isinstance(goal_value, torch.Tensor):
            goal_batch = goal_value.to(device=target_device, dtype=torch.float32)
        else:
            goal_batch = torch.as_tensor(
                np.asarray(goal_value, dtype=np.float32), dtype=torch.float32, device=target_device
            )
        if goal_batch.ndim == 1:
            goal_batch = goal_batch.unsqueeze(0)
        elif goal_batch.ndim != 2:
            raise ValueError(
                f"Goal must have shape [goal_dim] or [B, goal_dim]. Got {tuple(goal_batch.shape)}."
            )
        if goal_batch.shape[0] == 1:
            goal_batch = goal_batch.expand(batch_size, -1)
        elif goal_batch.shape[0] != batch_size:
            raise ValueError(
                f"Goal batch size does not match batch size: {goal_batch.shape[0]} vs {batch_size}."
            )
        return goal_batch

    def action_onehot_from_batch(batch):
        actions = batch.actions.long().view(-1)
        return F.one_hot(actions, num_classes=num_actions).to(dtype=batch.obs.dtype)

    def q_all_actions_from_goal(network, obs_t, goal_t):
        goal_batch = goal_batch_for(goal_t, obs_t.shape[0], obs_t.device)
        psi = network.encode_goal(goal_batch)
        return network.q_val_for_argmax_action_from_embedding(obs_t, psi, normalize_embedding=False)

    def moving_td_target(batch, batch_goal):
        """
        Compute TD targets using:
        - rewards recomputed from env.{compute_simple,compute_shaped}_reward
        - q_target_network for bootstrap

        batch_goal: numpy or torch goal for the task being trained.
        """
        (rewards, terminated) = env.relabel_transitions(batch, batch_goal)
        next_obs = batch.next_obs
        truncated = batch.truncated
        next_goal_batch = goal_batch_for(batch_goal, next_obs.shape[0], next_obs.device)
        with torch.no_grad():
            next_q_values = q_all_actions_from_goal(q_target_network, next_obs, next_goal_batch)
            next_q = next_q_values.max(dim=-1, keepdim=True).values
            if bootstrap_on_truncation:
                bootstrap_mask = 1.0 - terminated
            else:
                done = torch.clamp(terminated + truncated, min=0.0, max=1.0)
                bootstrap_mask = 1.0 - done
        return rewards + gamma * bootstrap_mask * next_q

    def factorised_q_from_batch(batch, batch_goal):
        action_onehot = action_onehot_from_batch(batch)
        goal_batch = goal_batch_for(batch_goal, batch.obs.shape[0], batch.obs.device)
        phi = q_network.encode_state_action(batch.obs, action_onehot)
        psi = q_network.encode_goal(goal_batch)
        return (phi * psi).sum(dim=-1, keepdim=True)

    def task_loss(batch, batch_goal):
        target = moving_td_target(batch, batch_goal)
        q_values = factorised_q_from_batch(batch, batch_goal)
        if q_values.shape != target.shape:
            raise RuntimeError(
                f"Q-values and targets have different shapes: {q_values.shape} vs {target.shape}."
            )
        return F.mse_loss(q_values, target)

    def sigreg_phi_loss(batch):
        action_onehot = action_onehot_from_batch(batch)
        phi = q_network.encode_state_action(batch.obs, action_onehot)
        return sigreg_loss(phi)

    (obs, _) = env.reset(seed=seed)
    global_step = 0
    success_streak = 0
    start_time = time.perf_counter()
    eval_returns = []
    solved = False
    min_steps = None
    min_time = None
    cosine_matrix = None
    while global_step < total_steps:
        fraction = min(1.0, global_step / max(1, eps_decay_steps))
        epsilon = eps_start + fraction * (eps_end - eps_start)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_single = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = q_all_actions_from_goal(q_network, obs_single, goal_tensor)
            action = int(q_values.argmax(dim=-1).item())
        (next_obs, reward, terminated, truncated, _) = env.step(action)
        done = terminated or truncated
        buffer.add_transition(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
        )
        obs = next_obs
        global_step += 1
        if done:
            (obs, _) = env.reset()
        if len(buffer) < warmup_steps or global_step % train_freq != 0:
            continue
        current_batch = buffer.sample(batch_size)
        eligible_old_task_ids = []
        for old_task_id, old_buffer in replay_task_buffers.items():
            if old_buffer is None:
                continue
            if len(old_buffer) < 1:
                continue
            if old_task_id not in task_goals:
                raise KeyError(f"Missing goal for replay task {old_task_id}. Add it to task_goals.")
            eligible_old_task_ids.append(old_task_id)
        replay_batches = []
        if len(eligible_old_task_ids) > 0 and replay_ratio > 0.0:
            if replay_tasks_per_batch is None:
                n_old_tasks = len(eligible_old_task_ids)
            else:
                n_old_tasks = min(int(replay_tasks_per_batch), len(eligible_old_task_ids))
            cycle_index = global_step // train_freq % len(eligible_old_task_ids)
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
                replay_batches.append((old_task_id, old_goal, old_batch))
        current_loss = task_loss(current_batch, goal)
        old_losses = []
        for old_task_id, old_goal, old_batch in replay_batches:
            old_losses.append(task_loss(old_batch, old_goal))
        if len(old_losses) > 0:
            old_replay_loss = torch.stack(old_losses).mean()
        else:
            old_replay_loss = torch.zeros((), device=device)
        if sigreg_coef > 0.0:
            current_sigreg = sigreg_phi_loss(current_batch)
        else:
            current_sigreg = torch.zeros((), device=device)
        if goal_separation_coef > 0.0 and len(task_goals) > 1:
            (goal_separation_loss, cosine_matrix) = online_goal_separation_loss(
                q_network=q_network,
                task_goals=task_goals,
                device=device,
                target_cosine=goal_separation_target_cosine,
            )
        else:
            goal_separation_loss = torch.zeros((), device=device)
        seen_task_ids = sorted(task_goals.keys())
        seen_goal_tensor = torch.as_tensor(
            np.asarray([task_goals[task_id] for task_id in seen_task_ids], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        raw_psi_all = q_network.goal_encoder(seen_goal_tensor)
        psi_raw_norm_loss = norm_penalty_loss_l1(
            raw_psi_all, target_norm=q_network.psi_max_norm - 0.1
        )
        actions = current_batch.actions.long().view(-1)
        action_onehot = F.one_hot(actions, num_classes=q_network.num_actions).to(
            dtype=current_batch.obs.dtype, device=current_batch.obs.device
        )
        sa_input = torch.cat([current_batch.obs, action_onehot], dim=-1)
        raw_phi_current = q_network.sa_encoder(sa_input)
        phi_raw_norm_loss = norm_penalty_loss_l1(
            raw_phi_current, target_norm=q_network.phi_max_norm - 0.1
        )
        total_loss = (
            current_loss
            + replay_loss_coef * old_replay_loss
            + sigreg_coef * current_sigreg
            + goal_separation_coef * goal_separation_loss
            + psi_raw_norm_coef * psi_raw_norm_loss
            + phi_raw_norm_coef * phi_raw_norm_loss
        )
        opt_sa.zero_grad(set_to_none=True)
        opt_goal.zero_grad(set_to_none=True)
        total_loss.backward()
        if lr_sa > 0.0:
            nn.utils.clip_grad_norm_(q_network.sa_encoder.parameters(), max_norm=10.0)
        if lr_goal_init > 0.0:
            nn.utils.clip_grad_norm_(q_network.goal_encoder.parameters(), max_norm=10.0)
        if lr_sa > 0.0:
            opt_sa.step()
        if lr_goal_init > 0.0:
            opt_goal.step()
        loss_history.append(
            {
                "step": global_step,
                "td": current_loss.detach().item(),
                "replay": old_replay_loss.detach().item(),
                "sigreg": current_sigreg.detach().item(),
                "goal_separation": goal_separation_loss.detach().item(),
                "phi_norm": phi_raw_norm_loss.detach().item(),
                "psi_norm": psi_raw_norm_loss.detach().item(),
                "total": total_loss.detach().item(),
                "replay_tasks": len(replay_batches),
            }
        )
        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                q_network.parameters(), q_target_network.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(tau * online_parameter)
        if global_step % eval_freq != 0:
            continue
        eval_env = make_env(goal=goal)

        def eval_policy(observation):
            observation_t = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                q_values = q_all_actions_from_goal(q_network, observation_t, goal_tensor)
            return int(q_values.argmax(dim=-1).item())

        (mean_return, mean_length) = evaluate_policy(
            eval_env, eval_policy, episodes=eval_episodes, seed=seed + 100000 + global_step
        )
        eval_returns.append((global_step, mean_return))
        logger.info(
            "TBTRL task=%s step=%s return=%.3f losses=%s",
            task_id,
            global_step,
            mean_return,
            loss_history[-1] if loss_history else {},
        )
        eval_env.close()
        if mean_return >= early_stop_reward:
            success_streak += 1
        else:
            success_streak = 0
        if enable_early_stop and success_streak >= early_stop_patience:
            solved = True
            min_steps = global_step
            min_time = time.perf_counter() - start_time
            logger.info(
                f"Early stopping at step={global_step}, return={mean_return:.3f}, streak={success_streak}"
            )
            break
    if min_steps is None:
        min_steps = global_step
        min_time = time.perf_counter() - start_time
    with torch.no_grad():
        final_task_embedding = q_network.encode_goal(goal_tensor).squeeze(0).detach().cpu().numpy()
    return DQNResult(
        q_network,
        q_target_network,
        eval_returns,
        global_step,
        min_steps if solved else None,
        min_time,
        buffer,
        final_task_embedding,
        cosine_matrix,
        loss_history,
    )
