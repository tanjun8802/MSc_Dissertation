import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import time
import random
from typing import Optional, Dict, Any

from utils import (
    TrajectoryReplayBufferDiscrete,
    TrajectoryReplayBuffer,
    AtariReplayBuffer,
    TrajectoryReplayBufferContinuous,
    evaluate_policy,
    set_seed,
    extract_fixed_probe_sa_embedding,
    extract_mean_sa_embedding,
    extract_sa_batch_for_isotropy,
    build_goal_batch,
    extract_fixed_probe_sa_embedding_td3,
    extract_mean_sa_embedding_td3,
    extract_sa_batch_for_isotropy_td3,
    encode_task_for_similarity,
    compute_task_similarity,
    retrieve_similar_task_embeddings,
    inspect_raw_phi_norms,
    evaluate_policy_with_success,
)

from loss_functions import (
    repulsion_loss_to_memory,
    sigreg_loss,
    orthogonal_loss,
    ewc_regulariser_loss,
    weight_regulariser_loss,
    goal_memory_contrastive_loss,
    goal_prototype_anchor_loss,
    online_goal_separation_loss,
    norm_penalty_loss_l2,
    norm_penalty_loss_l1,
)

from gym_robotics_networks import RunningMeanStd


def dqn_train(
    seed: int = 42,
    q_network=None,
    q_target_network=None,
    env=None,
    buffer_capacity=None,
    lr=None,
    obs_dim=None,
    device=None,
    total_steps=100000,
    warmup_steps=5000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50000,
    train_freq=4,
    goal=None,
    params=None,
    regulariser=None,
    embedding_memory=None,
    reference_params=None,
    fisher_diag=None,
    sa_reg_prefix_filter="sa_encoder",
    sigreg=False,
    td_steps=1,
    make_env=None,
    early_stop_reward=0.99,
    early_stop_patience=3,
    enable_early_stop=True,
):
    if q_network is None:
        raise ValueError("q_network must be provided")
    if q_target_network is None:
        raise ValueError("q_target_network must be provided")
    if env is None:
        raise ValueError("env must be provided")
    if goal is None:
        raise ValueError("goal must be provided")

    if buffer_capacity is None:
        raise ValueError("buffer_capacity must be provided")
    if lr is None:
        raise ValueError("lr must be provided")
    if make_env is None:
        raise ValueError("make_env function must be provided to create the evaluation environment.")

    if device is None:
        device = next(q_network.parameters()).device

    if obs_dim is None:
        obs_dim = int(env.observation_space.shape[0])

    if embedding_memory is None:
        embedding_memory = {}

    set_seed(seed)

    if params is None:
        opt = optim.Adam([
            {"params": q_network.sa_encoder.parameters(), "lr": lr},
            {"params": q_network.goal_encoder.parameters(), "lr": lr},
        ])
    else:
        opt = optim.Adam(params, lr=lr)

    buffer = TrajectoryReplayBufferDiscrete(buffer_capacity, obs_dim, 1, device=device)

    goal_arr = np.array(goal, dtype=np.float32)
    goal_t_single = torch.tensor(goal_arr, dtype=torch.float32, device=device).unsqueeze(0)

    obs, _ = env.reset()
    global_step = 0
    eval_returns = []
    start_time = time.perf_counter()
    min_steps = None
    min_time = None
    success_streak = 0

    ortho_loss = torch.tensor(0.0, device=device)
    sigreg_loss_val = torch.tensor(0.0, device=device)
    weight_loss = torch.tensor(0.0, device=device)
    ewc_loss = torch.tensor(0.0, device=device)
    loss = torch.tensor(0.0, device=device)

    num_actions = env.action_space.n

    while global_step < total_steps:
        frac = min(1.0, global_step / eps_decay_steps)
        eps = eps_start + frac * (eps_end - eps_start)

        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = q_network.q_val_for_argmax_action(obs_t, goal_t_single)
                action = int(q_vals.argmax(dim=-1).item())

        next_obs, rew, term, trunc, _ = env.step(action)
        done = term or trunc

        buffer.add_transition(obs, action, rew, next_obs, term, trunc)
        obs = next_obs
        global_step += 1

        if done:
            obs, _ = env.reset()

        if len(buffer) >= warmup_steps and global_step % train_freq == 0:
            batch = buffer.sample(batch_size)

            obs_t = batch.obs
            act_t = batch.actions.long()
            rew_t = batch.rewards
            next_obs_t = batch.next_obs
            term_t = batch.terminated
            trunc_t = batch.truncated

            goal_batch = goal_t_single.expand(obs_t.shape[0], -1)
            B = obs_t.shape[0]

            with torch.no_grad():
                next_q_vals = q_target_network.q_val_for_argmax_action(next_obs_t, goal_batch)
                next_q = next_q_vals.max(dim=-1, keepdim=True).values
                gamma_final = gamma ** td_steps
                target = rew_t + gamma_final * (1.0 - term_t) * next_q
 
            act_onehot = F.one_hot(
                act_t.squeeze(-1),
                num_classes=num_actions
            ).float()

            current_q = q_network(obs_t, act_onehot, goal_batch)
            td_loss = F.mse_loss(current_q, target)

            if sigreg:
                act_onehot_all = F.one_hot(
                    torch.arange(num_actions, device=device),
                    num_classes=num_actions
                ).float()
                act_onehot_all = act_onehot_all.unsqueeze(0).expand(B, -1, -1)

                obs_rep = obs_t.unsqueeze(1).expand(-1, num_actions, -1)
                obs_flat = obs_rep.reshape(B * num_actions, obs_dim)
                act_flat = act_onehot_all.reshape(B * num_actions, num_actions)

                phi_all = q_network.encode_state_action(obs_flat, act_flat)
                sigreg_loss_val = sigreg_loss(phi_all)

            # if reference_params is not None:
            #     # weight_loss = weight_regulariser_loss(
            #     #     q_network,
            #     #     reference_params=reference_params,
            #     #     prefix_filter=sa_reg_prefix_filter,
            #     # )

            if fisher_diag is not None:
                ewc_loss = ewc_regulariser_loss(
                    q_network,
                    reference_params=reference_params,
                    fisher_diag=fisher_diag,
                    prefix_filter=sa_reg_prefix_filter,
                )

            if regulariser is not None and regulariser == "repulsion":
                loss = td_loss + 0.1 * sigreg_loss_val + 10000 * ewc_loss
            else:
                loss = td_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
            opt.step()

            for p, p_tgt in zip(q_network.parameters(), q_target_network.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)

        if global_step % 1000 == 0 and reference_params is not None:
            delta_means = []
            for name, p in q_network.named_parameters():
                if name in reference_params:
                    delta_sq = (p.detach() - reference_params[name]).pow(2)
                    delta_means.append(delta_sq.mean().item())

            if len(delta_means) > 0:
                print("Mean squared deltas (first few):", delta_means[:5])

        if global_step % 1000 == 0:
            eval_env = make_env(goal)
            goal_eval_arr = np.array(goal, dtype=np.float32)
            goal_eval_t = torch.tensor(goal_eval_arr, dtype=torch.float32, device=device).unsqueeze(0)

            def eval_policy(o):
                o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_vals = q_network.q_val_for_argmax_action(o_t, goal_eval_t)
                    return int(q_vals.argmax(dim=-1).item())

            mean_ret, mean_len = evaluate_policy(eval_env, eval_policy, episodes=8)
            eval_returns.append((global_step, mean_ret))
            print(
                f"[DQN-factorised] step={global_step:7d} | eps={eps:.3f} "
                f"| eval_return={mean_ret:.3f} | eval_len={mean_len:.1f}"
                f"| Ortho loss={ortho_loss.item():.3f} | SigReg loss={sigreg_loss_val.item():.3f} | Loss={loss.item():.3f}"
                f"| Weight loss={weight_loss.item():.10f} | EWC loss={ewc_loss.item():.10f}"
            )

            if mean_ret >= early_stop_reward:
                success_streak += 1
            else:
                success_streak = 0

            if enable_early_stop and success_streak >= early_stop_patience:
                min_steps = global_step
                min_time = time.perf_counter() - start_time
                print(f"Good policy achieved at step {global_step} with mean return {mean_ret:.3f}")
                print(
                    f"Early stopping triggered at step {global_step} "
                    f"after {success_streak} consecutive evals with "
                    f"mean return >= {early_stop_reward:.2f}"
                )
                eval_env.close()
                break
            eval_env.close()

    min_steps = global_step
    min_time = time.perf_counter() - start_time
    goal_tensor = torch.tensor(np.array(goal, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        psi_z = q_network.encode_goal(goal_tensor)
        task_embedding = psi_z.squeeze(0).cpu().numpy()

    obs_probe_np = np.array([5.0, 5.0], dtype=np.float32)

    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    env_action_names = getattr(base_env, "action_names", ["Up", "Down", "Left", "Right"])
    act_probe_idx = env_action_names.index("Up")

    sa_embedding_mean = extract_mean_sa_embedding(
        q_network=q_network,
        buffer=buffer,
        num_actions=num_actions,
        batch_size=256,
        device=device,
        as_numpy=True,
    )

    sa_embedding_fixed = extract_fixed_probe_sa_embedding(
        q_network=q_network,
        obs_probe=obs_probe_np,
        act_probe_idx=act_probe_idx,
        num_actions=num_actions,
        device=device,
        as_numpy=True,
    )

    sa_batch_final = extract_sa_batch_for_isotropy(
        q_network=q_network,
        buffer=buffer,
        num_actions=num_actions,
        batch_size=1024,
        device=device,
        as_numpy=True,
    )

    env.close()
    return (
        q_network,
        q_target_network,
        eval_returns,
        min_steps,
        min_time,
        task_embedding,
        sa_embedding_mean,
        sa_embedding_fixed,
        sa_batch_final,
        buffer,
    )

def dqn_train_multi_loss(
    seed: int = 42,
    q_network=None,
    q_target_network=None,
    env=None,
    buffer_capacity=None,
    lr_sa=None,
    lr_goal=None,
    obs_dim=None,
    device=None,
    total_steps=100000,
    warmup_steps=5000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50000,
    train_freq=4,
    goal=None,
    params=None,
    regulariser=None,
    reg_alpha=500.0,
    embedding_memory=None,
    memory_goals=None,
    reference_params=None,
    fisher_diag=None,
    sa_reg_prefix_filter="sa_encoder",
    sigreg=False,
    td_steps=1,
    make_env=None,
    replay_task_buffers=None,
    replay_ratio=0.5,
    replay_tasks_per_batch=None,
    replay_loss_coef=1.0,
    early_stop_reward=0.99,
    early_stop_patience=5,
    enable_early_stop=True,
):
    if q_network is None:
        raise ValueError("q_network must be provided")
    if q_target_network is None:
        raise ValueError("q_target_network must be provided")
    if env is None:
        raise ValueError("env must be provided")
    if goal is None:
        raise ValueError("goal must be provided")
    if buffer_capacity is None:
        raise ValueError("buffer_capacity must be provided")
    if lr_sa is None:
        raise ValueError("lr_sa must be provided")
    if lr_goal is None:
        raise ValueError("lr_goal must be provided")
    if make_env is None:
        raise ValueError("make_env function must be provided")

    if device is None:
        device = next(q_network.parameters()).device

    if obs_dim is None:
        obs_dim = int(env.observation_space.shape[0])

    if embedding_memory is None:
        embedding_memory = {}

    if replay_task_buffers is None:
        replay_task_buffers = {}

    set_seed(seed)

    if params is None:
        opt = optim.Adam([
            {
                "params": q_network.sa_encoder.parameters(),
                "lr": lr_sa,
            },
            {
                "params": q_network.goal_encoder.parameters(),
                "lr": lr_goal,
            },
        ])
    else:
        opt = optim.Adam(params, lr=lr_goal)

    buffer = TrajectoryReplayBufferDiscrete(
        buffer_capacity,
        obs_dim,
        1,
        device=device,
    )

    goal_arr = np.asarray(goal, dtype=np.float32)

    goal_t_single = torch.tensor(
        goal_arr,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    obs, _ = env.reset()

    global_step = 0
    eval_returns = []
    start_time = time.perf_counter()

    min_steps = None
    min_time = None
    success_streak = 0

    ortho_loss = torch.tensor(0.0, device=device)
    sigreg_loss_val = torch.tensor(0.0, device=device)
    ewc_loss = torch.tensor(0.0, device=device)
    replay_td_loss_val = torch.tensor(0.0, device=device)
    loss = torch.tensor(0.0, device=device)
    td_loss = torch.tensor(0.0, device=device)
    goal_reg_loss = torch.tensor(0.0, device=device)

    num_actions = env.action_space.n

    def compute_td_loss_from_batch(batch, batch_goal):
        obs_t = batch.obs
        act_t = batch.actions.long()
        rew_t = batch.rewards
        next_obs_t = batch.next_obs
        term_t = batch.terminated

        batch_size_local = obs_t.shape[0]

        goal_batch = build_goal_batch(
            batch_goal,
            batch_size_local,
            device,
        )

        with torch.no_grad():
            next_q_vals = (
                q_target_network.q_val_for_argmax_action(
                    next_obs_t,
                    goal_batch,
                )
            )

            next_q = next_q_vals.max(
                dim=-1,
                keepdim=True,
            ).values

            target = (
                rew_t
                + (gamma ** td_steps)
                * (1.0 - term_t)
                * next_q
            )

        act_onehot = F.one_hot(
            act_t.squeeze(-1),
            num_classes=num_actions,
        ).float()

        current_q = q_network(
            obs_t,
            act_onehot,
            goal_batch,
        )

        td_loss_local = F.smooth_l1_loss(
            current_q,
            target,
        )

        return (
            td_loss_local,
            obs_t,
            act_t,
            goal_batch,
        )

    while global_step < total_steps:
        frac = min(
            1.0,
            global_step / eps_decay_steps,
        )

        eps = (
            eps_start
            + frac * (eps_end - eps_start)
        )

        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_vals = (
                    q_network.q_val_for_argmax_action(
                        obs_t,
                        goal_t_single,
                    )
                )

                action = int(
                    q_vals.argmax(
                        dim=-1,
                    ).item()
                )

        next_obs, rew, term, trunc, _ = env.step(action)
        done = term or trunc

        buffer.add_transition(
            obs,
            action,
            rew,
            next_obs,
            term,
            trunc,
        )

        obs = next_obs
        global_step += 1

        if done:
            obs, _ = env.reset()

        if len(buffer) < warmup_steps:
            continue

        if global_step % train_freq != 0:
            continue

        current_batch = buffer.sample(batch_size)

        td_loss, obs_t, act_t, goal_batch = (
            compute_td_loss_from_batch(
                current_batch,
                goal,
            )
        )

        B = obs_t.shape[0]

        if sigreg:
            act_onehot_all = F.one_hot(
                torch.arange(
                    num_actions,
                    device=device,
                ),
                num_classes=num_actions,
            ).float()

            act_onehot_all = (
                act_onehot_all
                .unsqueeze(0)
                .expand(B, -1, -1)
            )

            obs_rep = (
                obs_t
                .unsqueeze(1)
                .expand(-1, num_actions, -1)
            )

            obs_flat = obs_rep.reshape(
                B * num_actions,
                obs_dim,
            )

            act_flat = act_onehot_all.reshape(
                B * num_actions,
                num_actions,
            )

            phi_all = (
                q_network.encode_state_action(
                    obs_flat,
                    act_flat,
                )
            )

            sigreg_loss_val = sigreg_loss(phi_all)

        replay_losses = []
        replay_td_loss_val = torch.tensor(
            0.0,
            device=device,
        )

        eligible_replay_goals = [
            replay_goal
            for replay_goal, replay_buffer in (
                replay_task_buffers.items()
            )
            if (
                replay_buffer is not None
                and len(replay_buffer) >= batch_size
            )
        ]

        if (
            len(eligible_replay_goals) > 0
            and replay_ratio > 0.0
        ):
            if replay_tasks_per_batch is None:
                n_replay_tasks = len(
                    eligible_replay_goals
                )
            else:
                n_replay_tasks = min(
                    int(replay_tasks_per_batch),
                    len(eligible_replay_goals),
                )

            # Deterministic cyclic selection ensures that old
            # tasks are not randomly omitted indefinitely.
            cycle_index = (
                global_step // train_freq
            ) % len(eligible_replay_goals)

            ordered_goals = (
                eligible_replay_goals[cycle_index:]
                + eligible_replay_goals[:cycle_index]
            )

            sampled_replay_goals = ordered_goals[
                :n_replay_tasks
            ]

            replay_batch_size = max(
                1,
                int(
                    batch_size
                    * replay_ratio
                    / n_replay_tasks
                ),
            )

            for replay_goal in sampled_replay_goals:
                replay_buffer = (
                    replay_task_buffers[replay_goal]
                )

                replay_batch = replay_buffer.sample(
                    replay_batch_size
                )

                replay_loss_k, _, _, _ = (
                    compute_td_loss_from_batch(
                        replay_batch,
                        replay_goal,
                    )
                )

                replay_losses.append(replay_loss_k)

            if len(replay_losses) > 0:
                # Important: sum task-specific losses.
                # Do not use torch.stack(...).mean().
                replay_td_loss_val = torch.stack(
                    replay_losses
                ).sum()

        if (
            fisher_diag is not None
            and reference_params is not None
        ):
            ewc_loss = ewc_regulariser_loss(
                q_network,
                reference_params=reference_params,
                fisher_diag=fisher_diag,
                prefix_filter=sa_reg_prefix_filter,
            )
        else:
            ewc_loss = torch.tensor(
                0.0,
                device=device,
            )

        loss = td_loss

        if regulariser == "goal_memory_contrastive":
            if len(embedding_memory) > 0:
                if memory_goals is None:
                    raise ValueError(
                        "memory_goals must be provided"
                    )

                goal_tensor = torch.tensor(
                    goal_arr,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                current_goal_embedding = (
                    q_network.encode_goal(
                        goal_tensor
                    )
                )

                goal_reg_loss = (
                    goal_memory_contrastive_loss(
                        current_embedding=(
                            current_goal_embedding
                        ),
                        embedding_memory=embedding_memory,
                        current_goal=goal,
                        memory_goals=memory_goals,
                        similarity_mode="euclidean",
                        temperature=0.5,
                        pos_threshold=0.6,
                        neg_threshold=0.3,
                        margin=1.0,
                    )
                )

                loss = (
                    loss
                    + reg_alpha * goal_reg_loss
                )

        if len(replay_losses) > 0:
            loss = (
                loss
                + replay_loss_coef
                * replay_td_loss_val
            )

        if regulariser == "repulsion":
            loss = (
                loss
                + reg_alpha * ortho_loss
                + 0.1 * sigreg_loss_val
                + 10000 * ewc_loss
            )

        opt.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            q_network.parameters(),
            10.0,
        )

        opt.step()

        for p, p_target in zip(
            q_network.parameters(),
            q_target_network.parameters(),
        ):
            p_target.data.mul_(
                1.0 - tau
            ).add_(
                tau * p.data
            )

        if (
            global_step % 1000 == 0
            and reference_params is not None
        ):
            delta_means = []

            for name, p in q_network.named_parameters():
                if name in reference_params:
                    delta_sq = (
                        p.detach()
                        - reference_params[name]
                    ).pow(2)

                    delta_means.append(
                        delta_sq.mean().item()
                    )

            if len(delta_means) > 0:
                print(
                    "Mean squared deltas:",
                    delta_means[:5],
                )

        if global_step % 1000 == 0:
            eval_env = make_env(goal)

            goal_eval_t = torch.tensor(
                goal_arr,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            def eval_policy(observation):
                obs_eval_t = torch.tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    q_vals = (
                        q_network.q_val_for_argmax_action(
                            obs_eval_t,
                            goal_eval_t,
                        )
                    )

                    return int(
                        q_vals.argmax(
                            dim=-1,
                        ).item()
                    )

            mean_ret, mean_len = evaluate_policy(
                eval_env,
                eval_policy,
                episodes=8,
            )

            eval_returns.append(
                (global_step, mean_ret)
            )

            print(
                f"[DQN-factorised] "
                f"step={global_step:7d} "
                f"| eps={eps:.3f} "
                f"| eval_return={mean_ret:.3f} "
                f"| eval_len={mean_len:.1f} "
                f"| TD={td_loss.item():.6f} "
                f"| ReplayTD="
                f"{replay_td_loss_val.item():.6f} "
                f"| ReplayTasks="
                f"{len(replay_losses)} "
                f"| Loss={loss.item():.6f}"
            )

            if mean_ret >= early_stop_reward:
                success_streak += 1
            else:
                success_streak = 0

            if (
                enable_early_stop
                and success_streak
                >= early_stop_patience
            ):
                min_steps = global_step
                min_time = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"Good policy achieved at "
                    f"step {global_step} "
                    f"with mean return "
                    f"{mean_ret:.3f}"
                )

                print(
                    f"Early stopping triggered at "
                    f"step {global_step} "
                    f"after {success_streak} "
                    f"consecutive evaluations"
                )

                eval_env.close()
                break

            eval_env.close()

    min_steps = global_step
    min_time = (
        time.perf_counter()
        - start_time
    )

    goal_tensor = torch.tensor(
        goal_arr,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        psi_z = q_network.encode_goal(
            goal_tensor
        )

        task_embedding = (
            psi_z
            .squeeze(0)
            .cpu()
            .numpy()
        )

    obs_probe_np = np.array(
        [5.0, 5.0],
        dtype=np.float32,
    )

    base_env = (
        env.unwrapped
        if hasattr(env, "unwrapped")
        else env
    )

    env_action_names = getattr(
        base_env,
        "action_names",
        ["Up", "Down", "Left", "Right"],
    )

    act_probe_idx = env_action_names.index(
        "Up"
    )

    sa_embedding_mean = (
        extract_mean_sa_embedding(
            q_network=q_network,
            buffer=buffer,
            num_actions=num_actions,
            batch_size=256,
            device=device,
            as_numpy=True,
        )
    )

    sa_embedding_fixed = (
        extract_fixed_probe_sa_embedding(
            q_network=q_network,
            obs_probe=obs_probe_np,
            act_probe_idx=act_probe_idx,
            num_actions=num_actions,
            device=device,
            as_numpy=True,
        )
    )

    sa_batch_final = (
        extract_sa_batch_for_isotropy(
            q_network=q_network,
            buffer=buffer,
            num_actions=num_actions,
            batch_size=1024,
            device=device,
            as_numpy=True,
        )
    )

    env.close()

    return (
        q_network,
        q_target_network,
        eval_returns,
        min_steps,
        min_time,
        task_embedding,
        sa_embedding_mean,
        sa_embedding_fixed,
        sa_batch_final,
        buffer,
    )

def td3_train(
    seed: int = 42,
    actor=None,
    actor_tgt=None,
    q1=None,
    q2=None,
    q1_tgt=None,
    q2_tgt=None,
    env=None,
    make_env=None,
    goal=None,
    buffer_capacity=100000,
    total_steps=500000,
    warmup_steps=10000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_delay=2,
    expl_noise=0.15,
    lr=3e-4,
    train_freq=1,
    gradient_steps=1,
    eval_every=5000,
    device=None,
    obs_dim=None,
    act_dim=None,
    early_stop_reward=0.99,
    early_stop_patience=5,
    enable_early_stop=True,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import time

    # -------------------------
    # Checks and basic setup
    # -------------------------
    if actor is None:
        raise ValueError("actor must be provided")
    if actor_tgt is None:
        raise ValueError("actor_tgt must be provided")
    if q1 is None or q2 is None or q1_tgt is None or q2_tgt is None:
        raise ValueError("q1, q2, q1_tgt, q2_tgt must all be provided")
    if env is None:
        raise ValueError("env must be provided")
    if make_env is None:
        raise ValueError("make_env must be provided")
    if goal is None:
        raise ValueError("goal must be provided")

    set_seed(seed)

    if device is None:
        device = next(actor.parameters()).device

    # -------------------------
    # Observation / action dims
    # -------------------------
    # With flatten_obs=True, env.observation_space is Box([16,]) = [obs(10), achieved_goal(3), desired_goal(3)]
    # We assume:
    #   - obs_dim (for actor) = 10
    #   - goal_dim = 3 (desired_goal, last 3 entries)
    #   - full_obs_dim = 16 (what env returns)
    #
    # If obs_dim is passed, treat it as the actor's obs_dim (10).
    # Otherwise, infer from env.observation_space assuming the layout above.

    if obs_dim is None:
        # Infer assuming flat [obs(10), achieved_goal(3), desired_goal(3)]
        full_obs_dim = env.observation_space.shape[0]
        obs_dim = full_obs_dim - 6  # 16 - 6 = 10
        goal_dim = 3
    else:
        # obs_dim is given (e.g. 10); assume goal_dim = 3
        goal_dim = 3

    if act_dim is None:
        act_dim = env.action_space.shape[0]

    actor = actor.to(device)
    actor_tgt = actor_tgt.to(device)
    q1 = q1.to(device)
    q2 = q2.to(device)
    q1_tgt = q1_tgt.to(device)
    q2_tgt = q2_tgt.to(device)

    actor_tgt.load_state_dict(actor.state_dict())
    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())

    for p in (
        list(actor_tgt.parameters())
        + list(q1_tgt.parameters())
        + list(q2_tgt.parameters())
    ):
        p.requires_grad_(False)

    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    critic_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)

    # Replay buffer stores only the 'obs' part (10D), not the full 16D
    replay = TrajectoryReplayBuffer(
        capacity=buffer_capacity,
        obs_dim=obs_dim,
        action_dim=act_dim,
        device=device,
    )

    goal_arr = np.array(goal, dtype=np.float32)
    goal_t_single = torch.tensor(goal_arr, dtype=torch.float32, device=device).unsqueeze(0)

    total_env_steps = 0
    train_it = 0
    eval_returns = []
    eval_returns_time = []
    start_time = time.perf_counter()
    min_steps = None
    min_time = None
    success_streak = 0

    critic_loss_val = torch.tensor(0.0, device=device)
    actor_loss_val = torch.tensor(0.0, device=device)

    # -------------------------
    # Initial reset
    # -------------------------
    full_obs, _ = env.reset()  # full_obs: [16]
    # Split into obs and goal
    obs = full_obs[:obs_dim]          # [10]
    # goal part (last 3) is ignored here; we use the fixed goal_t_single

    # -------------------------
    # Warmup: random actions
    # -------------------------
    while total_env_steps < warmup_steps:
        action = env.action_space.sample().astype(np.float32)
        next_full_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_obs = next_full_obs[:obs_dim]

        replay.add_transition(obs, action, reward, next_obs, terminated, truncated)

        obs = next_obs
        total_env_steps += 1

        if done:
            full_obs, _ = env.reset()
            obs = full_obs[:obs_dim]

    # -------------------------
    # Main training loop
    # -------------------------
    while total_env_steps < total_steps:
        # Use obs (10D) + fixed goal for actor
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_t = actor(obs_t, goal_t_single)

        action = action_t[0].cpu().numpy()
        action = action + np.random.normal(0.0, expl_noise, size=act_dim)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        next_full_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        next_obs = next_full_obs[:obs_dim]

        replay.add_transition(obs, action, reward, next_obs, terminated, truncated)

        obs = next_obs
        total_env_steps += 1

        if done:
            full_obs, _ = env.reset()
            obs = full_obs[:obs_dim]

        # -------------------------
        # Training step
        # -------------------------
        if len(replay) >= batch_size and total_env_steps % train_freq == 0:
            for _ in range(gradient_steps):
                batch = replay.sample(batch_size)

                obs_b = batch.obs.float()          # [B, 10]
                act_b = batch.actions.float()      # [B, act_dim]
                rew_b = batch.rewards.float()      # [B, 1]
                next_obs_b = batch.next_obs.float()# [B, 10]
                term_b = batch.terminated.float()  # [B, 1]
                trunc_b = batch.truncated.float()  # [B, 1]

                if rew_b.ndim == 1:
                    rew_b = rew_b.unsqueeze(-1)
                if term_b.ndim == 1:
                    term_b = term_b.unsqueeze(-1)
                if trunc_b.ndim == 1:
                    trunc_b = trunc_b.unsqueeze(-1)

                goal_b = goal_t_single.expand(obs_b.shape[0], -1)  # [B, 3]

                with torch.no_grad():
                    noise = torch.randn_like(act_b) * policy_noise
                    noise = torch.clamp(noise, -noise_clip, noise_clip)

                    next_a_b = actor_tgt(next_obs_b, goal_b) + noise
                    next_a_b = torch.clamp(next_a_b, -1.0, 1.0)

                    target_q1 = q1_tgt(next_obs_b, next_a_b, goal_b)
                    target_q2 = q2_tgt(next_obs_b, next_a_b, goal_b)
                    target_q = torch.min(target_q1, target_q2)

                    y = rew_b + gamma * (1.0 - term_b) * target_q

                q1_pred = q1(obs_b, act_b, goal_b)
                q2_pred = q2(obs_b, act_b, goal_b)
                critic_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
                critic_loss_val = critic_loss.detach()

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(q1.parameters()) + list(q2.parameters()), 10.0
                )
                critic_opt.step()

                train_it += 1

                if train_it % policy_delay == 0:
                    actor_actions = actor(obs_b, goal_b)
                    actor_loss = -q1(obs_b, actor_actions, goal_b).mean()
                    actor_loss_val = actor_loss.detach()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                    actor_opt.step()

                    with torch.no_grad():
                        for p, pt in zip(actor.parameters(), actor_tgt.parameters()):
                            pt.data.mul_(1 - tau).add_(tau * p.data)
                        for p, pt in zip(q1.parameters(), q1_tgt.parameters()):
                            pt.data.mul_(1 - tau).add_(tau * p.data)
                        for p, pt in zip(q2.parameters(), q2_tgt.parameters()):
                            pt.data.mul_(1 - tau).add_(tau * p.data)

        # -------------------------
        # Evaluation
        # -------------------------
        if total_env_steps % eval_every == 0:
            eval_env = make_env()

            def eval_actor(o):
                # o is the full flat obs [16]
                obs_part = o[:obs_dim]        # [10]
                # goal_part = o[-goal_dim:]   # if you ever want to use env's goal
                obs_t = torch.as_tensor(
                    obs_part, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    a_t = actor(obs_t, goal_t_single)
                return a_t[0].cpu().numpy().astype(np.float32)

            mean_ret, mean_len = evaluate_policy(eval_env, eval_actor, episodes=8)

            eval_returns.append((total_env_steps, mean_ret))

            print(
                f"[TD3-factorised] step={total_env_steps:7d} | "
                f"train_it={train_it:7d} | "
                f"eval_return={mean_ret:.3f} | eval_len={mean_len:.1f} | "
                f"critic_loss={critic_loss_val.item():.3f} | "
                f"actor_loss={actor_loss_val.item():.3f}"
            )

            if mean_ret >= early_stop_reward:
                success_streak += 1
            else:
                success_streak = 0

            if enable_early_stop and success_streak >= early_stop_patience:
                min_steps = total_env_steps
                min_time = time.perf_counter() - start_time
                print(
                    f"Good policy achieved at step {total_env_steps} "
                    f"with mean return {mean_ret:.3f}"
                )
                print(
                    f"Early stopping triggered at step {total_env_steps} after "
                    f"{success_streak} consecutive evals with "
                    f"mean return >= {early_stop_reward:.2f}"
                )
                eval_env.close()
                break

            eval_env.close()

    # -------------------------
    # Post-training analysis
    # -------------------------
    min_steps = total_env_steps
    min_time = time.perf_counter() - start_time

    goal_tensor = torch.tensor(
        np.array(goal, dtype=np.float32), dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        psi_z = q1.encode_goal(goal_tensor)
        task_embedding = psi_z.squeeze(0).cpu().numpy()

    obs_probe_np = np.array([5.0, 5.0], dtype=np.float32)
    act_probe_np = np.zeros(act_dim, dtype=np.float32)

    sa_embedding_mean = extract_mean_sa_embedding_td3(
        critic=q1,
        buffer=replay,
        batch_size=256,
        device=device,
        as_numpy=True,
    )

    sa_embedding_fixed = extract_fixed_probe_sa_embedding_td3(
        critic=q1,
        obs_probe=obs_probe_np,
        act_probe=act_probe_np,
        device=device,
        as_numpy=True,
    )

    sa_batch_final = extract_sa_batch_for_isotropy_td3(
        critic=q1,
        buffer=replay,
        batch_size=1024,
        device=device,
        as_numpy=True,
    )

    env.close()

    return (
        actor,
        actor_tgt,
        q1,
        q1_tgt,
        q2,
        q2_tgt,
        eval_returns,
        min_steps,
        min_time,
        task_embedding,
        sa_embedding_mean,
        sa_embedding_fixed,
        sa_batch_final,
        replay,
    )

def sac_train(
    seed: int = 42,
    actor=None,
    actor_tgt=None,  # not strictly needed for SAC, but kept for symmetry
    q1=None,
    q2=None,
    q1_tgt=None,
    q2_tgt=None,
    env=None,
    make_env=None,
    goal=None,
    buffer_capacity=100000,
    total_steps=500000,
    warmup_steps=10000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    lr=3e-4,
    train_freq=1,
    gradient_steps=1,
    eval_every=1000,
    device=None,
    obs_dim=None,
    act_dim=None,
    alpha=0.2,           # entropy coefficient
    fixed_alpha=True,    # if False, tune alpha automatically
    target_entropy=None, # used if fixed_alpha=False
    early_stop_reward=-5.0,
    early_stop_patience=5,
    enable_early_stop=True,
):

    if actor is None:
        raise ValueError("actor must be provided")
    if q1 is None or q2 is None or q1_tgt is None or q2_tgt is None:
        raise ValueError("q1, q2, q1_tgt, q2_tgt must all be provided")
    if env is None:
        raise ValueError("env must be provided")
    if make_env is None:
        raise ValueError("make_env must be provided")
    if goal is None:
        raise ValueError("goal must be provided")

    set_seed(seed)

    if device is None:
        device = next(actor.parameters()).device

    # -------------------------
    # Observation / action dims
    # -------------------------
    if obs_dim is None:
        full_obs_dim = env.observation_space.shape[0]
        obs_dim = full_obs_dim - 6  # assuming [obs(10), achieved_goal(3), desired_goal(3)]
        goal_dim = 3
    else:
        goal_dim = 3

    if act_dim is None:
        act_dim = env.action_space.shape[0]

    actor = actor.to(device)
    if actor_tgt is not None:
        actor_tgt = actor_tgt.to(device)
    q1 = q1.to(device)
    q2 = q2.to(device)
    q1_tgt = q1_tgt.to(device)
    q2_tgt = q2_tgt.to(device)

    # Initialize target critics
    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())

    for p in list(q1_tgt.parameters()) + list(q2_tgt.parameters()):
        p.requires_grad_(False)

    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    critic_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)

    if not fixed_alpha:
        log_alpha = torch.zeros((), device=device, requires_grad=True)
        alpha_opt = optim.Adam([log_alpha], lr=lr)
        if target_entropy is None:
            target_entropy = -act_dim
    else:
        log_alpha = torch.tensor(np.log(alpha), device=device)
        alpha_opt = None

    def get_alpha():
        return log_alpha.exp()

    replay = TrajectoryReplayBuffer(
        capacity=buffer_capacity,
        obs_dim=obs_dim,
        action_dim=act_dim,
        device=device,
    )

    goal_arr = np.array(goal, dtype=np.float32)
    goal_t_single = torch.tensor(goal_arr, dtype=torch.float32, device=device).unsqueeze(0)

    total_env_steps = 0
    train_it = 0
    eval_returns = []
    start_time = time.perf_counter()
    min_steps = None
    min_time = None
    success_streak = 0

    critic_loss_val = torch.tensor(0.0, device=device)
    actor_loss_val = torch.tensor(0.0, device=device)
    alpha_val = get_alpha().item()

    # -------------------------
    # Initial reset
    # -------------------------
    full_obs, _ = env.reset()
    obs = full_obs[:obs_dim]

    # -------------------------
    # Warmup: random actions
    # -------------------------
    while total_env_steps < warmup_steps:
        action = env.action_space.sample().astype(np.float32)
        next_full_obs, reward, terminated, truncated, info = env.step(action)
        reward = float(reward) * 0.1  # scale down dense rewards
        done = terminated or truncated

        next_obs = next_full_obs[:obs_dim]
        replay.add_transition(obs, action, reward, next_obs, terminated, truncated)

        obs = next_obs
        total_env_steps += 1

        if done:
            full_obs, _ = env.reset()
            obs = full_obs[:obs_dim]

    # -------------------------
    # Main training loop
    # -------------------------
    while total_env_steps < total_steps:
        # Sample action from stochastic policy
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean_t, logstd_t = actor(obs_t, goal_t_single)
            std_t = logstd_t.exp()
            normal = torch.distributions.Normal(mean_t, std_t)
            action_t = normal.rsample()  # reparameterized
            action = action_t[0].cpu().numpy()

        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        next_full_obs, reward, terminated, truncated, info = env.step(action)
        reward = float(reward) * 0.1  # scale down dense rewards
        done = terminated or truncated

        next_obs = next_full_obs[:obs_dim]
        replay.add_transition(obs, action, reward, next_obs, terminated, truncated)

        obs = next_obs
        total_env_steps += 1

        if done:
            full_obs, _ = env.reset()
            obs = full_obs[:obs_dim]

        # -------------------------
        # Training step
        # -------------------------
        if len(replay) >= batch_size and total_env_steps % train_freq == 0:
            for _ in range(gradient_steps):
                batch = replay.sample(batch_size)

                obs_b = batch.obs.float()           # [B, obs_dim]
                act_b = batch.actions.float()       # [B, act_dim]
                rew_b = batch.rewards.float()       # [B, 1]
                next_obs_b = batch.next_obs.float() # [B, obs_dim]
                term_b = batch.terminated.float()   # [B, 1]
                trunc_b = batch.truncated.float()   # [B, 1]

                if rew_b.ndim == 1:
                    rew_b = rew_b.unsqueeze(-1)
                if term_b.ndim == 1:
                    term_b = term_b.unsqueeze(-1)
                if trunc_b.ndim == 1:
                    trunc_b = trunc_b.unsqueeze(-1)

                goal_b = goal_t_single.expand(obs_b.shape[0], -1)  # [B, goal_dim]

                # ---------------------
                # Critic update
                # ---------------------
                with torch.no_grad():
                    next_mean, next_logstd = actor_tgt(next_obs_b, goal_b)
                    next_std = next_logstd.exp()
                    next_dist = torch.distributions.Normal(next_mean, next_std)
                    next_action = next_dist.rsample()
                    next_log_prob = next_dist.log_prob(next_action).sum(dim=-1, keepdim=True)

                    target_q1 = q1_tgt(next_obs_b, next_action, goal_b)
                    target_q2 = q2_tgt(next_obs_b, next_action, goal_b)
                    target_q = torch.min(target_q1, target_q2)

                    y = rew_b + gamma * (1.0 - term_b) * (target_q - alpha * next_log_prob)

                q1_pred = q1(obs_b, act_b, goal_b)
                q2_pred = q2(obs_b, act_b, goal_b)
                critic_loss = F.mse_loss(q1_pred, y) + F.mse_loss(q2_pred, y)
                critic_loss_val = critic_loss.detach()

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(q1.parameters()) + list(q2.parameters()), 10.0
                )
                critic_opt.step()

                train_it += 1

                # ---------------------
                # Actor update
                # ---------------------
                mean_b, logstd_b = actor(obs_b, goal_b)
                std_b = logstd_b.exp()
                dist = torch.distributions.Normal(mean_b, std_b)
                action_b = dist.rsample()
                log_prob = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

                q1_pi = q1(obs_b, action_b, goal_b)
                q2_pi = q2(obs_b, action_b, goal_b)
                q_pi = torch.min(q1_pi, q2_pi)

                alpha = get_alpha()
                actor_loss = (alpha * log_prob - q_pi).mean()
                actor_loss_val = actor_loss.detach()

                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                actor_opt.step()

                # ---------------------
                # Alpha update (if tunable)
                # ---------------------
                if not fixed_alpha:
                    with torch.no_grad():
                        _, logstd_curr = actor(obs_b, goal_b)
                        std_curr = logstd_curr.exp()
                        dist_curr = torch.distributions.Normal(
                            torch.zeros_like(logstd_curr), std_curr
                        )
                        curr_log_prob = dist_curr.log_prob(
                            torch.zeros_like(logstd_curr)
                        ).sum(dim=-1, keepdim=True)
                        # Approximate entropy via -log_prob under current policy
                        # More correctly, sample actions and compute log_prob, but this is a common shortcut.
                        # For simplicity, use mean log_prob of sampled actions:
                        _, ls = actor(obs_b, goal_b)
                        sd = ls.exp()
                        d = torch.distributions.Normal(torch.zeros_like(ls), sd)
                        lp = d.log_prob(torch.zeros_like(ls)).sum(dim=-1, keepdim=True)
                        entropy = -lp

                    alpha_loss = -(log_alpha * (entropy + target_entropy).detach()).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()
                    alpha_val = get_alpha().item()

                # ---------------------
                # Target network update
                # ---------------------
                with torch.no_grad():
                    for p, pt in zip(q1.parameters(), q1_tgt.parameters()):
                        pt.data.mul_(1 - tau).add_(tau * p.data)
                    for p, pt in zip(q2.parameters(), q2_tgt.parameters()):
                        pt.data.mul_(1 - tau).add_(tau * p.data)

        # -------------------------
        # Evaluation
        # -------------------------
        if total_env_steps % eval_every == 0:
            eval_env = make_env()()

            def eval_actor(o):
                obs_part = o[:obs_dim]
                obs_t = torch.as_tensor(
                    obs_part, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    mean_t, logstd_t = actor(obs_t, goal_t_single)
                return mean_t[0].cpu().numpy().astype(np.float32)

            mean_ret, mean_len = evaluate_policy(eval_env, eval_actor, episodes=8)

            eval_returns.append((total_env_steps, mean_ret))

            print(
                f"[SAC-factorised] step={total_env_steps:7d} | "
                f"train_it={train_it:7d} | "
                f"eval_return={mean_ret:.3f} | eval_len={mean_len:.1f} | "
                f"critic_loss={critic_loss_val.item():.3f} | "
                f"actor_loss={actor_loss_val.item():.3f} | "
                f"alpha={alpha_val:.3f}"
            )

            if mean_ret >= early_stop_reward:
                success_streak += 1
            else:
                success_streak = 0

            if enable_early_stop and success_streak >= early_stop_patience:
                min_steps = total_env_steps
                min_time = time.perf_counter() - start_time
                print(
                    f"Good policy achieved at step {total_env_steps} "
                    f"with mean return {mean_ret:.3f}"
                )
                print(
                    f"Early stopping triggered at step {total_env_steps} after "
                    f"{success_streak} consecutive evals with "
                    f"mean return >= {early_stop_reward:.2f}"
                )
                eval_env.close()
                break

            eval_env.close()

    # -------------------------
    # Post-training analysis
    # -------------------------
    min_steps = total_env_steps
    min_time = time.perf_counter() - start_time

    goal_tensor = torch.tensor(
        np.array(goal, dtype=np.float32), dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        psi_z = q1.encode_goal(goal_tensor)
        task_embedding = psi_z.squeeze(0).cpu().numpy()

    obs_probe_np = np.array([5.0, 5.0], dtype=np.float32)
    act_probe_np = np.zeros(act_dim, dtype=np.float32)

    sa_embedding_mean = extract_mean_sa_embedding_td3(
        critic=q1,
        buffer=replay,
        batch_size=256,
        device=device,
        as_numpy=True,
    )

    sa_embedding_fixed = extract_fixed_probe_sa_embedding_td3(
        critic=q1,
        obs_probe=obs_probe_np,
        act_probe=act_probe_np,
        device=device,
        as_numpy=True,
    )

    sa_batch_final = extract_sa_batch_for_isotropy_td3(
        critic=q1,
        buffer=replay,
        batch_size=1024,
        device=device,
        as_numpy=True,
    )

    env.close()

    return (
        actor,
        actor_tgt,
        q1,
        q1_tgt,
        q2,
        q2_tgt,
        eval_returns,
        min_steps,
        min_time,
        task_embedding,
        sa_embedding_mean,
        sa_embedding_fixed,
        sa_batch_final,
        replay,
    )

def dqn_train_new(
    seed: int = 42,
    q_network=None,
    q_target_network=None,
    env=None,
    buffer_capacity=None,
    lr_sa=None,
    lr_goal=None,
    obs_dim=None,
    device=None,
    total_steps=100000,
    warmup_steps=5000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50000,
    train_freq=4,
    goal=None,
    params=None,
    embedding_memory=None,
    td_steps=1,
    make_env=None,
    early_stop_reward=0.99,
    early_stop_patience=3,
    enable_early_stop=True,

    # SA representation-stability settings
    use_sa_stability=True,
    project_sa_gradients=True,
    lambda_sa_stability=1.0,
    sa_anchor_batch_size=128,
    sa_anchor_memory_per_goal=512,
    max_anchor_sets=20,

    # Goal-similarity settings
    retrieved_prototype=None,
    use_goal_similarity_anchor=False,
    goal_anchor_coef=0.05,
    goal_anchor_steps=10000,
):
    if q_network is None:
        raise ValueError("q_network must be provided")

    if q_target_network is None:
        raise ValueError("q_target_network must be provided")

    if env is None:
        raise ValueError("env must be provided")

    if goal is None:
        raise ValueError("goal must be provided")

    if buffer_capacity is None:
        raise ValueError("buffer_capacity must be provided")

    if lr_sa is None:
        raise ValueError("lr_sa must be provided")

    if lr_goal is None:
        raise ValueError("lr_goal must be provided")

    if make_env is None:
        raise ValueError(
            "make_env function must be provided to create "
            "the evaluation environment."
        )

    if device is None:
        device = next(q_network.parameters()).device

    if obs_dim is None:
        obs_dim = int(
            env.observation_space.shape[0]
        )

    if embedding_memory is None:
        embedding_memory = {}

    if "sa_anchors" not in embedding_memory:
        embedding_memory["sa_anchors"] = []


    set_seed(seed)


    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    if params is None:
        optimizer = optim.Adam(
            [
                {
                    "params": q_network.sa_encoder.parameters(),
                    "lr": lr_sa,
                },
                {
                    "params": q_network.goal_encoder.parameters(),
                    "lr": lr_goal,
                },
            ]
        )
    else:
        optimizer = optim.Adam(
            params,
            lr=lr_goal,
        )


    # ---------------------------------------------------------
    # Replay buffer
    # ---------------------------------------------------------

    buffer = TrajectoryReplayBufferDiscrete(
        buffer_capacity,
        obs_dim,
        1,
        device=device,
    )


    # ---------------------------------------------------------
    # Goal setup
    # ---------------------------------------------------------

    goal_arr = np.asarray(
        goal,
        dtype=np.float32,
    )

    goal_t_single = torch.tensor(
        goal_arr,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    num_actions = env.action_space.n


    # ---------------------------------------------------------
    # Nested SA-stability helpers
    # ---------------------------------------------------------

    def sample_old_sa_anchors(batch_size):
        memories = embedding_memory.get(
            "sa_anchors",
            [],
        )

        if len(memories) == 0:
            return None

        valid_memories = [
            memory
            for memory in memories
            if (
                memory is not None
                and memory["states"].shape[0] > 0
            )
        ]

        if len(valid_memories) == 0:
            return None

        selected_memory = valid_memories[
            np.random.randint(
                low=0,
                high=len(valid_memories),
            )
        ]

        num_available = (
            selected_memory["states"].shape[0]
        )

        sample_size = min(
            batch_size,
            num_available,
        )

        indices = torch.randint(
            low=0,
            high=num_available,
            size=(sample_size,),
        )

        states = selected_memory["states"][indices]
        actions = selected_memory["actions"][indices]
        old_features = selected_memory["features"][indices]

        return {
            "states": states.to(device),
            "actions": actions.to(device),
            "features": old_features.to(device),
        }


    def compute_sa_stability_loss(anchor_batch):
        """
        Preserve phi(s,a) on old environment anchors.
        """

        if anchor_batch is None:
            return None

        states = anchor_batch["states"]
        actions = anchor_batch["actions"]
        old_features = anchor_batch["features"]

        action_onehot = F.one_hot(
            actions.long().view(-1),
            num_classes=num_actions,
        ).float()

        current_features = (
            q_network.encode_state_action(
                states,
                action_onehot,
            )
        )

        current_features = F.normalize(
            current_features,
            dim=-1,
        )

        old_features = F.normalize(
            old_features,
            dim=-1,
        )

        return F.mse_loss(
            current_features,
            old_features,
        )


    def flatten_grads(parameters):
        """
        Flatten gradients into one vector.
        """

        flat_grads = []

        for parameter in parameters:
            if parameter.grad is None:
                flat_grads.append(
                    torch.zeros_like(
                        parameter
                    ).reshape(-1)
                )
            else:
                flat_grads.append(
                    parameter.grad.detach()
                    .clone()
                    .reshape(-1)
                )

        if len(flat_grads) == 0:
            return torch.empty(
                0,
                device=device,
            )

        return torch.cat(flat_grads)


    def assign_flat_grad(
        parameters,
        flat_grad,
    ):
        """
        Assign a flattened gradient back to parameters.
        """

        offset = 0

        for parameter in parameters:
            numel = parameter.numel()

            parameter.grad = flat_grad[
                offset:offset + numel
            ].view_as(
                parameter
            ).clone()

            offset += numel


    def projected_gradient_step(
        new_loss,
        stability_loss,
    ):
        """
        Compute the normal new-task gradient and the
        stability gradient.

        Project only the SA-encoder gradient if the
        two objectives conflict.

        Goal-encoder gradients are not projected.
        """

        all_parameters = [
            parameter
            for parameter in q_network.parameters()
            if parameter.requires_grad
        ]

        sa_parameters = [
            parameter
            for parameter in q_network.sa_encoder.parameters()
            if parameter.requires_grad
        ]

        # -----------------------------------------------------
        # 1. New-task gradients
        # -----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True,
        )

        new_loss.backward(
            retain_graph=True,
        )

        saved_new_grads = {}

        for parameter in all_parameters:
            if parameter.grad is None:
                saved_new_grads[id(parameter)] = None
            else:
                saved_new_grads[id(parameter)] = (
                    parameter.grad.detach().clone()
                )

        new_sa_grad = flatten_grads(
            sa_parameters,
        )

        # -----------------------------------------------------
        # 2. Stability gradients
        # -----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True,
        )

        if stability_loss is not None:
            stability_loss.backward()

        stability_sa_grad = flatten_grads(
            sa_parameters,
        )

        # -----------------------------------------------------
        # 3. Conflict test
        # -----------------------------------------------------

        dot_product = torch.dot(
            new_sa_grad,
            stability_sa_grad,
        )

        stability_norm_sq = torch.dot(
            stability_sa_grad,
            stability_sa_grad,
        )

        projected_sa_grad = new_sa_grad.clone()
        was_projected = False

        if (
            dot_product < 0.0
            and stability_norm_sq > 1e-12
        ):
            projection_coefficient = (
                dot_product
                / (stability_norm_sq + 1e-12)
            )

            projected_sa_grad = (
                new_sa_grad
                - projection_coefficient
                * stability_sa_grad
            )

            was_projected = True

        # -----------------------------------------------------
        # 4. Restore normal new-task gradients
        # -----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True,
        )

        for parameter in all_parameters:
            saved_grad = saved_new_grads[
                id(parameter)
            ]

            if saved_grad is None:
                parameter.grad = None
            else:
                parameter.grad = saved_grad.clone()

        assign_flat_grad(
            sa_parameters,
            projected_sa_grad,
        )

        return {
            "gradient_dot": float(
                dot_product.detach().cpu()
            ),
            "new_sa_grad_norm": float(
                new_sa_grad.norm().detach().cpu()
            ),
            "stability_sa_grad_norm": float(
                stability_sa_grad.norm().detach().cpu()
            ),
            "projected_sa_grad_norm": float(
                projected_sa_grad.norm().detach().cpu()
            ),
            "was_projected": was_projected,
        }


    def create_anchor_memory():
        """
        Create compact SA functional memory after training.

        Stores:
            states
            actions
            phi(s,a)

        Does not store a network checkpoint.
        """

        if len(buffer) == 0:
            return None

        sample_size = min(
            sa_anchor_memory_per_goal,
            len(buffer),
        )

        batch = buffer.sample(
            sample_size
        )

        states = batch.obs.detach().to(device)
        actions = batch.actions.long().detach().to(device)
        actions = actions.view(-1)

        action_onehot = F.one_hot(
            actions,
            num_classes=num_actions,
        ).float()

        with torch.no_grad():
            features = (
                q_network.encode_state_action(
                    states,
                    action_onehot,
                )
            )

        return {
            "states": states.detach().cpu(),
            "actions": actions.detach().cpu(),
            "features": features.detach().cpu(),
        }


    # ---------------------------------------------------------
    # Goal-similarity anchor helper
    # ---------------------------------------------------------

    def compute_goal_anchor_loss(
        goal_batch,
    ):
        """
        Match the current goal-encoder output to the
        retrieved prototype embedding.

        The retrieved prototype is treated as a fixed
        target and receives no gradient.
        """

        if retrieved_prototype is None:
            return torch.zeros(
                (),
                device=device,
            )

        current_goal_embedding = (
            q_network.encode_goal(
                goal_batch
            )
        )

        prototype = torch.as_tensor(
            retrieved_prototype,
            dtype=current_goal_embedding.dtype,
            device=device,
        )

        prototype = prototype.view(
            1,
            -1,
        ).expand_as(
            current_goal_embedding
        )

        prototype = prototype.detach()

        return F.mse_loss(
            current_goal_embedding,
            prototype,
        )


    # ---------------------------------------------------------
    # Training state
    # ---------------------------------------------------------

    obs, _ = env.reset()

    global_step = 0
    eval_returns = []
    eval_returns_time = []

    start_time = time.perf_counter()

    min_steps = None
    min_time = None

    success_streak = 0

    loss = torch.zeros(
        (),
        device=device,
    )

    td_loss = torch.zeros(
        (),
        device=device,
    )

    goal_anchor_loss = torch.zeros(
        (),
        device=device,
    )

    goal_anchor_loss_value = 0.0
    goal_anchor_weight = 0.0

    stability_loss = None
    stability_loss_value = 0.0
    projection_stats = None

    has_old_sa_memory = (
        use_sa_stability
        and len(
            embedding_memory["sa_anchors"]
        ) > 0
    )


    # ---------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------

    while global_step < total_steps:

        frac = min(
            1.0,
            global_step / max(
                1,
                eps_decay_steps,
            ),
        )

        eps = (
            eps_start
            + frac * (eps_end - eps_start)
        )


        # -----------------------------------------------------
        # Action selection
        # -----------------------------------------------------

        if np.random.random() < eps:
            action = env.action_space.sample()

        else:
            obs_t_single = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = (
                    q_network.q_val_for_argmax_action(
                        obs_t_single,
                        goal_t_single,
                    )
                )

                action = int(
                    q_values.argmax(
                        dim=-1,
                    ).item()
                )


        # -----------------------------------------------------
        # Environment transition
        # -----------------------------------------------------

        next_obs, reward, terminated, truncated, _ = (
            env.step(action)
        )

        done = terminated or truncated

        buffer.add_transition(
            obs,
            action,
            reward,
            next_obs,
            terminated,
            truncated,
        )

        obs = next_obs
        global_step += 1

        if done:
            obs, _ = env.reset()


        # -----------------------------------------------------
        # Training update
        # -----------------------------------------------------

        if (
            len(buffer) >= warmup_steps
            and global_step % train_freq == 0
        ):
            batch = buffer.sample(
                batch_size
            )

            obs_t = batch.obs
            act_t = batch.actions.long()
            rew_t = batch.rewards
            next_obs_t = batch.next_obs
            term_t = batch.terminated

            if rew_t.ndim == 1:
                rew_t = rew_t.unsqueeze(-1)

            if term_t.ndim == 1:
                term_t = term_t.unsqueeze(-1)

            goal_batch = goal_t_single.expand(
                obs_t.shape[0],
                -1,
            )


            # -------------------------------------------------
            # Double-DQN target
            # -------------------------------------------------

            with torch.no_grad():
                next_q_online = (
                    q_network.q_val_for_argmax_action(
                        next_obs_t,
                        goal_batch,
                    )
                )

                next_actions = next_q_online.argmax(
                    dim=-1,
                    keepdim=True,
                )

                next_action_onehot = F.one_hot(
                    next_actions.squeeze(-1),
                    num_classes=num_actions,
                ).float()

                next_q_target = q_target_network(
                    next_obs_t,
                    next_action_onehot,
                    goal_batch,
                )

                if next_q_target.ndim == 1:
                    next_q_target = (
                        next_q_target.unsqueeze(-1)
                    )

                gamma_final = gamma ** td_steps

                target = (
                    rew_t
                    + gamma_final
                    * (1.0 - term_t)
                    * next_q_target
                )


            # -------------------------------------------------
            # Current Q-value
            # -------------------------------------------------

            act_onehot = F.one_hot(
                act_t.squeeze(-1),
                num_classes=num_actions,
            ).float()

            current_q = q_network(
                obs_t,
                act_onehot,
                goal_batch,
            )

            if current_q.ndim == 1:
                current_q = (
                    current_q.unsqueeze(-1)
                )

            td_loss = F.mse_loss(
                current_q,
                target,
            )


            # -------------------------------------------------
            # Goal-similarity anchor
            # -------------------------------------------------

            goal_anchor_loss = torch.zeros(
                (),
                device=device,
            )

            goal_anchor_weight = 0.0

            anchor_is_active = (
                use_goal_similarity_anchor
                and retrieved_prototype is not None
                and goal_anchor_coef > 0.0
                and goal_anchor_steps > 0
                and global_step < goal_anchor_steps
            )

            if anchor_is_active:
                anchor_progress = (
                    global_step
                    / float(goal_anchor_steps)
                )

                goal_anchor_weight = (
                    goal_anchor_coef
                    * max(
                        0.0,
                        1.0 - anchor_progress,
                    )
                )

                if goal_anchor_weight > 0.0:
                    goal_anchor_loss = (
                        compute_goal_anchor_loss(
                            goal_batch,
                        )
                    )

            goal_anchor_loss_value = (
                goal_anchor_loss.detach().item()
            )


            # -------------------------------------------------
            # Combined new-task loss
            # -------------------------------------------------

            loss = (
                td_loss
                + goal_anchor_weight
                * goal_anchor_loss
            )


            # -------------------------------------------------
            # SA-stability loss
            # -------------------------------------------------

            stability_loss = None
            stability_loss_value = 0.0
            projection_stats = None

            if has_old_sa_memory:
                anchor_batch = sample_old_sa_anchors(
                    sa_anchor_batch_size,
                )

                if anchor_batch is not None:
                    stability_loss = (
                        compute_sa_stability_loss(
                            anchor_batch,
                        )
                    )

                    stability_loss = (
                        lambda_sa_stability
                        * stability_loss
                    )

                    stability_loss_value = (
                        stability_loss.detach().item()
                    )


            # -------------------------------------------------
            # Parameter update
            # -------------------------------------------------

            if (
                project_sa_gradients
                and stability_loss is not None
            ):
                projection_stats = (
                    projected_gradient_step(
                        new_loss=loss,
                        stability_loss=stability_loss,
                    )
                )

                nn.utils.clip_grad_norm_(
                    q_network.parameters(),
                    max_norm=10.0,
                )

                optimizer.step()

            else:
                optimizer.zero_grad(
                    set_to_none=True,
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    q_network.parameters(),
                    max_norm=10.0,
                )

                optimizer.step()


            # -------------------------------------------------
            # Soft target-network update
            # -------------------------------------------------

            with torch.no_grad():
                for parameter, target_parameter in zip(
                    q_network.parameters(),
                    q_target_network.parameters(),
                ):
                    target_parameter.data.mul_(
                        1.0 - tau,
                    )

                    target_parameter.data.add_(
                        tau * parameter.data,
                    )


        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        if global_step % 1000 == 0:
            eval_env = make_env(goal)

            goal_eval_arr = np.asarray(
                goal,
                dtype=np.float32,
            )

            goal_eval_t = torch.tensor(
                goal_eval_arr,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)


            def eval_policy(observation):
                observation_t = torch.tensor(
                    observation,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)

                with torch.no_grad():
                    q_values = (
                        q_network.q_val_for_argmax_action(
                            observation_t,
                            goal_eval_t,
                        )
                    )

                    return int(
                        q_values.argmax(
                            dim=-1,
                        ).item()
                    )


            mean_return, mean_length = (
                evaluate_policy(
                    eval_env,
                    eval_policy,
                    episodes=8,
                )
            )

            eval_returns.append(
                (
                    global_step,
                    mean_return,
                )
            )

            eval_returns_time.append(
                (
                    time.perf_counter()
                    - start_time,
                    mean_return,
                )
            )


            if projection_stats is None:
                projection_text = (
                    "projection=n/a"
                )
            else:
                projection_text = (
                    f"projection="
                    f"{projection_stats['was_projected']} "
                    f"dot="
                    f"{projection_stats['gradient_dot']:.3e} "
                    f"g_new="
                    f"{projection_stats['new_sa_grad_norm']:.3e} "
                    f"g_stable="
                    f"{projection_stats['stability_sa_grad_norm']:.3e}"
                )


            print(
                f"[DQN-factorised] "
                f"step={global_step:7d} | "
                f"eps={eps:.3f} | "
                f"eval_return={mean_return:.3f} | "
                f"eval_len={mean_length:.1f} | "
                f"TD={td_loss.item():.4e} | "
                f"GoalAnchor="
                f"{goal_anchor_loss_value:.4e} | "
                f"AnchorWeight="
                f"{goal_anchor_weight:.4e} | "
                f"SA-stability="
                f"{stability_loss_value:.4e} | "
                f"{projection_text}"
            )


            if mean_return >= early_stop_reward:
                success_streak += 1
            else:
                success_streak = 0


            if (
                enable_early_stop
                and success_streak >= early_stop_patience
            ):
                min_steps = global_step

                min_time = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"Good policy achieved at step "
                    f"{global_step} with mean return "
                    f"{mean_return:.3f}"
                )

                print(
                    f"Early stopping triggered at step "
                    f"{global_step} after "
                    f"{success_streak} consecutive evaluations."
                )

                eval_env.close()
                break


            eval_env.close()


    # ---------------------------------------------------------
    # Final statistics
    # ---------------------------------------------------------

    min_steps = global_step

    min_time = (
        time.perf_counter()
        - start_time
    )

    goal_tensor = torch.tensor(
        np.asarray(
            goal,
            dtype=np.float32,
        ),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        psi_z = q_network.encode_goal(
            goal_tensor,
        )

        task_embedding = (
            psi_z.squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )


    # ---------------------------------------------------------
    # Save compact SA anchor memory
    # ---------------------------------------------------------

    anchor_memory = create_anchor_memory()

    if anchor_memory is not None:
        embedding_memory["sa_anchors"].append(
            anchor_memory
        )

        if len(
            embedding_memory["sa_anchors"]
        ) > max_anchor_sets:
            embedding_memory["sa_anchors"] = (
                embedding_memory["sa_anchors"]
                [-max_anchor_sets:]
            )


    # ---------------------------------------------------------
    # SA embedding diagnostics
    # ---------------------------------------------------------

    obs_probe_np = np.array(
        [5.0, 5.0],
        dtype=np.float32,
    )

    base_env = (
        env.unwrapped
        if hasattr(env, "unwrapped")
        else env
    )

    env_action_names = getattr(
        base_env,
        "action_names",
        ["Up", "Down", "Left", "Right"],
    )

    act_probe_idx = env_action_names.index(
        "Up",
    )

    sa_embedding_mean = (
        extract_mean_sa_embedding(
            q_network=q_network,
            buffer=buffer,
            num_actions=num_actions,
            batch_size=256,
            device=device,
            as_numpy=True,
        )
    )

    sa_embedding_fixed = (
        extract_fixed_probe_sa_embedding(
            q_network=q_network,
            obs_probe=obs_probe_np,
            act_probe_idx=act_probe_idx,
            num_actions=num_actions,
            device=device,
            as_numpy=True,
        )
    )

    sa_batch_final = (
        extract_sa_batch_for_isotropy(
            q_network=q_network,
            buffer=buffer,
            num_actions=num_actions,
            batch_size=1024,
            device=device,
            as_numpy=True,
        )
    )


    env.close()


    return (
        q_network,
        q_target_network,
        eval_returns,
        min_steps,
        min_time,
        task_embedding,
        sa_embedding_mean,
        sa_embedding_fixed,
        sa_batch_final,
        buffer,
    )

def dqn_train_rotation(
    seed: int = 42,
    q_network=None,
    q_target_network=None,
    env=None,
    buffer_capacity=None,
    lr=None,
    obs_dim=None,
    device=None,
    total_steps=100000,
    warmup_steps=5000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50000,
    train_freq=4,
    goal=None,
    params=None,
    regulariser=None,
    reference_params=None,
    fisher_diag=None,
    sa_reg_prefix_filter="sa_encoder",
    sigreg=False,
    td_steps=1,
    make_env=None,
    early_stop_reward=0.99,
    early_stop_patience=3,
    enable_early_stop=True,
    save_embeddings=True,
    task_id=None,
    embedding_memory=None,  # Must be passed in from outside
):
    # === Validation ===
    if q_network is None:
        raise ValueError("q_network must be provided")
    if q_target_network is None:
        raise ValueError("q_target_network must be provided")
    if env is None:
        raise ValueError("env must be provided")
    if goal is None:
        raise ValueError("goal must be provided")
    if buffer_capacity is None:
        raise ValueError("buffer_capacity must be provided")
    if lr is None:
        raise ValueError("lr must be provided")
    if make_env is None:
        raise ValueError("make_env function must be provided")
    if device is None:
        device = next(q_network.parameters()).device
    if obs_dim is None:
        obs_dim = int(env.observation_space.shape[0])
    if embedding_memory is None:
        embedding_memory = {}
    
    set_seed(seed)
    
    if params is None:
        opt = optim.Adam([
            {"params": q_network.sa_encoder.parameters(), "lr": lr},
            {"params": q_network.goal_encoder.parameters(), "lr": lr},
        ])
    else:
        opt = optim.Adam(params, lr=lr)
    
    buffer = TrajectoryReplayBufferDiscrete(buffer_capacity, obs_dim, 1, device=device)
    
    goal_arr = np.array(goal, dtype=np.float32)
    goal_t_single = torch.tensor(goal_arr, dtype=torch.float32, device=device).unsqueeze(0)
    
    obs, _ = env.reset()
    global_step = 0
    eval_returns = []
    start_time = time.perf_counter()
    min_steps = None
    min_time = None
    loss = torch.zeros((), device=device)
    td_loss = torch.zeros((), device=device)
    success_streak = 0
    
    num_actions = env.action_space.n
    
    # === Training loop (unchanged from your original) ===
    while global_step < total_steps:
        frac = min(1.0, global_step / eps_decay_steps)
        eps = eps_start + frac * (eps_end - eps_start)
        
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = q_network.q_val_for_argmax_action(obs_t, goal_t_single)
                action = int(q_vals.argmax(dim=-1).item())
        
        next_obs, rew, term, trunc, _ = env.step(action)
        done = term or trunc
        
        buffer.add_transition(obs, action, rew, next_obs, term, trunc)
        obs = next_obs
        global_step += 1
        
        if done:
            obs, _ = env.reset()
        
        if len(buffer) >= warmup_steps and global_step % train_freq == 0:
            batch = buffer.sample(batch_size)
            
            obs_t = batch.obs
            act_t = batch.actions.long()
            rew_t = batch.rewards
            next_obs_t = batch.next_obs
            term_t = batch.terminated
            trunc_t = batch.truncated
            
            goal_batch = goal_t_single.expand(obs_t.shape[0], -1)
            B = obs_t.shape[0]
            
            with torch.no_grad():
                next_q_vals = q_target_network.q_val_for_argmax_action(next_obs_t, goal_batch)
                next_q = next_q_vals.max(dim=-1, keepdim=True).values
                gamma_final = gamma ** td_steps
                target = rew_t + gamma_final * (1.0 - term_t) * next_q
            
            act_onehot = F.one_hot(act_t.squeeze(-1), num_classes=num_actions).float()
            current_q = q_network(obs_t, act_onehot, goal_batch)
            td_loss = F.mse_loss(current_q, target)
            
            if sigreg:
                act_onehot_all = F.one_hot(
                    torch.arange(num_actions, device=device),
                    num_classes=num_actions
                ).float()
                act_onehot_all = act_onehot_all.unsqueeze(0).expand(B, -1, -1)
                obs_rep = obs_t.unsqueeze(1).expand(-1, num_actions, -1)
                obs_flat = obs_rep.reshape(B * num_actions, obs_dim)
                act_flat = act_onehot_all.reshape(B * num_actions, num_actions)
                phi_all = q_network.encode_state_action(obs_flat, act_flat)
                sigreg_loss_val = sigreg_loss(phi_all)
            
            if fisher_diag is not None:
                ewc_loss = ewc_regulariser_loss(
                    q_network,
                    reference_params=reference_params,
                    fisher_diag=fisher_diag,
                    prefix_filter=sa_reg_prefix_filter,
                )
            
            if regulariser is not None and regulariser == "repulsion":
                loss = td_loss + 0.1 * sigreg_loss_val + 10000 * ewc_loss
            else:
                loss = td_loss
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
            opt.step()
            
            for p, p_tgt in zip(q_network.parameters(), q_target_network.parameters()):
                p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)
        
        # Evaluation
        if global_step % 1000 == 0:
            eval_env = make_env(goal)
            goal_eval_arr = np.array(goal, dtype=np.float32)
            goal_eval_t = torch.tensor(goal_eval_arr, dtype=torch.float32, device=device).unsqueeze(0)
            
            def eval_policy(o):
                o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_vals = q_network.q_val_for_argmax_action(o_t, goal_eval_t)
                    return int(q_vals.argmax(dim=-1).item())
            
            mean_ret, mean_len = evaluate_policy(eval_env, eval_policy, episodes=8)
            eval_returns.append((global_step, mean_ret))
            print(
                f"[DQN] step={global_step:7d} | eps={eps:.3f} "
                f"| eval_return={mean_ret:.3f} | eval_len={mean_len:.1f} | loss={loss.item():.3f}"
            )
            
            if mean_ret >= early_stop_reward:
                success_streak += 1
            else:
                success_streak = 0
            
            if enable_early_stop and success_streak >= early_stop_patience:
                min_steps = global_step
                min_time = time.perf_counter() - start_time
                print(f"Early stopping at step {global_step}")
                eval_env.close()
                break
            eval_env.close()
        
    # === NEW: Save embeddings AFTER training completes ===
    if save_embeddings and task_id is not None:
        print(f"\n=== Saving embeddings for Task {task_id} ===")
        
        # Sample representative (s,a) pairs from buffer
        num_samples = min(1000, len(buffer))
        
        # Sample random indices
        sampled_indices = np.random.choice(buffer.size, num_samples, replace=False)
        
        phi_embeddings_list = []
        
        for idx in sampled_indices:
            # === Access buffer arrays directly ===
            obs_sample = buffer.obs[idx]  # (obs_dim,)
            act_sample = buffer.actions[idx]  # scalar (int)
            
            obs_t = torch.tensor(obs_sample, dtype=torch.float32, device=device).unsqueeze(0)
            act_onehot = F.one_hot(
                torch.tensor(act_sample, device=device).unsqueeze(0),
                num_classes=num_actions
            ).float()
            
            with torch.no_grad():
                phi_z = q_network.encode_state_action(obs_t, act_onehot)
                phi_embeddings_list.append(phi_z.squeeze(0).cpu().numpy())
        
        phi_embeddings = np.stack(phi_embeddings_list)  # Shape: (num_samples, embed_dim)
        
        # Save goal (psi) embedding
        goal_tensor = torch.tensor(goal_arr, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            psi_z = q_network.encode_goal(goal_tensor)
            psi_embedding = psi_z.squeeze(0).cpu().numpy()  # Shape: (embed_dim,)
        
        # Store everything
        embedding_memory[task_id] = {
            'phi_embeddings': phi_embeddings,  # (N, d)
            'psi_embedding': psi_embedding,  # (d,)
            'goal': goal_arr.copy(),
            'num_samples': num_samples,
        }
        
        print(f"Saved {num_samples} phi embeddings: shape {phi_embeddings.shape}")
        print(f"Saved psi embedding: shape {psi_embedding.shape}")
        print(f"Goal: {goal_arr}")
    
    # === Return values ===
    env.close()
    
    return (
        q_network,
        q_target_network,
        eval_returns,
        min_steps,
        min_time,
        buffer,
        embedding_memory,  # Return the updated embedding_memory
    )



def dqn_train_shared_sa_task_codes(
    seed=42,
    q_network=None,
    q_target_network=None,
    env=None,
    buffer_capacity=None,
    lr_sa=1e-3,
    lr_goal_init=1e-3,
    lr_task_code=1e-3,
    obs_dim=None,
    device=None,
    total_steps=100_000,
    warmup_steps=5_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50_000,
    train_freq=4,
    goal=None,
    task_id=None,
    make_env=None,
    td_steps=1,

    # Persistent dictionaries carried across the task sequence.
    task_embedding_memory=None,
    replay_task_buffers=None,

    # Old-task replay.
    replay_ratio=0.5,
    replay_tasks_per_batch=None,
    replay_loss_coef=1.0,

    # Old-code adaptation after phi changes.
    old_code_recalibration_coef=1.0,

    # Shared coordinate-goal encoder is only an initialiser.
    update_shared_goal_encoder_for_current=True,

    bootstrap_on_truncation=True,
    early_stop_reward=0.99,
    early_stop_patience=5,
    enable_early_stop=True,

    # SIGReg regularisation on SA encoder.
    sigreg_coef=1e-3,
    sketch_dim=64,
):
    """
    Factorised critic:

        Q_i(s, a) = phi_theta(s, a)^T psi_i

    Parameter ownership:
      - sa_encoder:
          current TD + selected old-task TD losses + SIGReg.

      - current task code psi_k:
          current task TD loss only.

      - old task code psi_i:
          only Task i replay TD loss.

      - shared goal_encoder:
          optional current-task-only update. It is used to
          initialise a code for a new task, but is not the
          memory for old task embeddings.

    Required network methods:
      q_network.encode_state_action(obs, action_onehot)
      q_network.encode_goal(goal)
      q_network.q_val_for_argmax_action_from_embedding(
          obs, task_embedding, normalize_embedding=True
      )

    `task_embedding_memory`:
        {integer_task_id: CPU tensor [rep_dim]}

    `replay_task_buffers`:
        {integer_task_id: replay_buffer}
    """

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if q_network is None:
        raise ValueError("q_network must be provided")

    if q_target_network is None:
        raise ValueError("q_target_network must be provided")

    if env is None:
        raise ValueError("env must be provided")

    if buffer_capacity is None:
        raise ValueError("buffer_capacity must be provided")

    if goal is None:
        raise ValueError("goal must be provided")

    if task_id is None:
        raise ValueError("task_id must be provided")

    if not isinstance(task_id, (int, np.integer)):
        raise TypeError("task_id must be an integer")

    if make_env is None:
        raise ValueError("make_env must be provided")

    if train_freq < 1:
        raise ValueError("train_freq must be >= 1")

    if device is None:
        device = next(q_network.parameters()).device

    if obs_dim is None:
        obs_dim = int(env.observation_space.shape[0])

    set_seed(seed)

    if task_embedding_memory is None:
        task_embedding_memory = {}

    if replay_task_buffers is None:
        replay_task_buffers = {}

    num_actions = env.action_space.n

    q_network = q_network.to(device)
    q_target_network = q_target_network.to(device)

    # ---------------------------------------------------------
    # DQN target network: never receives gradients
    # ---------------------------------------------------------

    q_target_network.load_state_dict(
        q_network.state_dict()
    )

    q_target_network.eval()

    for parameter in q_target_network.parameters():
        parameter.requires_grad_(False)

    goal_tensor = torch.as_tensor(
        np.asarray(goal, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    # ---------------------------------------------------------
    # Current task-local code
    #
    # This is the actual psi_k used for the critic. It is not
    # the shared goal_encoder output after initialisation.
    # ---------------------------------------------------------

    if task_id in task_embedding_memory:
        current_code_value = torch.as_tensor(
            task_embedding_memory[task_id],
            dtype=torch.float32,
            device=device,
        ).detach().clone()

    else:
        # Initialise a new task code using the shared goal MLP.
        with torch.no_grad():
            current_code_value = (
                q_network.encode_goal(goal_tensor)
                .squeeze(0)
                .detach()
                .clone()
            )

    current_code = nn.Parameter(current_code_value)

    # ---------------------------------------------------------
    # Old task-local codes
    #
    # They are independent trainable parameters for this call.
    # Their values are saved back to memory after each update.
    # ---------------------------------------------------------

    old_codes = {}

    for old_task_id, stored_code in (
        task_embedding_memory.items()
    ):
        if old_task_id == task_id:
            continue

        old_codes[old_task_id] = nn.Parameter(
            torch.as_tensor(
                stored_code,
                dtype=torch.float32,
                device=device,
            ).detach().clone()
        )

    # ---------------------------------------------------------
    # Optimisers with distinct parameter ownership
    # ---------------------------------------------------------

    opt_sa = optim.Adam(
        q_network.sa_encoder.parameters(),
        lr=lr_sa,
    )

    opt_current_code = optim.Adam(
        [current_code],
        lr=lr_task_code,
    )

    opt_old_codes = {
        old_task_id: optim.Adam(
            [old_code],
            lr=lr_task_code,
        )
        for old_task_id, old_code in old_codes.items()
    }

    # Optional: only trains the shared coordinate-to-code
    # initialiser on current task data.
    opt_goal_init = None

    if update_shared_goal_encoder_for_current:
        opt_goal_init = optim.Adam(
            q_network.goal_encoder.parameters(),
            lr=lr_goal_init,
        )

    # ---------------------------------------------------------
    # Current task transition buffer
    # ---------------------------------------------------------

    buffer = TrajectoryReplayBufferDiscrete(
        buffer_capacity,
        obs_dim,
        1,
        device=device,
    )

    obs, _ = env.reset()

    global_step = 0
    success_streak = 0

    start_time = time.perf_counter()

    eval_returns = []
    min_steps = None
    min_time = None

    current_code_td_loss = torch.zeros(
        (),
        device=device,
    )

    current_sa_td_loss = torch.zeros(
        (),
        device=device,
    )

    replay_sa_td_loss = torch.zeros(
        (),
        device=device,
    )

    old_code_td_loss = torch.zeros(
        (),
        device=device,
    )

    goal_initialiser_loss = torch.zeros(
        (),
        device=device,
    )

    current_sigreg_loss_val = torch.zeros(
        (),
        device=device,
    )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def q_all_actions_from_code(
        network,
        obs_t,
        code,
    ):
        """
        Returns [B, num_actions].

        The network normalises `code` internally when
        normalize_embedding=True. This matches encode_goal().
        """
        return network.q_val_for_argmax_action_from_embedding(
            obs_t,
            code,
            normalize_embedding=True,
        )

    def td_target_from_batch(
        batch,
        code,
    ):
        """
        Constructs a fixed Bellman target for the supplied
        task code. Returns state, one-hot action, and target.
        """
        obs_t = batch.obs
        actions_t = batch.actions.long().view(-1)

        rewards_t = batch.rewards
        next_obs_t = batch.next_obs

        terminated_t = batch.terminated.float()
        truncated_t = batch.truncated.float()

        if rewards_t.ndim == 1:
            rewards_t = rewards_t.unsqueeze(-1)

        if terminated_t.ndim == 1:
            terminated_t = terminated_t.unsqueeze(-1)

        if truncated_t.ndim == 1:
            truncated_t = truncated_t.unsqueeze(-1)

        with torch.no_grad():
            next_q_values = q_all_actions_from_code(
                q_target_network,
                next_obs_t,
                code,
            )

            next_q = next_q_values.max(
                dim=-1,
                keepdim=True,
            ).values

            if bootstrap_on_truncation:
                bootstrap_mask = 1.0 - terminated_t

            else:
                done_t = torch.clamp(
                    terminated_t + truncated_t,
                    min=0.0,
                    max=1.0,
                )

                bootstrap_mask = 1.0 - done_t

            target = rewards_t + (
                gamma ** td_steps
            ) * bootstrap_mask * next_q

        if global_step % 1000 == 0:
            print(
                "Target statistics: "
                f"min={target.min().item():.4f}, "
                f"max={target.max().item():.4f}, "
                f"mean={target.mean().item():.4f}, "
                f"alpha={q_network.alpha.item():.4f}"
            )

        action_onehot = F.one_hot(
            actions_t,
            num_classes=num_actions,
        ).float()

        return (
            obs_t,
            action_onehot,
            target,
        )

    def sa_td_loss(
        batch,
        code,
    ):
        """
        TD loss for phi only.

        The task code is detached. Its gradient cannot be
        updated by this loss.
        """
        obs_t, action_onehot, target = td_target_from_batch(
            batch,
            code,
        )

        with torch.no_grad():
            fixed_code = F.normalize(
                code,
                p=2,
                dim=-1,
                eps=1e-8,
            )

        # with torch.no_grad():
        #     fixed_code = code.detach()

        phi = q_network.encode_state_action(
            obs_t,
            action_onehot,
        )

        q_pred = q_network.alpha * (
            phi * fixed_code.unsqueeze(0)
        ).sum(
            dim=-1,
            keepdim=True,
        )

        assert q_pred.shape == target.shape

        return F.smooth_l1_loss(
            q_pred,
            target,
        )

    def code_td_loss(
        batch,
        code,
    ):
        """
        TD loss for exactly one task-local code only.

        phi(s,a) is detached, so this cannot update SA encoder.
        """
        obs_t, action_onehot, target = td_target_from_batch(
            batch,
            code,
        )

        with torch.no_grad():
            phi_fixed = q_network.encode_state_action(
                obs_t,
                action_onehot,
            )

        code_norm = F.normalize(
            code,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        q_pred = q_network.alpha * (
            phi_fixed * code_norm.unsqueeze(0)
        ).sum(
            dim=-1,
            keepdim=True,
        )

        assert q_pred.shape == target.shape

        return F.smooth_l1_loss(
            q_pred,
            target,
        )
    # def code_td_loss(batch, code, target_norm=1.0, norm_coef=0.0):
    #     obs_t, action_onehot, target = td_target_from_batch(batch, code)

    #     with torch.no_grad():
    #         phi_fixed = q_network.encode_state_action(obs_t, action_onehot)

    #     q_pred = (phi_fixed * code.unsqueeze(0)).sum(dim=-1, keepdim=True)
    #     td = F.smooth_l1_loss(q_pred, target)

    #     if norm_coef > 0.0:
    #         norm = code.norm(p=2, dim=-1, keepdim=True)
    #         norm_reg = ((norm - target_norm) ** 2).mean()
    #         return td + norm_coef * norm_reg
    #     return td

    def shared_goal_initialiser_loss(
        batch,
    ):
        """
        Optional current-task-only loss for goal_encoder.

        This lets the goal-coordinate MLP remain useful as
        an initialiser for future task codes. It does not
        replace stored old task codes.
        """
        obs_t, action_onehot, target = td_target_from_batch(
            batch,
            current_code,
        )

        with torch.no_grad():
            phi_fixed = q_network.encode_state_action(
                obs_t,
                action_onehot,
            )

        shared_goal_embedding = q_network.encode_goal(
            goal_tensor.expand(
                obs_t.shape[0],
                -1,
            )
        )

        q_pred = q_network.alpha * (
            phi_fixed * shared_goal_embedding
        ).sum(
            dim=-1,
            keepdim=True,
        )

        assert q_pred.shape == target.shape

        return F.smooth_l1_loss(
            q_pred,
            target,
        )

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------

    while global_step < total_steps:

        fraction = min(
            1.0,
            global_step / max(1, eps_decay_steps),
        )

        epsilon = eps_start + fraction * (
            eps_end - eps_start
        )

        # -----------------------------------------------------
        # Collect current-task experience
        # -----------------------------------------------------

        if np.random.random() < epsilon:
            action = env.action_space.sample()

        else:
            obs_single = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = q_all_actions_from_code(
                    q_network,
                    obs_single,
                    current_code,
                )

            action = int(
                q_values.argmax(dim=-1).item()
            )

        next_obs, reward, terminated, truncated, _ = env.step(
            action
        )

        done = terminated or truncated

        buffer.add_transition(
            obs,
            action,
            reward,
            next_obs,
            terminated,
            truncated,
        )

        obs = next_obs
        global_step += 1

        if done:
            obs, _ = env.reset()

        if (
            len(buffer) < warmup_steps
            or global_step % train_freq != 0
        ):
            continue

        current_batch = buffer.sample(batch_size)

        # -----------------------------------------------------
        # Retrieve a bounded number of old tasks
        # -----------------------------------------------------

        eligible_old_task_ids = [
            old_task_id
            for old_task_id, old_buffer in (
                replay_task_buffers.items()
            )
            if (
                old_task_id in old_codes
                and old_buffer is not None
                and len(old_buffer) > 0
            )
        ]

        replay_batches = []

        if (
            len(eligible_old_task_ids) > 0
            and replay_ratio > 0.0
        ):
            if replay_tasks_per_batch is None:
                n_old_tasks = len(
                    eligible_old_task_ids
                )

            else:
                n_old_tasks = min(
                    int(replay_tasks_per_batch),
                    len(eligible_old_task_ids),
                )

            cycle_index = (
                global_step // train_freq
            ) % len(eligible_old_task_ids)

            ordered_old_task_ids = (
                eligible_old_task_ids[cycle_index:]
                + eligible_old_task_ids[:cycle_index]
            )

            selected_old_task_ids = ordered_old_task_ids[
                :n_old_tasks
            ]

            replay_batch_size = max(
                1,
                int(
                    batch_size
                    * replay_ratio
                    / n_old_tasks
                ),
            )

            for old_task_id in selected_old_task_ids:
                old_batch = replay_task_buffers[
                    old_task_id
                ].sample(
                    replay_batch_size
                )

                replay_batches.append(
                    (
                        old_task_id,
                        old_batch,
                    )
                )

        # =====================================================
        # A. Shared SA encoder update
        #
        # phi sees current TD + selected old-task TD losses.
        # All task codes are fixed for this update.
        #
        # If lr_sa == 0.0, skip the update and just compute
        # losses for logging (no backward pass).
        # =====================================================

        if lr_sa > 0.0:
            # ---- Trainable SA encoder path ----

            # Current task TD loss
            current_sa_td_loss = sa_td_loss(
                current_batch,
                current_code,
            )

            # SIGReg on current task's phi
            with torch.no_grad():
                obs_cur = current_batch.obs
                act_cur = F.one_hot(
                    current_batch.actions.long().view(-1),
                    num_classes=num_actions,
                ).float()

            phi_cur = q_network.encode_state_action(obs_cur, act_cur)
            current_sigreg = sigreg_loss(phi_cur, sketch_dim=sketch_dim)

            # Old tasks TD loss
            old_sa_losses = []

            for old_task_id, old_batch in replay_batches:
                old_sa_losses.append(
                    sa_td_loss(
                        old_batch,
                        old_codes[old_task_id],
                    )
                )

            if len(old_sa_losses) > 0:
                replay_sa_td_loss = torch.stack(
                    old_sa_losses
                ).mean()
            else:
                replay_sa_td_loss = torch.zeros(
                    (),
                    device=device,
                )

            shared_sa_loss = (
                current_sa_td_loss
                + replay_loss_coef * replay_sa_td_loss
                + sigreg_coef * current_sigreg
            )

            opt_sa.zero_grad(set_to_none=True)
            shared_sa_loss.backward()

            nn.utils.clip_grad_norm_(
                q_network.sa_encoder.parameters(),
                max_norm=10.0,
            )

            opt_sa.step()

            current_sigreg_loss_val = current_sigreg.detach()

        else:
            # ---- Frozen SA encoder path: no gradients ----
            with torch.no_grad():
                current_sa_td_loss = sa_td_loss(
                    current_batch,
                    current_code,
                )

                old_sa_losses = []

                for old_task_id, old_batch in replay_batches:
                    old_sa_losses.append(
                        sa_td_loss(
                            old_batch,
                            old_codes[old_task_id],
                        )
                    )

                if len(old_sa_losses) > 0:
                    replay_sa_td_loss = torch.stack(
                        old_sa_losses
                    ).mean()
                else:
                    replay_sa_td_loss = torch.zeros(
                        (),
                        device=device,
                    )

                shared_sa_loss = (
                    current_sa_td_loss
                    + replay_loss_coef
                    * replay_sa_td_loss
                )

                # Still compute SIGReg for logging (no gradient)
                obs_cur = current_batch.obs
                act_cur = F.one_hot(
                    current_batch.actions.long().view(-1),
                    num_classes=num_actions,
                ).float()

                phi_cur = q_network.encode_state_action(obs_cur, act_cur)
                current_sigreg_loss_val = sigreg_loss(
                    phi_cur, sketch_dim=sketch_dim
                ).detach()

        # =====================================================
        # B. Current task code update
        #
        # psi_k sees current task TD only.
        # phi is fixed.
        # =====================================================

        current_code_td_loss = code_td_loss(
            current_batch,
            current_code,
        )

        opt_current_code.zero_grad(set_to_none=True)
        current_code_td_loss.backward()

        nn.utils.clip_grad_norm_(
            [current_code],
            max_norm=10.0,
        )

        opt_current_code.step()

        # =====================================================
        # C. Old task code recalibration
        #
        # Each psi_i sees only Task i's own replay TD loss.
        # phi is fixed.
        # =====================================================

        old_code_losses = []

        for old_task_id, old_batch in replay_batches:
            this_old_code_loss = code_td_loss(
                old_batch,
                old_codes[old_task_id],
            )

            old_code_losses.append(
                this_old_code_loss.detach()
            )

            opt_old_codes[old_task_id].zero_grad(
                set_to_none=True
            )

            (
                old_code_recalibration_coef
                * this_old_code_loss
            ).backward()

            nn.utils.clip_grad_norm_(
                [old_codes[old_task_id]],
                max_norm=10.0,
            )

            opt_old_codes[old_task_id].step()

        if len(old_code_losses) > 0:
            old_code_td_loss = torch.stack(
                old_code_losses
            ).mean()
        else:
            old_code_td_loss = torch.zeros(
                (),
                device=device,
            )

        # =====================================================
        # D. Optional shared goal encoder update
        #
        # Current task only. It cannot overwrite old codes,
        # because old codes are independent task parameters.
        # =====================================================

        if opt_goal_init is not None:
            goal_initialiser_loss = (
                shared_goal_initialiser_loss(
                    current_batch
                )
            )

            opt_goal_init.zero_grad(set_to_none=True)
            goal_initialiser_loss.backward()

            nn.utils.clip_grad_norm_(
                q_network.goal_encoder.parameters(),
                max_norm=10.0,
            )

            opt_goal_init.step()
        else:
            goal_initialiser_loss = torch.zeros(
                (),
                device=device,
            )

        # -----------------------------------------------------
        # Save current values of all local task codes
        # -----------------------------------------------------

        with torch.no_grad():
            task_embedding_memory[task_id] = (
                F.normalize(
                    current_code,
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )
                .detach()
                .cpu()
            )

            # task_embedding_memory[task_id] = (
            #     current_code.detach().cpu()  # no normalisation
            # )

            for old_task_id, old_code in old_codes.items():
                task_embedding_memory[old_task_id] = (
                    F.normalize(
                        old_code,
                        p=2,
                        dim=-1,
                        eps=1e-8,
                    )
                    .detach()
                    .cpu()
                )
                # task_embedding_memory[old_task_id] = (
                #     old_code.detach().cpu()  # no normalisation
                # )

        # -----------------------------------------------------
        # Polyak target critic update
        # -----------------------------------------------------

        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                q_network.parameters(),
                q_target_network.parameters(),
            ):
                target_parameter.mul_(1.0 - tau).add_(
                    tau * online_parameter
                )

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        if global_step % 1000 != 0:
            continue

        eval_env = make_env(goal)

        def eval_policy(observation):
            observation_t = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = q_all_actions_from_code(
                    q_network,
                    observation_t,
                    current_code,
                )

            return int(
                q_values.argmax(dim=-1).item()
            )

        mean_return, mean_length = evaluate_policy(
            eval_env,
            eval_policy,
            episodes=8,
        )

        eval_returns.append(
            (global_step, mean_return)
        )

        print(
            f"[DQN-shared-SA-task-codes] "
            f"step={global_step:7d} | "
            f"eps={epsilon:.3f} | "
            f"return={mean_return:.3f} | "
            f"len={mean_length:.1f} | "
            f"CodeCur={current_code_td_loss.item():.5f} | "
            f"SACur={current_sa_td_loss.item():.5f} | "
            f"SAOld={replay_sa_td_loss.item():.5f} | "
            f"SIGReg={current_sigreg_loss_val.item():.5f} | "
            f"CodeOld={old_code_td_loss.item():.5f} | "
            f"GoalInit={goal_initialiser_loss.item():.5f} | "
            f"ReplayTasks={len(replay_batches)}"
        )

        eval_env.close()

        if mean_return >= early_stop_reward:
            success_streak += 1
        else:
            success_streak = 0

        if (
            enable_early_stop
            and success_streak >= early_stop_patience
        ):
            min_steps = global_step
            min_time = (
                time.perf_counter()
                - start_time
            )
            print(
                f"Early stopping at step={global_step}, "
                f"return={mean_return:.3f}, "
                f"streak={success_streak}"
            )
            break

    # ---------------------------------------------------------
    # Final outputs
    # ---------------------------------------------------------

    if min_steps is None:
        min_steps = global_step
        min_time = (
            time.perf_counter()
            - start_time
        )

    with torch.no_grad():
        final_task_embedding = F.normalize(
            current_code,
            p=2,
            dim=-1,
            eps=1e-8,
        ).detach().cpu().numpy()

    # with torch.no_grad():
    #     final_task_embedding = current_code.detach().cpu().numpy()

    obs_probe_np = np.array(
        [5.0, 5.0],
        dtype=np.float32,
    )

    base_env = (
        env.unwrapped
        if hasattr(env, "unwrapped")
        else env
    )

    action_names = getattr(
        base_env,
        "action_names",
        ["Up", "Down", "Left", "Right"],
    )

    act_probe_idx = action_names.index("Up")

    sa_embedding_mean = extract_mean_sa_embedding(
        q_network=q_network,
        buffer=buffer,
        num_actions=num_actions,
        batch_size=256,
        device=device,
        as_numpy=True,
    )

    sa_embedding_fixed = extract_fixed_probe_sa_embedding(
        q_network=q_network,
        obs_probe=obs_probe_np,
        act_probe_idx=act_probe_idx,
        num_actions=num_actions,
        device=device,
        as_numpy=True,
    )

    sa_batch_final = extract_sa_batch_for_isotropy(
        q_network=q_network,
        buffer=buffer,
        num_actions=num_actions,
        batch_size=1024,
        device=device,
        as_numpy=True,
    )

    env.close()

    return (
        q_network,
        q_target_network,
        eval_returns,
        min_steps,
        min_time,
        final_task_embedding,
        sa_embedding_mean,
        sa_embedding_fixed,
        sa_batch_final,
        buffer,
        task_embedding_memory,
    )


def dqn_train_phi_psi_adjustment(
    seed=42,
    q_network=None,
    q_target_network=None,
    env=None,
    buffer_capacity=None,
    lr_sa=1e-3,
    lr_goal_init=1e-3,
    lr_task_code=1e-3,
    obs_dim=None,
    device=None,
    total_steps=100_000,
    warmup_steps=5_000,
    batch_size=256,
    eval_freq=1_000,
    gamma=0.99,
    tau=0.005,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50_000,
    train_freq=4,
    goal=None,
    task_id=None,
    make_env=None,
    td_steps=1,

    # {task_id: old_task_buffer}
    replay_task_buffers=None,

    # {task_id: goal}
    task_goals=None,

    replay_ratio=0.5,
    replay_tasks_per_batch=None,
    replay_loss_coef=1.0,

    bootstrap_on_truncation=True,
    early_stop_reward=0.99,
    early_stop_patience=5,
    enable_early_stop=True,

    sigreg_coef=1e-3,
    sketch_dim=64,

    goal_separation_coef=0.001,
    goal_separation_target_cosine=0.85,
    psi_raw_norm_coef=1e-4,
    phi_raw_norm_coef=1e-4,

):
    """
    Functional factorised critic:

        Q(s, a, g)
        =
        phi_theta(s, a)^T psi_eta(g)

    This version uses the old replay-buffer class.

    Each replay buffer contains transitions from one task
    only. The task's goal is obtained from task_goals[task_id].

    No fixed targets are used or stored.

    Required old buffer batch fields:

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

    # =========================================================
    # Validation
    # =========================================================

    if q_network is None:
        raise ValueError(
            "q_network must be provided."
        )

    if q_target_network is None:
        raise ValueError(
            "q_target_network must be provided."
        )

    if env is None:
        raise ValueError(
            "env must be provided."
        )

    if buffer_capacity is None:
        raise ValueError(
            "buffer_capacity must be provided."
        )

    if goal is None:
        raise ValueError(
            "goal must be provided."
        )

    if task_id is None:
        raise ValueError(
            "task_id must be provided."
        )

    if not isinstance(
        task_id,
        (int, np.integer),
    ):
        raise TypeError(
            "task_id must be an integer."
        )

    if make_env is None:
        raise ValueError(
            "make_env must be provided."
        )

    if train_freq < 1:
        raise ValueError(
            "train_freq must be >= 1."
        )

    if td_steps < 1:
        raise ValueError(
            "td_steps must be >= 1."
        )

    if device is None:
        device = next(
            q_network.parameters()
        ).device

    device = torch.device(device)

    if obs_dim is None:
        obs_dim = int(
            env.observation_space.shape[0]
        )

    set_seed(seed)

    if replay_task_buffers is None:
        replay_task_buffers = {}

    if task_goals is None:
        task_goals = {}

    if isinstance(
    goal,
    torch.Tensor,
    ):
        task_goals[task_id] = (
            goal.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=True,
            )
        )

    else:
        task_goals[task_id] = np.asarray(
            goal,
            dtype=np.float32,
        ).copy()

    num_actions = env.action_space.n

    q_network = q_network.to(device)

    q_target_network = (
        q_target_network.to(device)
    )

    # =========================================================
    # Target-network setup
    # =========================================================

    q_target_network.load_state_dict(
        q_network.state_dict()
    )

    q_target_network.eval()

    for parameter in q_target_network.parameters():
        parameter.requires_grad_(False)

    # =========================================================
    # Current task goal
    # =========================================================

    goal_tensor = torch.as_tensor(
        np.asarray(
            goal,
            dtype=np.float32,
        ),
        dtype=torch.float32,
        device=device,
    ).view(
        1,
        -1,
    )

    # =========================================================
    # Optimisers
    # =========================================================

    opt_sa = optim.Adam(
        q_network.sa_encoder.parameters(),
        lr=lr_sa,
    )

    opt_goal = optim.Adam(
        q_network.goal_encoder.parameters(),
        lr=lr_goal_init,
    )

    # =========================================================
    # Current task buffer
    # =========================================================

    #
    # Old buffer constructor: no goal_dim and no fixed targets.
    #

    buffer = TrajectoryReplayBufferDiscrete(
        buffer_capacity,
        obs_dim,
        1,
        device=device,
    )

    collected_episodes: list[dict] = []

    # =========================================================
    # Helper functions
    # =========================================================

    def goal_batch_for(
        goal_value,
        batch_size,
        target_device,
    ):
        if isinstance(
            goal_value,
            torch.Tensor,
        ):
            goal_batch = goal_value.to(
                device=target_device,
                dtype=torch.float32,
            )

        else:
            goal_batch = torch.as_tensor(
                np.asarray(
                    goal_value,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=target_device,
            )

        if goal_batch.ndim == 1:
            goal_batch = goal_batch.unsqueeze(0)

        elif goal_batch.ndim != 2:
            raise ValueError(
                "Goal must have shape [goal_dim] or "
                "[B, goal_dim]. Got "
                f"{tuple(goal_batch.shape)}."
            )

        if goal_batch.shape[0] == 1:
            goal_batch = goal_batch.expand(
                batch_size,
                -1,
            )

        elif goal_batch.shape[0] != batch_size:
            raise ValueError(
                "Goal batch size does not match batch size: "
                f"{goal_batch.shape[0]} vs {batch_size}."
            )

        return goal_batch

    def action_onehot_from_batch(
        batch,
    ):
        actions = (
            batch.actions
            .long()
            .view(-1)
        )

        return F.one_hot(
            actions,
            num_classes=num_actions,
        ).to(
            dtype=batch.obs.dtype
        )

    def q_all_actions_from_goal(
        network,
        obs_t,
        goal_t,
    ):
        goal_batch = goal_batch_for(
            goal_t,
            obs_t.shape[0],
            obs_t.device,
        )

        psi = network.encode_goal(
            goal_batch
        )

        return network.q_val_for_argmax_action_from_embedding(
            obs_t,
            psi,
            normalize_embedding=False,
        )

    # def moving_td_target(
    #     batch,
    #     batch_goal,
    # ):
    #     rewards = batch.rewards
    #     next_obs = batch.next_obs

    #     terminated = (
    #         batch.terminated.float()
    #     )

    #     truncated = (
    #         batch.truncated.float()
    #     )

    #     if rewards.ndim == 1:
    #         rewards = rewards.unsqueeze(-1)

    #     if terminated.ndim == 1:
    #         terminated = terminated.unsqueeze(-1)

    #     if truncated.ndim == 1:
    #         truncated = truncated.unsqueeze(-1)

    #     next_goal_batch = goal_batch_for(
    #         batch_goal,
    #         next_obs.shape[0],
    #         next_obs.device,
    #     )

    #     with torch.no_grad():
    #         next_q_values = (
    #             q_all_actions_from_goal(
    #                 q_target_network,
    #                 next_obs,
    #                 next_goal_batch,
    #             )
    #         )

    #         next_q = next_q_values.max(
    #             dim=-1,
    #             keepdim=True,
    #         ).values

    #         if bootstrap_on_truncation:
    #             bootstrap_mask = (
    #                 1.0 - terminated
    #             )

    #         else:
    #             done = torch.clamp(
    #                 terminated + truncated,
    #                 min=0.0,
    #                 max=1.0,
    #             )

    #             bootstrap_mask = (
    #                 1.0 - done
    #             )

    #         return rewards + (
    #             gamma ** td_steps
    #         ) * bootstrap_mask * next_q

    def moving_td_target(batch, batch_goal):
        """
        Compute TD targets using:
        - rewards recomputed from env.{compute_simple,compute_shaped}_reward
        - q_target_network for bootstrap

        batch_goal: numpy or torch goal for the task being trained.
        """
        # Recompute rewards from env instead of using batch.rewards
        obs_np = batch.obs.detach().cpu().numpy()        # states
        next_obs_np = batch.next_obs.detach().cpu().numpy()  # next_states
        actions_np = batch.actions.long().detach().cpu().numpy()

        if isinstance(batch_goal, torch.Tensor):
            goal_np = batch_goal.detach().cpu().numpy()
        else:
            goal_np = np.asarray(batch_goal, dtype=np.float32)

        # Broadcast goal to batch size if needed
        if goal_np.ndim == 1:
            goal_np = np.broadcast_to(goal_np, (obs_np.shape[0], goal_np.shape[0]))
        elif goal_np.shape[0] == 1:
            goal_np = np.broadcast_to(goal_np, (obs_np.shape[0], goal_np.shape[1]))

        rewards_np = np.empty((obs_np.shape[0], 1), dtype=np.float32)

        use_simple = getattr(env, "reward_mode", "simple") == "simple"

        for i in range(obs_np.shape[0]):
            s = obs_np[i]
            sp = next_obs_np[i]
            a = int(actions_np[i])
            g = goal_np[i]

            if use_simple:
                r = env.compute_simple_reward(s, a, sp, g)
            else:
                r = env.compute_shaped_reward(s, a, sp, g)

            rewards_np[i, 0] = r

        rewards = torch.as_tensor(
            rewards_np,
            dtype=batch.obs.dtype,
            device=batch.obs.device,
        )

        next_obs = batch.next_obs

        # Termination: ONLY when next state == goal
        reached = (
            (next_obs_np[:, 0] == goal_np[:, 0])
            & (next_obs_np[:, 1] == goal_np[:, 1])
        )

        terminated = torch.as_tensor(
            reached[:, None],                      # [B, 1]
            dtype=batch.obs.dtype,
            device=batch.obs.device,
        )

        # No truncation logic; treat truncation as always 0
        truncated = torch.zeros_like(terminated)

        if terminated.ndim == 1:
            terminated = terminated.unsqueeze(-1)
        if truncated.ndim == 1:
            truncated = truncated.unsqueeze(-1)

        next_goal_batch = goal_batch_for(
            batch_goal,
            next_obs.shape[0],
            next_obs.device,
        )

        with torch.no_grad():
            next_q_values = q_all_actions_from_goal(
                q_target_network,
                next_obs,
                next_goal_batch,
            )
            next_q = next_q_values.max(
                dim=-1,
                keepdim=True,
            ).values

            if bootstrap_on_truncation:
                bootstrap_mask = 1.0 - terminated
            else:
                done = torch.clamp(
                    terminated + truncated,
                    min=0.0,
                    max=1.0,
                )
                bootstrap_mask = 1.0 - done

        return rewards + (gamma ** td_steps) * bootstrap_mask * next_q

    def factorised_q_from_batch(
        batch,
        batch_goal,
    ):
        action_onehot = (
            action_onehot_from_batch(
                batch
            )
        )

        goal_batch = goal_batch_for(
            batch_goal,
            batch.obs.shape[0],
            batch.obs.device,
        )

        phi = q_network.encode_state_action(
            batch.obs,
            action_onehot,
        )

        psi = q_network.encode_goal(
            goal_batch
        )

        return (
            phi * psi
        ).sum(
            dim=-1,
            keepdim=True,
        )

    def task_loss(
        batch,
        batch_goal,
    ):
        target = moving_td_target(
            batch,
            batch_goal,
        )

        q_values = factorised_q_from_batch(
            batch,
            batch_goal,
        )

        if q_values.shape != target.shape:
            raise RuntimeError(
                "Q-values and targets have different "
                f"shapes: {q_values.shape} vs "
                f"{target.shape}."
            )

        # return F.smooth_l1_loss(
        #     q_values,
        #     target,
        # )
        return F.mse_loss(
            q_values,
            target,
        )

    def sigreg_phi_loss(
        batch,
    ):
        action_onehot = (
            action_onehot_from_batch(
                batch
            )
        )

        phi = q_network.encode_state_action(
            batch.obs,
            action_onehot,
        )

        return sigreg_loss(
            phi,
            sketch_dim=sketch_dim,
        )


    # =========================================================
    # Main training loop
    # =========================================================

    obs, _ = env.reset()
    global_step = 0
    success_streak = 0

    start_time = time.perf_counter()

    eval_returns = []
    min_steps = None
    min_time = None
    cosine_matrix = None

    current_loss_value = torch.zeros((), device=device)
    old_replay_loss_value = torch.zeros((), device=device)
    current_sigreg_loss_value = torch.zeros((), device=device)

    # =========================================================
    # Main training loop
    # =========================================================

    while global_step < total_steps:
        fraction = min(
            1.0,
            global_step / max(1, eps_decay_steps),
        )

        epsilon = (
            eps_start
            + fraction * (eps_end - eps_start)
        )

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs_single = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = q_all_actions_from_goal(
                    q_network,
                    obs_single,
                    goal_tensor,
                )

            action = int(q_values.argmax(dim=-1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
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
            obs, _ = env.reset()

        if (
            len(buffer) < warmup_steps
            or global_step % train_freq != 0
        ):
            continue

        current_batch = buffer.sample(
            batch_size
        )

        # -----------------------------------------------------
        # Select old replay tasks
        # -----------------------------------------------------

        eligible_old_task_ids = []

        for old_task_id, old_buffer in (
            replay_task_buffers.items()
        ):
            if old_buffer is None:
                continue

            if len(old_buffer) < 1:
                continue

            if old_task_id not in task_goals:
                raise KeyError(
                    "Missing goal for replay task "
                    f"{old_task_id}. Add it to task_goals."
                )

            eligible_old_task_ids.append(
                old_task_id
            )

        replay_batches = []

        if (
            len(eligible_old_task_ids) > 0
            and replay_ratio > 0.0
        ):
            if replay_tasks_per_batch is None:
                n_old_tasks = len(
                    eligible_old_task_ids
                )

            else:
                n_old_tasks = min(
                    int(replay_tasks_per_batch),
                    len(eligible_old_task_ids),
                )

            cycle_index = (
                global_step // train_freq
            ) % len(eligible_old_task_ids)

            ordered_old_task_ids = (
                eligible_old_task_ids[
                    cycle_index:
                ]
                + eligible_old_task_ids[
                    :cycle_index
                ]
            )

            selected_old_task_ids = (
                ordered_old_task_ids[
                    :n_old_tasks
                ]
            )

            replay_batch_size = max(
                1,
                int(
                    batch_size
                    * replay_ratio
                    / n_old_tasks
                ),
            )

            for old_task_id in (
                selected_old_task_ids
            ):
                old_buffer = (
                    replay_task_buffers[
                        old_task_id
                    ]
                )

                if len(old_buffer) < replay_batch_size:
                    continue

                old_batch = old_buffer.sample(
                    replay_batch_size
                )

                old_goal = task_goals[
                    old_task_id
                ]

                replay_batches.append(
                    (
                        old_task_id,
                        old_goal,
                        old_batch,
                    )
                )

        # -----------------------------------------------------
        # Current task loss
        # -----------------------------------------------------

        current_loss = task_loss(
            current_batch,
            goal,
        )

        # -----------------------------------------------------
        # Old-task replay loss
        # -----------------------------------------------------

        old_losses = []

        for old_task_id, old_goal, old_batch in (
            replay_batches
        ):
            old_losses.append(
                task_loss(
                    old_batch,
                    old_goal,
                )
            )

        if len(old_losses) > 0:
            old_replay_loss = torch.stack(
                old_losses
            ).mean()

        else:
            old_replay_loss = torch.zeros(
                (),
                device=device,
            )

        # -----------------------------------------------------
        # SIGReg
        # -----------------------------------------------------

        if sigreg_coef > 0.0:
            current_sigreg = (
                sigreg_phi_loss(
                    current_batch
                )
            )

        else:
            current_sigreg = torch.zeros(
                (),
                device=device,
            )

        if (
            goal_separation_coef > 0.0
            and len(task_goals) > 1
        ):
            goal_separation_loss, cosine_matrix = (
                online_goal_separation_loss(
                    q_network=q_network,
                    task_goals=task_goals,
                    device=device,
                    target_cosine=(
                        goal_separation_target_cosine
                    ),
                )
            )

        else:
            goal_separation_loss = torch.zeros(
                (),
                device=device,
            )

        # Norm regularisation

        seen_task_ids = sorted(
            task_goals.keys()
        )

        seen_goal_tensor = torch.as_tensor(
            np.asarray(
                [
                    task_goals[task_id]
                    for task_id in seen_task_ids
                ],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=device,
        )

        raw_psi_all = q_network.goal_encoder(seen_goal_tensor)
        psi_raw_norm_loss = norm_penalty_loss_l1(raw_psi_all, target_norm=(q_network.psi_max_norm - 0.1))

        actions = (current_batch.actions.long().view(-1))
        action_onehot = F.one_hot(actions,num_classes=q_network.num_actions).to(dtype=current_batch.obs.dtype,device=current_batch.obs.device)
        sa_input = torch.cat([current_batch.obs,action_onehot,],dim=-1)

        raw_phi_current = q_network.sa_encoder(sa_input)
        phi_raw_norm_loss = norm_penalty_loss_l1(raw_phi_current,target_norm=(q_network.phi_max_norm - 0.1))
        
        # -----------------------------------------------------
        # Joint loss
        # -----------------------------------------------------

        total_loss = (
            current_loss
            + replay_loss_coef
            * old_replay_loss
            + sigreg_coef
            * current_sigreg
            + goal_separation_coef
            * goal_separation_loss
            + psi_raw_norm_coef
            * psi_raw_norm_loss
            + phi_raw_norm_coef
            * phi_raw_norm_loss
        )

        opt_sa.zero_grad(
            set_to_none=True
        )

        opt_goal.zero_grad(
            set_to_none=True
        )

        total_loss.backward()

        if lr_sa > 0.0:
            nn.utils.clip_grad_norm_(
                q_network.sa_encoder.parameters(),
                max_norm=10.0,
            )

        if lr_goal_init > 0.0:
            nn.utils.clip_grad_norm_(
                q_network.goal_encoder.parameters(),
                max_norm=10.0,
            )

        if lr_sa > 0.0:
            opt_sa.step()

        if lr_goal_init > 0.0:
            opt_goal.step()

        current_loss_value = (
            current_loss.detach()
        )

        old_replay_loss_value = (
            old_replay_loss.detach()
        )

        current_sigreg_loss_value = (
            current_sigreg.detach()
        )
        psi_raw_norm_loss_value = (
            psi_raw_norm_loss.detach()
        )

        goal_separation_value = (
            goal_separation_loss.detach().item()
        )

        # -----------------------------------------------------
        # Polyak target update
        # -----------------------------------------------------

        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                q_network.parameters(),
                q_target_network.parameters(),
            ):
                target_parameter.mul_(
                    1.0 - tau
                ).add_(
                    tau * online_parameter
                )

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        if global_step % eval_freq != 0:
            continue

        eval_env = make_env(
            goal=goal
        )

        def eval_policy(observation):
            observation_t = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = (
                    q_all_actions_from_goal(
                        q_network,
                        observation_t,
                        goal_tensor,
                    )
                )

            return int(
                q_values.argmax(
                    dim=-1
                ).item()
            )

        mean_return, mean_length = (
            evaluate_policy(
                eval_env,
                eval_policy,
                episodes=8,
            )
        )

        eval_returns.append(
            (
                global_step,
                mean_return,
            )
        )

        q_bound = (
            q_network.phi_max_norm
            * q_network.psi_max_norm
        )

        print(
            "[DQN-functional-phi-psi] "
            f"step={global_step:7d} | "
            f"eps={epsilon:.3f} | "
            f"return={mean_return:.3f} | "
            f"len={mean_length:.1f} | "
            f"Current="
            f"{current_loss_value.item():.5f} | "
            f"OldReplay="
            f"{old_replay_loss_value.item():.5f} | "
            f"SIGReg="
            f"{current_sigreg_loss_value.item():.5f} | "
            f"QBound={q_bound:.3f} | "
            f"ReplayTasks="
            f"{len(replay_batches)} | "
            f"GoalSeparation="
            f"{goal_separation_value:.5f} | "
            f"RawPsiNormLoss="
            f"{psi_raw_norm_loss_value:.6f} | "
            f"RawPhiNormLoss="
            f"{phi_raw_norm_loss.detach().item():.6f}"

        )

        eval_env.close()

        if mean_return >= early_stop_reward:
            success_streak += 1

        else:
            success_streak = 0

        if (
            enable_early_stop
            and success_streak
            >= early_stop_patience
        ):
            min_steps = global_step

            min_time = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Early stopping at step={global_step}, "
                f"return={mean_return:.3f}, "
                f"streak={success_streak}"
            )

            break

    # =========================================================
    # Final outputs
    # =========================================================

    if min_steps is None:
        min_steps = global_step

        min_time = (
            time.perf_counter()
            - start_time
        )

    with torch.no_grad():
        final_task_embedding = (
            q_network.encode_goal(
                goal_tensor
            )
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

    obs_probe_np = np.array(
        [5.0, 5.0],
        dtype=np.float32,
    )

    base_env = (
        env.unwrapped
        if hasattr(
            env,
            "unwrapped",
        )
        else env
    )

    action_names = getattr(
        base_env,
        "action_names",
        [
            "Up",
            "Down",
            "Left",
            "Right",
        ],
    )

    act_probe_idx = action_names.index(
        "Up"
    )

    sa_embedding_mean = (
        extract_mean_sa_embedding(
            q_network=q_network,
            buffer=buffer,
            num_actions=num_actions,
            batch_size=256,
            device=device,
            as_numpy=True,
        )
    )

    sa_embedding_fixed = (
        extract_fixed_probe_sa_embedding(
            q_network=q_network,
            obs_probe=obs_probe_np,
            act_probe_idx=act_probe_idx,
            num_actions=num_actions,
            device=device,
            as_numpy=True,
        )
    )

    sa_batch_final = (
        extract_sa_batch_for_isotropy(
            q_network=q_network,
            buffer=buffer,
            num_actions=num_actions,
            batch_size=1024,
            device=device,
            as_numpy=True,
        )
    )

    if len(buffer) > 0:
        final_phi_batch = buffer.sample(
            min(
                1024,
                len(buffer),
            )
        )

        final_raw_phi_stats = (
            inspect_raw_phi_norms(
                q_network,
                final_phi_batch,
            )
        )

        print(
            f"\nFinal raw phi norms "
            f"for task {task_id}:"
        )

        print(
            "Raw phi norms | "
            f"mean="
            f"{final_raw_phi_stats['mean']:.4f} | "
            f"std="
            f"{final_raw_phi_stats['std']:.4f} | "
            f"min="
            f"{final_raw_phi_stats['min']:.4f} | "
            f"max="
            f"{final_raw_phi_stats['max']:.4f}"
        )

    else:
        final_raw_phi_stats = None



    env.close()

    return (
        q_network,
        q_target_network,
        eval_returns,
        min_steps,
        min_time,
        final_task_embedding,
        sa_embedding_mean,
        sa_embedding_fixed,
        sa_batch_final,
        buffer,
        cosine_matrix,
    )

def dqn_train_atari(
    seed=42,
    q_network=None,
    q_target_network=None,
    env=None,
    buffer_capacity=None,
    lr_sa=1e-3,
    lr_goal_init=1e-3,
    lr_task_code=1e-3,
    obs_dim=None,
    device=None,
    total_steps=100_000,
    warmup_steps=5_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50_000,
    train_freq=4,
    goal=None,
    task_id=None,
    make_env=None,
    td_steps=1,
    replay_task_buffers=None,
    task_goals=None,
    replay_ratio=0.5,
    replay_tasks_per_batch=None,
    replay_loss_coef=1.0,
    bootstrap_on_truncation=True,
    early_stop_reward=0.99,
    early_stop_patience=5,
    enable_early_stop=True,
    sigreg_coef=1e-3,
    sketch_dim=64,
    goal_separation_coef=0.001,
    goal_separation_target_cosine=0.85,
    psi_raw_norm_coef=1e-4,
    phi_raw_norm_coef=1e-4,
):
    """
    Atari-compatible sequential continual DQN trainer.

    Factorised critic:

        Q(s, a, g) = phi(s, a)^T psi(g)

    Expected observation format:

        [B, C, H, W]

    For standard Atari preprocessing:

        [B, 4, 84, 84]

    The trainer does not flatten observations and does not use obs_dim.
    """

    # =========================================================
    # Validation
    # =========================================================

    if q_network is None:
        raise ValueError("q_network must be provided.")

    if q_target_network is None:
        raise ValueError(
            "q_target_network must be provided."
        )

    if env is None:
        raise ValueError("env must be provided.")

    if buffer_capacity is None:
        raise ValueError(
            "buffer_capacity must be provided."
        )

    if goal is None:
        raise ValueError("goal must be provided.")

    if task_id is None:
        raise ValueError("task_id must be provided.")

    if not isinstance(
        task_id,
        (int, np.integer),
    ):
        raise TypeError(
            "task_id must be an integer."
        )

    if make_env is None:
        raise ValueError(
            "make_env must be provided."
        )

    if train_freq < 1:
        raise ValueError(
            "train_freq must be >= 1."
        )

    if td_steps < 1:
        raise ValueError(
            "td_steps must be >= 1."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive."
        )

    if device is None:
        device = next(
            q_network.parameters()
        ).device

    device = torch.device(device)

    set_seed(seed)

    if replay_task_buffers is None:
        replay_task_buffers = {}

    if task_goals is None:
        task_goals = {}

    goal_np = np.asarray(
        goal,
        dtype=np.float32,
    ).reshape(-1)

    task_goals[task_id] = goal_np.copy()

    q_network = q_network.to(device)

    q_target_network = (
        q_target_network.to(device)
    )

    obs_shape = tuple(
        env.observation_space.shape
    )

    if len(obs_shape) != 3:
        raise ValueError(
            "Expected Atari observation shape "
            "[C,H,W] or [H,W,C]. "
            f"Got {obs_shape}."
        )

    num_actions = env.action_space.n

    # =========================================================
    # Target network
    # =========================================================

    q_target_network.load_state_dict(
        q_network.state_dict()
    )

    q_target_network.eval()

    for parameter in q_target_network.parameters():
        parameter.requires_grad_(False)

    # =========================================================
    # Current task goal
    # =========================================================

    goal_tensor = torch.as_tensor(
        goal_np,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    # =========================================================
    # Optimizer
    # =========================================================
    #
    # The original trainer only optimised sa_encoder and
    # goal_encoder. For Atari, the convolutional observation
    # encoder must also be trained.
    #

    trainable_parameters = [
        parameter
        for parameter in q_network.parameters()
        if parameter.requires_grad
    ]

    if len(trainable_parameters) == 0:
        raise ValueError(
            "q_network has no trainable parameters."
        )

    optimizer = optim.Adam(
        trainable_parameters,
        lr=lr_sa,
    )

    # =========================================================
    # Current task replay buffer
    # =========================================================
    #
    # This assumes the buffer accepts an observation shape
    # rather than a flattened obs_dim.
    #

    buffer = AtariReplayBuffer(
        capacity=buffer_capacity,
        obs_shape=obs_shape,
        device=device,
    )

    reset_output = env.reset()
    obs = extract_observation(reset_output)

    validate_atari_observation(
        obs,
        obs_shape,
    )

    global_step = 0
    success_streak = 0

    start_time = time.perf_counter()

    eval_returns = []
    min_steps = None
    min_time = None

    current_loss_value = torch.zeros(
        (),
        device=device,
    )

    old_replay_loss_value = torch.zeros(
        (),
        device=device,
    )

    current_sigreg_loss_value = torch.zeros(
        (),
        device=device,
    )

    psi_raw_norm_loss_value = torch.zeros(
        (),
        device=device,
    )

    phi_raw_norm_loss_value = torch.zeros(
        (),
        device=device,
    )

    goal_separation_value = 0.0

    # =========================================================
    # Helper functions
    # =========================================================

    def goal_batch_for(
        goal_value,
        batch_size_value,
        target_device,
    ):
        if isinstance(
            goal_value,
            torch.Tensor,
        ):
            goal_batch = goal_value.to(
                device=target_device,
                dtype=torch.float32,
            )

        else:
            goal_batch = torch.as_tensor(
                np.asarray(
                    goal_value,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=target_device,
            )

        if goal_batch.ndim == 1:
            goal_batch = goal_batch.unsqueeze(0)

        elif goal_batch.ndim != 2:
            raise ValueError(
                "Goal must have shape "
                "[goal_dim] or [B, goal_dim]."
            )

        if goal_batch.shape[0] == 1:
            goal_batch = goal_batch.expand(
                batch_size_value,
                -1,
            )

        elif goal_batch.shape[0] != batch_size_value:
            raise ValueError(
                "Goal batch size does not match "
                "observation batch size."
            )

        return goal_batch

    def action_onehot_from_batch(batch):
        actions = (
            batch.actions
            .long()
            .view(-1)
        )

        return F.one_hot(
            actions,
            num_classes=num_actions,
        ).to(
            dtype=batch.obs.dtype,
            device=batch.obs.device,
        )

    def q_all_actions_from_goal(
        network,
        obs_t,
        goal_t,
    ):
        goal_batch = goal_batch_for(
            goal_t,
            obs_t.shape[0],
            obs_t.device,
        )

        psi = network.encode_goal(
            goal_batch
        )

        return network.q_val_for_argmax_action_from_embedding(
            obs_t,
            psi,
            normalize_embedding=False,
        )

    def moving_td_target(
        batch,
        batch_goal,
    ):
        rewards = batch.rewards.float()
        next_obs = batch.next_obs

        terminated = (
            batch.terminated.float()
        )

        truncated = (
            batch.truncated.float()
        )

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)

        if terminated.ndim == 1:
            terminated = terminated.unsqueeze(-1)

        if truncated.ndim == 1:
            truncated = truncated.unsqueeze(-1)

        next_goal_batch = goal_batch_for(
            batch_goal,
            next_obs.shape[0],
            next_obs.device,
        )

        with torch.no_grad():
            next_q_values = (
                q_all_actions_from_goal(
                    q_target_network,
                    next_obs,
                    next_goal_batch,
                )
            )

            next_q = next_q_values.max(
                dim=-1,
                keepdim=True,
            ).values

            if bootstrap_on_truncation:
                bootstrap_mask = (
                    1.0 - terminated
                )

            else:
                done = torch.clamp(
                    terminated + truncated,
                    min=0.0,
                    max=1.0,
                )

                bootstrap_mask = (
                    1.0 - done
                )

            return rewards + (
                gamma ** td_steps
            ) * bootstrap_mask * next_q

    def factorised_q_from_batch(
        batch,
        batch_goal,
    ):
        action_onehot = (
            action_onehot_from_batch(
                batch
            )
        )

        goal_batch = goal_batch_for(
            batch_goal,
            batch.obs.shape[0],
            batch.obs.device,
        )

        phi = q_network.encode_state_action(
            batch.obs,
            action_onehot,
        )

        psi = q_network.encode_goal(
            goal_batch
        )

        return (
            phi * psi
        ).sum(
            dim=-1,
            keepdim=True,
        )

    def task_loss(
        batch,
        batch_goal,
    ):
        target = moving_td_target(
            batch,
            batch_goal,
        )

        q_values = factorised_q_from_batch(
            batch,
            batch_goal,
        )

        if q_values.shape != target.shape:
            raise RuntimeError(
                "Q-values and targets have different "
                f"shapes: {q_values.shape} vs "
                f"{target.shape}."
            )

        return F.smooth_l1_loss(
            q_values,
            target,
        )

    def sigreg_phi_loss(batch):
        action_onehot = (
            action_onehot_from_batch(
                batch
            )
        )

        phi = q_network.encode_state_action(
            batch.obs,
            action_onehot,
        )

        return sigreg_loss(
            phi,
            sketch_dim=sketch_dim,
        )

    def raw_phi_from_batch(batch):
        """
        Compute unprojected phi for Atari inputs.
        """

        actions = (
            batch.actions
            .long()
            .view(-1)
        )

        action_onehot = F.one_hot(
            actions,
            num_classes=num_actions,
        ).to(
            dtype=batch.obs.dtype,
            device=batch.obs.device,
        )

        required_methods = [
            "_encode_obs_features",
            "action_encoder",
            "sa_encoder",
        ]

        for method_name in required_methods:
            if not hasattr(
                q_network,
                method_name,
            ):
                raise AttributeError(
                    "Atari network must expose "
                    f"{method_name}."
                )

        obs_features = (
            q_network._encode_obs_features(
                batch.obs
            )
        )

        action_features = (
            q_network.action_encoder(
                action_onehot
            )
        )

        sa = torch.cat(
            [
                obs_features,
                action_features,
            ],
            dim=-1,
        )

        return q_network.sa_encoder(sa)

    def raw_psi_norm_loss():
        seen_task_ids = sorted(
            task_goals.keys()
        )

        seen_goal_tensor = torch.as_tensor(
            np.asarray(
                [
                    task_goals[i]
                    for i in seen_task_ids
                ],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=device,
        )

        raw_psi = q_network.goal_encoder(
            seen_goal_tensor
        )

        return norm_penalty_loss_l1(
            raw_psi,
            target_norm=(
                q_network.psi_max_norm
                - 0.1
            ),
        )

    # =========================================================
    # Main training loop
    # =========================================================

    while global_step < total_steps:
        fraction = min(
            1.0,
            global_step
            / max(
                1,
                eps_decay_steps,
            ),
        )

        epsilon = (
            eps_start
            + fraction
            * (
                eps_end
                - eps_start
            )
        )

        # -----------------------------------------------------
        # Collect current-task experience
        # -----------------------------------------------------

        if np.random.random() < epsilon:
            action = env.action_space.sample()

        else:
            obs_single = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = (
                    q_all_actions_from_goal(
                        q_network,
                        obs_single,
                        goal_tensor,
                    )
                )

            action = int(
                q_values.argmax(
                    dim=-1
                ).item()
            )

        next_obs, reward, terminated, truncated, _ = (
            env.step(action)
        )

        validate_atari_observation(
            next_obs,
            obs_shape,
        )

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

        done = bool(
            terminated
            or truncated
        )

        if done:
            if hasattr(
                buffer,
                "end_episode",
            ):
                buffer.end_episode()

            obs = extract_observation(
                env.reset()
            )

            validate_atari_observation(
                obs,
                obs_shape,
            )

        if (
            len(buffer) < warmup_steps
            or global_step % train_freq != 0
        ):
            continue

        current_batch = buffer.sample(
            batch_size
        )

        # -----------------------------------------------------
        # Select old replay tasks
        # -----------------------------------------------------

        eligible_old_task_ids = []

        for old_task_id, old_buffer in (
            replay_task_buffers.items()
        ):
            if old_buffer is None:
                continue

            if len(old_buffer) < 1:
                continue

            if old_task_id not in task_goals:
                raise KeyError(
                    "Missing goal for replay task "
                    f"{old_task_id}."
                )

            eligible_old_task_ids.append(
                old_task_id
            )

        replay_batches = []

        if (
            len(eligible_old_task_ids) > 0
            and replay_ratio > 0.0
        ):
            if replay_tasks_per_batch is None:
                n_old_tasks = len(
                    eligible_old_task_ids
                )

            else:
                n_old_tasks = min(
                    int(replay_tasks_per_batch),
                    len(eligible_old_task_ids),
                )

            cycle_index = (
                global_step // train_freq
            ) % len(eligible_old_task_ids)

            ordered_old_task_ids = (
                eligible_old_task_ids[
                    cycle_index:
                ]
                + eligible_old_task_ids[
                    :cycle_index
                ]
            )

            selected_old_task_ids = (
                ordered_old_task_ids[
                    :n_old_tasks
                ]
            )

            replay_batch_size = max(
                1,
                int(
                    batch_size
                    * replay_ratio
                    / n_old_tasks
                ),
            )

            for old_task_id in (
                selected_old_task_ids
            ):
                old_buffer = (
                    replay_task_buffers[
                        old_task_id
                    ]
                )

                if len(old_buffer) < replay_batch_size:
                    continue

                old_batch = old_buffer.sample(
                    replay_batch_size
                )

                old_goal = task_goals[
                    old_task_id
                ]

                replay_batches.append(
                    (
                        old_task_id,
                        old_goal,
                        old_batch,
                    )
                )

        # -----------------------------------------------------
        # Current and replay losses
        # -----------------------------------------------------

        current_loss = task_loss(
            current_batch,
            goal,
        )

        old_losses = []

        for old_task_id, old_goal, old_batch in (
            replay_batches
        ):
            old_losses.append(
                task_loss(
                    old_batch,
                    old_goal,
                )
            )

        if len(old_losses) > 0:
            old_replay_loss = torch.stack(
                old_losses
            ).mean()

        else:
            old_replay_loss = torch.zeros(
                (),
                device=device,
            )

        # -----------------------------------------------------
        # Representation regularisation
        # -----------------------------------------------------

        if sigreg_coef > 0.0:
            current_sigreg = (
                sigreg_phi_loss(
                    current_batch
                )
            )

        else:
            current_sigreg = torch.zeros(
                (),
                device=device,
            )

        if (
            goal_separation_coef > 0.0
            and len(task_goals) > 1
        ):
            goal_separation_loss, _ = (
                online_goal_separation_loss(
                    q_network=q_network,
                    task_goals=task_goals,
                    device=device,
                    target_cosine=(
                        goal_separation_target_cosine
                    ),
                )
            )

        else:
            goal_separation_loss = torch.zeros(
                (),
                device=device,
            )

        psi_raw_norm_loss = (
            raw_psi_norm_loss()
        )

        raw_phi_current = (
            raw_phi_from_batch(
                current_batch
            )
        )

        phi_raw_norm_loss = (
            norm_penalty_loss_l1(
                raw_phi_current,
                target_norm=(
                    q_network.phi_max_norm
                    - 0.1
                ),
            )
        )

        total_loss = (
            current_loss
            + replay_loss_coef
            * old_replay_loss
            + sigreg_coef
            * current_sigreg
            + goal_separation_coef
            * goal_separation_loss
            + psi_raw_norm_coef
            * psi_raw_norm_loss
            + phi_raw_norm_coef
            * phi_raw_norm_loss
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss.backward()

        nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=10.0,
        )

        optimizer.step()

        current_loss_value = (
            current_loss.detach()
        )

        old_replay_loss_value = (
            old_replay_loss.detach()
        )

        current_sigreg_loss_value = (
            current_sigreg.detach()
        )

        psi_raw_norm_loss_value = (
            psi_raw_norm_loss.detach()
        )

        phi_raw_norm_loss_value = (
            phi_raw_norm_loss.detach()
        )

        goal_separation_value = (
            goal_separation_loss.detach().item()
        )

        # -----------------------------------------------------
        # Polyak target update
        # -----------------------------------------------------

        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                q_network.parameters(),
                q_target_network.parameters(),
            ):
                target_parameter.mul_(
                    1.0 - tau
                ).add_(
                    tau * online_parameter
                )

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        if global_step % 1000 != 0:
            continue

        eval_env = make_env(
            goal=goal
        )

        def eval_policy(observation):
            observation_t = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                values = (
                    q_all_actions_from_goal(
                        q_network,
                        observation_t,
                        goal_tensor,
                    )
                )

            return int(
                values.argmax(
                    dim=-1
                ).item()
            )

        mean_return, mean_length = (
            evaluate_policy(
                eval_env,
                eval_policy,
                episodes=8,
            )
        )

        eval_returns.append(
            (
                global_step,
                mean_return,
            )
        )

        q_bound = (
            q_network.phi_max_norm
            * q_network.psi_max_norm
        )

        print(
            "[DQN-Atari-factorised] "
            f"step={global_step:7d} | "
            f"eps={epsilon:.3f} | "
            f"return={mean_return:.3f} | "
            f"len={mean_length:.1f} | "
            f"Current="
            f"{current_loss_value.item():.5f} | "
            f"OldReplay="
            f"{old_replay_loss_value.item():.5f} | "
            f"SIGReg="
            f"{current_sigreg_loss_value.item():.5f} | "
            f"QBound={q_bound:.3f} | "
            f"ReplayTasks="
            f"{len(replay_batches)} | "
            f"GoalSeparation="
            f"{goal_separation_value:.5f} | "
            f"RawPsiNormLoss="
            f"{psi_raw_norm_loss_value.item():.6f} | "
            f"RawPhiNormLoss="
            f"{phi_raw_norm_loss_value.item():.6f}"
        )

        eval_env.close()

        if mean_return >= early_stop_reward:
            success_streak += 1
        else:
            success_streak = 0

        if (
            enable_early_stop
            and success_streak
            >= early_stop_patience
        ):
            min_steps = global_step
            min_time = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Early stopping at step={global_step}, "
                f"return={mean_return:.3f}, "
                f"streak={success_streak}"
            )

            break

    # =========================================================
    # Final outputs
    # =========================================================

    if min_steps is None:
        min_steps = global_step
        min_time = (
            time.perf_counter()
            - start_time
        )

    with torch.no_grad():
        final_task_embedding = (
            q_network.encode_goal(
                goal_tensor
            )
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

    if len(buffer) > 0:
        final_probe_batch = buffer.sample(
            min(
                256,
                len(buffer),
            )
        )

        sa_embedding_fixed = (
            extract_fixed_probe_sa_embedding_atari(
                q_network=q_network,
                batch=final_probe_batch,
                as_numpy=True,
            )
        )

    else:
        sa_embedding_fixed = None

    sa_embedding_mean = (
        extract_mean_sa_embedding_atari(
            q_network=q_network,
            buffer=buffer,
            num_actions=num_actions,
            batch_size=256,
            device=device,
            as_numpy=True,
        )
    )

    sa_batch_final = (
        extract_sa_batch_for_isotropy_atari(
            q_network=q_network,
            buffer=buffer,
            num_actions=num_actions,
            batch_size=1024,
            device=device,
            as_numpy=True,
        )
    )

    if len(buffer) > 0:
        final_phi_batch = buffer.sample(
            min(
                1024,
                len(buffer),
            )
        )

        final_raw_phi_stats = (
            inspect_raw_phi_norms_atari(
                q_network,
                final_phi_batch,
            )
        )

        print(
            f"\nFinal raw phi norms "
            f"for task {task_id}:"
        )

        print(
            "Raw phi norms | "
            f"mean="
            f"{final_raw_phi_stats['mean']:.4f} | "
            f"std="
            f"{final_raw_phi_stats['std']:.4f} | "
            f"min="
            f"{final_raw_phi_stats['min']:.4f} | "
            f"max="
            f"{final_raw_phi_stats['max']:.4f}"
        )

    env.close()

    return (
        q_network,
        q_target_network,
        eval_returns,
        min_steps,
        min_time,
        final_task_embedding,
        sa_embedding_mean,
        sa_embedding_fixed,
        sa_batch_final,
        buffer,
    )


def extract_observation(reset_output):
    """
    Extract the observation from Gymnasium reset output.

    Supports:
        obs, info

    and dictionary-style observations containing:
        observation
        obs
    """

    obs, _ = reset_output

    if isinstance(obs, dict):
        if "observation" in obs:
            return obs["observation"]

        if "obs" in obs:
            return obs["obs"]

        raise KeyError(
            "Dictionary observation must contain "
            "'observation' or 'obs'."
        )

    return obs


def validate_atari_observation(
    obs,
    obs_shape,
):
    actual_shape = tuple(
        np.asarray(obs).shape
    )

    if actual_shape != tuple(obs_shape):
        raise ValueError(
            f"Expected observation shape {obs_shape}, "
            f"got {actual_shape}."
        )


def extract_mean_sa_embedding_atari(
    q_network,
    buffer,
    num_actions,
    batch_size,
    device,
    as_numpy=True,
):
    if len(buffer) == 0:
        return None

    batch = buffer.sample(
        min(
            batch_size,
            len(buffer),
        )
    )

    actions = (
        batch.actions
        .long()
        .view(-1)
    )

    action_onehot = F.one_hot(
        actions,
        num_classes=num_actions,
    ).to(
        dtype=batch.obs.dtype,
        device=batch.obs.device,
    )

    with torch.no_grad():
        phi = q_network.encode_state_action(
            batch.obs,
            action_onehot,
        )

        result = phi.mean(
            dim=0
        )

    if as_numpy:
        return result.cpu().numpy()

    return result


def extract_fixed_probe_sa_embedding_atari(
    q_network,
    batch,
    as_numpy=True,
):
    actions = (
        batch.actions
        .long()
        .view(-1)
    )

    action_onehot = F.one_hot(
        actions,
        num_classes=q_network.num_actions,
    ).to(
        dtype=batch.obs.dtype,
        device=batch.obs.device,
    )

    with torch.no_grad():
        phi = q_network.encode_state_action(
            batch.obs,
            action_onehot,
        )

        result = phi.mean(
            dim=0
        )

    if as_numpy:
        return result.cpu().numpy()

    return result


def extract_sa_batch_for_isotropy_atari(
    q_network,
    buffer,
    num_actions,
    batch_size,
    device,
    as_numpy=True,
):
    if len(buffer) == 0:
        return None

    batch = buffer.sample(
        min(
            batch_size,
            len(buffer),
        )
    )

    actions = (
        batch.actions
        .long()
        .view(-1)
    )

    action_onehot = F.one_hot(
        actions,
        num_classes=num_actions,
    ).to(
        dtype=batch.obs.dtype,
        device=batch.obs.device,
    )

    with torch.no_grad():
        embeddings = (
            q_network.encode_state_action(
                batch.obs,
                action_onehot,
            )
        )

    if as_numpy:
        return embeddings.cpu().numpy()

    return embeddings


def inspect_raw_phi_norms_atari(
    q_network,
    batch,
):
    actions = (
        batch.actions
        .long()
        .view(-1)
    )

    action_onehot = F.one_hot(
        actions,
        num_classes=q_network.num_actions,
    ).to(
        dtype=batch.obs.dtype,
        device=batch.obs.device,
    )

    with torch.no_grad():
        obs_features = (
            q_network._encode_obs_features(
                batch.obs
            )
        )

        action_features = (
            q_network.action_encoder(
                action_onehot
            )
        )

        sa = torch.cat(
            [
                obs_features,
                action_features,
            ],
            dim=-1,
        )

        raw_phi = q_network.sa_encoder(
            sa
        )

        norms = raw_phi.norm(
            dim=-1
        )

    return {
        "mean": norms.mean().item(),
        "std": norms.std(
            unbiased=False
        ).item(),
        "min": norms.min().item(),
        "max": norms.max().item(),
    }


def sac_train_tbtrl(
    seed: int = 42,
    actor: nn.Module = None,
    critic: nn.Module = None,
    critic_target: nn.Module = None,
    env=None,
    buffer_capacity: int = None,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    lr_ent_coef: float = 3e-4,
    obs_dim: int = None,
    action_dim: int = None,
    device: torch.device = None,
    total_steps: int = 300_000,
    warmup_steps: int = 25_000,
    batch_size: int = 256,
    eval_freq: int = 5_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    train_freq: int = 1,
    gradient_steps: int = 1,
    goal: np.ndarray = None,
    task_id: int = None,
    make_env=None,
    env_id: str = None,

    # {task_id: old_task_buffer}
    replay_task_buffers: Optional[Dict[int, Any]] = None,

    # {task_id: fixed_goal_np_array}
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
    enable_early_stop: bool = True,

    # =========================================================
    # TBTRL critic-only regularization
    # =========================================================

    # SIGReg applied to phi1 and phi2, then averaged.
    sigreg_coef: float = 0.0,
    sketch_dim: int = 64,

    # Pairwise goal embedding anti-collapse.
    # This contributes no useful signal for Task 0 alone.
    goal_separation_coef: float = 0.0,
    goal_separation_target_cosine: float = 0.85,

    # Soft excess-norm penalties.
    # Penalty is zero below target; no hard projection.
    phi_raw_norm_coef: float = 0.0,
    psi_raw_norm_coef: float = 0.0,

    # Explicit soft-range thresholds.
    # If None, use critic.phi_max_norm - 0.1 and
    # critic.psi_max_norm - 0.1 when those attributes exist.
    phi_norm_target: float | None = None,
    psi_norm_target: float | None = None,
    initial_alpha: float = 1.0,

    reset_target_from_critic: bool = True
):
    """
    Factorised SAC TBTRL trainer.

    SAC actor objective remains unchanged:

        L_actor =
            E[
                alpha * log pi(a | s, g)
                - min(Q1(s, a, g), Q2(s, a, g))
            ]

    Critic objective:

        L_critic =
            L_TD_current
            + replay_loss_coef * L_TD_old
            + sigreg_coef * L_SIGReg(phi)
            + goal_separation_coef * L_goal_sep(psi)
            + phi_raw_norm_coef * L_phi_norm
            + psi_raw_norm_coef * L_psi_norm

    Required factorised critic API:

        critic.q1_forward(state, action, goal)
        critic.q2_forward(state, action, goal)

        critic.phi1_forward(state, action)
        critic.psi1_forward(goal)

        critic.phi2_forward(state, action)
        critic.psi2_forward(goal)
    """

    # =========================================================
    # Validation
    # =========================================================

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
        raise ValueError(
            "ent_coef must be a positive float or 'auto'."
        )

    if isinstance(ent_coef, float) and ent_coef <= 0.0:
        raise ValueError(
            "Fixed ent_coef must be positive."
        )

    if sigreg_coef < 0.0:
        raise ValueError("sigreg_coef must be >= 0.")

    if goal_separation_coef < 0.0:
        raise ValueError(
            "goal_separation_coef must be >= 0."
        )

    if phi_raw_norm_coef < 0.0:
        raise ValueError(
            "phi_raw_norm_coef must be >= 0."
        )

    if psi_raw_norm_coef < 0.0:
        raise ValueError(
            "psi_raw_norm_coef must be >= 0."
        )

    if not -1.0 <= goal_separation_target_cosine <= 1.0:
        raise ValueError(
            "goal_separation_target_cosine must be "
            "within [-1, 1]."
        )

    # =========================================================
    # Device, dimensions, seed
    # =========================================================

    if device is None:
        device = next(
            critic.parameters()
        ).device

    device = torch.device(device)

    if obs_dim is None:
        obs_dim = (
            int(
                env.observation_space[
                    "observation"
                ].shape[0]
            )
            + int(
                env.observation_space[
                    "achieved_goal"
                ].shape[0]
            )
        )

    if action_dim is None:
        action_dim = int(
            env.action_space.shape[0]
        )

    goal_dim = int(
        env.observation_space[
            "desired_goal"
        ].shape[0]
    )

    set_seed(seed)

    if replay_task_buffers is None:
        replay_task_buffers = {}

    if task_goals is None:
        task_goals = {}

    training_goal = np.asarray(
        goal,
        dtype=np.float32,
    ).copy()

    if training_goal.shape != (goal_dim,):
        raise ValueError(
            f"goal has shape {training_goal.shape}; "
            f"expected ({goal_dim},)."
        )

    task_goals[task_id] = training_goal.copy()

    actor = actor.to(device)
    critic = critic.to(device)
    critic_target = critic_target.to(device)

    # =========================================================
    # Target critic setup
    # =========================================================

    critic_target.load_state_dict(
        critic.state_dict()
    )

    actor.train()
    critic.train()
    critic_target.eval()

    for parameter in critic_target.parameters():
        parameter.requires_grad_(False)

    # =========================================================
    # Verify factorised critic API only if TBTRL is active
    # =========================================================

    tbtrl_active = any(
        coefficient > 0.0
        for coefficient in [
            sigreg_coef,
            goal_separation_coef,
            phi_raw_norm_coef,
            psi_raw_norm_coef,
        ]
    )

    if tbtrl_active:
        required_methods = [
            "phi1_forward",
            "psi1_forward",
            "phi2_forward",
            "psi2_forward",
        ]

        missing_methods = [
            method_name
            for method_name in required_methods
            if not hasattr(critic, method_name)
        ]

        if len(missing_methods) > 0:
            raise AttributeError(
                "TBTRL regularizers require a factorised "
                "critic exposing: "
                f"{missing_methods}."
            )

    # =========================================================
    # Action bounds
    # =========================================================

    action_low = torch.as_tensor(
        env.action_space.low,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    action_high = torch.as_tensor(
        env.action_space.high,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    if action_low.shape[-1] != action_dim:
        raise RuntimeError(
            "Action-space bounds do not match action_dim."
        )

    if not (
        torch.allclose(
            action_low,
            -torch.ones_like(action_low),
        )
        and torch.allclose(
            action_high,
            torch.ones_like(action_high),
        )
    ):
        raise ValueError(
            "This SAC actor assumes action bounds [-1, 1]."
        )

    # =========================================================
    # Optimizers and replay buffer
    # =========================================================

    opt_actor = optim.Adam(
        actor.parameters(),
        lr=lr_actor,
    )

    opt_critic = optim.Adam(
        critic.parameters(),
        lr=lr_critic,
    )

    buffer = TrajectoryReplayBufferContinuous(
        buffer_capacity,
        obs_dim,
        action_dim,
        device=device,
    )

    # =========================================================
    # Entropy coefficient
    # =========================================================

    if target_entropy is None:
        target_entropy = -float(action_dim)

    if ent_coef == "auto":

        log_ent_coef = torch.tensor(
            np.log(initial_alpha),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        opt_ent_coef = optim.Adam(
            [log_ent_coef],
            lr=lr_ent_coef,
        )

        opt_ent_coef = optim.Adam(
            [log_ent_coef],
            lr=lr_ent_coef,
        )

        fixed_ent_coef = None

    else:
        fixed_ent_coef = torch.as_tensor(
            float(ent_coef),
            dtype=torch.float32,
            device=device,
        )

        log_ent_coef = None
        opt_ent_coef = None

    # =========================================================
    # Input normalizers
    # =========================================================

    state_rms = RunningMeanStd(
        shape=(obs_dim,),
        device=device,
    )

    goal_rms = RunningMeanStd(
        shape=(goal_dim,),
        device=device,
    )

    # =========================================================
    # Generic helpers
    # =========================================================

    def split_fetch_obs(obs_dict):
        observation = np.asarray(
            obs_dict["observation"],
            dtype=np.float32,
        )

        achieved_goal = np.asarray(
            obs_dict["achieved_goal"],
            dtype=np.float32,
        )

        desired_goal = np.asarray(
            obs_dict["desired_goal"],
            dtype=np.float32,
        )

        state = np.concatenate(
            [
                observation,
                achieved_goal,
            ],
            axis=-1,
        ).astype(np.float32)

        return state, desired_goal

    def goal_batch_for(
        goal_value,
        requested_batch_size: int,
        target_device: torch.device,
    ) -> torch.Tensor:
        if isinstance(goal_value, torch.Tensor):
            goal_batch = goal_value.to(
                device=target_device,
                dtype=torch.float32,
            )

        else:
            goal_batch = torch.as_tensor(
                np.asarray(
                    goal_value,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=target_device,
            )

        if goal_batch.ndim == 1:
            goal_batch = goal_batch.unsqueeze(0)

        if goal_batch.ndim != 2:
            raise ValueError(
                "Goal must be [goal_dim] or [B, goal_dim]. "
                f"Got {tuple(goal_batch.shape)}."
            )

        if goal_batch.shape[0] == 1:
            goal_batch = goal_batch.expand(
                requested_batch_size,
                -1,
            )

        elif goal_batch.shape[0] != requested_batch_size:
            raise ValueError(
                "Goal batch size does not match transition "
                "batch size."
            )

        return goal_batch

    def normalize_state(
        state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not normalize_state_inputs:
            return state_tensor

        return state_rms.normalize(
            state_tensor,
            clip=obs_norm_clip,
        )

    def normalize_goal(
        goal_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not normalize_goal_inputs:
            return goal_tensor

        return goal_rms.normalize(
            goal_tensor,
            clip=obs_norm_clip,
        )

    def current_ent_coef() -> torch.Tensor:
        if log_ent_coef is not None:
            return log_ent_coef.exp()

        return fixed_ent_coef

    def polyak_update_critic() -> None:
        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                critic.parameters(),
                critic_target.parameters(),
            ):
                target_parameter.mul_(1.0 - tau).add_(
                    online_parameter,
                    alpha=tau,
                )

    # =========================================================
    # SAC target and TD loss: unchanged algorithmically
    # =========================================================

    def td_target(
        batch,
        raw_goal_batch: torch.Tensor,
        ent_coef_tensor: torch.Tensor,
    ) -> torch.Tensor:
        rewards = batch.rewards
        terminated = batch.terminated

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)

        if terminated.ndim == 1:
            terminated = terminated.unsqueeze(-1)

        next_state = normalize_state(
            batch.next_obs
        )

        next_goal = normalize_goal(
            goal_batch_for(
                raw_goal_batch,
                batch.next_obs.shape[0],
                batch.next_obs.device,
            )
        )

        with torch.no_grad():
            next_action, next_log_prob, _ = actor.sample(
                next_state,
                next_goal,
            )

            next_q1 = critic_target.q1_forward(
                next_state,
                next_action,
                next_goal,
            )

            next_q2 = critic_target.q2_forward(
                next_state,
                next_action,
                next_goal,
            )

            next_q = torch.minimum(
                next_q1,
                next_q2,
            )

            next_soft_value = (
                next_q
                - ent_coef_tensor * next_log_prob
            )

            # SB3-equivalent treatment of Fetch time-limit
            # truncation: bootstrap unless actually terminated.
            if bootstrap_on_truncation:
                bootstrap_mask = 1.0 - terminated.float()

            else:
                truncated = batch.truncated

                if truncated.ndim == 1:
                    truncated = truncated.unsqueeze(-1)

                done = torch.logical_or(
                    terminated.bool(),
                    truncated.bool(),
                ).float()

                bootstrap_mask = 1.0 - done

            target = (
                rewards
                + gamma
                * bootstrap_mask
                * next_soft_value
            )

        return target

    def critic_td_loss(
        batch,
        raw_goal_batch: torch.Tensor,
        ent_coef_tensor: torch.Tensor,
    ) -> torch.Tensor:
        target = td_target(
            batch,
            raw_goal_batch,
            ent_coef_tensor,
        )

        state = normalize_state(
            batch.obs
        )

        goal_batch = normalize_goal(
            raw_goal_batch
        )

        q1 = critic.q1_forward(
            state,
            batch.actions,
            goal_batch,
        )

        q2 = critic.q2_forward(
            state,
            batch.actions,
            goal_batch,
        )

        if q1.shape != target.shape:
            raise RuntimeError(
                f"Q1 shape {q1.shape} does not match "
                f"target shape {target.shape}."
            )

        if q2.shape != target.shape:
            raise RuntimeError(
                f"Q2 shape {q2.shape} does not match "
                f"target shape {target.shape}."
            )
        
        return 0.5 * (
            F.mse_loss(q1, target)
            + F.mse_loss(q2, target)
        )

    # =========================================================
    # TBTRL helpers: critic-only regularization
    # =========================================================

    def seen_goals_tensor() -> torch.Tensor:
        seen_task_ids = sorted(
            task_goals.keys()
        )

        return torch.as_tensor(
            np.asarray(
                [
                    task_goals[seen_task_id]
                    for seen_task_id in seen_task_ids
                ],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=device,
        )

    def resolve_norm_target(
        name: str,
        supplied_target: float | None,
    ) -> float:
        if supplied_target is not None:
            target = float(supplied_target)

        elif name == "phi":
            target = float(
                getattr(
                    critic,
                    "phi_max_norm",
                    10.1,
                )
            ) - 0.1

        elif name == "psi":
            target = float(
                getattr(
                    critic,
                    "psi_max_norm",
                    10.1,
                )
            ) - 0.1

        else:
            raise ValueError(
                f"Unknown embedding name: {name}."
            )

        if target <= 0.0:
            raise ValueError(
                f"{name}_norm_target must be positive, "
                f"got {target}."
            )

        return target

    def soft_excess_norm_loss(
        embeddings: torch.Tensor,
        target_norm: float,
    ) -> torch.Tensor:
        """
        DQN-equivalent soft range-control objective:

        - zero penalty when ||z|| <= target_norm;
        - positive linear penalty on only the excess norm;
        - no hard constraint or embedding projection.
        """
        norms = embeddings.norm(
            p=2,
            dim=-1,
        )

        return F.relu(
            norms - target_norm
        ).mean()

    def factorised_embeddings(
        state: torch.Tensor,
        action: torch.Tensor,
        goal_batch: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,

    ]:
        phi1 = critic.phi1_forward(
            state,
            action,
        )

        psi1 = critic.psi1_forward(
            goal_batch,
        )

        phi2 = critic.phi2_forward(
            state,
            action,
        )

        psi2 = critic.psi2_forward(
            goal_batch,
        )

        return phi1, psi1, phi2, psi2

    def tbtrl_sigreg_loss(
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if sigreg_coef <= 0.0:
            return torch.zeros(
                (),
                dtype=torch.float32,
                device=device,
            )

        phi1 = critic.phi1_forward(
            state,
            action,
        )

        phi2 = critic.phi2_forward(
            state,
            action,
        )
        
        return 0.5 * (
            sigreg_loss(
                phi1,
                sketch_dim=sketch_dim,
            )
            + sigreg_loss(
                phi2,
                sketch_dim=sketch_dim,
            )
        )

    def tbtrl_goal_separation_loss() -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        zero = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )

        # No meaningful pairwise goal-separation objective
        # until at least two distinct task goals exist.
        if (
            goal_separation_coef <= 0.0
            or len(task_goals) <= 1
        ):
            return zero, zero, zero

        all_seen_goals = normalize_goal(
            seen_goals_tensor()
        )

        psi1_all = critic.psi1_forward(
            all_seen_goals
        )

        psi2_all = critic.psi2_forward(
            all_seen_goals
        )

        def head_separation_loss(
            psi: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
        ]:
            normalized_psi = F.normalize(
                psi,
                p=2,
                dim=-1,
                eps=1e-8,
            )

            cosine_matrix = (
                normalized_psi
                @ normalized_psi.T
            )

            n_goals = cosine_matrix.shape[0]

            off_diagonal_mask = ~torch.eye(
                n_goals,
                dtype=torch.bool,
                device=device,
            )

            off_diagonal_cosines = cosine_matrix[
                off_diagonal_mask
            ]

            separation = F.relu(
                off_diagonal_cosines
                - goal_separation_target_cosine
            ).mean()

            return (
                separation,
                off_diagonal_cosines.max(),
            )

        separation_1, max_cosine_1 = (
            head_separation_loss(
                psi1_all
            )
        )

        separation_2, max_cosine_2 = (
            head_separation_loss(
                psi2_all
            )
        )

        return (
            0.5 * (
                separation_1
                + separation_2
            ),
            max_cosine_1.detach(),
            max_cosine_2.detach(),
        )

    def tbtrl_norm_losses(
        state: torch.Tensor,
        action: torch.Tensor,
        goal_batch: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:

        phi1, _, phi2, _ = factorised_embeddings(
            state,
            action,
            goal_batch,
        )

        all_seen_goals = normalize_goal(
            seen_goals_tensor()
        )

        psi1_all = critic.psi1_forward(
            all_seen_goals
        )

        psi2_all = critic.psi2_forward(
            all_seen_goals
        )

        phi_target = resolve_norm_target(
            "phi",
            phi_norm_target,
        )

        psi_target = resolve_norm_target(
            "psi",
            psi_norm_target,
        )

        phi_loss = 0.5 * (
            soft_excess_norm_loss(
                phi1,
                phi_target,
            )
            + soft_excess_norm_loss(
                phi2,
                phi_target,
            )
        )


        psi_loss = 0.5 * (
            soft_excess_norm_loss(
                psi1_all,
                psi_target,
            )
            + soft_excess_norm_loss(
                psi2_all,
                psi_target,
            )
        )

        statistics = {
            "phi1_norm": phi1.norm(
                p=2,
                dim=-1,
            ).mean().detach(),

            "phi2_norm": phi2.norm(
                p=2,
                dim=-1,
            ).mean().detach(),

            "psi1_norm": psi1_all.norm(
                p=2,
                dim=-1,
            ).mean().detach(),

            "psi2_norm": psi2_all.norm(
                p=2,
                dim=-1,
            ).mean().detach(),
        }

        return (
            phi_loss,
            psi_loss,
            statistics,
        )

    # =========================================================
    # Training state
    # =========================================================

    obs_dict, _ = env.reset()

    global_step = 0
    update_count = 0
    success_streak = 0

    start_time = time.perf_counter()

    eval_returns = []
    eval_success_rates = []
    eval_final_distances = []

    min_steps = None
    min_time = None

    zero = torch.zeros(
        (),
        dtype=torch.float32,
        device=device,
    )

    current_critic_loss_value = zero
    old_replay_loss_value = zero
    current_actor_loss_value = zero
    current_ent_coef_value = torch.ones(
        (),
        dtype=torch.float32,
        device=device,
    )
    current_ent_coef_loss_value = zero
    current_log_prob_value = zero
    current_entropy_value = zero
    current_q_pi_value = zero
    current_q_data_value = zero
    current_pi_abs_value = zero
    current_pi_saturation_value = zero
    current_actor_grad_norm_value = zero

    current_sigreg_value = zero
    current_goal_separation_value = zero
    current_phi_raw_norm_value = zero
    current_psi_raw_norm_value = zero

    current_phi1_norm_value = zero
    current_phi2_norm_value = zero
    current_psi1_norm_value = zero
    current_psi2_norm_value = zero

    current_psi1_max_cosine_value = zero
    current_psi2_max_cosine_value = zero

    # =========================================================
    # Main SAC training loop
    # =========================================================

    while global_step < total_steps:
        # -----------------------------------------------------
        # Data collection: unchanged SAC logic
        # -----------------------------------------------------

        state, current_goal = split_fetch_obs(
            obs_dict
        )

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        goal_t = torch.as_tensor(
            current_goal,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            state_rms.update(state_t)

            if normalize_goal_inputs:
                goal_rms.update(goal_t)

        if global_step < warmup_steps:
            action = env.action_space.sample().astype(
                np.float32
            )

        else:
            with torch.no_grad():
                action_t, _, _ = actor.sample(
                    normalize_state(state_t),
                    normalize_goal(goal_t),
                )

                action = (
                    action_t.squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

        (
            next_obs_dict,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action)

        next_state, next_goal = split_fetch_obs(
            next_obs_dict
        )

        with torch.no_grad():
            next_state_t = torch.as_tensor(
                next_state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            state_rms.update(next_state_t)

            if normalize_goal_inputs:
                next_goal_t = torch.as_tensor(
                    next_goal,
                    dtype=torch.float32,
                    device=device,
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
            obs_dict, _ = env.reset()

        if len(buffer) < warmup_steps:
            continue

        if global_step % train_freq != 0:
            continue

        # -----------------------------------------------------
        # Gradient updates
        # -----------------------------------------------------

        for _ in range(gradient_steps):
            current_batch = buffer.sample(
                batch_size
            )

            # Immutable task goal, not a mutable live env variable.
            current_goal_tensor = goal_batch_for(
                training_goal,
                current_batch.obs.shape[0],
                device,
            )

            # -------------------------------------------------
            # Select old-task replay batches
            # -------------------------------------------------

            eligible_old_task_ids = []

            for old_task_id, old_buffer in (
                replay_task_buffers.items()
            ):
                if old_buffer is None:
                    continue

                if len(old_buffer) < 1:
                    continue

                if old_task_id not in task_goals:
                    raise KeyError(
                        f"Missing goal for replay task "
                        f"{old_task_id}."
                    )

                eligible_old_task_ids.append(
                    old_task_id
                )

            replay_batches = []

            if (
                len(eligible_old_task_ids) > 0
                and replay_ratio > 0.0
            ):
                if replay_tasks_per_batch is None:
                    n_old_tasks = len(
                        eligible_old_task_ids
                    )

                else:
                    n_old_tasks = min(
                        int(replay_tasks_per_batch),
                        len(eligible_old_task_ids),
                    )

                cycle_index = (
                    update_count
                    % len(eligible_old_task_ids)
                )

                ordered_old_task_ids = (
                    eligible_old_task_ids[cycle_index:]
                    + eligible_old_task_ids[:cycle_index]
                )

                selected_old_task_ids = (
                    ordered_old_task_ids[:n_old_tasks]
                )

                replay_batch_size = max(
                    1,
                    int(
                        batch_size
                        * replay_ratio
                        / n_old_tasks
                    ),
                )

                for old_task_id in selected_old_task_ids:
                    old_buffer = replay_task_buffers[
                        old_task_id
                    ]

                    if len(old_buffer) < replay_batch_size:
                        continue

                    old_batch = old_buffer.sample(
                        replay_batch_size
                    )

                    old_goal = task_goals[
                        old_task_id
                    ]

                    replay_batches.append(
                        (
                            old_goal,
                            old_batch,
                        )
                    )

            # -------------------------------------------------
            # SAC critic TD losses: unchanged
            # -------------------------------------------------

            ent_coef_tensor = current_ent_coef().detach()

            current_td_loss = critic_td_loss(
                current_batch,
                current_goal_tensor,
                ent_coef_tensor,
            )

            old_td_losses = []

            for old_goal, old_batch in replay_batches:
                old_goal_tensor = goal_batch_for(
                    old_goal,
                    old_batch.obs.shape[0],
                    device,
                )

                old_td_losses.append(
                    critic_td_loss(
                        old_batch,
                        old_goal_tensor,
                        ent_coef_tensor,
                    )
                )

            if len(old_td_losses) > 0:
                old_replay_td_loss = torch.stack(
                    old_td_losses
                ).mean()

            else:
                old_replay_td_loss = torch.zeros(
                    (),
                    device=device,
                )

            # -------------------------------------------------
            # TBTRL critic-only regularisation
            # -------------------------------------------------

            state_for_reg = normalize_state(
                current_batch.obs
            )

            goal_for_reg = normalize_goal(
                current_goal_tensor
            )

            current_sigreg = tbtrl_sigreg_loss(
                state_for_reg,
                current_batch.actions,
            )

            (
                current_goal_separation,
                psi1_max_cosine,
                psi2_max_cosine,
            ) = tbtrl_goal_separation_loss()

            (
                phi_raw_norm_loss,
                psi_raw_norm_loss,
                norm_statistics,
            ) = tbtrl_norm_losses(
                state_for_reg,
                current_batch.actions,
                goal_for_reg,
            )

            critic_total_loss = (
                current_td_loss
                + replay_loss_coef
                * old_replay_td_loss
                + sigreg_coef
                * current_sigreg
                + goal_separation_coef
                * current_goal_separation
                + phi_raw_norm_coef
                * phi_raw_norm_loss
                + psi_raw_norm_coef
                * psi_raw_norm_loss
            )

            opt_critic.zero_grad(
                set_to_none=True
            )

            critic_total_loss.backward()

            nn.utils.clip_grad_norm_(
                critic.parameters(),
                max_norm=10.0,
            )

            opt_critic.step()

            # -------------------------------------------------
            # SAC actor update: unchanged
            # -------------------------------------------------

            state_batch = normalize_state(
                current_batch.obs
            )

            goal_batch = normalize_goal(
                current_goal_tensor
            )

            sampled_actions, log_prob, _ = actor.sample(
                state_batch,
                goal_batch,
            )

            q1_pi = critic.q1_forward(
                state_batch,
                sampled_actions,
                goal_batch,
            )

            q2_pi = critic.q2_forward(
                state_batch,
                sampled_actions,
                goal_batch,
            )

            min_q_pi = torch.minimum(
                q1_pi,
                q2_pi,
            )

            ent_coef_tensor = current_ent_coef().detach()

            actor_loss = (
                ent_coef_tensor * log_prob
                - min_q_pi
            ).mean()

            opt_actor.zero_grad(
                set_to_none=True
            )

            actor_loss.backward()

            actor_grad_norm = nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=10.0,
            )

            opt_actor.step()

            # -------------------------------------------------
            # SAC automatic entropy update: unchanged
            # -------------------------------------------------

            if log_ent_coef is not None:
                ent_coef_loss = -(
                    log_ent_coef
                    * (
                        log_prob.detach()
                        + target_entropy
                    )
                ).mean()

                opt_ent_coef.zero_grad(
                    set_to_none=True
                )

                ent_coef_loss.backward()

                opt_ent_coef.step()

            else:
                ent_coef_loss = torch.zeros(
                    (),
                    device=device,
                )

            # -------------------------------------------------
            # Target critic update: unchanged
            # -------------------------------------------------

            polyak_update_critic()

            update_count += 1

            # -------------------------------------------------
            # Diagnostics
            # -------------------------------------------------

            with torch.no_grad():
                q_data_1 = critic.q1_forward(
                    state_batch,
                    current_batch.actions,
                    goal_batch,
                )

                q_data_2 = critic.q2_forward(
                    state_batch,
                    current_batch.actions,
                    goal_batch,
                )

                q_data = torch.minimum(
                    q_data_1,
                    q_data_2,
                )

                pi_abs = sampled_actions.abs().mean()

                pi_saturation = (
                    sampled_actions.abs() > 0.95
                ).float().mean()

            current_critic_loss_value = (
                current_td_loss.detach()
            )

            old_replay_loss_value = (
                old_replay_td_loss.detach()
            )

            current_actor_loss_value = actor_loss.detach()

            current_ent_coef_value = (
                current_ent_coef().detach()
            )

            current_ent_coef_loss_value = (
                ent_coef_loss.detach()
            )

            current_log_prob_value = (
                log_prob.mean().detach()
            )

            current_entropy_value = (
                -log_prob.mean().detach()
            )

            current_q_pi_value = (
                min_q_pi.mean().detach()
            )

            current_q_data_value = (
                q_data.mean().detach()
            )

            current_pi_abs_value = pi_abs.detach()

            current_pi_saturation_value = (
                pi_saturation.detach()
            )

            current_actor_grad_norm_value = (
                torch.as_tensor(
                    actor_grad_norm,
                    dtype=torch.float32,
                    device=device,
                ).detach()
            )

            current_sigreg_value = current_sigreg.detach()

            current_goal_separation_value = (
                current_goal_separation.detach()
            )

            current_phi_raw_norm_value = (
                phi_raw_norm_loss.detach()
            )

            current_psi_raw_norm_value = (
                psi_raw_norm_loss.detach()
            )

            current_phi1_norm_value = norm_statistics[
                "phi1_norm"
            ]

            current_phi2_norm_value = norm_statistics[
                "phi2_norm"
            ]

            current_psi1_norm_value = norm_statistics[
                "psi1_norm"
            ]

            current_psi2_norm_value = norm_statistics[
                "psi2_norm"
            ]

            current_psi1_max_cosine_value = (
                psi1_max_cosine
            )

            current_psi2_max_cosine_value = (
                psi2_max_cosine
            )

        # -----------------------------------------------------
        # Evaluation: same as working SAC trainer
        # -----------------------------------------------------

        if global_step % eval_freq != 0:
            continue

        eval_env = make_env(
            goal=training_goal,
            env_id=env_id,
        )

        def eval_policy(observation_dict):
            eval_state, eval_goal = split_fetch_obs(
                observation_dict
            )

            eval_state_t = torch.as_tensor(
                eval_state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            eval_goal_t = torch.as_tensor(
                eval_goal,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                eval_action = actor.deterministic(
                    normalize_state(eval_state_t),
                    normalize_goal(eval_goal_t),
                )

            return (
                eval_action
                .squeeze(0)
                .cpu()
                .numpy()
            )

        (
            mean_return,
            mean_length,
            success_rate,
            mean_final_distance,
        ) = evaluate_policy_with_success(
            eval_env,
            eval_policy,
            episodes=8,
            seed=seed + 100_000 + global_step,
        )

        eval_returns.append(
            (
                global_step,
                mean_return,
            )
        )

        eval_success_rates.append(
            (
                global_step,
                success_rate,
            )
        )

        eval_final_distances.append(
            (
                global_step,
                mean_final_distance,
            )
        )

        if normalize_state_inputs:
            state_std = torch.sqrt(
                state_rms.var
            ).detach().cpu().numpy()

            state_norm_string = (
                f"StateNormCount="
                f"{state_rms.count.item():.0f} | "
                f"StateStdMin={state_std.min():.6f} | "
                f"StateStdMax={state_std.max():.6f} | "
            )

        else:
            state_norm_string = ""

        print(
            "[SAC-TBTRL] "
            f"step={global_step:7d} | "
            f"return={mean_return:.3f} | "
            f"len={mean_length:.1f} | "
            f"Success={success_rate:.3f} | "
            f"FinalDist={mean_final_distance:.4f} | "
            f"CriticTD={current_critic_loss_value.item():.5f} | "
            f"OldReplay={old_replay_loss_value.item():.5f} | "
            f"SIGReg={current_sigreg_value.item():.6f} | "
            f"GoalSep={current_goal_separation_value.item():.6f} | "
            f"PhiNormReg={current_phi_raw_norm_value.item():.6f} | "
            f"PsiNormReg={current_psi_raw_norm_value.item():.6f} | "
            f"Phi1Norm={current_phi1_norm_value.item():.3f} | "
            f"Phi2Norm={current_phi2_norm_value.item():.3f} | "
            f"Psi1Norm={current_psi1_norm_value.item():.3f} | "
            f"Psi2Norm={current_psi2_norm_value.item():.3f} | "
            f"PsiMaxCos=("
            f"{current_psi1_max_cosine_value.item():.3f},"
            f"{current_psi2_max_cosine_value.item():.3f}"
            f") | "
            f"ActorLoss={current_actor_loss_value.item():.5f} | "
            f"Alpha={current_ent_coef_value.item():.5f} | "
            f"AlphaLoss={current_ent_coef_loss_value.item():.5f} | "
            f"LogProb={current_log_prob_value.item():.5f} | "
            f"Entropy={current_entropy_value.item():.5f} | "
            f"Qpi={current_q_pi_value.item():.5f} | "
            f"Qdata={current_q_data_value.item():.5f} | "
            f"MeanAbsPi={current_pi_abs_value.item():.5f} | "
            f"PiSat={current_pi_saturation_value.item():.3f} | "
            f"ActorGrad={current_actor_grad_norm_value.item():.5f} | "
            f"{state_norm_string}"
            f"ReplayTasks={len(replay_batches)}"
        )

        eval_env.close()

        if early_stop_success_rate is not None:
            criterion_met = (
                success_rate
                >= early_stop_success_rate
            )

        else:
            criterion_met = (
                mean_return
                >= early_stop_reward
            )

        if criterion_met:
            success_streak += 1

        else:
            success_streak = 0

        if (
            enable_early_stop
            and success_streak
            >= early_stop_patience
        ):
            min_steps = global_step

            min_time = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Early stopping at step={global_step}, "
                f"return={mean_return:.3f}, "
                f"success={success_rate:.3f}, "
                f"streak={success_streak}"
            )

            break

    # =========================================================
    # Final outputs: identical structure to standard trainer
    # =========================================================

    if min_steps is None:
        min_steps = global_step

        min_time = (
            time.perf_counter()
            - start_time
        )

    env.close()

    return (
        actor,
        critic,
        critic_target,
        eval_returns,
        eval_success_rates,
        eval_final_distances,
        min_steps,
        min_time,
        buffer,
    )

def sac_train_standard(
    seed: int = 42,
    actor: nn.Module = None,
    critic: nn.Module = None,
    critic_target: nn.Module = None,
    env=None,
    buffer_capacity: int = None,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    lr_ent_coef: float = 3e-4,
    obs_dim: int = None,
    action_dim: int = None,
    device: torch.device = None,
    total_steps: int = 300_000,
    warmup_steps: int = 25_000,
    batch_size: int = 256,
    eval_freq: int = 5_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    train_freq: int = 1,
    gradient_steps: int = 1,
    goal: np.ndarray = None,
    task_id: int = None,
    make_env=None,

    # {task_id: old_task_buffer}
    replay_task_buffers: Optional[Dict[int, Any]] = None,
    # {task_id: goal}
    task_goals: Optional[Dict[int, np.ndarray]] = None,

    replay_ratio: float = 0.0,
    replay_tasks_per_batch: Optional[int] = None,
    replay_loss_coef: float = 1.0,

    bootstrap_on_truncation: bool = False,
    ent_coef: str | float = "auto",
    target_entropy: float | None = None,

    normalize_state_inputs: bool = True,
    normalize_goal_inputs: bool = False,
    obs_norm_clip: float = 10.0,

    early_stop_reward: float = -0.05,
    early_stop_success_rate: float | None = None,
    early_stop_patience: int = 5,
    enable_early_stop: bool = True,
):
    """
    Goal-conditioned SAC trainer using a standard twin critic.

    Actor:
        pi(a | state, goal), represented by a diagonal Gaussian followed
        by tanh squashing.

    Critic:
        Q_k(state, action, goal), k in {1, 2}.

    SAC target:
        y = r + gamma * mask *
            [min(Q1_target(s', a', g), Q2_target(s', a', g))
             - alpha * log pi(a' | s', g)]

    Data collection:
        - Uniform random legal actions before warmup_steps.
        - Stochastic policy samples thereafter.

    The replay buffer stores raw transitions. State/goal normalization
    is applied only before network calls, with persistent statistics.
    """

    # =========================================================
    # Validation
    # =========================================================

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
        raise ValueError(
            "ent_coef must be a positive float or the string 'auto'."
        )

    if isinstance(ent_coef, float) and ent_coef <= 0.0:
        raise ValueError(
            "Fixed ent_coef must be positive."
        )

    # =========================================================
    # Device, dimensions, seed
    # =========================================================

    if device is None:
        device = next(critic.parameters()).device

    device = torch.device(device)

    if obs_dim is None:
        obs_dim = (
            int(env.observation_space["observation"].shape[0])
            + int(env.observation_space["achieved_goal"].shape[0])
        )

    if action_dim is None:
        action_dim = int(env.action_space.shape[0])

    goal_dim = int(
        env.observation_space["desired_goal"].shape[0]
    )

    set_seed(seed)

    if replay_task_buffers is None:
        replay_task_buffers = {}

    if task_goals is None:
        task_goals = {}

    task_goals[task_id] = np.asarray(
        goal,
        dtype=np.float32,
    ).copy()

    actor = actor.to(device)
    critic = critic.to(device)
    critic_target = critic_target.to(device)

    # =========================================================
    # Target critic setup
    # =========================================================

    critic_target.load_state_dict(
        critic.state_dict()
    )

    actor.train()
    critic.train()
    critic_target.eval()

    for parameter in critic_target.parameters():
        parameter.requires_grad_(False)

    # =========================================================
    # Action bounds
    # =========================================================

    action_low = torch.as_tensor(
        env.action_space.low,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    action_high = torch.as_tensor(
        env.action_space.high,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    if action_low.shape[-1] != action_dim:
        raise RuntimeError(
            "Action-space bounds do not match action_dim: "
            f"{action_low.shape[-1]} vs {action_dim}."
        )

    # SAC actor emits tanh actions in [-1, 1].
    # FetchPush normally has exactly this action range.
    if not (
        torch.allclose(
            action_low,
            -torch.ones_like(action_low),
        )
        and torch.allclose(
            action_high,
            torch.ones_like(action_high),
        )
    ):
        raise ValueError(
            "This SAC actor currently assumes action bounds [-1, 1]. "
            "Add affine action scaling if your environment differs."
        )

    # =========================================================
    # Optimisers and replay
    # =========================================================

    opt_actor = optim.Adam(
        actor.parameters(),
        lr=lr_actor,
    )

    opt_critic = optim.Adam(
        critic.parameters(),
        lr=lr_critic,
    )

    buffer = TrajectoryReplayBufferContinuous(
        buffer_capacity,
        obs_dim,
        action_dim,
        device=device,
    )

    # =========================================================
    # Entropy coefficient
    # =========================================================

    if target_entropy is None:
        target_entropy = -float(action_dim)

    if ent_coef == "auto":
        log_ent_coef = torch.zeros(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        opt_ent_coef = optim.Adam(
            [log_ent_coef],
            lr=lr_ent_coef,
        )

        fixed_ent_coef = None

    else:
        fixed_ent_coef = torch.as_tensor(
            float(ent_coef),
            dtype=torch.float32,
            device=device,
        )

        log_ent_coef = None
        opt_ent_coef = None

    # =========================================================
    # Input normalizers
    # =========================================================

    state_rms = RunningMeanStd(
        shape=(obs_dim,),
        device=device,
    )

    goal_rms = RunningMeanStd(
        shape=(goal_dim,),
        device=device,
    )

    # =========================================================
    # Helpers
    # =========================================================

    def split_fetch_obs(obs_dict):
        observation = np.asarray(
            obs_dict["observation"],
            dtype=np.float32,
        )

        achieved_goal = np.asarray(
            obs_dict["achieved_goal"],
            dtype=np.float32,
        )

        desired_goal = np.asarray(
            obs_dict["desired_goal"],
            dtype=np.float32,
        )

        state = np.concatenate(
            [
                observation,
                achieved_goal,
            ],
            axis=-1,
        ).astype(np.float32)

        return state, desired_goal

    def goal_batch_for(
        goal_value,
        requested_batch_size: int,
        target_device: torch.device,
    ) -> torch.Tensor:
        if isinstance(goal_value, torch.Tensor):
            goal_batch = goal_value.to(
                device=target_device,
                dtype=torch.float32,
            )
        else:
            goal_batch = torch.as_tensor(
                np.asarray(
                    goal_value,
                    dtype=np.float32,
                ),
                dtype=torch.float32,
                device=target_device,
            )

        if goal_batch.ndim == 1:
            goal_batch = goal_batch.unsqueeze(0)

        if goal_batch.ndim != 2:
            raise ValueError(
                "Goal must have shape [goal_dim] or [B, goal_dim]. "
                f"Got {tuple(goal_batch.shape)}."
            )

        if goal_batch.shape[0] == 1:
            goal_batch = goal_batch.expand(
                requested_batch_size,
                -1,
            )

        elif goal_batch.shape[0] != requested_batch_size:
            raise ValueError(
                "Goal batch size does not match requested batch size: "
                f"{goal_batch.shape[0]} vs {requested_batch_size}."
            )

        return goal_batch

    def normalize_state(
        state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not normalize_state_inputs:
            return state_tensor

        return state_rms.normalize(
            state_tensor,
            clip=obs_norm_clip,
        )

    def normalize_goal(
        goal_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not normalize_goal_inputs:
            return goal_tensor

        return goal_rms.normalize(
            goal_tensor,
            clip=obs_norm_clip,
        )

    def polyak_update_critic() -> None:
        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                critic.parameters(),
                critic_target.parameters(),
            ):
                target_parameter.mul_(1.0 - tau).add_(
                    online_parameter,
                    alpha=tau,
                )

    def current_ent_coef() -> torch.Tensor:
        if log_ent_coef is not None:
            return log_ent_coef.exp()

        return fixed_ent_coef

    def td_target(
        batch,
        raw_goal_batch: torch.Tensor,
        ent_coef_tensor: torch.Tensor,
    ) -> torch.Tensor:
        raw_next_state = batch.next_obs

        rewards = batch.rewards
        terminated = batch.terminated
        truncated = batch.truncated

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)

        if terminated.ndim == 1:
            terminated = terminated.unsqueeze(-1)

        if truncated.ndim == 1:
            truncated = truncated.unsqueeze(-1)

        raw_next_goal = goal_batch_for(
            raw_goal_batch,
            raw_next_state.shape[0],
            raw_next_state.device,
        )

        next_state = normalize_state(
            raw_next_state
        )

        next_goal = normalize_goal(
            raw_next_goal
        )

        with torch.no_grad():
            next_action, next_log_prob, _ = actor.sample(
                next_state,
                next_goal,
            )

            next_q1 = critic_target.q1_forward(
                next_state,
                next_action,
                next_goal,
            )

            next_q2 = critic_target.q2_forward(
                next_state,
                next_action,
                next_goal,
            )

            next_q = torch.minimum(
                next_q1,
                next_q2,
            )

            next_soft_value = (
                next_q
                - ent_coef_tensor * next_log_prob
            )

            if bootstrap_on_truncation:
                bootstrap_mask = 1.0 - terminated.float()

            else:
                done = torch.logical_or(
                    terminated.bool(),
                    truncated.bool(),
                ).float()

                bootstrap_mask = 1.0 - done

            target = (
                rewards
                + gamma
                * bootstrap_mask
                * next_soft_value
            )

        return target

    def critic_td_loss(
        batch,
        raw_goal_batch: torch.Tensor,
        ent_coef_tensor: torch.Tensor,
    ) -> torch.Tensor:
        target = td_target(
            batch,
            raw_goal_batch,
            ent_coef_tensor,
        )

        state = normalize_state(
            batch.obs
        )

        goal_batch = normalize_goal(
            raw_goal_batch
        )

        q1 = critic.q1_forward(
            state,
            batch.actions,
            goal_batch,
        )

        q2 = critic.q2_forward(
            state,
            batch.actions,
            goal_batch,
        )

        if q1.shape != target.shape:
            raise RuntimeError(
                f"Q1 shape {q1.shape} does not match target "
                f"shape {target.shape}."
            )

        if q2.shape != target.shape:
            raise RuntimeError(
                f"Q2 shape {q2.shape} does not match target "
                f"shape {target.shape}."
            )

        return 0.5 *(
            F.mse_loss(q1, target)
            + F.mse_loss(q2, target)
        ) 

    # =========================================================
    # Training state
    # =========================================================

    obs_dict, _ = env.reset()

    global_step = 0
    update_count = 0
    success_streak = 0

    start_time = time.perf_counter()

    eval_returns = []
    eval_success_rates = []
    eval_final_distances = []

    min_steps = None
    min_time = None

    current_critic_loss_value = torch.zeros(
        (),
        device=device,
    )

    old_replay_loss_value = torch.zeros(
        (),
        device=device,
    )

    current_actor_loss_value = torch.zeros(
        (),
        device=device,
    )

    current_ent_coef_value = torch.as_tensor(
        1.0,
        dtype=torch.float32,
        device=device,
    )

    current_ent_coef_loss_value = torch.zeros(
        (),
        device=device,
    )

    current_log_prob_value = torch.zeros(
        (),
        device=device,
    )

    current_entropy_value = torch.zeros(
        (),
        device=device,
    )

    current_q_pi_value = torch.zeros(
        (),
        device=device,
    )

    current_q_data_value = torch.zeros(
        (),
        device=device,
    )

    current_pi_abs_value = torch.zeros(
        (),
        device=device,
    )

    current_pi_saturation_value = torch.zeros(
        (),
        device=device,
    )

    current_actor_grad_norm_value = torch.zeros(
        (),
        device=device,
    )

    # =========================================================
    # Main training loop
    # =========================================================

    while global_step < total_steps:
        # -----------------------------------------------------
        # Environment interaction
        # -----------------------------------------------------

        state, current_goal = split_fetch_obs(
            obs_dict
        )

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        goal_t = torch.as_tensor(
            current_goal,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        # Update normalization moments using raw environment data.
        with torch.no_grad():
            state_rms.update(state_t)

            if normalize_goal_inputs:
                goal_rms.update(goal_t)

        # Uniform random warm-up, then stochastic SAC behaviour policy.
        if global_step < warmup_steps:
            action = env.action_space.sample().astype(
                np.float32
            )

        else:
            with torch.no_grad():
                normalized_state_t = normalize_state(
                    state_t
                )

                normalized_goal_t = normalize_goal(
                    goal_t
                )

                action_t, _, _ = actor.sample(
                    normalized_state_t,
                    normalized_goal_t,
                )

                action = (
                    action_t.squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

        next_obs_dict, reward, terminated, truncated, _ = env.step(
            action
        )

        next_state, next_goal = split_fetch_obs(
            next_obs_dict
        )

        with torch.no_grad():
            next_state_t = torch.as_tensor(
                next_state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            state_rms.update(next_state_t)

            if normalize_goal_inputs:
                next_goal_t = torch.as_tensor(
                    next_goal,
                    dtype=torch.float32,
                    device=device,
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
            obs_dict, _ = env.reset()

        # -----------------------------------------------------
        # Delay learning until warm-up finishes
        # -----------------------------------------------------

        if len(buffer) < warmup_steps:
            continue

        if global_step % train_freq != 0:
            continue

        # -----------------------------------------------------
        # Multiple gradient updates if requested
        # -----------------------------------------------------

        for _ in range(gradient_steps):
            current_batch = buffer.sample(batch_size)

            current_goal_tensor = goal_batch_for(
                current_goal,
                current_batch.obs.shape[0],
                device,
            )

            # -------------------------------------------------
            # Optional old-task replay selection
            # -------------------------------------------------

            eligible_old_task_ids = []

            for old_task_id, old_buffer in replay_task_buffers.items():
                if old_buffer is None:
                    continue

                if len(old_buffer) < 1:
                    continue

                if old_task_id not in task_goals:
                    raise KeyError(
                        f"Missing goal for replay task "
                        f"{old_task_id}."
                    )

                eligible_old_task_ids.append(old_task_id)

            replay_batches = []

            if (
                len(eligible_old_task_ids) > 0
                and replay_ratio > 0.0
            ):
                if replay_tasks_per_batch is None:
                    n_old_tasks = len(
                        eligible_old_task_ids
                    )
                else:
                    n_old_tasks = min(
                        int(replay_tasks_per_batch),
                        len(eligible_old_task_ids),
                    )

                cycle_index = update_count % len(
                    eligible_old_task_ids
                )

                ordered_old_task_ids = (
                    eligible_old_task_ids[cycle_index:]
                    + eligible_old_task_ids[:cycle_index]
                )

                selected_old_task_ids = (
                    ordered_old_task_ids[:n_old_tasks]
                )

                replay_batch_size = max(
                    1,
                    int(
                        batch_size
                        * replay_ratio
                        / n_old_tasks
                    ),
                )

                for old_task_id in selected_old_task_ids:
                    old_buffer = replay_task_buffers[
                        old_task_id
                    ]

                    if len(old_buffer) < replay_batch_size:
                        continue

                    old_batch = old_buffer.sample(
                        replay_batch_size
                    )

                    old_goal = task_goals[old_task_id]

                    replay_batches.append(
                        (old_goal, old_batch)
                    )

            # -------------------------------------------------
            # Entropy coefficient value, held fixed for critic
            # and actor gradients except for alpha optimization
            # -------------------------------------------------

            ent_coef_tensor = current_ent_coef().detach()

            # -------------------------------------------------
            # Critic update
            # -------------------------------------------------

            current_td_loss = critic_td_loss(
                current_batch,
                current_goal_tensor,
                ent_coef_tensor,
            )

            old_td_losses = []

            for old_goal, old_batch in replay_batches:
                old_goal_tensor = goal_batch_for(
                    old_goal,
                    old_batch.obs.shape[0],
                    device,
                )

                old_td_losses.append(
                    critic_td_loss(
                        old_batch,
                        old_goal_tensor,
                        ent_coef_tensor,
                    )
                )

            if len(old_td_losses) > 0:
                old_replay_td_loss = torch.stack(
                    old_td_losses
                ).mean()

            else:
                old_replay_td_loss = torch.zeros(
                    (),
                    device=device,
                )

            critic_total_loss = (
                current_td_loss
                + replay_loss_coef * old_replay_td_loss
            )

            opt_critic.zero_grad(set_to_none=True)

            critic_total_loss.backward()

            nn.utils.clip_grad_norm_(
                critic.parameters(),
                max_norm=10.0,
            )

            opt_critic.step()

            # -------------------------------------------------
            # Actor update
            # -------------------------------------------------

            raw_state_batch = current_batch.obs

            raw_goal_batch = goal_batch_for(
                current_goal,
                raw_state_batch.shape[0],
                device,
            )

            state_batch = normalize_state(
                raw_state_batch
            )

            goal_batch = normalize_goal(
                raw_goal_batch
            )

            sampled_actions, log_prob, _ = actor.sample(
                state_batch,
                goal_batch,
            )

            q1_pi = critic.q1_forward(
                state_batch,
                sampled_actions,
                goal_batch,
            )

            q2_pi = critic.q2_forward(
                state_batch,
                sampled_actions,
                goal_batch,
            )

            min_q_pi = torch.minimum(
                q1_pi,
                q2_pi,
            )

            ent_coef_tensor = current_ent_coef().detach()

            actor_loss = (
                ent_coef_tensor * log_prob
                - min_q_pi
            ).mean()

            opt_actor.zero_grad(set_to_none=True)

            actor_loss.backward()

            actor_grad_norm = nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=10.0,
            )

            opt_actor.step()

            # -------------------------------------------------
            # Automatic entropy-temperature update
            # -------------------------------------------------

            if log_ent_coef is not None:
                ent_coef_loss = -(
                    log_ent_coef
                    * (
                        log_prob.detach()
                        + target_entropy
                    )
                ).mean()

                opt_ent_coef.zero_grad(
                    set_to_none=True
                )

                ent_coef_loss.backward()

                opt_ent_coef.step()

            else:
                ent_coef_loss = torch.zeros(
                    (),
                    device=device,
                )

            # -------------------------------------------------
            # Target critic update
            # -------------------------------------------------

            polyak_update_critic()

            update_count += 1

            # -------------------------------------------------
            # Diagnostics
            # -------------------------------------------------

            with torch.no_grad():
                q_data_1 = critic.q1_forward(
                    state_batch,
                    current_batch.actions,
                    goal_batch,
                )

                q_data_2 = critic.q2_forward(
                    state_batch,
                    current_batch.actions,
                    goal_batch,
                )

                q_data = torch.minimum(
                    q_data_1,
                    q_data_2,
                )

                pi_abs = sampled_actions.abs().mean()

                pi_saturation = (
                    sampled_actions.abs() > 0.95
                ).float().mean()

            current_critic_loss_value = (
                current_td_loss.detach()
            )

            old_replay_loss_value = (
                old_replay_td_loss.detach()
            )

            current_actor_loss_value = actor_loss.detach()

            current_ent_coef_value = (
                current_ent_coef().detach()
            )

            current_ent_coef_loss_value = (
                ent_coef_loss.detach()
            )

            current_log_prob_value = log_prob.mean().detach()

            current_entropy_value = (
                -log_prob.mean().detach()
            )

            current_q_pi_value = min_q_pi.mean().detach()

            current_q_data_value = q_data.mean().detach()

            current_pi_abs_value = pi_abs.detach()

            current_pi_saturation_value = (
                pi_saturation.detach()
            )

            current_actor_grad_norm_value = torch.as_tensor(
                actor_grad_norm,
                dtype=torch.float32,
                device=device,
            ).detach()

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        if global_step % eval_freq != 0:
            continue

        eval_env = make_env(goal=current_goal)

        def eval_policy(observation_dict):
            eval_state, eval_goal = split_fetch_obs(
                observation_dict
            )

            eval_state_t = torch.as_tensor(
                eval_state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            eval_goal_t = torch.as_tensor(
                eval_goal,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                eval_state_t = normalize_state(
                    eval_state_t
                )

                eval_goal_t = normalize_goal(
                    eval_goal_t
                )

                eval_action = actor.deterministic(
                    eval_state_t,
                    eval_goal_t,
                )

            return eval_action.squeeze(0).cpu().numpy()

        (
            mean_return,
            mean_length,
            success_rate,
            mean_final_distance,
        ) = evaluate_policy_with_success(
            eval_env,
            eval_policy,
            episodes=8,
            seed=seed + 100_000 + global_step,
        )

        eval_returns.append(
            (global_step, mean_return)
        )

        eval_success_rates.append(
            (global_step, success_rate)
        )

        eval_final_distances.append(
            (global_step, mean_final_distance)
        )

        if normalize_state_inputs:
            state_std = torch.sqrt(
                state_rms.var
            ).detach().cpu().numpy()

            state_norm_string = (
                f"StateNormCount="
                f"{state_rms.count.item():.0f} | "
                f"StateStdMin={state_std.min():.6f} | "
                f"StateStdMax={state_std.max():.6f} | "
            )

        else:
            state_norm_string = ""

        print(
            "[SAC-Standard-Norm] "
            f"step={global_step:7d} | "
            f"return={mean_return:.3f} | "
            f"len={mean_length:.1f} | "
            f"Success={success_rate:.3f} | "
            f"FinalDist={mean_final_distance:.4f} | "
            f"CriticTD={current_critic_loss_value.item():.5f} | "
            f"OldReplay={old_replay_loss_value.item():.5f} | "
            f"ActorLoss={current_actor_loss_value.item():.5f} | "
            f"Alpha={current_ent_coef_value.item():.5f} | "
            f"AlphaLoss={current_ent_coef_loss_value.item():.5f} | "
            f"LogProb={current_log_prob_value.item():.5f} | "
            f"Entropy={current_entropy_value.item():.5f} | "
            f"Qpi={current_q_pi_value.item():.5f} | "
            f"Qdata={current_q_data_value.item():.5f} | "
            f"MeanAbsPi={current_pi_abs_value.item():.5f} | "
            f"PiSat={current_pi_saturation_value.item():.3f} | "
            f"ActorGrad={current_actor_grad_norm_value.item():.5f} | "
            f"{state_norm_string}"
            f"ReplayTasks={len(replay_batches)}"
        )

        eval_env.close()

        if early_stop_success_rate is not None:
            criterion_met = (
                success_rate >= early_stop_success_rate
            )
        else:
            criterion_met = (
                mean_return >= early_stop_reward
            )

        if criterion_met:
            success_streak += 1
        else:
            success_streak = 0

        if (
            enable_early_stop
            and success_streak >= early_stop_patience
        ):
            min_steps = global_step

            min_time = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Early stopping at step={global_step}, "
                f"return={mean_return:.3f}, "
                f"success={success_rate:.3f}, "
                f"streak={success_streak}"
            )

            break

    # =========================================================
    # Final outputs
    # =========================================================

    if min_steps is None:
        min_steps = global_step

        min_time = (
            time.perf_counter()
            - start_time
        )

    env.close()

    return (
        actor,
        critic,
        critic_target,
        eval_returns,
        eval_success_rates,
        eval_final_distances,
        min_steps,
        min_time,
        buffer,
    )


def sac_train_tbtrl_v1(
    seed: int = 42,
    actor: nn.Module = None,
    critic: nn.Module = None,
    critic_target: nn.Module = None,
    env=None,
    buffer_capacity: int = None,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    lr_ent_coef: float = 3e-4,
    obs_dim: int = None,
    action_dim: int = None,
    device: torch.device = None,
    total_steps: int = 300_000,
    warmup_steps: int = 25_000,
    batch_size: int = 256,
    eval_freq: int = 5_000,
    gamma: float = 0.99,
    tau: float = 0.005,
    train_freq: int = 1,
    gradient_steps: int = 1,
    goal: np.ndarray = None,
    task_id: int = None,
    make_env=None,
    env_id: str = None,

    # {task_id: retained old-task replay buffer}
    replay_task_buffers: Optional[Dict[int, Any]] = None,

    # {task_id: fixed goal np.ndarray}
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
    enable_early_stop: bool = True,

    # =========================================================
    # TBTRL critic-only regularisation
    # =========================================================

    sigreg_coef: float = 0.0,
    sketch_dim: int = 64,

    goal_separation_coef: float = 0.0,
    goal_separation_target_cosine: float = 0.85,

    phi_raw_norm_coef: float = 0.0,
    psi_raw_norm_coef: float = 0.0,

    phi_norm_target: float | None = None,
    psi_norm_target: float | None = None,

    # =========================================================
    # Performance / resume controls
    # =========================================================

    # Apply expensive TBTRL penalties every N critic updates.
    # E.g. frequency=4 computes them every fourth update.
    # Their contribution is multiplied by this value when run,
    # retaining approximately the same average penalty strength.
    tbtrl_reg_freq: int = 4,

    # Print profiling data every N updates; None/0 disables it.
    profile_freq: int | None = None,

    # Important for crash recovery:
    # False preserves the checkpoint's existing target critic.
    reset_target_from_critic: bool = True,
):
    """
    Optimised factorised SAC-TBTRL trainer.

    Performance changes relative to the original version:
    - Old-task samples are concatenated and processed in one TD pass.
    - phi1/phi2 and psi1/psi2 embeddings are computed once per
      regularisation update and reused across relevant penalties.
    - Expensive TBTRL regularisers run every tbtrl_reg_freq updates.
    - Q(data) logging forward passes occur only at evaluation frequency.
    - reset_target_from_critic is honoured for checkpoint recovery.
    """

    # =========================================================
    # Validation
    # =========================================================

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

    if tbtrl_reg_freq < 1:
        raise ValueError("tbtrl_reg_freq must be >= 1.")

    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1].")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    if obs_norm_clip <= 0.0:
        raise ValueError("obs_norm_clip must be positive.")

    if isinstance(ent_coef, str) and ent_coef != "auto":
        raise ValueError(
            "ent_coef must be a positive float or 'auto'."
        )

    if isinstance(ent_coef, (int, float)) and ent_coef <= 0.0:
        raise ValueError(
            "Fixed ent_coef must be positive."
        )

    if sigreg_coef < 0.0:
        raise ValueError("sigreg_coef must be >= 0.")

    if goal_separation_coef < 0.0:
        raise ValueError(
            "goal_separation_coef must be >= 0."
        )

    if phi_raw_norm_coef < 0.0:
        raise ValueError(
            "phi_raw_norm_coef must be >= 0."
        )

    if psi_raw_norm_coef < 0.0:
        raise ValueError(
            "psi_raw_norm_coef must be >= 0."
        )

    if not -1.0 <= goal_separation_target_cosine <= 1.0:
        raise ValueError(
            "goal_separation_target_cosine must be within [-1, 1]."
        )

    # =========================================================
    # Device, dimensions, task state
    # =========================================================

    if device is None:
        device = next(critic.parameters()).device

    device = torch.device(device)

    if obs_dim is None:
        obs_dim = (
            int(env.observation_space["observation"].shape[0])
            + int(env.observation_space["achieved_goal"].shape[0])
        )

    if action_dim is None:
        action_dim = int(env.action_space.shape[0])

    goal_dim = int(
        env.observation_space["desired_goal"].shape[0]
    )

    set_seed(seed)

    if replay_task_buffers is None:
        replay_task_buffers = {}

    if task_goals is None:
        task_goals = {}

    training_goal = np.asarray(
        goal,
        dtype=np.float32,
    ).copy()

    if training_goal.shape != (goal_dim,):
        raise ValueError(
            f"goal has shape {training_goal.shape}; "
            f"expected ({goal_dim},)."
        )

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

    # =========================================================
    # Factorised critic API validation
    # =========================================================

    tbtrl_active = any(
        coefficient > 0.0
        for coefficient in (
            sigreg_coef,
            goal_separation_coef,
            phi_raw_norm_coef,
            psi_raw_norm_coef,
        )
    )

    if tbtrl_active:
        required_methods = [
            "phi1_forward",
            "psi1_forward",
            "phi2_forward",
            "psi2_forward",
        ]

        missing_methods = [
            method_name
            for method_name in required_methods
            if not hasattr(critic, method_name)
        ]

        if missing_methods:
            raise AttributeError(
                "TBTRL regularisers require critic methods: "
                f"{missing_methods}."
            )

    # =========================================================
    # Action bounds
    # =========================================================

    action_low = torch.as_tensor(
        env.action_space.low,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    action_high = torch.as_tensor(
        env.action_space.high,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    if action_low.shape[-1] != action_dim:
        raise RuntimeError(
            "Action-space bounds do not match action_dim."
        )

    if not (
        torch.allclose(
            action_low,
            -torch.ones_like(action_low),
        )
        and torch.allclose(
            action_high,
            torch.ones_like(action_high),
        )
    ):
        raise ValueError(
            "This SAC actor assumes action bounds [-1, 1]."
        )

    # =========================================================
    # Optimisers and replay buffer
    # =========================================================

    opt_actor = optim.Adam(
        actor.parameters(),
        lr=lr_actor,
    )

    opt_critic = optim.Adam(
        critic.parameters(),
        lr=lr_critic,
    )

    buffer = TrajectoryReplayBufferContinuous(
        buffer_capacity,
        obs_dim,
        action_dim,
        device=device,
    )

    if target_entropy is None:
        target_entropy = -float(action_dim)

    if ent_coef == "auto":
        log_ent_coef = torch.zeros(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        opt_ent_coef = optim.Adam(
            [log_ent_coef],
            lr=lr_ent_coef,
        )

        fixed_ent_coef = None

    else:
        fixed_ent_coef = torch.as_tensor(
            float(ent_coef),
            dtype=torch.float32,
            device=device,
        )

        log_ent_coef = None
        opt_ent_coef = None

    # =========================================================
    # Input normalisers
    # =========================================================

    state_rms = RunningMeanStd(
        shape=(obs_dim,),
        device=device,
    )

    goal_rms = RunningMeanStd(
        shape=(goal_dim,),
        device=device,
    )

    # =========================================================
    # Generic helpers
    # =========================================================

    def split_fetch_obs(obs_dict):
        observation = np.asarray(
            obs_dict["observation"],
            dtype=np.float32,
        )

        achieved_goal = np.asarray(
            obs_dict["achieved_goal"],
            dtype=np.float32,
        )

        desired_goal = np.asarray(
            obs_dict["desired_goal"],
            dtype=np.float32,
        )

        state = np.concatenate(
            [observation, achieved_goal],
            axis=-1,
        ).astype(np.float32)

        return state, desired_goal

    def goal_batch_for(
        goal_value,
        requested_batch_size: int,
        target_device: torch.device,
    ) -> torch.Tensor:
        if isinstance(goal_value, torch.Tensor):
            goal_batch = goal_value.to(
                device=target_device,
                dtype=torch.float32,
            )
        else:
            goal_batch = torch.as_tensor(
                np.asarray(goal_value, dtype=np.float32),
                dtype=torch.float32,
                device=target_device,
            )

        if goal_batch.ndim == 1:
            goal_batch = goal_batch.unsqueeze(0)

        if goal_batch.ndim != 2:
            raise ValueError(
                "Goal must have shape [goal_dim] or [B, goal_dim]. "
                f"Got {tuple(goal_batch.shape)}."
            )

        if goal_batch.shape[0] == 1:
            goal_batch = goal_batch.expand(
                requested_batch_size,
                -1,
            )

        elif goal_batch.shape[0] != requested_batch_size:
            raise ValueError(
                "Goal batch size does not match transition batch size."
            )

        return goal_batch

    def normalize_state(
        state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not normalize_state_inputs:
            return state_tensor

        return state_rms.normalize(
            state_tensor,
            clip=obs_norm_clip,
        )

    def normalize_goal(
        goal_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not normalize_goal_inputs:
            return goal_tensor

        return goal_rms.normalize(
            goal_tensor,
            clip=obs_norm_clip,
        )

    def current_ent_coef() -> torch.Tensor:
        if log_ent_coef is not None:
            return log_ent_coef.exp()

        return fixed_ent_coef

    def polyak_update_critic() -> None:
        with torch.no_grad():
            for online_parameter, target_parameter in zip(
                critic.parameters(),
                critic_target.parameters(),
            ):
                target_parameter.mul_(1.0 - tau).add_(
                    online_parameter,
                    alpha=tau,
                )

    def seen_goals_tensor() -> torch.Tensor:
        seen_task_ids = sorted(task_goals.keys())

        return torch.as_tensor(
            np.asarray(
                [
                    task_goals[seen_task_id]
                    for seen_task_id in seen_task_ids
                ],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=device,
        )

    def resolve_norm_target(
        name: str,
        supplied_target: float | None,
    ) -> float:
        if supplied_target is not None:
            target = float(supplied_target)

        elif name == "phi":
            target = float(
                getattr(
                    critic,
                    "phi_max_norm",
                    10.1,
                )
            ) - 0.1

        elif name == "psi":
            target = float(
                getattr(
                    critic,
                    "psi_max_norm",
                    10.1,
                )
            ) - 0.1

        else:
            raise ValueError(
                f"Unknown embedding name: {name}."
            )

        if target <= 0.0:
            raise ValueError(
                f"{name}_norm_target must be positive, got {target}."
            )

        return target

    def soft_excess_norm_loss(
        embeddings: torch.Tensor,
        target_norm: float,
    ) -> torch.Tensor:
        return F.relu(
            embeddings.norm(p=2, dim=-1) - target_norm
        ).mean()

    def head_goal_separation_loss(
        psi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_goals = psi.shape[0]

        if n_goals < 2:
            return zero, zero

        normalized_psi = F.normalize(
            psi,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        cosine_matrix = normalized_psi @ normalized_psi.T

        off_diagonal_mask = ~torch.eye(
            n_goals,
            dtype=torch.bool,
            device=device,
        )

        off_diagonal_cosines = cosine_matrix[
            off_diagonal_mask
        ]

        separation_loss = F.relu(
            off_diagonal_cosines
            - goal_separation_target_cosine
        ).mean()

        return separation_loss, off_diagonal_cosines.max()

    def td_target(
        batch,
        normalized_goal_batch: torch.Tensor,
        ent_coef_tensor: torch.Tensor,
    ) -> torch.Tensor:
        rewards = batch.rewards
        terminated = batch.terminated

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)

        if terminated.ndim == 1:
            terminated = terminated.unsqueeze(-1)

        next_state = normalize_state(batch.next_obs)

        with torch.no_grad():
            next_action, next_log_prob, _ = actor.sample(
                next_state,
                normalized_goal_batch,
            )

            next_q1 = critic_target.q1_forward(
                next_state,
                next_action,
                normalized_goal_batch,
            )

            next_q2 = critic_target.q2_forward(
                next_state,
                next_action,
                normalized_goal_batch,
            )

            next_q = torch.minimum(next_q1, next_q2)

            next_soft_value = (
                next_q - ent_coef_tensor * next_log_prob
            )

            if bootstrap_on_truncation:
                bootstrap_mask = 1.0 - terminated.float()

            else:
                truncated = batch.truncated

                if truncated.ndim == 1:
                    truncated = truncated.unsqueeze(-1)

                done = torch.logical_or(
                    terminated.bool(),
                    truncated.bool(),
                ).float()

                bootstrap_mask = 1.0 - done

            return (
                rewards
                + gamma * bootstrap_mask * next_soft_value
            )

    def critic_td_loss(
        batch,
        raw_goal_batch: torch.Tensor,
        ent_coef_tensor: torch.Tensor,
    ) -> torch.Tensor:
        normalized_goal_batch = normalize_goal(
            raw_goal_batch
        )

        target = td_target(
            batch,
            normalized_goal_batch,
            ent_coef_tensor,
        )

        state = normalize_state(batch.obs)

        q1 = critic.q1_forward(
            state,
            batch.actions,
            normalized_goal_batch,
        )

        q2 = critic.q2_forward(
            state,
            batch.actions,
            normalized_goal_batch,
        )

        if q1.shape != target.shape:
            raise RuntimeError(
                f"Q1 shape {q1.shape} does not match "
                f"target shape {target.shape}."
            )

        if q2.shape != target.shape:
            raise RuntimeError(
                f"Q2 shape {q2.shape} does not match "
                f"target shape {target.shape}."
            )

        return 0.5 * (
            F.mse_loss(q1, target)
            + F.mse_loss(q2, target)
        )

    def concatenate_batches(batches):
        """
        Concatenate sampled old-task batches into a single ReplayBatch.

        Matches your ReplayBatch signature:
        obs, actions, rewards, next_obs, terminated, truncated,
        episode_id, timestep, indices
        """
        if len(batches) == 0:
            return None

        if len(batches) == 1:
            return batches[0]

        return type(batches[0])(
            obs=torch.cat(
                [batch.obs for batch in batches],
                dim=0,
            ),
            actions=torch.cat(
                [batch.actions for batch in batches],
                dim=0,
            ),
            rewards=torch.cat(
                [batch.rewards for batch in batches],
                dim=0,
            ),
            next_obs=torch.cat(
                [batch.next_obs for batch in batches],
                dim=0,
            ),
            terminated=torch.cat(
                [batch.terminated for batch in batches],
                dim=0,
            ),
            truncated=torch.cat(
                [batch.truncated for batch in batches],
                dim=0,
            ),
            episode_id=torch.cat(
                [batch.episode_id for batch in batches],
                dim=0,
            ),
            timestep=torch.cat(
                [batch.timestep for batch in batches],
                dim=0,
            ),
            indices=torch.cat(
                [batch.indices for batch in batches],
                dim=0,
            ),
        )

    def device_synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elif device.type == "mps":
            torch.mps.synchronize()

    # =========================================================
    # Training state
    # =========================================================

    obs_dict, _ = env.reset()

    global_step = 0
    update_count = 0
    success_streak = 0

    start_time = time.perf_counter()

    eval_returns = []
    eval_success_rates = []
    eval_final_distances = []

    min_steps = None
    min_time = None

    zero = torch.zeros(
        (),
        dtype=torch.float32,
        device=device,
    )

    current_critic_loss_value = zero
    old_replay_loss_value = zero
    current_actor_loss_value = zero
    current_ent_coef_value = torch.ones(
        (),
        dtype=torch.float32,
        device=device,
    )
    current_ent_coef_loss_value = zero
    current_log_prob_value = zero
    current_entropy_value = zero
    current_q_pi_value = zero
    current_q_data_value = zero
    current_pi_abs_value = zero
    current_pi_saturation_value = zero
    current_actor_grad_norm_value = zero

    current_sigreg_value = zero
    current_goal_separation_value = zero
    current_phi_raw_norm_value = zero
    current_psi_raw_norm_value = zero

    current_phi1_norm_value = zero
    current_phi2_norm_value = zero
    current_psi1_norm_value = zero
    current_psi2_norm_value = zero

    current_psi1_max_cosine_value = zero
    current_psi2_max_cosine_value = zero

    replay_tasks_used = 0

    # =========================================================
    # Main training loop
    # =========================================================

    while global_step < total_steps:
        # -----------------------------------------------------
        # Data collection
        # -----------------------------------------------------

        state, current_goal = split_fetch_obs(obs_dict)

        if global_step < warmup_steps:
            action = env.action_space.sample().astype(
                np.float32
            )

        else:
            state_t = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            goal_t = torch.as_tensor(
                current_goal,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                action_t, _, _ = actor.sample(
                    normalize_state(state_t),
                    normalize_goal(goal_t),
                )

            action = (
                action_t.squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        (
            next_obs_dict,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action)

        next_state, next_goal = split_fetch_obs(
            next_obs_dict
        )

        # Only create/update RMS tensors when normalisation is enabled.
        if normalize_state_inputs:
            with torch.no_grad():
                state_rms.update(
                    torch.as_tensor(
                        state,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                )

                state_rms.update(
                    torch.as_tensor(
                        next_state,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                )

        if normalize_goal_inputs:
            with torch.no_grad():
                goal_rms.update(
                    torch.as_tensor(
                        current_goal,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                )

                goal_rms.update(
                    torch.as_tensor(
                        next_goal,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                )

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
            obs_dict, _ = env.reset()

        if len(buffer) < warmup_steps:
            continue

        if global_step % train_freq != 0:
            continue

        # -----------------------------------------------------
        # Gradient updates
        # -----------------------------------------------------

        for _ in range(gradient_steps):
            if (
                profile_freq is not None
                and profile_freq > 0
                and update_count % profile_freq == 0
            ):
                device_synchronize()
                profile_start = time.perf_counter()

            current_batch = buffer.sample(batch_size)

            current_goal_tensor = goal_batch_for(
                training_goal,
                current_batch.obs.shape[0],
                device,
            )

            # -------------------------------------------------
            # Build one concatenated old-task replay batch
            # -------------------------------------------------

            eligible_old_task_ids = [
                old_task_id
                for old_task_id, old_buffer in (
                    replay_task_buffers.items()
                )
                if (
                    old_buffer is not None
                    and len(old_buffer) > 0
                )
            ]

            for old_task_id in eligible_old_task_ids:
                if old_task_id not in task_goals:
                    raise KeyError(
                        f"Missing goal for replay task "
                        f"{old_task_id}."
                    )

            selected_old_task_ids = []

            if (
                replay_ratio > 0.0
                and len(eligible_old_task_ids) > 0
            ):
                if replay_tasks_per_batch is None:
                    n_old_tasks = len(eligible_old_task_ids)
                else:
                    n_old_tasks = min(
                        int(replay_tasks_per_batch),
                        len(eligible_old_task_ids),
                    )

                cycle_index = (
                    update_count
                    % len(eligible_old_task_ids)
                )

                ordered_old_task_ids = (
                    eligible_old_task_ids[cycle_index:]
                    + eligible_old_task_ids[:cycle_index]
                )

                selected_old_task_ids = (
                    ordered_old_task_ids[:n_old_tasks]
                )

            old_batches = []
            old_goal_batches = []

            if len(selected_old_task_ids) > 0:
                replay_batch_size = max(
                    1,
                    int(
                        batch_size
                        * replay_ratio
                        / len(selected_old_task_ids)
                    ),
                )

                for old_task_id in selected_old_task_ids:
                    old_buffer = replay_task_buffers[
                        old_task_id
                    ]

                    if len(old_buffer) < replay_batch_size:
                        continue

                    old_batch = old_buffer.sample(
                        replay_batch_size
                    )

                    old_goal_tensor = goal_batch_for(
                        task_goals[old_task_id],
                        old_batch.obs.shape[0],
                        device,
                    )

                    old_batches.append(old_batch)
                    old_goal_batches.append(old_goal_tensor)

            replay_tasks_used = len(old_batches)

            concatenated_old_batch = concatenate_batches(
                old_batches
            )

            if len(old_goal_batches) > 0:
                concatenated_old_goals = torch.cat(
                    old_goal_batches,
                    dim=0,
                )
            else:
                concatenated_old_goals = None

            # -------------------------------------------------
            # Critic TD losses
            # -------------------------------------------------

            ent_coef_tensor = current_ent_coef().detach()

            current_td_loss = critic_td_loss(
                current_batch,
                current_goal_tensor,
                ent_coef_tensor,
            )

            if concatenated_old_batch is not None:
                old_replay_td_loss = critic_td_loss(
                    concatenated_old_batch,
                    concatenated_old_goals,
                    ent_coef_tensor,
                )
            else:
                old_replay_td_loss = zero

            # -------------------------------------------------
            # TBTRL regularisers
            # -------------------------------------------------

            run_tbtrl_regularisers = (
                tbtrl_active
                and update_count % tbtrl_reg_freq == 0
            )

            if run_tbtrl_regularisers:
                state_for_reg = normalize_state(
                    current_batch.obs
                )

                phi1 = critic.phi1_forward(
                    state_for_reg,
                    current_batch.actions,
                )

                phi2 = critic.phi2_forward(
                    state_for_reg,
                    current_batch.actions,
                )

                if sigreg_coef > 0.0:
                    current_sigreg = 0.5 * (
                        sigreg_loss(
                            phi1,
                            sketch_dim=sketch_dim,
                        )
                        + sigreg_loss(
                            phi2,
                            sketch_dim=sketch_dim,
                        )
                    )
                else:
                    current_sigreg = zero

                phi_target = resolve_norm_target(
                    "phi",
                    phi_norm_target,
                )

                if phi_raw_norm_coef > 0.0:
                    phi_raw_norm_loss = 0.5 * (
                        soft_excess_norm_loss(
                            phi1,
                            phi_target,
                        )
                        + soft_excess_norm_loss(
                            phi2,
                            phi_target,
                        )
                    )
                else:
                    phi_raw_norm_loss = zero

                psi_required = (
                    goal_separation_coef > 0.0
                    or psi_raw_norm_coef > 0.0
                )

                if psi_required:
                    all_seen_goals = normalize_goal(
                        seen_goals_tensor()
                    )

                    psi1_all = critic.psi1_forward(
                        all_seen_goals
                    )

                    psi2_all = critic.psi2_forward(
                        all_seen_goals
                    )
                else:
                    psi1_all = None
                    psi2_all = None

                if (
                    goal_separation_coef > 0.0
                    and len(task_goals) > 1
                ):
                    separation_1, psi1_max_cosine = (
                        head_goal_separation_loss(psi1_all)
                    )

                    separation_2, psi2_max_cosine = (
                        head_goal_separation_loss(psi2_all)
                    )

                    current_goal_separation = 0.5 * (
                        separation_1 + separation_2
                    )

                else:
                    current_goal_separation = zero
                    psi1_max_cosine = zero
                    psi2_max_cosine = zero

                psi_target = resolve_norm_target(
                    "psi",
                    psi_norm_target,
                )

                if psi_raw_norm_coef > 0.0:
                    psi_raw_norm_loss = 0.5 * (
                        soft_excess_norm_loss(
                            psi1_all,
                            psi_target,
                        )
                        + soft_excess_norm_loss(
                            psi2_all,
                            psi_target,
                        )
                    )
                else:
                    psi_raw_norm_loss = zero

                current_phi1_norm_value = (
                    phi1.norm(p=2, dim=-1).mean().detach()
                )

                current_phi2_norm_value = (
                    phi2.norm(p=2, dim=-1).mean().detach()
                )

                if psi1_all is not None:
                    current_psi1_norm_value = (
                        psi1_all.norm(
                            p=2,
                            dim=-1,
                        ).mean().detach()
                    )

                    current_psi2_norm_value = (
                        psi2_all.norm(
                            p=2,
                            dim=-1,
                        ).mean().detach()
                    )
                else:
                    current_psi1_norm_value = zero
                    current_psi2_norm_value = zero

                # Because penalties are computed intermittently,
                # scale their active-update contribution to retain
                # approximately the original average strength.
                regulariser_scale = float(tbtrl_reg_freq)

            else:
                current_sigreg = zero
                current_goal_separation = zero
                phi_raw_norm_loss = zero
                psi_raw_norm_loss = zero
                psi1_max_cosine = zero
                psi2_max_cosine = zero
                regulariser_scale = 1.0

            critic_total_loss = (
                current_td_loss
                + replay_loss_coef * old_replay_td_loss
                + regulariser_scale
                * sigreg_coef
                * current_sigreg
                + regulariser_scale
                * goal_separation_coef
                * current_goal_separation
                + regulariser_scale
                * phi_raw_norm_coef
                * phi_raw_norm_loss
                + regulariser_scale
                * psi_raw_norm_coef
                * psi_raw_norm_loss
            )

            opt_critic.zero_grad(set_to_none=True)
            critic_total_loss.backward()

            nn.utils.clip_grad_norm_(
                critic.parameters(),
                max_norm=10.0,
            )

            opt_critic.step()

            # -------------------------------------------------
            # Actor update
            # -------------------------------------------------

            state_batch = normalize_state(
                current_batch.obs
            )

            goal_batch = normalize_goal(
                current_goal_tensor
            )

            # Freeze critic parameters while differentiating
            # Q(s, pi(s)) with respect to actor parameters.
            # This avoids allocating useless critic gradients.
            for parameter in critic.parameters():
                parameter.requires_grad_(False)

            sampled_actions, log_prob, _ = actor.sample(
                state_batch,
                goal_batch,
            )

            q1_pi = critic.q1_forward(
                state_batch,
                sampled_actions,
                goal_batch,
            )

            q2_pi = critic.q2_forward(
                state_batch,
                sampled_actions,
                goal_batch,
            )

            min_q_pi = torch.minimum(q1_pi, q2_pi)

            actor_loss = (
                current_ent_coef().detach() * log_prob
                - min_q_pi
            ).mean()

            opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()

            actor_grad_norm = nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=10.0,
            )

            opt_actor.step()

            for parameter in critic.parameters():
                parameter.requires_grad_(True)

            # -------------------------------------------------
            # Entropy coefficient
            # -------------------------------------------------

            if log_ent_coef is not None:
                ent_coef_loss = -(
                    log_ent_coef
                    * (
                        log_prob.detach()
                        + target_entropy
                    )
                ).mean()

                opt_ent_coef.zero_grad(set_to_none=True)
                ent_coef_loss.backward()
                opt_ent_coef.step()

            else:
                ent_coef_loss = zero

            # -------------------------------------------------
            # Target critic and cheap scalar logging state
            # -------------------------------------------------

            polyak_update_critic()
            update_count += 1

            current_critic_loss_value = (
                current_td_loss.detach()
            )

            old_replay_loss_value = (
                old_replay_td_loss.detach()
            )

            current_actor_loss_value = actor_loss.detach()

            current_ent_coef_value = (
                current_ent_coef().detach()
            )

            current_ent_coef_loss_value = (
                ent_coef_loss.detach()
            )

            current_log_prob_value = (
                log_prob.mean().detach()
            )

            current_entropy_value = (
                -log_prob.mean().detach()
            )

            current_q_pi_value = (
                min_q_pi.mean().detach()
            )

            current_pi_abs_value = (
                sampled_actions.abs().mean().detach()
            )

            current_pi_saturation_value = (
                (
                    sampled_actions.abs() > 0.95
                ).float().mean().detach()
            )

            current_actor_grad_norm_value = (
                torch.as_tensor(
                    actor_grad_norm,
                    dtype=torch.float32,
                    device=device,
                ).detach()
            )

            current_sigreg_value = current_sigreg.detach()

            current_goal_separation_value = (
                current_goal_separation.detach()
            )

            current_phi_raw_norm_value = (
                phi_raw_norm_loss.detach()
            )

            current_psi_raw_norm_value = (
                psi_raw_norm_loss.detach()
            )

            current_psi1_max_cosine_value = (
                psi1_max_cosine.detach()
            )

            current_psi2_max_cosine_value = (
                psi2_max_cosine.detach()
            )

            if (
                profile_freq is not None
                and profile_freq > 0
                and update_count % profile_freq == 0
            ):
                device_synchronize()

                update_seconds = (
                    time.perf_counter()
                    - profile_start
                )

                print(
                    "[SAC-TBTRL profile] "
                    f"update={update_count} | "
                    f"seconds={update_seconds:.4f} | "
                    f"updates_per_sec={1.0 / update_seconds:.2f}"
                )

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        if global_step % eval_freq != 0:
            continue

        eval_env = make_env(
            goal=training_goal,
            env_id=env_id,
        )

        def eval_policy(observation_dict):
            eval_state, eval_goal = split_fetch_obs(
                observation_dict
            )

            eval_state_t = torch.as_tensor(
                eval_state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            eval_goal_t = torch.as_tensor(
                eval_goal,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                eval_action = actor.deterministic(
                    normalize_state(eval_state_t),
                    normalize_goal(eval_goal_t),
                )

            return (
                eval_action.squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        (
            mean_return,
            mean_length,
            success_rate,
            mean_final_distance,
        ) = evaluate_policy_with_success(
            eval_env,
            eval_policy,
            episodes=8,
            seed=seed + 100_000 + global_step,
        )

        eval_returns.append(
            (global_step, mean_return)
        )

        eval_success_rates.append(
            (global_step, success_rate)
        )

        eval_final_distances.append(
            (global_step, mean_final_distance)
        )

        # These quantities are useful, but were previously
        # computed after every update. Compute them only here.
        diagnostic_batch = buffer.sample(batch_size)

        diagnostic_goal_batch = goal_batch_for(
            training_goal,
            diagnostic_batch.obs.shape[0],
            device,
        )

        with torch.no_grad():
            diagnostic_state_batch = normalize_state(
                diagnostic_batch.obs
            )

            diagnostic_goal_batch = normalize_goal(
                diagnostic_goal_batch
            )

            q_data_1 = critic.q1_forward(
                diagnostic_state_batch,
                diagnostic_batch.actions,
                diagnostic_goal_batch,
            )

            q_data_2 = critic.q2_forward(
                diagnostic_state_batch,
                diagnostic_batch.actions,
                diagnostic_goal_batch,
            )

            current_q_data_value = torch.minimum(
                q_data_1,
                q_data_2,
            ).mean()

        if normalize_state_inputs:
            state_std = torch.sqrt(
                state_rms.var
            ).detach().cpu().numpy()

            state_norm_string = (
                f"StateNormCount={state_rms.count.item():.0f} | "
                f"StateStdMin={state_std.min():.6f} | "
                f"StateStdMax={state_std.max():.6f} | "
            )

        else:
            state_norm_string = ""

        print(
            "[SAC-TBTRL] "
            f"step={global_step:7d} | "
            f"return={mean_return:.3f} | "
            f"len={mean_length:.1f} | "
            f"Success={success_rate:.3f} | "
            f"FinalDist={mean_final_distance:.4f} | "
            f"CriticTD={current_critic_loss_value.item():.5f} | "
            f"OldReplay={old_replay_loss_value.item():.5f} | "
            f"SIGReg={current_sigreg_value.item():.6f} | "
            f"GoalSep={current_goal_separation_value.item():.6f} | "
            f"PhiNormReg={current_phi_raw_norm_value.item():.6f} | "
            f"PsiNormReg={current_psi_raw_norm_value.item():.6f} | "
            f"Phi1Norm={current_phi1_norm_value.item():.3f} | "
            f"Phi2Norm={current_phi2_norm_value.item():.3f} | "
            f"Psi1Norm={current_psi1_norm_value.item():.3f} | "
            f"Psi2Norm={current_psi2_norm_value.item():.3f} | "
            f"PsiMaxCos=("
            f"{current_psi1_max_cosine_value.item():.3f},"
            f"{current_psi2_max_cosine_value.item():.3f}"
            f") | "
            f"ActorLoss={current_actor_loss_value.item():.5f} | "
            f"Alpha={current_ent_coef_value.item():.5f} | "
            f"AlphaLoss={current_ent_coef_loss_value.item():.5f} | "
            f"LogProb={current_log_prob_value.item():.5f} | "
            f"Entropy={current_entropy_value.item():.5f} | "
            f"Qpi={current_q_pi_value.item():.5f} | "
            f"Qdata={current_q_data_value.item():.5f} | "
            f"MeanAbsPi={current_pi_abs_value.item():.5f} | "
            f"PiSat={current_pi_saturation_value.item():.3f} | "
            f"ActorGrad={current_actor_grad_norm_value.item():.5f} | "
            f"{state_norm_string}"
            f"ReplayTasks={replay_tasks_used}"
        )

        eval_env.close()

        if early_stop_success_rate is not None:
            criterion_met = (
                success_rate >= early_stop_success_rate
            )
        else:
            criterion_met = (
                mean_return >= early_stop_reward
            )

        if criterion_met:
            success_streak += 1
        else:
            success_streak = 0

        if (
            enable_early_stop
            and success_streak >= early_stop_patience
        ):
            min_steps = global_step

            min_time = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Early stopping at step={global_step}, "
                f"return={mean_return:.3f}, "
                f"success={success_rate:.3f}, "
                f"streak={success_streak}"
            )

            break

    # =========================================================
    # Final outputs
    # =========================================================

    if min_steps is None:
        min_steps = global_step
        min_time = time.perf_counter() - start_time

    env.close()

    return (
        actor,
        critic,
        critic_target,
        eval_returns,
        eval_success_rates,
        eval_final_distances,
        min_steps,
        min_time,
        buffer,
    )