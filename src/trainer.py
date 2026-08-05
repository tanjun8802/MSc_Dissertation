import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import time
import random

from utils import (
    TrajectoryReplayBufferDiscrete,
    TrajectoryReplayBuffer,
    evaluate_policy,
    set_seed,
    extract_fixed_probe_sa_embedding,
    extract_mean_sa_embedding,
    extract_sa_batch_for_isotropy,
    build_goal_batch,
    extract_fixed_probe_sa_embedding_td3,
    extract_mean_sa_embedding_td3,
    extract_sa_batch_for_isotropy_td3
)

from loss_functions import (
    repulsion_loss_to_memory,
    sigreg_loss,
    orthogonal_loss,
    ewc_regulariser_loss,
    weight_regulariser_loss,
    goal_memory_contrastive_loss
)


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
    reg_alpha=500,
    embedding_memory=None,
    reference_params=None,
    fisher_diag=None,
    sa_reg_prefix_filter="sa_encoder",
    sigreg=1,
    td_steps=1,
    make_env=None,
    epsilon_reduction=False,
    eps_start_transfer=0.2,
    eps_end_transfer=0.05,
    eps_decay_steps_transfer=20000,
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
    eval_returns_time = []
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
        if epsilon_reduction:
            frac = min(1.0, global_step / eps_decay_steps_transfer)
            eps = eps_start_transfer + frac * (eps_end_transfer - eps_start_transfer)
        else:
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

            if sigreg is not None:
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
                loss = td_loss + reg_alpha * ortho_loss + 0.1 * sigreg_loss_val + 10000 * ewc_loss
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
    reg_alpha=500.0,
    embedding_memory=None,
    memory_goals=None,
    reference_params=None,
    fisher_diag=None,
    sa_reg_prefix_filter="sa_encoder",
    sigreg=1,
    td_steps=1,
    make_env=None,
    replay_task_buffers=None,
    replay_ratio=0.25,
    replay_tasks_per_batch=2,
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

    if replay_task_buffers is None:
        replay_task_buffers = {}

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
    eval_returns_time = []
    start_time = time.perf_counter()
    min_steps = None
    min_time = None
    success_streak = 0
    ortho_loss = torch.tensor(0.0, device=device)
    sigreg_loss_val = torch.tensor(0.0, device=device)
    weight_loss = torch.tensor(0.0, device=device)
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
        trunc_t = batch.truncated

        B = obs_t.shape[0]
        goal_batch = build_goal_batch(batch_goal, B, device)

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
        td_loss_local = F.mse_loss(current_q, target)
        return td_loss_local, obs_t, act_t, goal_batch

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
            current_batch = buffer.sample(batch_size)
            td_loss, obs_t, act_t, goal_batch = compute_td_loss_from_batch(current_batch, goal)

            B = obs_t.shape[0]

            if sigreg is not None:
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

            replay_td_loss_val = torch.tensor(0.0, device=device)

            eligible_replay_goals = [
                g for g, rb in replay_task_buffers.items()
                if rb is not None and len(rb) >= batch_size
            ]

            if len(eligible_replay_goals) > 0 and replay_ratio > 0.0:
                n_replay_tasks = min(replay_tasks_per_batch, len(eligible_replay_goals))
                sampled_replay_goals = random.sample(eligible_replay_goals, n_replay_tasks)

                replay_losses = []
                replay_batch_size = max(1, int(batch_size * replay_ratio / n_replay_tasks))

                for replay_goal in sampled_replay_goals:
                    replay_buffer = replay_task_buffers[replay_goal]
                    replay_batch = replay_buffer.sample(replay_batch_size)
                    replay_loss_k, _, _, _ = compute_td_loss_from_batch(replay_batch, replay_goal)
                    replay_losses.append(replay_loss_k)

                if len(replay_losses) > 0:
                    replay_td_loss_val = torch.stack(replay_losses).mean()

            if fisher_diag is not None and reference_params is not None:
                ewc_loss = ewc_regulariser_loss(
                    q_network,
                    reference_params=reference_params,
                    fisher_diag=fisher_diag,
                    prefix_filter=sa_reg_prefix_filter,
                )
            else:
                ewc_loss = torch.tensor(0.0, device=device)

            loss = td_loss

            if regulariser == "goal_memory_contrastive":
                if embedding_memory is not None and len(embedding_memory) > 0:
                    if memory_goals is None:
                        raise ValueError("memory_goals must be provided for goal_memory_contrastive")

                    goal_tensor = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
                    current_goal_embedding = q_network.encode_goal(goal_tensor)

                    goal_reg_loss = goal_memory_contrastive_loss(
                        current_embedding=current_goal_embedding,
                        embedding_memory=embedding_memory,
                        current_goal=goal,
                        memory_goals=memory_goals,
                        similarity_mode="euclidean",
                        temperature=2.0,
                        pos_threshold=0.6,
                        neg_threshold=0.3,
                        margin=1.0,
                    )

                    loss = loss + reg_alpha * goal_reg_loss

            if len(replay_task_buffers) > 0:
                loss = loss + replay_loss_coef * replay_td_loss_val

            if regulariser is not None and regulariser == "repulsion":
                loss = loss + reg_alpha * ortho_loss + 0.1 * sigreg_loss_val + 10000 * ewc_loss

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
                f"| eval_return={mean_ret:.3f} | eval_len={mean_len:.1f} "
                f"| TD={td_loss.item():.3f} | ReplayTD={replay_td_loss_val.item():.3f} "
                f"| Ortho={ortho_loss.item():.3f} | SigReg={sigreg_loss_val.item():.3f} "
                f"| EWC={ewc_loss.item():.6f} | Loss={loss.item():.3f}"
                f"| goal_reg_loss={goal_reg_loss.item():.4f}"
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

    if obs_dim is None:
        obs_dim = env.observation_space.shape[0]
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

    for p in list(actor_tgt.parameters()) + list(q1_tgt.parameters()) + list(q2_tgt.parameters()):
        p.requires_grad_(False)

    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    critic_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)

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

    obs, _ = env.reset()

    while total_env_steps < warmup_steps:
        action = env.action_space.sample().astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay.add_transition(obs, action, reward, next_obs, terminated, truncated)

        obs = next_obs
        total_env_steps += 1

        if done:
            obs, _ = env.reset()

    while total_env_steps < total_steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_t = actor(obs_t, goal_t_single)

        action = action_t[0].cpu().numpy()
        action = action + np.random.normal(0.0, expl_noise, size=act_dim)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay.add_transition(obs, action, reward, next_obs, terminated, truncated)

        obs = next_obs
        total_env_steps += 1

        if done:
            obs, _ = env.reset()

        if len(replay) >= batch_size and total_env_steps % train_freq == 0:
            for _ in range(gradient_steps):
                batch = replay.sample(batch_size)

                obs_b = batch.obs.float()
                act_b = batch.actions.float()
                rew_b = batch.rewards.float()
                next_obs_b = batch.next_obs.float()
                term_b = batch.terminated.float()
                trunc_b = batch.truncated.float()

                if rew_b.ndim == 1:
                    rew_b = rew_b.unsqueeze(-1)
                if term_b.ndim == 1:
                    term_b = term_b.unsqueeze(-1)
                if trunc_b.ndim == 1:
                    trunc_b = trunc_b.unsqueeze(-1)

                goal_b = goal_t_single.expand(obs_b.shape[0], -1)

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
                torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10.0)
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

        if total_env_steps % eval_every == 0:
            eval_env = make_env(goal=goal)

            def eval_actor(o):
                o_t = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    a_t = actor(o_t, goal_t_single)
                return a_t[0].cpu().numpy().astype(np.float32)

            mean_ret, mean_len = evaluate_policy(eval_env, eval_actor, episodes=8)

            eval_returns.append((total_env_steps, mean_ret))

            print(
                f"[TD3-factorised] step={total_env_steps:7d} | "
                f"train_it={train_it:7d} | "
                f"eval_return={mean_ret:.3f} | eval_len={mean_len:.1f} | "
                f"critic_loss={critic_loss_val.item():.3f} | actor_loss={actor_loss_val.item():.3f}"
            )

            if mean_ret >= early_stop_reward:
                success_streak += 1
            else:
                success_streak = 0

            if enable_early_stop and success_streak >= early_stop_patience:
                min_steps = total_env_steps
                min_time = time.perf_counter() - start_time
                print(f"Good policy achieved at step {total_env_steps} with mean return {mean_ret:.3f}")
                print(
                    f"Early stopping triggered at step {total_env_steps} "
                    f"after {success_streak} consecutive evals with "
                    f"mean return >= {early_stop_reward:.2f}"
                )
                eval_env.close()
                break

            eval_env.close()

    min_steps = total_env_steps
    min_time = time.perf_counter() - start_time
    goal_tensor = torch.tensor(np.array(goal, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0)
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
        q1,                  # to align with q_network slot
        q1_tgt,              # to align with q_target_network slot
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