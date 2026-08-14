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
    extract_sa_batch_for_isotropy_td3,
    encode_task_for_similarity,
    compute_task_similarity,
    retrieve_similar_task_embeddings
)

from loss_functions import (
    repulsion_loss_to_memory,
    sigreg_loss,
    orthogonal_loss,
    ewc_regulariser_loss,
    weight_regulariser_loss,
    goal_memory_contrastive_loss,
    goal_prototype_anchor_loss
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
    if lr_sa is None:
        raise ValueError("lr_sa must be provided")
    if lr_goal is None:
        raise ValueError("lr_goal must be provided")
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
            {"params": q_network.sa_encoder.parameters(), "lr": lr_sa},
            {"params": q_network.goal_encoder.parameters(), "lr": lr_goal},
        ])
    else:
        opt = optim.Adam(params, lr=lr_goal)

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
        td_loss_local = F.smooth_l1_loss(current_q, target)
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
                        temperature=0.5,
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
                f"| TD={td_loss.item():.6f} | ReplayTD={replay_td_loss_val.item():.6f} "
                f"| Ortho={ortho_loss.item():.6f} | SigReg={sigreg_loss_val.item():.6f} "
                f"| EWC={ewc_loss.item():.6f} | Loss={loss.item():.6f}"
                f"| goal_reg_loss={goal_reg_loss.item():.6f}"
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