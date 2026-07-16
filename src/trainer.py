import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import time

from utils import (
    TrajectoryReplayBufferDiscrete,
    evaluate_policy,
    set_seed,
    extract_fixed_probe_sa_embedding,
    extract_mean_sa_embedding,
    extract_sa_batch_for_isotropy,
)

from loss_functions import (
    repulsion_loss_to_memory,
    sigreg_loss,
    orthogonal_loss,
    ewc_regulariser_loss,
    weight_regulariser_loss,
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
    total_steps=150000,
    warmup_steps=5000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=100000,
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
    ortho_loss = torch.tensor(0.0, device=device)
    sigreg_loss_val = torch.tensor(0.0, device=device)
    weight_loss = torch.tensor(0.0, device=device)
    ewc_loss = torch.tensor(0.0, device=device)
    loss = torch.tensor(0.0, device=device)

    ep_obs, ep_actions, ep_rewards, ep_next_obs, ep_terminated, ep_truncated = (
        [], [], [], [], [], []
    )

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

        ep_obs.append(obs.copy())
        ep_actions.append(action)
        ep_rewards.append(float(rew))
        ep_next_obs.append(next_obs.copy())
        ep_terminated.append(float(term))
        ep_truncated.append(float(trunc))

        obs = next_obs
        global_step += 1

        if done:
            episode = {
                "obs": ep_obs,
                "actions": ep_actions,
                "rewards": ep_rewards,
                "next_obs": ep_next_obs,
                "terminated": ep_terminated,
                "truncated": ep_truncated,
            }
            buffer.add_episode(episode)
            ep_obs, ep_actions, ep_rewards, ep_next_obs, ep_terminated, ep_truncated = (
                [], [], [], [], [], []
            )
            obs, _ = env.reset()

        if len(buffer) >= warmup_steps and global_step % train_freq == 0:
            batch = buffer.sample(batch_size)

            obs_t = batch.obs
            act_t = batch.actions.long()
            rew_t = batch.rewards
            next_obs_t = batch.next_obs
            term_t = batch.terminated
            trunc_t = batch.truncated
            done_t = torch.clamp(term_t + trunc_t, 0.0, 1.0)

            goal_batch = goal_t_single.expand(obs_t.shape[0], -1)
            B = obs_t.shape[0]

            with torch.no_grad():
                next_q_vals = q_target_network.q_val_for_argmax_action(next_obs_t, goal_batch)
                next_q = next_q_vals.max(dim=-1, keepdim=True).values
                gamma_final = gamma ** td_steps
                target = rew_t + gamma_final * (1.0 - done_t) * next_q

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

            if reference_params is not None:
                weight_loss = weight_regulariser_loss(
                    q_network,
                    reference_params=reference_params,
                    prefix_filter=sa_reg_prefix_filter,
                )

            if fisher_diag is not None:
                ewc_loss = ewc_regulariser_loss(
                    q_network,
                    reference_params=reference_params,
                    fisher_diag=fisher_diag,
                    prefix_filter=sa_reg_prefix_filter,
                )

            if regulariser is not None and regulariser == "repulsion":
                loss = td_loss + reg_alpha * ortho_loss + 0.1 * sigreg_loss_val + 1000 * weight_loss + 100000 * ewc_loss
            else:
                loss = td_loss + 0.1 * sigreg_loss_val + 10 * ewc_loss

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
            eval_time = time.perf_counter() - start_time
            eval_returns_time.append((eval_time, mean_ret))
            eval_returns.append((global_step, mean_ret))
            print(
                f"[DQN-factorised] step={global_step:7d} | eps={eps:.3f} "
                f"| eval_return={mean_ret:.3f} | eval_len={mean_len:.1f}"
                f"| Ortho loss={ortho_loss.item():.3f} | SigReg loss={sigreg_loss_val.item():.3f} | Loss={loss.item():.3f}"
                f"| Weight loss={weight_loss.item():.10f} | EWC loss={ewc_loss.item():.10f}"
            )
            if mean_ret >= 0.99 and min_steps is None:
                min_steps = global_step
                min_time = eval_time
                print(f"Good policy achieved at step {global_step} with mean return {mean_ret:.3f}")
            eval_env.close()

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
        eval_returns_time,
        min_steps,
        min_time,
        task_embedding,
        sa_embedding_mean,
        sa_embedding_fixed,
        sa_batch_final,
        buffer,
    )



