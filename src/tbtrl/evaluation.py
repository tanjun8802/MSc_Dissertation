"""Evaluation on separate, seeded environments without parameter updates."""

import numpy as np
import torch


def evaluate_policy_with_success(env, policy_fn, goal=None, episodes=8, seed=0):
    """Return mean return, length, success and final goal distance.

    Success comes from the environment's info contract. Environments may expose
    `goal_distance(observation, goal)` for observations unlike goal vectors.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    returns, lengths, successes, distances = [], [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, length, succeeded = 0.0, 0, False
        while True:
            obs, reward, terminated, truncated, info = env.step(policy_fn(obs))
            total += float(reward)
            length += 1
            succeeded |= bool(info.get("success", info.get("is_success", False)))
            if terminated or truncated:
                break
        distance_fn = getattr(env, "goal_distance", None)
        distance = distance_fn(obs, goal) if goal is not None and distance_fn else float("nan")
        returns.append(total)
        lengths.append(length)
        successes.append(float(succeeded))
        distances.append(distance)
    return tuple(float(np.mean(values)) for values in (returns, lengths, successes, distances))


def evaluate_policy(env, policy_fn, episodes=8, seed=0):
    return evaluate_policy_with_success(env, policy_fn, episodes=episodes, seed=seed)[:2]


def make_policy(result, goal, algorithm, *, normalize_state=False, normalize_goal=False, clip=10.0):
    model = result.network if algorithm == "dqn" else result.actor
    device = next(model.parameters()).device
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device).view(1, -1)

    @torch.no_grad()
    def policy(obs):
        state = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1)
        if algorithm == "dqn":
            return int(model.q_val_for_argmax_action(state, goal_t).argmax(-1).item())
        state = result.state_normalizer.normalize(state, clip=clip) if normalize_state else state
        task = result.goal_normalizer.normalize(goal_t, clip=clip) if normalize_goal else goal_t
        return model.deterministic(state, task).squeeze(0).cpu().numpy()

    return policy
