"""Evaluation on separate, seeded environments without parameter updates."""

import numpy as np
import torch


def task_evaluation_seed(seed, task_id):
    """Shared start seed for post-training metrics and matching visual rollouts."""
    return seed + 1_000_000 + task_id * 100


def evaluate_policy_metrics(
    env, policy_fn, goal=None, episodes=8, seed=0, *, success_metric="ever", hold_steps=10
):
    """Reach, final-position and sustained end-of-episode success on the same rollouts."""
    if episodes < 1 or hold_steps < 1:
        raise ValueError("episodes and hold_steps must be positive.")
    if success_metric not in ("ever", "final", "hold"):
        raise ValueError("success_metric must be ever, final or hold.")
    returns, lengths, ever, final, hold, distances = [], [], [], [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total, successes = 0.0, []
        while True:
            obs, reward, terminated, truncated, info = env.step(policy_fn(obs))
            total += float(reward)
            successes.append(bool(info.get("success", info.get("is_success", False))))
            if terminated or truncated:
                break
        distance_fn = getattr(env, "goal_distance", None)
        distances.append(
            distance_fn(obs, goal) if goal is not None and distance_fn else float("nan")
        )
        returns.append(total)
        lengths.append(len(successes))
        ever.append(any(successes))
        final.append(successes[-1])
        hold.append(len(successes) >= hold_steps and all(successes[-hold_steps:]))
    result = dict(
        mean_return=float(np.mean(returns)),
        mean_length=float(np.mean(lengths)),
        ever_success_rate=float(np.mean(ever)),
        final_success_rate=float(np.mean(final)),
        hold_success_rate=float(np.mean(hold)),
        final_distance=float(np.mean(distances)),
    )
    result["success_rate"] = result[f"{success_metric}_success_rate"]
    return result


def evaluate_policy_with_success(env, policy_fn, goal=None, episodes=8, seed=0, **options):
    """Compatibility tuple; by default success means reaching at least once."""
    metrics = evaluate_policy_metrics(env, policy_fn, goal, episodes, seed, **options)
    return tuple(
        metrics[k] for k in ("mean_return", "mean_length", "success_rate", "final_distance")
    )


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
