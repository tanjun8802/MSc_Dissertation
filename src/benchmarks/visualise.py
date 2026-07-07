from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def random_policy(env, obs):
    del obs
    return env.action_space.sample()


def collect_episode_frames(
    env,
    policy_fn=None,
    max_steps: int = 500,
    reset_kwargs: dict | None = None,
):
    if policy_fn is None:
        policy_fn = random_policy
    if reset_kwargs is None:
        reset_kwargs = {}

    obs, info = env.reset(**reset_kwargs)
    del info

    frames = []
    first_frame = env.render()
    if first_frame is not None:
        frames.append(np.asarray(first_frame))

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while steps < max_steps and not (terminated or truncated):
        action = policy_fn(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        del info

        total_reward += float(reward)
        steps += 1

        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

    return {
        "frames": frames,
        "num_steps": steps,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def save_gif(frames, output_path: str | Path, fps: int = 30):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("No frames to save.")

    duration_ms = int(1000 / fps)
    imageio.mimsave(output_path, frames, duration=duration_ms / 1000.0)
    return output_path


def save_mp4(frames, output_path: str | Path, fps: int = 30):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("No frames to save.")

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    return output_path


def preview_frame(env, reset_kwargs: dict | None = None):
    if reset_kwargs is None:
        reset_kwargs = {}

    env.reset(**reset_kwargs)
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "env.render() returned None. Make sure the environment was created with render_mode='rgb_array'."
        )
    return np.asarray(frame)