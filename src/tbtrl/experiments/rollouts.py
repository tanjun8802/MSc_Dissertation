"""Inline RGB rollout GIFs on independent environments, using saved policies."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tbtrl.evaluation import make_policy
from tbtrl.random import preserve_rng_state

from .figure_style import MODES


@preserve_rng_state()
def render_rollout_gifs(result, config, record, factory, directory, *, display=True, episodes=1):
    from matplotlib.font_manager import findfont

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    goal = config.goals[record["task_id"]]
    policy = make_policy(
        result,
        goal,
        config.algorithm,
        normalize_state=config.training.get("normalize_state_inputs", False),
        normalize_goal=config.training.get("normalize_goal_inputs", False),
        clip=config.training.get("obs_norm_clip", 10.0),
    )
    font = ImageFont.truetype(findfont("DejaVu Sans"), 14)
    summaries = []
    for episode in range(episodes):
        seed = record["seed"] + 500_000 + episode
        env = factory(goal=goal, render_mode="rgb_array")
        frames, total, steps, success = [], 0.0, 0, False
        try:
            obs, reset_info = env.reset(seed=seed)
            initial_observation = obs.tolist()
            fps = env.metadata.get("render_fps", 25)

            def capture():
                frame = np.asarray(env.render())
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                    raise ValueError("RGB diagnostics require uint8 frames with shape (H, W, 3).")
                canvas = Image.new("RGB", (frame.shape[1], frame.shape[0] + 66), "white")
                canvas.paste(Image.fromarray(frame), (0, 66))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (10, 5),
                    f"{MODES[record['mode']][0]} | Goal {record['task_id']} | Seed {record['seed']} | Step {steps}",
                    fill="black",
                    font=font,
                )
                draw.text(
                    (10, 25),
                    f"Target (m): {tuple(round(v, 3) for v in goal)} | Return: {total:.2f}",
                    fill="black",
                    font=font,
                )
                draw.text(
                    (10, 45),
                    f"Success reached: {success} | Distance: {env.goal_distance(obs, goal):.3f} m",
                    fill="black",
                    font=font,
                )
                frames.append(canvas)

            capture()
            while True:
                obs, reward, terminated, truncated, info = env.step(policy(obs))
                total += reward
                steps += 1
                success |= bool(info.get("success", info.get("is_success", False)))
                capture()
                if terminated or truncated:
                    break
            summary = dict(
                episode=episode + 1,
                seed=seed,
                initial_state_seed=reset_info.get("initial_state_seed"),
                initial_observation=initial_observation,
                goal=goal,
                steps=steps,
                episode_return=total,
                success=success,
                final_success=bool(info.get("success", info.get("is_success", False))),
                final_distance=env.goal_distance(obs, goal),
                fps=fps,
                frame_count=len(frames),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        finally:
            env.close()
        destination = directory / f"rollout-{episode + 1:02d}.gif"
        frames[0].save(
            destination,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / fps),
            loop=0,
            disposal=2,
        )
        frames[-1].save(directory / f"rollout-{episode + 1:02d}-final.png", dpi=(300, 300))
        summary["gif"] = destination.name
        summaries.append(summary)
        print(
            f"Rollout GIF: {destination.name} | goal={record['task_id']} | steps={steps} | return={total:.3f} | success={success}",
            flush=True,
        )
        if display:
            from IPython.display import Image as InlineImage
            from IPython.display import display as show

            show(InlineImage(filename=str(destination), format="gif"))
    (directory / "rollouts.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries
