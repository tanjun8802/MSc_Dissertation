from __future__ import annotations

from pathlib import Path

from .atari import make_atari_env
from .exorl import (
    DEFAULT_EXORL_DATA_ROOT,
    download_exorl_dataset,
    get_exorl_replay_dir,
    load_exorl_dataset_into_buffer,
    make_exorl_env,
)
from .ogbench import (
    DEFAULT_OGBENCH_DATA_ROOT,
    load_ogbench_dataset_into_buffer,
    make_ogbench_env,
)
from .procgen import make_procgen_env
from .wrappers import GoalInfoWrapper


def make_env(
    benchmark: str,
    task: str,
    *,
    seed: int = 0,
    add_goal_wrapper: bool = False,
    goal_fn=None,
    render_mode: str | None = None,
    render_height: int = 240,
    render_width: int = 320,
    camera_id: int = 0,
    # Atari-specific
    frame_stack: int = 4,
    noop_max: int = 30,
    frame_skip: int = 4,
    episodic_life: bool = True,
    clip_rewards: bool = True,
    grayscale: bool = True,
    image_size: int = 84,
    full_action_space: bool = False,
    max_episode_steps: int | None = None,
    # Procgen-specific
    num_levels: int = 0,
    start_level: int = 0,
    distribution_mode: str = "easy",
    normalize_obs: bool = True,
):
    benchmark = benchmark.lower()

    if benchmark == "exorl":
        env = make_exorl_env(
            task,
            seed=seed,
            render_mode=render_mode,
            render_height=render_height,
            render_width=render_width,
            camera_id=camera_id,
        )
    elif benchmark == "ogbench":
        env = make_ogbench_env(task)
    elif benchmark == "atari":
        env = make_atari_env(
            task,
            seed=seed,
            render_mode=render_mode,
            frame_stack=frame_stack,
            noop_max=noop_max,
            frame_skip=frame_skip,
            episodic_life=episodic_life,
            clip_rewards=clip_rewards,
            grayscale=grayscale,
            image_size=image_size,
            full_action_space=full_action_space,
            max_episode_steps=max_episode_steps if max_episode_steps is not None else 108_000,
        )
    elif benchmark == "procgen":
        env = make_procgen_env(
            task,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            seed=seed,
            render_mode=render_mode,
            normalize_obs=normalize_obs,
        )
    else:
        raise ValueError(
            f"Unsupported benchmark '{benchmark}'. "
            "Expected 'exorl', 'ogbench', 'atari', or 'procgen'."
        )

    if add_goal_wrapper:
        env = GoalInfoWrapper(env, goal_fn=goal_fn)

    return env


def load_dataset_into_buffer(
    benchmark: str,
    replay_buffer,
    task: str,
    *,
    data_root: str | Path | None = None,
    algorithm: str | None = None,
    split: str = "train",
    max_episodes: int | None = None,
    auto_download: bool = False,
):
    benchmark = benchmark.lower()

    if benchmark == "exorl":
        if algorithm is None:
            raise ValueError("For ExORL dataset loading, `algorithm` must be provided.")

        root = Path(data_root).expanduser() if data_root is not None else DEFAULT_EXORL_DATA_ROOT

        if auto_download:
            download_exorl_dataset(
                domain=task,
                algorithm=algorithm,
                data_root=root,
                force_download=False,
            )

        replay_dir = get_exorl_replay_dir(
            domain=task,
            algorithm=algorithm,
            data_root=root,
        )
        return load_exorl_dataset_into_buffer(
            replay_buffer=replay_buffer,
            replay_dir=replay_dir,
            max_episodes=max_episodes,
        )

    if benchmark == "ogbench":
        root = Path(data_root).expanduser() if data_root is not None else DEFAULT_OGBENCH_DATA_ROOT
        return load_ogbench_dataset_into_buffer(
            replay_buffer=replay_buffer,
            dataset_name=task,
            dataset_dir=root,
            split=split,
            max_episodes=max_episodes,
        )

    raise ValueError(
        f"Unsupported benchmark '{benchmark}'. "
        "Expected 'exorl' or 'ogbench'.  "
        "(Atari and Procgen benchmarks are online-only and do not support offline dataset loading.)"
    )