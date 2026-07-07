from __future__ import annotations

from pathlib import Path

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
    else:
        raise ValueError(f"Unsupported benchmark '{benchmark}'. Expected 'exorl' or 'ogbench'.")

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

    raise ValueError(f"Unsupported benchmark '{benchmark}'. Expected 'exorl' or 'ogbench'.")