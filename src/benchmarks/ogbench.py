from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_OGBENCH_DATA_ROOT = Path.home() / ".ogbench" / "data"

OGBENCH_EXAMPLE_DATASETS = (
    "pointmaze-medium-navigate-v0",
    "antmaze-large-navigate-v0",
    "humanoidmaze-medium-navigate-v0",
    "antsoccer-arena-navigate-v0",
    "cube-double-play-v0",
    "scene-play-v0",
    "puzzle-3x3-play-v0",
    "powderworld-easy-play-v0",
)


def require_ogbench():
    try:
        import ogbench
    except ImportError as exc:
        raise ImportError(
            "ogbench is not installed. Install benchmark dependencies with "
            "`uv sync --group benchmarks`."
        ) from exc
    return ogbench


def make_ogbench_env(dataset_name: str):
    ogbench = require_ogbench()
    return ogbench.make_env_and_datasets(dataset_name, env_only=True)


def download_ogbench_datasets(
    dataset_names,
    dataset_dir: str | Path = DEFAULT_OGBENCH_DATA_ROOT,
) -> Path:
    ogbench = require_ogbench()
    output_dir = Path(dataset_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    ogbench.download_datasets(list(dataset_names), dataset_dir=str(output_dir))
    return output_dir


def load_ogbench_datasets(
    dataset_name: str,
    dataset_dir: str | Path = DEFAULT_OGBENCH_DATA_ROOT,
    compact_dataset: bool = False,
):
    ogbench = require_ogbench()
    return ogbench.make_env_and_datasets(
        dataset_name,
        dataset_dir=str(Path(dataset_dir).expanduser()),
        compact_dataset=compact_dataset,
    )


def iter_ogbench_trajectories(dataset: dict[str, np.ndarray]):
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    num_transitions = len(actions)
    rewards = np.asarray(
        dataset.get("rewards", np.zeros(num_transitions, dtype=np.float32)),
        dtype=np.float32,
    )
    terminals = np.asarray(dataset["terminals"], dtype=bool)

    if len(observations) != len(actions):
        raise ValueError("OGBench observations and actions must have the same number of rows.")

    if "next_observations" in dataset:
        next_observations = np.asarray(dataset["next_observations"], dtype=np.float32)
    elif "valids" in dataset:
        next_observations = _reconstruct_next_observations(
            observations=observations,
            valids=np.asarray(dataset["valids"], dtype=bool),
        )
    else:
        raise ValueError("OGBench datasets must include `next_observations` or `valids`.")

    episode_start = 0
    for index, is_terminal in enumerate(terminals):
        if not is_terminal and index != len(terminals) - 1:
            continue

        episode_length = index + 1 - episode_start
        episode_slice = slice(episode_start, index + 1)

        yield {
            "obs": [obs for obs in observations[episode_slice]],
            "actions": [action.reshape(-1) for action in actions[episode_slice]],
            "rewards": [float(value) for value in rewards[episode_slice]],
            "next_obs": [obs for obs in next_observations[episode_slice]],
            "terminated": [False for _ in range(episode_length - 1)] + [bool(is_terminal)],
            "truncated": [False for _ in range(episode_length)],
        }

        episode_start = index + 1


def load_ogbench_dataset_into_buffer(
    replay_buffer,
    dataset: dict[str, np.ndarray] | None = None,
    *,
    dataset_name: str | None = None,
    dataset_dir: str | Path = DEFAULT_OGBENCH_DATA_ROOT,
    split: str = "train",
    max_episodes: int | None = None,
) -> int:
    selected_dataset = dataset

    if selected_dataset is None:
        if dataset_name is None:
            raise ValueError("Provide either a pre-loaded dataset or a dataset_name.")

        _, train_dataset, val_dataset = load_ogbench_datasets(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            compact_dataset=False,
        )
        selected_dataset = train_dataset if split == "train" else val_dataset

    loaded_episodes = 0
    for trajectory in iter_ogbench_trajectories(selected_dataset):
        replay_buffer.add_episode(trajectory)
        loaded_episodes += 1
        if max_episodes is not None and loaded_episodes >= max_episodes:
            break

    return loaded_episodes


def _reconstruct_next_observations(
    observations: np.ndarray,
    valids: np.ndarray,
) -> np.ndarray:
    next_observations = np.empty_like(observations)
    next_observations[:-1] = observations[1:]
    next_observations[-1] = observations[-1]
    next_observations[~valids] = observations[~valids]
    return next_observations