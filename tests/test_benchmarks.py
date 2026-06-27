from __future__ import annotations

from pathlib import Path

import numpy as np

from src.benchmarks.exorl import (
    exorl_episode_length,
    exorl_episode_to_trajectory,
    get_exorl_dataset_url,
    load_exorl_dataset_into_buffer,
)
from src.benchmarks.ogbench import iter_ogbench_trajectories, load_ogbench_dataset_into_buffer
from src.utils import TrajectoryReplayBuffer


def test_exorl_dataset_url_shape():
    assert get_exorl_dataset_url("walker", "proto") == "https://dl.fbaipublicfiles.com/exorl/walker/proto.zip"


def test_exorl_episode_conversion_matches_repo_buffer_contract():
    episode = {
        "observation": np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=np.float32,
        ),
        "action": np.asarray([[0.0], [0.1], [0.2], [0.3]], dtype=np.float32),
        "reward": np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
    }

    assert exorl_episode_length(episode) == 3
    trajectory = exorl_episode_to_trajectory(episode)

    assert len(trajectory["obs"]) == 3
    assert np.allclose(trajectory["obs"][0], np.asarray([0.0, 0.0], dtype=np.float32))
    assert np.allclose(trajectory["next_obs"][-1], np.asarray([3.0, 3.0], dtype=np.float32))
    assert trajectory["truncated"] == [False, False, True]


def test_exorl_dataset_loader_populates_replay_buffer(tmp_path: Path):
    replay_dir = tmp_path / "walker" / "proto" / "buffer"
    replay_dir.mkdir(parents=True)
    np.savez_compressed(
        replay_dir / "episode_000_3.npz",
        observation=np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        action=np.asarray([[0.0], [0.1], [0.2], [0.3]], dtype=np.float32),
        reward=np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
    )

    buffer = TrajectoryReplayBuffer(capacity=16, obs_dim=2, action_dim=1, device="cpu")
    loaded_episodes = load_exorl_dataset_into_buffer(buffer, replay_dir.parent, max_episodes=1)

    assert loaded_episodes == 1
    assert len(buffer) == 3


def test_ogbench_trajectory_iteration_segments_by_terminal_flag():
    dataset = {
        "observations": np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        "actions": np.asarray([[0.0], [0.1], [0.2], [0.3], [0.4]], dtype=np.float32),
        "next_observations": np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32),
        "terminals": np.asarray([0, 0, 1, 0, 1], dtype=np.float32),
        "rewards": np.asarray([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    }

    trajectories = list(iter_ogbench_trajectories(dataset))

    assert len(trajectories) == 2
    assert len(trajectories[0]["obs"]) == 3
    assert len(trajectories[1]["obs"]) == 2
    assert trajectories[0]["terminated"][-1] is True
    assert trajectories[1]["terminated"][-1] is True


def test_ogbench_dataset_loader_populates_replay_buffer():
    dataset = {
        "observations": np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        "actions": np.asarray([[0.0], [0.1], [0.2], [0.3]], dtype=np.float32),
        "next_observations": np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        "terminals": np.asarray([0, 1, 0, 1], dtype=np.float32),
    }

    buffer = TrajectoryReplayBuffer(capacity=16, obs_dim=1, action_dim=1, device="cpu")
    loaded_episodes = load_ogbench_dataset_into_buffer(buffer, dataset=dataset)

    assert loaded_episodes == 2
    assert len(buffer) == 4
