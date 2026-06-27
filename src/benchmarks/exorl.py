from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import urllib.request
import zipfile

import gymnasium as gym
import numpy as np


DEFAULT_EXORL_DATA_ROOT = Path.home() / ".msc_dissertation" / "benchmarks" / "exorl"
EXORL_ALGORITHMS = (
    "aps",
    "diayn",
    "disagreement",
    "icm",
    "icm_apt",
    "proto",
    "random",
    "rnd",
    "smm",
)
EXORL_TASKS_BY_DOMAIN = {
    "cartpole": (
        "cartpole_balance",
        "cartpole_balance_sparse",
        "cartpole_swingup",
        "cartpole_swingup_sparse",
    ),
    "cheetah": (
        "cheetah_run",
        "cheetah_run_backward",
    ),
    "jaco": (
        "jaco_reach_top_left",
        "jaco_reach_top_right",
        "jaco_reach_bottom_left",
        "jaco_reach_bottom_right",
    ),
    "point_mass_maze": (
        "point_mass_maze_reach_top_left",
        "point_mass_maze_reach_top_right",
        "point_mass_maze_reach_bottom_left",
        "point_mass_maze_reach_bottom_right",
    ),
    "quadruped": (
        "quadruped_walk",
        "quadruped_run",
    ),
    "walker": (
        "walker_stand",
        "walker_walk",
        "walker_run",
    ),
}
EXORL_SUITE_TASKS = {
    "cartpole_balance": ("cartpole", "balance"),
    "cartpole_balance_sparse": ("cartpole", "balance_sparse"),
    "cartpole_swingup": ("cartpole", "swingup"),
    "cartpole_swingup_sparse": ("cartpole", "swingup_sparse"),
    "cheetah_run": ("cheetah", "run"),
    "quadruped_walk": ("quadruped", "walk"),
    "quadruped_run": ("quadruped", "run"),
    "walker_stand": ("walker", "stand"),
    "walker_walk": ("walker", "walk"),
    "walker_run": ("walker", "run"),
}
EXORL_DATASET_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/exorl/{domain}/{algorithm}.zip"


class DMControlGymnasiumAdapter(gym.Env):
    """State-only Gymnasium adapter for dm_control tasks.

    Rendering is intentionally omitted here because benchmark setup varies across
    machines and this repo only needs reset/step compatibility for training.
    """

    metadata = {"render_modes": []}

    def __init__(self, env_factory, seed: int = 0):
        super().__init__()
        self._env_factory = env_factory
        self._seed = seed
        self._env = self._env_factory(seed)
        initial_time_step = self._env.reset()
        initial_obs = self._flatten_observation(initial_time_step.observation)
        action_spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=np.asarray(action_spec.minimum, dtype=np.float32),
            high=np.asarray(action_spec.maximum, dtype=np.float32),
            shape=action_spec.shape,
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_obs.shape,
            dtype=np.float32,
        )

    @staticmethod
    def _flatten_observation(observation) -> np.ndarray:
        if isinstance(observation, dict):
            flattened = [np.asarray(value, dtype=np.float32).reshape(-1) for value in observation.values()]
            return np.concatenate(flattened, axis=0)
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None and seed != self._seed:
            self._seed = seed
            self._env = self._env_factory(seed)
        time_step = self._env.reset()
        obs = self._flatten_observation(time_step.observation)
        return obs, {"backend": "dm_control"}

    def step(self, action):
        typed_action = np.asarray(action, dtype=np.float32)
        time_step = self._env.step(typed_action)
        obs = self._flatten_observation(time_step.observation)
        reward = float(0.0 if time_step.reward is None else time_step.reward)
        terminated = bool(time_step.last())
        truncated = False
        info = {"discount": float(1.0 if time_step.discount is None else time_step.discount)}
        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()


@dataclass(frozen=True)
class ExORLDatasetSpec:
    domain: str
    algorithm: str
    replay_dir: Path


def get_exorl_dataset_url(domain: str, algorithm: str) -> str:
    _validate_domain(domain)
    _validate_algorithm(algorithm)
    return EXORL_DATASET_URL_TEMPLATE.format(domain=domain, algorithm=algorithm)


def get_exorl_replay_dir(
    domain: str,
    algorithm: str,
    data_root: str | Path = DEFAULT_EXORL_DATA_ROOT,
) -> Path:
    _validate_domain(domain)
    _validate_algorithm(algorithm)
    return Path(data_root).expanduser() / domain / algorithm


def describe_exorl_dataset(
    domain: str,
    algorithm: str,
    data_root: str | Path = DEFAULT_EXORL_DATA_ROOT,
) -> ExORLDatasetSpec:
    return ExORLDatasetSpec(
        domain=domain,
        algorithm=algorithm,
        replay_dir=get_exorl_replay_dir(domain=domain, algorithm=algorithm, data_root=data_root),
    )


def download_exorl_dataset(
    domain: str,
    algorithm: str,
    data_root: str | Path = DEFAULT_EXORL_DATA_ROOT,
    force_download: bool = False,
) -> Path:
    replay_dir = get_exorl_replay_dir(domain=domain, algorithm=algorithm, data_root=data_root)
    buffer_dir = replay_dir / "buffer"
    if buffer_dir.exists() and any(buffer_dir.glob("*.npz")) and not force_download:
        return replay_dir

    if replay_dir.exists() and force_download:
        shutil.rmtree(replay_dir)

    replay_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path = replay_dir.parent / f"{algorithm}.zip"
    with urllib.request.urlopen(get_exorl_dataset_url(domain, algorithm)) as response, archive_path.open("wb") as output:
        shutil.copyfileobj(response, output)

    replay_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(replay_dir)
    archive_path.unlink(missing_ok=True)
    return replay_dir


def iter_exorl_episode_files(replay_dir: str | Path):
    buffer_dir = Path(replay_dir).expanduser() / "buffer"
    return sorted(buffer_dir.glob("*.npz"))


def load_exorl_episode(episode_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(episode_path).expanduser(), allow_pickle=False) as episode:
        return {key: episode[key] for key in episode.files}


def exorl_episode_length(episode: dict[str, np.ndarray]) -> int:
    return int(next(iter(episode.values())).shape[0] - 1)


def exorl_episode_to_trajectory(episode: dict[str, np.ndarray]) -> dict[str, list[np.ndarray | float | bool]]:
    total_steps = exorl_episode_length(episode)
    if total_steps <= 0:
        raise ValueError("ExORL episodes must contain at least one transition.")

    observations = np.asarray(episode["observation"], dtype=np.float32)
    actions = _strip_exorl_dummy_transition(np.asarray(episode["action"], dtype=np.float32), total_steps)
    rewards = _strip_exorl_dummy_transition(np.asarray(episode["reward"], dtype=np.float32), total_steps)

    return {
        "obs": [observations[index] for index in range(total_steps)],
        "actions": [actions[index].reshape(-1) for index in range(total_steps)],
        "rewards": [float(np.asarray(rewards[index]).reshape(-1)[0]) for index in range(total_steps)],
        "next_obs": [observations[index + 1] for index in range(total_steps)],
        "terminated": [False for _ in range(total_steps)],
        "truncated": [index == total_steps - 1 for index in range(total_steps)],
    }


def load_exorl_dataset_into_buffer(
    replay_buffer,
    replay_dir: str | Path,
    max_episodes: int | None = None,
) -> int:
    loaded_episodes = 0
    for episode_path in iter_exorl_episode_files(replay_dir):
        replay_buffer.add_episode(exorl_episode_to_trajectory(load_exorl_episode(episode_path)))
        loaded_episodes += 1
        if max_episodes is not None and loaded_episodes >= max_episodes:
            break
    return loaded_episodes


def make_exorl_env(task_name: str, seed: int = 0) -> gym.Env:
    if task_name not in EXORL_SUITE_TASKS:
        available = ", ".join(sorted(EXORL_SUITE_TASKS))
        raise NotImplementedError(
            "This repo only provides direct dm_control-backed ExORL environments for suite tasks. "
            f"Supported tasks: {available}. Requested: {task_name}."
        )

    try:
        from dm_control import suite
    except ImportError as exc:
        raise ImportError(
            "dm_control is required for ExORL environments. Install benchmark dependencies with `uv sync --group benchmarks`."
        ) from exc

    domain_name, suite_task_name = EXORL_SUITE_TASKS[task_name]

    def env_factory(random_seed: int):
        return suite.load(
            domain_name,
            suite_task_name,
            task_kwargs={"random": random_seed},
            environment_kwargs={"flat_observation": True},
        )

    return DMControlGymnasiumAdapter(env_factory=env_factory, seed=seed)


def _validate_algorithm(algorithm: str) -> None:
    if algorithm not in EXORL_ALGORITHMS:
        available = ", ".join(EXORL_ALGORITHMS)
        raise ValueError(f"Unsupported ExORL algorithm '{algorithm}'. Expected one of: {available}.")


def _validate_domain(domain: str) -> None:
    if domain not in EXORL_TASKS_BY_DOMAIN:
        available = ", ".join(sorted(EXORL_TASKS_BY_DOMAIN))
        raise ValueError(f"Unsupported ExORL domain '{domain}'. Expected one of: {available}.")


def _strip_exorl_dummy_transition(values: np.ndarray, total_steps: int) -> np.ndarray:
    if values.shape[0] == total_steps:
        return values
    if values.shape[0] == total_steps + 1:
        return values[1:]
    raise ValueError(
        f"ExORL field has unexpected first dimension {values.shape[0]}; expected {total_steps} or {total_steps + 1}."
    )
