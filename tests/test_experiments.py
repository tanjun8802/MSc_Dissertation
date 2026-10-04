import json
from pathlib import Path

import nbformat
import numpy as np
import pytest

from tbtrl.experiments.checkpoints import load_policy
from tbtrl.experiments.config import load_config
from tbtrl.experiments.runner import run_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.json")), ids=lambda p: p.stem)
def test_supported_experiment_runs_all_phases_and_saves_loadable_checkpoints(path, tmp_path):
    config = load_config(path).smoke()
    records = run_experiment(config, tmp_path / "run")
    assert len(records) == 6
    assert {record["mode"] for record in records} == {"scratch", "sequential", "recovery"}
    assert all(len(record["evaluation"]) == 2 for record in records)
    assert all(record["steps"] > 0 for record in records)
    assert (tmp_path / "run" / "manifest.json").is_file()
    for checkpoint in (tmp_path / "run").rglob("checkpoint.pt"):
        policy = load_policy(checkpoint)
        action = policy(np.array([1.0, 1.0], dtype=np.float32))
        assert np.isfinite(action).all()
    with pytest.raises(FileExistsError):
        run_experiment(config, tmp_path / "run")


def test_same_seed_reproduces_metrics_and_losses(tmp_path):
    config = load_config(ROOT / "configs/tbtrl_gridworldmaze_dqn.json").smoke()
    runs = [run_experiment(config, tmp_path / str(i), modes=("sequential",)) for i in range(2)]
    for first, second in zip(*runs):
        assert first["learning_curve"] == second["learning_curve"]
        assert first["evaluation"] == second["evaluation"]
    for task_id in (0, 1):
        relative = f"seed_42/sequential/task_{task_id}/losses.jsonl"
        assert (tmp_path / "0" / relative).read_text() == (tmp_path / "1" / relative).read_text()


def test_unknown_config_option_fails_before_training():
    config = load_config(ROOT / "configs/tbtrl_gridworldmaze_sac.json")
    config.training["typo_learning_rate"] = 1e-3
    with pytest.raises(ValueError, match="typo_learning_rate"):
        config.validate()


def test_working_notebooks_have_no_hidden_state_or_outputs():
    notebooks = list((ROOT / "experiments/working").glob("*.ipynb"))
    assert len(notebooks) == 3
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")
                assert cell.execution_count is None and cell.outputs == []
                assert "sys.path" not in cell.source
        assert "run_experiment" in json.dumps(notebook)


def test_new_environment_dimensions_and_normalized_checkpoint(tmp_path):
    import gymnasium as gym
    import torch

    from tbtrl.environments.registry import register_environment
    from tbtrl.experiments.config import ExperimentConfig
    from tbtrl.models.sac import GaussianActor

    class VectorGoalEnv(gym.Env):
        observation_space = gym.spaces.Box(-10, 10, shape=(3,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

        def __init__(self, goal):
            self.goal = goal

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.array([1, 2, 3], dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            return np.array([1, 2, 3], dtype=np.float32), 0.0, False, self.steps >= 2, {}

    register_environment("test-vector-goal", VectorGoalEnv)
    config = ExperimentConfig(
        name="different-dimensions",
        algorithm="sac",
        environment="test-vector-goal",
        goals=[[1], [2]],
        model={"hidden_dim": 8, "latent_dim": 4},
        training={
            "total_steps": 6,
            "warmup_steps": 2,
            "buffer_capacity": 16,
            "batch_size": 2,
            "eval_freq": 2,
            "normalize_state_inputs": True,
            "normalize_goal_inputs": True,
            "enable_early_stop": False,
            "replay_ratio": 1,
        },
        eval_episodes=1,
    )
    run_experiment(config, tmp_path / "run", modes=("sequential",))
    path = tmp_path / "run/seed_42/sequential/task_1/checkpoint.pt"
    checkpoint = torch.load(path, weights_only=True)
    actor = GaussianActor(3, 1, 1, net_arch=(8, 8))
    actor.load_state_dict(checkpoint["actor"])
    observation = np.array([2, 3, 4], dtype=np.float32)

    def normalize(value, key):
        state = checkpoint[key]
        return (
            ((torch.tensor(value) - state["mean"]) / (state["var"] + 1e-8).sqrt())
            .clamp(-10, 10)
            .view(1, -1)
        )

    with torch.no_grad():
        expected = (
            actor.deterministic(
                normalize(observation, "state_normalizer"),
                normalize([2.0], "goal_normalizer"),
            )
            .numpy()
            .squeeze(0)
        )
    np.testing.assert_allclose(load_policy(path)(observation), expected, atol=1e-7)
