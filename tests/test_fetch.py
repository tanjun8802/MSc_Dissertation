"""Real Fetch physics/reset/relabelling, plus optional real graphics checks."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("gymnasium_robotics")

from tbtrl.environments.fetch import FixedStartFetch
from tbtrl.environments.registry import make_environment
from tbtrl.experiments.config import load_config
from tbtrl.experiments.diagnostics import NotebookDiagnostics
from tbtrl.experiments.figure_style import FigureStyle
from tbtrl.experiments.render import render_saved_run
from tbtrl.experiments.runner import run_experiment
from tbtrl.replay import ReplayBatch

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["fetch_reach", "fetch_pick_and_place"]
FAST = FigureStyle(dpi=70, formats=("png",))


def config_for(name):
    return load_config(ROOT / f"configs/tbtrl_{name}_sac.json")


@pytest.mark.parametrize("name", NAMES)
def test_physical_state_and_action_trace_are_identical_across_goals_and_reset_seeds(name):
    config = config_for(name)
    reference = trajectory = None
    for goal in config.goals:
        options = dict(config.environment_options, randomize_object=False)
        env = make_environment(config.environment, goal, **options)
        try:
            assert env.observation_space.shape == ((10,) if name == "fetch_reach" else (25,))
            assert env.action_space.shape == (4,)
            for seed in (1, 42, 987):
                obs, _ = env.reset(seed=seed)
                state = np.concatenate(
                    [
                        obs,
                        env.unwrapped.data.qpos,
                        env.unwrapped.data.qvel,
                        env.unwrapped.data.mocap_pos.ravel(),
                        env.unwrapped.data.mocap_quat.ravel(),
                        env.unwrapped.data.ctrl,
                    ]
                )
                if reference is None:
                    reference = state.copy()
                np.testing.assert_array_equal(state, reference)
                np.testing.assert_array_equal(
                    env.unwrapped.goal, np.asarray(goal, dtype=np.float32)
                )
                trace = []
                for _ in range(6):
                    obs, _, terminated, _, _ = env.step(
                        np.array([0.2, -0.1, 0.1, -0.5], dtype=np.float32)
                    )
                    assert not terminated
                    trace.append(obs.copy())
                if trajectory is None:
                    trajectory = np.array(trace)
                np.testing.assert_array_equal(trace, trajectory)
        finally:
            env.close()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("reward_type", ["dense", "sparse"])
def test_relabelling_matches_native_rewards_and_preserves_continuing_task(name, reward_type):
    config = config_for(name)
    options = dict(config.environment_options, max_horizon=3, reward_type=reward_type)
    env = make_environment(config.environment, config.goals[0], **options)
    try:
        obs, _ = env.reset(seed=0)
        transitions, observed_rewards = [], []
        for _ in range(3):
            following, reward, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
            assert not terminated
            transitions.append((obs.copy(), following.copy()))
            observed_rewards.append(reward)
            obs = following
        assert truncated
        batch = ReplayBatch(
            obs=torch.tensor(np.array([p[0] for p in transitions])),
            actions=torch.zeros(3, 4),
            next_obs=torch.tensor(np.array([p[1] for p in transitions])),
            rewards=torch.full((3, 1), 999.0),
            terminated=torch.ones(3, 1),
            truncated=torch.ones(3, 1),
        )
        achieved = batch.next_obs[:, env.achieved_slice].numpy()
        for goal in (config.goals[0], config.goals[1], achieved[0].tolist()):
            rewards, terminated = env.relabel_transitions(batch, goal)
            expected = env.unwrapped.compute_reward(
                achieved, np.broadcast_to(goal, achieved.shape), {}
            )
            np.testing.assert_allclose(rewards.numpy().ravel(), expected, atol=1e-6)
            assert not terminated.any()  # including success; only time limit ends Fetch
            assert rewards.dtype == torch.float32
        actual, _ = env.relabel_transitions(batch, config.goals[0])
        np.testing.assert_allclose(actual.numpy().ravel(), observed_rewards, atol=1e-6)
        assert (batch.rewards == 999).all() and batch.truncated.all()
    finally:
        env.close()


def test_pick_block_varies_but_arm_is_fixed_and_seeded_sequences_reproduce():
    config = config_for("fetch_pick_and_place")
    assert config.environment_options["randomize_object"] is True
    reference_arm = reference_sequence = None
    for goal in config.goals:
        env = make_environment(config.environment, goal, **config.environment_options)
        try:
            base = env.unwrapped
            joint = base.model.joint("object0:joint")
            qpos_start, qvel_start = int(joint.qposadr[0]), int(joint.dofadr[0])
            block_sequences = []
            for _ in range(2):
                positions = []
                for episode in range(12):
                    obs, info = env.reset(seed=123 if episode == 0 else None)
                    assert info["randomize_object"]
                    arm = np.concatenate(
                        [
                            obs[:3],
                            np.delete(base.data.qpos, np.s_[qpos_start : qpos_start + 7]),
                            np.delete(base.data.qvel, np.s_[qvel_start : qvel_start + 6]),
                            base.data.mocap_pos.ravel(),
                            base.data.mocap_quat.ravel(),
                            base.data.ctrl,
                        ]
                    )
                    if reference_arm is None:
                        reference_arm = arm.copy()
                    np.testing.assert_array_equal(arm, reference_arm)
                    xy = obs[3:5]
                    delta = xy - base.initial_gripper_xpos[:2]
                    assert np.all(np.abs(delta) <= base.obj_range + 1e-6)
                    assert np.linalg.norm(delta) >= 0.1 - 1e-6
                    positions.append(obs[3:6].copy())
                    # Verify the next reset restores the arm after movement.
                    for _ in range(6):
                        env.step(np.array([0.3, -0.2, 0.2, -1], dtype=np.float32))
                block_sequences.append(np.array(positions))
            np.testing.assert_array_equal(*block_sequences)
            assert len(np.unique(block_sequences[0][:, :2], axis=0)) == 12
            np.testing.assert_array_equal(block_sequences[0][:, 2], block_sequences[0][0, 2])
            if reference_sequence is None:
                reference_sequence = block_sequences[0]
            np.testing.assert_array_equal(block_sequences[0], reference_sequence)
            different, _ = env.reset(seed=987)
            assert not np.array_equal(different[3:5], reference_sequence[0, :2])
        finally:
            env.close()


@pytest.mark.parametrize("name", NAMES)
def test_configured_goal_fetch_pipeline_uses_latest_global_sac(name, tmp_path):
    full = config_for(name)
    assert len(full.goals) >= 3 and full.environment_options["reward_type"] == "dense"
    config = full.smoke()
    config.goals = full.goals
    rows = run_experiment(config, tmp_path / name, verbose=False)
    assert len(rows) == 3 * len(full.goals)
    assert all(len(row["evaluation"]) == len(full.goals) for row in rows)
    sequential = [r for r in rows if r["mode"] == "sequential"]
    for i, row in enumerate(sequential):
        assert row["replay_memory"]["tasks"] == i + 1
        if i:
            assert row["initial_alpha"] == sequential[i - 1]["final_alpha"]
    manifest = json.loads((tmp_path / name / "manifest.json").read_text())
    assert manifest["gymnasium_robotics"] and manifest["mujoco"]


@pytest.mark.parametrize("name", NAMES)
def test_fetch_diagnostics_with_stub_frames_do_not_change_training(name, tmp_path, monkeypatch):
    # This verifies GIF encoding, inline delivery and RNG isolation, NOT OpenGL.
    # The separate opt-in test below and CI notebook runs exercise native rendering.
    monkeypatch.setattr(
        FixedStartFetch, "render", lambda self: np.full((120, 640, 3), 100, dtype=np.uint8)
    )
    config = config_for(name).smoke()
    config.eval_episodes = 8
    # Match the eight rendered episodes to the held-out numerical evaluation.
    # Smoke mode separately reduces an explicit post_eval_episodes to one.
    config.post_eval_episodes = 8
    baseline = tmp_path / "baseline"
    actual = tmp_path / "actual"
    run_experiment(config, baseline, modes=("sequential",), verbose=False)
    shown = []
    monkeypatch.setattr("IPython.display.display", lambda value: shown.append(value))
    records = run_experiment(
        config,
        actual,
        modes=("sequential",),
        verbose=False,
        on_task_end=NotebookDiagnostics(display=True, style=FAST),
    )
    assert sum(getattr(value, "format", None) == "gif" for value in shown) == 16
    for task in (0, 1):
        relative = Path(f"seed_42/sequential/task_{task}")
        old = torch.load(baseline / relative / "checkpoint.pt", weights_only=True)
        new = torch.load(actual / relative / "checkpoint.pt", weights_only=True)
        for component in ("actor", "critic", "target", "state_normalizer", "goal_normalizer"):
            for key in old[component]:
                torch.testing.assert_close(old[component][key], new[component][key], rtol=0, atol=0)
        with Image.open(actual / relative / "rollout-01.gif") as gif:
            assert gif.n_frames == 9 and gif.info["duration"] == 40
        rollouts = json.loads((actual / relative / "rollouts.json").read_text())
        assert len(rollouts) == 8
        assert [row["seed"] for row in rollouts] == list(
            range(1_000_042 + task * 100, 1_000_050 + task * 100)
        )
        assert len(list((actual / relative).glob("rollout-*.gif"))) == 8
        if name == "fetch_pick_and_place":
            starts = [row["initial_observation"][3:5] for row in rollouts]
            assert len(np.unique(starts, axis=0)) == 8
        evaluation = records[task]["evaluation"][task]
        for metric, rollout_key in (
            ("mean_return", "episode_return"),
            ("mean_length", "steps"),
            ("success_rate", "success"),
            ("final_distance", "final_distance"),
        ):
            assert np.mean([row[rollout_key] for row in rollouts]) == pytest.approx(
                evaluation[metric], abs=1e-6
            )
        assert not (actual / relative / "q-policy.png").exists()
        assert (actual / relative / "embeddings-actor_mu.png").exists()
    destination = render_saved_run(actual, style=FAST)
    assert (destination / "seed_42/sequential/task_1/rollout-08.gif").exists()


@pytest.mark.skipif(
    os.environ.get("TBTRL_TEST_RENDER") != "1",
    reason="Native graphics check requires TBTRL_TEST_RENDER=1 and a working GL context",
)
@pytest.mark.parametrize("name", NAMES)
def test_native_fetch_rendering_produces_animated_gifs(name, tmp_path):
    config = config_for(name).smoke()
    config.goals = config.goals[:1]
    output = tmp_path / name
    run_experiment(
        config,
        output,
        modes=("sequential",),
        verbose=False,
        on_task_end=NotebookDiagnostics(display=False, style=FAST),
    )
    directory = output / "seed_42/sequential/task_0"
    with Image.open(directory / "rollout-01.gif") as gif:
        assert gif.n_frames == 9 and gif.size == (640, 546)
        frame = np.array(gif.convert("RGB"))[66:]
        assert frame.std() > 10  # actual scene, not a uniform placeholder
    summary = json.loads((directory / "rollouts.json").read_text())[0]
    expected_seed = summary["seed"] if name == "fetch_pick_and_place" else 42
    assert summary["initial_state_seed"] == expected_seed and summary["truncated"]
    if artifact_root := os.environ.get("TBTRL_RENDER_ARTIFACTS"):
        destination = Path(artifact_root) / name
        destination.mkdir(parents=True, exist_ok=True)
        for source in directory.glob("rollout*"):
            shutil.copy2(source, destination / source.name)
