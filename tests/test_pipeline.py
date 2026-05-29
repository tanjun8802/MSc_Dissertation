from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.agents import GoalConditionedActor
from src.environments.fourrooms import FourRoomsGridWorld
from src.networks import GoalRepresentationModel, StateActionRepresentationModel
from src.utils import TrajectoryReplayBuffer


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EXPERIMENT_NOTEBOOKS = [
    SRC_ROOT / "experiments" / "SGCRL_GridWorld4Rooms.ipynb",
    SRC_ROOT / "experiments" / "SGCRL_MountainCar.ipynb",
]


def _make_env(max_episode_steps: int = 50) -> FourRoomsGridWorld:
    return FourRoomsGridWorld(room_size=11, max_episode_steps=max_episode_steps)


def _build_episode(env: FourRoomsGridWorld, num_steps: int = 16):
    obs, _ = env.reset(seed=7)
    episode = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "next_obs": [],
        "terminated": [],
        "truncated": [],
    }
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode["obs"].append(obs.copy())
        episode["actions"].append(np.asarray(action, dtype=env.action_space.dtype))
        episode["rewards"].append(float(reward))
        episode["next_obs"].append(next_obs.copy())
        episode["terminated"].append(bool(terminated))
        episode["truncated"].append(bool(truncated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    return episode


def _extract_import_targets_from_notebook(notebook_path: Path):
    notebook_json = json.loads(notebook_path.read_text(encoding="utf-8"))
    module_names = set()
    for cell in notebook_json.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_names.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names.add(node.module)
    return module_names


def test_environment_reset_step_contract():
    env = _make_env(max_episode_steps=20)
    obs, info = env.reset(seed=123)
    assert isinstance(info, dict)
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all()

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_space_compatibility_with_env_and_actor():
    env = _make_env()
    obs, _ = env.reset(seed=99)

    sampled_action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(sampled_action)
    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    actor = GoalConditionedActor(obs_dim=obs_dim, action_dim=action_dim)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    goal_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

    policy_action, log_prob, entropy = actor.sample_action(obs_tensor, goal_tensor)
    assert policy_action.shape == (1, action_dim)
    assert log_prob.shape == (1, 1)
    assert entropy.shape == (1, 1)
    assert torch.isfinite(policy_action).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()

    env_action_np = policy_action.detach().cpu().numpy()
    env_action_typed = env_action_np.astype(env.action_space.dtype)
    env_action = env_action_typed.reshape(env.action_space.shape)
    step_obs, step_reward, step_terminated, step_truncated, _ = env.step(env_action)
    assert np.isfinite(step_obs).all()
    assert np.isfinite(step_reward)
    assert isinstance(step_terminated, bool)
    assert isinstance(step_truncated, bool)


def test_replay_buffer_contract_shapes_and_keys():
    env = _make_env()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    buffer = TrajectoryReplayBuffer(capacity=256, obs_dim=obs_dim, action_dim=action_dim, device="cpu")
    buffer.add_episode(_build_episode(env, num_steps=32))
    assert len(buffer) > 0

    batch_size = 8
    batch = buffer.sample(batch_size=batch_size)
    assert batch.obs.shape == (batch_size, obs_dim)
    assert batch.actions.shape == (batch_size, action_dim)
    assert batch.rewards.shape == (batch_size, 1)
    assert batch.next_obs.shape == (batch_size, obs_dim)
    assert batch.terminated.shape == (batch_size, 1)
    assert batch.truncated.shape == (batch_size, 1)
    assert batch.episode_id.shape == (batch_size,)
    assert batch.timestep.shape == (batch_size,)
    assert batch.indices.shape == (batch_size,)

    expected_future_keys = {
        "obs",
        "actions",
        "next_obs",
        "goals",
        "rewards",
        "terminated",
        "truncated",
        "episode_id",
        "timestep",
        "future_timestep",
        "indices",
        "goal_indices",
    }
    future_batch = buffer.sample_future_goal_batch(batch_size=batch_size)
    assert set(future_batch.keys()) == expected_future_keys
    assert future_batch["obs"].shape == (batch_size, obs_dim)
    assert future_batch["actions"].shape == (batch_size, action_dim)
    assert future_batch["goals"].shape == (batch_size, obs_dim)


def test_model_forward_pass_shapes_and_finite_values():
    obs_dim = 2
    action_dim = 2
    embedding_dim = 32
    batch_size = 7

    actor = GoalConditionedActor(obs_dim=obs_dim, action_dim=action_dim)
    phi = StateActionRepresentationModel(obs_dim=obs_dim, action_dim=action_dim, output_dim=embedding_dim)
    psi = GoalRepresentationModel(obs_dim=obs_dim, output_dim=embedding_dim)

    obs = torch.randn(batch_size, obs_dim, dtype=torch.float32)
    goals = torch.randn(batch_size, obs_dim, dtype=torch.float32)
    actions, log_prob, entropy = actor.sample_action(obs, goals)
    assert actions.shape == (batch_size, action_dim)
    assert log_prob.shape == (batch_size, 1)
    assert entropy.shape == (batch_size, 1)

    phi_output = phi(torch.cat([obs, actions], dim=-1))
    psi_output = psi(goals)
    assert phi_output.shape == (batch_size, embedding_dim)
    assert psi_output.shape == (batch_size, embedding_dim)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(phi_output).all()
    assert torch.isfinite(psi_output).all()


def test_one_update_training_step_runs_and_is_finite():
    torch.manual_seed(0)
    np.random.seed(0)

    env = _make_env(max_episode_steps=40)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    actor = GoalConditionedActor(obs_dim=obs_dim, action_dim=action_dim)
    phi = StateActionRepresentationModel(obs_dim=obs_dim, action_dim=action_dim, output_dim=16)
    psi = GoalRepresentationModel(obs_dim=obs_dim, output_dim=16)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(phi.parameters()) + list(psi.parameters()),
        lr=3e-4,
    )

    batch_size = 16
    obs = torch.randn(batch_size, obs_dim)
    goals = torch.randn(batch_size, obs_dim)
    policy_actions, _, _ = actor.sample_action(obs, goals)
    phi_output = phi(torch.cat([obs, policy_actions], dim=-1))
    psi_output = psi(goals)
    logits = phi_output @ psi_output.T
    labels = torch.arange(batch_size)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss)

    tracked_params = list(actor.parameters()) + list(phi.parameters()) + list(psi.parameters())
    params_before = [param.detach().clone() for param in tracked_params]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        policy_actions_post, _, _ = actor.sample_action(obs, goals)
        phi_output_post = phi(torch.cat([obs, policy_actions_post], dim=-1))
        psi_output_post = psi(goals)
        logits_post = phi_output_post @ psi_output_post.T
        post_update_loss = torch.nn.functional.cross_entropy(logits_post, labels)

    any_changed = any(not torch.allclose(before, after.detach()) for before, after in zip(params_before, tracked_params))
    assert any_changed or torch.isfinite(post_update_loss)


def test_short_rollout_smoke_loop():
    env = _make_env(max_episode_steps=100)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    actor = GoalConditionedActor(obs_dim=obs_dim, action_dim=action_dim)
    buffer = TrajectoryReplayBuffer(capacity=300, obs_dim=obs_dim, action_dim=action_dim, device="cpu")

    obs, _ = env.reset(seed=1)
    episode = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "next_obs": [],
        "terminated": [],
        "truncated": [],
    }
    total_steps = 40
    for _ in range(total_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        goal_tensor = torch.zeros_like(obs_tensor)
        action_tensor, _, _ = actor.sample_action(obs_tensor, goal_tensor)
        action_np = action_tensor.detach().cpu().numpy().reshape(-1)
        action = action_np.astype(env.action_space.dtype)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        assert np.isfinite(next_obs).all()
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        episode["obs"].append(obs.copy())
        episode["actions"].append(action.copy())
        episode["rewards"].append(float(reward))
        episode["next_obs"].append(next_obs.copy())
        episode["terminated"].append(bool(terminated))
        episode["truncated"].append(bool(truncated))

        obs = next_obs
        if terminated or truncated:
            buffer.add_episode(episode)
            obs, _ = env.reset()
            episode = {k: [] for k in episode}

    if episode["obs"]:
        buffer.add_episode(episode)
    assert len(buffer) > 0
    sampled = buffer.sample(batch_size=min(8, len(buffer)))
    assert sampled.obs.shape[1] == obs_dim


def test_import_sanity_for_repo_modules():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    local_modules = [
        "src.agents",
        "src.networks",
        "src.utils",
        "src.environments.fourrooms",
        "src.environments.mountaincar",
    ]
    for module_name in local_modules:
        imported = importlib.import_module(module_name)
        assert imported is not None


def test_experiment_local_imports_resolve():
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    prefixes = ("environments", "utils", "agents", "networks")

    for notebook_path in EXPERIMENT_NOTEBOOKS:
        import_targets = _extract_import_targets_from_notebook(notebook_path)
        for target in sorted(import_targets):
            base_module = target.split(".")[0]
            if base_module in prefixes:
                imported = importlib.import_module(target)
                assert imported is not None