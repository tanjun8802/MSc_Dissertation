from copy import deepcopy
from functools import partial

import numpy as np
import pytest
import torch

from tbtrl.environments.registry import make_environment
from tbtrl.losses import CovarianceRegularizer, online_goal_separation_loss
from tbtrl.models.dqn import FactorisedQNetwork
from tbtrl.models.sac import FactorisedTwinCritic, GaussianActor
from tbtrl.replay import ReplayBuffer
from tbtrl.training.dqn import train_dqn
from tbtrl.training.sac import train_sac


def test_goal_separation_contract_and_gradients():
    model = FactorisedQNetwork(2, 4, hidden_dim=8, rep_dim=4)
    loss, cosines = online_goal_separation_loss(model, {0: [1, 1]}, "cpu")
    assert loss.item() == 0 and cosines.shape == (1, 1)
    loss, cosines = online_goal_separation_loss(model, {0: [1, 1], 1: [2, 2]}, "cpu", -1)
    loss.backward()
    assert cosines.shape == (2, 2)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.goal_encoder.parameters()
    )


def test_covariance_loss_matches_reference_and_handles_singleton():
    x = torch.randn(6, 4, requires_grad=True)
    centered = x - x.mean(0)
    expected = ((centered.T @ centered / (5 + 1e-6) - torch.eye(4)) ** 2).sum() / 4
    torch.testing.assert_close(CovarianceRegularizer(4)(x), expected)
    CovarianceRegularizer()(x[:1]).backward()
    assert torch.isfinite(x.grad).all()


def test_random_sketch_is_owned_by_regularizer_and_reproducible():
    x = torch.ones(4, 8)
    torch.manual_seed(17)
    first = CovarianceRegularizer(2)
    first(x)
    torch.manual_seed(17)
    second = CovarianceRegularizer(2)
    second(x)
    torch.testing.assert_close(first.projection, second.projection)
    assert first.projection.data_ptr() != second.projection.data_ptr()


def test_discrete_relabelling_uses_requested_goal_and_preserves_truncation():
    env = make_environment("maze-discrete", [1, 1])
    buffer = ReplayBuffer(2, 2, 1, discrete=True)
    buffer.add_transition([1, 1], 3, 99, [2, 1], False, True)
    batch = buffer.sample(1)
    rewards, terminal = env.relabel_transitions(batch, [2, 1])
    assert rewards.item() == 1 and terminal.item() == 1
    rewards, terminal = env.relabel_transitions(batch, [9, 9])
    assert rewards.item() == 0 and terminal.item() == 0
    assert batch.truncated.item() == 1
    env.close()


def old_buffer(discrete):
    buffer = ReplayBuffer(16, 2, 1 if discrete else 2, discrete=discrete)
    for _ in range(16):
        buffer.add_transition([1, 1], 3 if discrete else [0.1, 0.2], 0.0, [2, 1], False)
    return buffer


@pytest.mark.parametrize("algorithm", ["dqn", "sac"])
def test_real_updates_include_replay_and_all_regularizers(algorithm):
    torch.manual_seed(4)
    discrete = algorithm == "dqn"
    factory = partial(
        make_environment, "maze-discrete" if discrete else "maze-continuous", max_horizon=4
    )
    goal = [3, 1] if discrete else [3.5, 1.5]
    env = factory(goal=goal)
    common = dict(
        env=env,
        goal=goal,
        task_id=1,
        make_env=factory,
        seed=5,
        task_goals={0: np.array([1.0, 1.0])},
        replay_task_buffers={0: old_buffer(discrete)},
        buffer_capacity=32,
        total_steps=8,
        warmup_steps=2,
        batch_size=4,
        eval_freq=4,
        eval_episodes=1,
        train_freq=1,
        enable_early_stop=False,
        replay_ratio=1.0,
        replay_loss_coef=0.3,
        sigreg_coef=0.01,
        sketch_dim=4,
        goal_separation_coef=0.04,
        goal_separation_target_cosine=-1.0,
        phi_raw_norm_coef=0.1,
        psi_raw_norm_coef=0.1,
    )
    if discrete:
        model = FactorisedQNetwork(
            2, 4, hidden_dim=8, rep_dim=4, phi_max_norm=0.2, psi_max_norm=0.2
        )
        before = deepcopy(model.state_dict())
        result = train_dqn(q_network=model, q_target_network=deepcopy(model), **common)
        for row in result.losses:
            expected = (
                row["td"]
                + 0.3 * row["replay"]
                + 0.01 * row["sigreg"]
                + 0.04 * row["goal_separation"]
                + 0.1 * row["phi_norm"]
                + 0.1 * row["psi_norm"]
            )
            assert row["total"] == pytest.approx(expected, rel=1e-5, abs=1e-7)
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
        assert any(
            not torch.equal(v, result.target.state_dict()[k]) for k, v in model.state_dict().items()
        )
    else:
        model = FactorisedTwinCritic(2, 2, 2, hidden_dim=8, latent_dim=4)
        actor = GaussianActor(2, 2, 2, net_arch=(8, 8))
        before = deepcopy(model.state_dict())
        actor_before = deepcopy(actor.state_dict())
        result = train_sac(
            actor=actor,
            critic=model,
            critic_target=deepcopy(model),
            phi_norm_target=0.01,
            psi_norm_target=0.01,
            **common,
        )
        assert not {id(p) for p in model.critic1_parameters()} & {
            id(p) for p in model.critic2_parameters()
        }
        for row in result.losses:
            for head in (1, 2):
                expected = (
                    row[f"td{head}"]
                    + 0.3 * row[f"replay{head}"]
                    + 0.01 * row[f"sigreg{head}"]
                    + 0.04 * row[f"goal_separation{head}"]
                    + 0.1 * row[f"phi_norm{head}"]
                    + 0.1 * row[f"psi_norm{head}"]
                )
                assert row[f"total{head}"] == pytest.approx(expected, rel=1e-5, abs=1e-7)
                assert any(
                    not torch.equal(before[k], v)
                    for k, v in model.state_dict().items()
                    if k.startswith((f"phi{head}", f"psi{head}"))
                )
        assert any(not torch.equal(actor_before[k], v) for k, v in actor.state_dict().items())
    assert result.steps == 8 and result.steps_to_threshold is None
    assert result.losses and all(row["replay_tasks"] == 1 for row in result.losses)
    assert all(np.isfinite(value) for row in result.losses for value in row.values())
    assert all(p.grad is None for p in result.target.parameters())
    env.close()
