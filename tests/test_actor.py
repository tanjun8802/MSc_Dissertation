"""Actor factorisation, policy replay and TBTRL optimizer regression checks."""

from copy import deepcopy
from functools import partial

import numpy as np
import pytest
import torch

from tbtrl.environments.registry import make_environment
from tbtrl.models.sac import FactorisedGaussianActor, FactorisedTwinCritic, GaussianActor
from tbtrl.replay import ReplayBuffer
from tbtrl.training.actor import ActorRegularizer, ActorTBTRLOptions, policy_objective
from tbtrl.training.sac import train_sac


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS is not available"
            ),
        ),
    ],
)
def test_factorisation_shapes_squashed_density_and_gradients(device):
    torch.manual_seed(4)
    actor = FactorisedGaussianActor(3, 2, 1, hidden_dim=8, latent_dim=4).to(device)
    states = torch.randn(5, 3, device=device)
    goals = torch.randn(5, 1, device=device)
    mu_phi, std_phi = actor.state_factors(states)
    psi = actor.goal_factors(goals)
    mean, log_std = actor(states, goals)
    assert mu_phi.shape == std_phi.shape == (5, 2, 4)
    torch.testing.assert_close(mean, torch.einsum("bar,br->ba", mu_phi, psi))
    torch.testing.assert_close(log_std, torch.einsum("bar,br->ba", std_phi, psi).clamp(-20, 2))
    action, log_prob, deterministic = actor.sample(states, goals)
    assert action.shape == (5, 2) and log_prob.shape == (5, 1)
    assert action.dtype == torch.float32 and torch.isfinite(log_prob).all()
    assert (action.abs() <= 1).all()
    torch.testing.assert_close(deterministic, mean.tanh())
    normal = torch.distributions.Normal(mean, log_std.exp())
    # Initial outputs stay away from saturation, allowing an independent density check.
    expected = (normal.log_prob(action.atanh()) - (1 - action.square() + 1e-6).log()).sum(
        -1, keepdim=True
    )
    torch.testing.assert_close(log_prob, expected, rtol=1e-4, atol=1e-4)
    (action.square().mean() + log_prob.mean()).backward()
    for branch in (actor.phi_mu, actor.phi_log_std, actor.psi):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    with torch.no_grad():
        assert not torch.allclose(actor(states, goals)[0], actor(states, goals + 1)[0])


def test_policy_replay_uses_fresh_goal_conditioned_actions_and_keeps_dqda(monkeypatch):
    torch.manual_seed(2)
    actor = FactorisedGaussianActor(3, 2, 1, hidden_dim=8, latent_dim=4)
    critic = FactorisedTwinCritic(3, 2, 1, hidden_dim=8, latent_dim=4)
    critic.requires_grad_(False)
    before = deepcopy(critic.state_dict())
    states, goals = torch.randn(4, 3), torch.tensor([[2.0]]).expand(4, 1)
    sampled = []
    original_sample = actor.sample

    def sample(states, goals):
        sampled.append(goals.clone())
        return original_sample(states, goals)

    monkeypatch.setattr(actor, "sample", sample)
    # alpha=0 isolates the path through Q(s, sampled_action, goal).
    loss, log_prob = policy_objective(actor, critic, states, goals, 0.0)
    loss.backward()
    assert log_prob.shape == (4, 1) and len(sampled) == 1
    torch.testing.assert_close(sampled[0], goals)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert all(p.grad is None for p in critic.parameters())
    for key, value in critic.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_covariance_cannot_use_action_head_differences_to_hide_state_collapse():
    actor = FactorisedGaussianActor(2, 2, 2, hidden_dim=8, latent_dim=2)
    # Different action heads, but absolutely no variation across states.
    constant = torch.tensor([[[0.0, 0.0], [5.0, 5.0]]]).expand(4, -1, -1)
    actor.state_factors = lambda states: (constant, constant)
    regularizer = ActorRegularizer(actor, ActorTBTRLOptions(sigreg_coef=1.0, sketch_dim=2))
    total, terms, _ = regularizer(actor, torch.randn(4, 2), torch.randn(1, 2))
    assert terms["sigreg"].item() == pytest.approx(1.0)
    assert total.item() == pytest.approx(1.0)
    assert terms["goal_separation"].item() == 0


def test_actor_tbtrl_updates_all_factors_and_reports_weighted_loss():
    torch.manual_seed(4)
    actor = FactorisedGaussianActor(2, 2, 2, hidden_dim=8, latent_dim=4, output_init_scale=1.0)
    critic = FactorisedTwinCritic(2, 2, 2, hidden_dim=8, latent_dim=4)
    before = deepcopy(actor.state_dict())
    old = ReplayBuffer(16, 2, 2)
    for _ in range(16):
        old.add_transition([1, 1], [0.1, 0.2], 0, [2, 1], False)
    factory = partial(make_environment, "maze-continuous", max_horizon=4)
    env = factory(goal=[3.5, 1.5])
    options = dict(
        replay_loss_coef=0.7,
        sigreg_coef=0.03,
        sketch_dim=4,
        goal_separation_coef=0.02,
        goal_separation_target_cosine=-1.0,
        phi_norm_coef=0.1,
        psi_norm_coef=0.2,
        phi_norm_target=0.001,
        psi_norm_target=0.001,
    )
    try:
        result = train_sac(
            actor=actor,
            critic=critic,
            critic_target=deepcopy(critic),
            env=env,
            goal=[3.5, 1.5],
            task_id=1,
            task_goals={0: np.array([1.0, 1.0])},
            replay_task_buffers={0: old},
            replay_ratio=1.0,
            buffer_capacity=32,
            make_env=factory,
            total_steps=8,
            warmup_steps=2,
            batch_size=4,
            eval_freq=4,
            eval_episodes=1,
            enable_early_stop=False,
            actor_tbtrl=options,
        )
    finally:
        env.close()
    for row in result.losses:
        assert row["actor_replay_tasks"] == 1
        assert row["actor_sigreg"] > 0 and row["actor_phi_norm"] > 0 and row["actor_psi_norm"] > 0
        assert row["actor_goal_separation"] > 0
        expected = (
            row["actor_current"]
            + 0.7 * row["actor_replay"]
            + 0.03 * row["actor_sigreg"]
            + 0.02 * row["actor_goal_separation"]
            + 0.1 * row["actor_phi_norm"]
            + 0.2 * row["actor_psi_norm"]
        )
        assert row["actor"] == pytest.approx(expected, abs=1e-6)
        assert all(np.isfinite(value) for value in row.values())
    for prefix in ("phi_mu", "phi_log_std", "psi"):
        assert any(
            not torch.equal(before[k], v)
            for k, v in actor.state_dict().items()
            if k.startswith(prefix)
        )
    assert all(p.requires_grad for p in critic.parameters())
    assert all(p.grad is None for p in result.target.parameters())


def test_actor_option_errors_are_not_silently_ignored():
    with pytest.raises(ValueError, match="Unknown actor"):
        ActorTBTRLOptions.from_dict({"wrong_loss": 1})
    for values in ({"replay_loss_coef": -1}, {"sigreg_coef": float("nan")}, {"sketch_dim": 0}):
        with pytest.raises(ValueError):
            ActorTBTRLOptions.from_dict(values)
    with pytest.raises(ValueError, match="factorised actor"):
        ActorRegularizer(GaussianActor(2, 2, 2), ActorTBTRLOptions(sigreg_coef=1))
