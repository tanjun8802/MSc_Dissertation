"""Optional actor TBTRL terms, separate from the unchanged critic objectives."""

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from tbtrl.losses import CovarianceRegularizer


@dataclass(frozen=True)
class ActorTBTRLOptions:
    replay_loss_coef: float = 0.0
    retention_kl_coef: float = 0.0
    covariance_target_variance: float = 1.0
    sigreg_coef: float = 0.0
    sketch_dim: int = 16
    goal_separation_coef: float = 0.0
    goal_separation_target_cosine: float = 0.85
    phi_norm_coef: float = 0.0
    psi_norm_coef: float = 0.0
    phi_norm_target: float = 2.0
    psi_norm_target: float = 2.0

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown actor TBTRL settings: {sorted(unknown)}")
        result = cls(**values)
        for name in (
            "replay_loss_coef",
            "retention_kl_coef",
            "sigreg_coef",
            "goal_separation_coef",
            "phi_norm_coef",
            "psi_norm_coef",
        ):
            value = getattr(result, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"actor_tbtrl.{name} must be finite and nonnegative.")
        for name in ("phi_norm_target", "psi_norm_target", "covariance_target_variance"):
            value = getattr(result, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"actor_tbtrl.{name} must be finite and positive.")
        if not isinstance(result.sketch_dim, int) or result.sketch_dim < 1:
            raise ValueError("actor_tbtrl.sketch_dim must be a positive integer.")
        if not -1 <= result.goal_separation_target_cosine <= 1:
            raise ValueError("actor goal cosine threshold must be in [-1, 1].")
        return result

    @property
    def needs_factors(self):
        return any(
            (self.sigreg_coef, self.goal_separation_coef, self.phi_norm_coef, self.psi_norm_coef)
        )


def policy_objective(actor, critic, states, goals, alpha):
    """Fresh reparameterized actions; caller freezes critic parameters, not dQ/da."""
    actions, log_prob, _ = actor.sample(states, goals)
    q = torch.minimum(
        critic.q1_forward(states, actions, goals), critic.q2_forward(states, actions, goals)
    )
    return (alpha * log_prob - q).mean(), log_prob


class ActorRegularizer:
    """SAC-style linear excess penalties and per-action covariance sketches.

    Covariance is computed across states separately for every mean/std action
    component, then averaged. Mixing action heads as if they were independent
    state samples could hide a collapsed state representation.
    """

    def __init__(self, actor, options):
        self.options = options
        self.has_factors = all(
            callable(getattr(actor, name, None)) for name in ("state_factors", "goal_factors")
        )
        if options.needs_factors and not self.has_factors:
            raise ValueError("Actor representation penalties require a factorised actor.")
        self.covariance = (
            [CovarianceRegularizer(options.sketch_dim) for _ in range(2 * actor.action_dim)]
            if self.has_factors
            else []
        )

    def __call__(self, actor, states, goals):
        zero = states.new_zeros(())
        terms = dict(sigreg=zero, goal_separation=zero, phi_norm=zero, psi_norm=zero)
        statistics = {}
        o = self.options
        if not self.has_factors:
            return zero, terms, statistics
        mu, std = actor.state_factors(states)
        psi = actor.goal_factors(goals)
        components = [x[:, j, :] for x in (mu, std) for j in range(actor.action_dim)]
        if o.sigreg_coef:
            terms["sigreg"] = torch.stack(
                [
                    reg(phi / math.sqrt(o.covariance_target_variance))
                    for reg, phi in zip(self.covariance, components)
                ]
            ).mean()
        norms = torch.cat([mu.norm(dim=-1).flatten(), std.norm(dim=-1).flatten()])
        psi_norms = psi.norm(dim=-1)
        terms["phi_norm"] = F.relu(norms - o.phi_norm_target).mean()
        terms["psi_norm"] = F.relu(psi_norms - o.psi_norm_target).mean()
        max_cosine = zero
        if len(goals) > 1:
            unit = F.normalize(psi, dim=-1, eps=1e-8)
            cosines = (unit @ unit.T)[~torch.eye(len(goals), dtype=torch.bool, device=psi.device)]
            terms["goal_separation"] = F.relu(cosines - o.goal_separation_target_cosine).mean()
            max_cosine = cosines.max().detach()
        total = (
            o.sigreg_coef * terms["sigreg"]
            + o.goal_separation_coef * terms["goal_separation"]
            + o.phi_norm_coef * terms["phi_norm"]
            + o.psi_norm_coef * terms["psi_norm"]
        )
        statistics = dict(
            actor_mu_phi_norm=mu.norm(dim=-1).mean().detach(),
            actor_std_phi_norm=std.norm(dim=-1).mean().detach(),
            actor_psi_norm_mean=psi_norms.mean().detach(),
            actor_psi_max_cosine=max_cosine,
        )
        return total, terms, statistics


def policy_retention_kl(reference, actor, states, goals):
    """KL(previous policy || current policy); tanh is a shared bijection.

    Evaluate the analytic diagonal Gaussian KL before squashing, avoiding
    sampling noise and inverse-tanh numerical instability near action bounds.
    """
    with torch.no_grad():
        old_mean, old_log_std = reference(states, goals)
    mean, log_std = actor(states, goals)
    old = torch.distributions.Normal(old_mean, old_log_std.exp())
    current = torch.distributions.Normal(mean, log_std.exp())
    return torch.distributions.kl_divergence(old, current).sum(-1).mean()


def actor_gradient_statistics(components, parameters):
    """Weighted component gradient norms without writing parameter .grad fields."""
    parameters = list(parameters)
    vectors = {}
    for name, loss in components.items():
        grads = (
            torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            if loss.requires_grad
            else [None] * len(parameters)
        )
        vectors[name] = torch.cat(
            [
                torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                for p, g in zip(parameters, grads)
            ]
        )
    result = {f"actor_{name}_grad_norm": vector.norm().item() for name, vector in vectors.items()}
    current, replay = vectors["current"], vectors["replay"]
    denominator = current.norm() * replay.norm()
    result["actor_current_replay_grad_cosine"] = (
        (torch.dot(current, replay) / denominator).item() if denominator > 0 else 0.0
    )
    return result
