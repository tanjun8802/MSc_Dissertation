"""Task-boundary temperature, diagnostic isolation and retention regression tests."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tbtrl.experiments.config import load_config
from tbtrl.experiments.runner import run_experiment
from tbtrl.experiments.transfer import ACTORS, EXPERIMENTS, configure_transfer
from tbtrl.models.sac import FactorisedGaussianActor
from tbtrl.training.actor import ActorRegularizer, ActorTBTRLOptions, policy_retention_kl

ROOT = Path(__file__).resolve().parents[1]


def base():
    return load_config(ROOT / "configs/tbtrl_gridworldmaze_sac_factorised_actor.json")


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_transfer_presets_execute_and_record_actual_temperature(experiment, tmp_path):
    config = configure_transfer(base(), experiment).smoke()
    config.seeds = [42, 43]
    records = run_experiment(config, tmp_path / "run", verbose=False)
    for seed in config.seeds:
        rows = [r for r in records if r["seed"] == seed]
        scratch = [r for r in rows if r["mode"] == "scratch"]
        sequence = [r for r in rows if r["mode"] == "sequential"]
        recovery = [r for r in rows if r["mode"] == "recovery"]
        assert all(r["initial_alpha"] == 0.1 for r in scratch)
        assert sequence[0]["initial_alpha"] == (0.2 if experiment == "historical_reset" else 0.1)
        if experiment != "historical_reset":
            assert scratch[0]["learning_curve"] == sequence[0]["learning_curve"]
        if config.entropy_transfer == "carry":
            assert sequence[1]["initial_alpha"] == sequence[0]["final_alpha"]
            assert all(r["initial_alpha"] == sequence[-1]["final_alpha"] for r in recovery)
        else:
            assert sequence[1]["initial_alpha"] == sequence[0]["initial_alpha"]
            assert all(
                r["initial_alpha"] == (1.0 if experiment == "historical_reset" else 0.1)
                for r in recovery
            )
        for row in rows:
            probes = row["transfer_probes"]
            assert [p["step"] for p in probes][:4] == [0, 3, 4, 5]
            assert [p["updates"] for p in probes][:4] == [0, 0, 1, 2]
            assert probes[0]["alpha"] == pytest.approx(row["initial_alpha"])
            relative = f"seed_{seed}/{row['mode']}/task_{row['task_id']}"
            checkpoint = torch.load(
                tmp_path / "run" / relative / "checkpoint.pt", weights_only=True
            )
            assert checkpoint["config"]["training"]["initial_alpha"] == row["initial_alpha"]
            assert checkpoint["entropy_coefficient"] == row["final_alpha"]
            saved = json.loads((tmp_path / "run" / relative / "transfer-probes.json").read_text())
            assert saved == probes
        assert len(sequence[1]["transfer_probes"][0]["evaluation"]) == 2
        if experiment == "policy_retention":
            losses = [
                json.loads(line)
                for line in (tmp_path / "run" / f"seed_{seed}/sequential/task_1/losses.jsonl")
                .read_text()
                .splitlines()
            ]
            assert any(r["actor_retention_kl"] > 0 for r in losses)
            assert any(r["actor_retention_kl_grad_norm"] > 0 for r in losses)


def test_transfer_probes_and_component_gradients_leave_training_unchanged(tmp_path):
    config = configure_transfer(base()).smoke()
    observed = run_experiment(config, tmp_path / "observed", modes=("sequential",), verbose=False)
    config.training["transfer_probe_steps"] = None
    config.training["actor_gradient_diagnostics"] = False
    baseline = run_experiment(config, tmp_path / "baseline", modes=("sequential",), verbose=False)
    for first, second in zip(observed, baseline):
        assert first["learning_curve"] == second["learning_curve"]
        assert first["final_alpha"] == second["final_alpha"]
        relative = f"seed_42/sequential/task_{first['task_id']}/checkpoint.pt"
        left = torch.load(tmp_path / "observed" / relative, weights_only=True)
        right = torch.load(tmp_path / "baseline" / relative, weights_only=True)
        for group in ("actor", "critic", "target", "state_normalizer", "goal_normalizer"):
            for key in left[group]:
                torch.testing.assert_close(left[group][key], right[group][key], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ACTORS)
def test_actor_ablations_run_under_matched_temperature(variant, tmp_path):
    original = base()
    before = original.to_dict()
    config = configure_transfer(original, actor_variant=variant).smoke()
    assert original.to_dict() == before
    rows = run_experiment(config, tmp_path / variant, modes=("sequential",), verbose=False)
    assert rows[1]["initial_alpha"] == rows[0]["final_alpha"]


def test_policy_kl_is_zero_at_snapshot_then_penalizes_drift_without_teacher_gradients():
    actor = FactorisedGaussianActor(2, 2, 2, hidden_dim=8, latent_dim=4)
    reference = deepcopy(actor).eval().requires_grad_(False)
    states, goals = torch.randn(8, 2), torch.randn(8, 2)
    torch.testing.assert_close(
        policy_retention_kl(reference, actor, states, goals), torch.tensor(0.0)
    )
    with torch.no_grad():
        actor.phi_mu[-1].bias.add_(0.5)
        actor.phi_log_std[-1].bias.add_(0.1)
    loss = policy_retention_kl(reference, actor, states, goals)
    assert loss > 0
    loss.backward()
    assert all(p.grad is None for p in reference.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())


def test_covariance_target_matches_scaled_identity():
    actor = FactorisedGaussianActor(2, 2, 2, hidden_dim=8, latent_dim=2)
    # Unbiased covariance is exactly 0.125 I, with norms inside the bound 2.
    samples = (
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]) * (0.125 * 1.5) ** 0.5
    )
    factors = samples[:, None, :].expand(-1, 2, -1)
    actor.state_factors = lambda states: (factors, factors)
    reg = ActorRegularizer(
        actor, ActorTBTRLOptions(sigreg_coef=1, sketch_dim=2, covariance_target_variance=0.125)
    )
    _, terms, _ = reg(actor, samples, torch.zeros(1, 2))
    assert terms["sigreg"].item() < 1e-10
    assert terms["phi_norm"] == 0
    with pytest.raises(ValueError):
        ActorTBTRLOptions.from_dict({"covariance_target_variance": 0})


def test_transfer_config_rejects_incompatible_options():
    config = configure_transfer(base())
    config.recovery["ent_coef"] = 0.1
    with pytest.raises(ValueError, match="automatic entropy"):
        config.validate()
    with pytest.raises(ValueError, match="requires factorised_tbtrl"):
        configure_transfer(base(), "scaled_regularization", "baseline")
