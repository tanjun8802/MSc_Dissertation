from copy import deepcopy

import pytest
import torch

from tbtrl.models.dqn import FactorisedQNetwork


def test_soft_ball_preserves_raw_factors_and_all_q_interfaces():
    model = FactorisedQNetwork(
        2, 4, rep_dim=2, hidden_dim=4, phi_max_norm=2, psi_max_norm=1, embedding_constraint="soft"
    )
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.sa_encoder[-1].bias.copy_(torch.tensor([3.0, 4.0]))
        model.goal_encoder[-1].bias.copy_(torch.tensor([1.0, 2.0]))
    obs, actions, goal = torch.zeros(1, 2), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.ones(2)
    torch.testing.assert_close(model.encode_state_action(obs, actions), torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(model.encode_goal(goal), torch.tensor([1.0, 2.0]))
    assert model(obs, actions, goal).item() == 11
    assert model.q_val_for_argmax_action(obs, goal).tolist() == [[11.0] * 4]
    embedding = torch.tensor([2.0, 3.0])
    assert model.forward_with_task_embedding(obs, actions, embedding).item() == 18
    assert model.q_val_for_argmax_action_from_embedding(obs, embedding).tolist() == [[18.0] * 4]
    assert model.representation_norms(obs, actions, goal)["theoretical_q_bound"] is None
    with pytest.raises(ValueError, match="normalis"):
        model.forward_with_task_embedding(obs, actions, embedding, normalize_embedding=True)


def test_squared_excess_penalty_starts_at_radius_and_has_outward_growth():
    from tbtrl.losses import norm_penalty_loss_l2

    inside = torch.tensor([[0.3, 0.4]], requires_grad=True)
    loss = norm_penalty_loss_l2(inside, target_norm=2)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(inside.grad, torch.zeros_like(inside))
    outside = torch.tensor([[3.0, 4.0]], requires_grad=True)
    loss = norm_penalty_loss_l2(outside, target_norm=2)
    assert loss.item() == pytest.approx(9)
    loss.backward()
    torch.testing.assert_close(outside.grad, torch.tensor([[3.6, 4.8]]))


def test_legacy_projection_preserves_old_checkpoint_defaults():
    model = FactorisedQNetwork(2, 4, rep_dim=2, phi_max_norm=2, psi_max_norm=1)
    obs, actions, goal = (
        torch.ones(1, 2) * 100,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.ones(2) * 100,
    )
    assert model.encode_state_action(obs, actions).norm() <= 2.000001
    assert model.encode_goal(goal).norm() <= 1.000001
    assert deepcopy(model).representation_norms(obs, actions, goal)["theoretical_q_bound"] == 2


def test_soft_ball_trainer_reports_excess_without_claiming_a_hard_bound():
    from functools import partial

    from tbtrl.environments.registry import make_environment
    from tbtrl.training.dqn import train_dqn

    factory = partial(make_environment, "maze-discrete", max_horizon=4)
    model = FactorisedQNetwork(
        2,
        4,
        hidden_dim=4,
        rep_dim=2,
        phi_max_norm=0.01,
        psi_max_norm=0.01,
        embedding_constraint="soft",
    )
    env = factory(goal=[3, 3])
    try:
        result = train_dqn(
            q_network=model,
            q_target_network=deepcopy(model),
            env=env,
            goal=[3, 3],
            task_id=0,
            make_env=factory,
            buffer_capacity=16,
            total_steps=6,
            warmup_steps=2,
            batch_size=4,
            eval_freq=2,
            eval_episodes=1,
            train_freq=1,
            enable_early_stop=False,
            goal_separation_coef=0,
            sigreg_coef=0,
        )
    finally:
        env.close()
    assert result.losses
    assert all(row["q_bound"] is None for row in result.losses)
    assert all(row["goal_separation"] == 0 for row in result.losses)
    assert result.losses[0]["phi_radius_excess_fraction"] > 0
    assert result.losses[0]["psi_radius_excess_fraction"] > 0
    assert result.losses[0]["phi_norm"] > 0


def test_revised_study_uses_soft_balls_and_disables_unjustified_geometry(tmp_path):
    from pathlib import Path

    import matplotlib.pyplot as plt

    from tbtrl.experiments.config import load_config
    from tbtrl.experiments.dqn_study import configure_k_replay, plot_soft_ball_norms
    from tbtrl.experiments.figure_style import FigureStyle
    from tbtrl.experiments.runner import run_experiment

    config = configure_k_replay(
        load_config(
            Path(__file__).resolve().parents[1] / "configs/tbtrl_gridworldmaze_dqn.json"
        ).smoke(),
        max_tasks=1,
        sample_budget=7,
        candidate_capacity=4,
        refresh_updates=2,
        capacity=32,
        recent_capacity=16,
        max_groups=2,
    )
    assert config.model["embedding_constraint"] == "soft"
    assert config.training["goal_separation_coef"] == config.training["sigreg_coef"] == 0
    rows = run_experiment(config, tmp_path / "study", modes=("sequential",), verbose=False)
    last = rows[-1]["replay_diagnostics"][-1]
    assert last["phi_norm_samples"] == 11  # Four current plus seven historical samples.
    assert last["phi_radius"] == config.model["phi_max_norm"]
    figure = plot_soft_ball_norms(tmp_path / "study", style=FigureStyle(dpi=60, formats=("png",)))
    assert len(figure.axes) == 4
    plt.close(figure)
    assert (tmp_path / "study/figures/soft-ball-norms.png").exists()
