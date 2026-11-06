from copy import deepcopy

import numpy as np
import pytest
import torch

from tbtrl.training.task_selection import KReplayOptions, TaskSelector


def test_basis_selects_real_independent_tasks_and_survives_resume():
    options = KReplayOptions(max_tasks=2, sample_budget=7, candidate_capacity=4)
    selector = TaskSelector(options)
    for task in range(4):
        selector.observe_completed_task(task)
    vectors = {0: [1, 0], 1: [-1, 0], 2: [0, 1], 3: [1, 0]}
    chosen, metrics = selector.select(vectors)
    assert chosen == [0, 2]
    assert metrics["rank"] == 2
    assert metrics["max_residual"] < 1e-8
    assert selector.sample_counts(chosen) == [4, 3]
    restored = TaskSelector(options, deepcopy(selector.state_dict()))
    for task in range(4, 30):
        selector.observe_completed_task(task)
        restored.observe_completed_task(task)
        assert selector.candidate_ids == restored.candidate_ids
        assert len(selector.candidate_ids) <= 4
    values = {i: [1, i % 2] for i in selector.candidate_ids}
    assert selector.select(values) == restored.select(values)


def test_selector_does_not_consume_training_rng():
    np.random.seed(8)
    torch.manual_seed(8)
    expected_np = np.random.rand(3)
    expected_torch = torch.rand(3)
    np.random.seed(8)
    torch.manual_seed(8)
    selector = TaskSelector(KReplayOptions(selector="random", max_tasks=2, candidate_capacity=4))
    for task in range(20):
        selector.observe_completed_task(task)
    selector.select({i: [1, i] for i in selector.candidate_ids})
    np.testing.assert_equal(np.random.rand(3), expected_np)
    torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)


def test_real_dqn_bounds_samples_goals_and_updates_with_adaptive_replay():
    from functools import partial

    from tbtrl.environments.registry import make_environment
    from tbtrl.models.dqn import FactorisedQNetwork
    from tbtrl.replay import ReplayBuffer
    from tbtrl.training.dqn import train_dqn

    options = dict(
        max_tasks=2,
        sample_budget=7,
        candidate_capacity=4,
        refresh_updates=2,
        weighting="interference",
        weight_smoothing=0.0,
    )
    selector = TaskSelector(KReplayOptions(**options))
    goals = {i: np.array([i + 1, 1], dtype=np.float32) for i in range(10)}
    replay = ReplayBuffer(8, 2, 1, discrete=True)
    for _ in range(8):
        replay.add_transition([1, 1], 3, 999, [2, 1], False)
    for i in goals:
        selector.observe_completed_task(i)
    factory = partial(make_environment, "maze-discrete", max_horizon=4)
    model = FactorisedQNetwork(2, 4, hidden_dim=8, rep_dim=4)
    before = deepcopy(model.state_dict())
    env = factory(goal=[3, 3])
    try:
        result = train_dqn(
            q_network=model,
            q_target_network=deepcopy(model),
            env=env,
            goal=[3, 3],
            task_id=10,
            make_env=factory,
            task_goals=goals,
            replay_task_buffers={i: replay for i in goals},
            buffer_capacity=16,
            current_buffer=ReplayBuffer(16, 2, 1, discrete=True),
            total_steps=8,
            warmup_steps=2,
            train_freq=1,
            batch_size=4,
            eval_freq=4,
            eval_episodes=1,
            enable_early_stop=False,
            replay_ratio=0,
            k_replay=options,
            selector_state=selector.state_dict(),
        )
    finally:
        env.close()
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert result.selector_state["completed"] == 11
    assert result.replay_diagnostics
    assert result.replay_diagnostics[0]["weighting_gradient_passes"] == 3
    assert len(result.replay_diagnostics[0]["replay_shared_grad_norms"]) == 2
    assert max(row["mean_value_drift"] for row in result.replay_diagnostics) > 0
    for row in result.losses:
        assert row["replay_tasks"] == 2
        assert row["replay_samples"] == 7
        assert row["regularised_goals"] == 3
        assert len(row["selected_tasks"]) == 2
        assert sum(row["replay_weights"]) == pytest.approx(1)
        assert min(row["replay_weights"]) >= 0.1


def test_interference_weights_protect_conflicting_task_with_a_floor():
    options = KReplayOptions(
        max_tasks=2, weighting="interference", weight_smoothing=0, weight_temperature=0.2
    )
    selector = TaskSelector(options)
    selector.select({0: [1, 0], 1: [0, 1]})
    weights = selector.update_weights(
        torch.tensor([1.0, 0.0]), [torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0])]
    )
    assert weights[0] > 0.8
    assert weights[1] >= 0.1
    assert sum(weights) == pytest.approx(1)
    assert selector.update_weights(torch.zeros(2), [torch.ones(2)] * 2) == [0.5, 0.5]


def test_runner_continues_bounded_selector_without_reusing_scratch_state(tmp_path):
    from pathlib import Path

    from tbtrl.experiments.config import load_config
    from tbtrl.experiments.continuation import load_sequence
    from tbtrl.experiments.runner import run_experiment

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/tbtrl_gridworldmaze_dqn.json").smoke()
    config.replay_mode, config.similarity_threshold = "global", 0.8
    config.replay_schedule = "constant"
    config.global_replay_capacity, config.recent_replay_capacity = 32, 16
    config.replay_max_groups = 2
    config.training.update(
        replay_ratio=0,
        k_replay=dict(max_tasks=1, sample_budget=5, candidate_capacity=3, refresh_updates=2),
    )
    config.scratch["replay_ratio"] = 0
    config.validate()
    output = tmp_path / "continued"
    run_experiment(config, output, modes=("scratch", "sequential"), verbose=False)
    config.goals.append([6, 3])
    rows = run_experiment(
        config, output, modes=("scratch", "sequential", "recovery"), verbose=False, resume=True
    )
    sequential = [r for r in rows if r["mode"] == "sequential"]
    assert sequential[-1]["bounded_replay"]["completed"] == 3
    assert all(r["bounded_replay"]["completed"] == 1 for r in rows if r["mode"] == "scratch")
    state = load_sequence(output / "seed_42/sequential/boundary.pt", "cpu")["result"].selector_state
    assert state["completed"] == 3
    assert len(state["reservoir"]) <= 2
    reference = tmp_path / "reference"
    run_experiment(config, reference, modes=("scratch", "sequential", "recovery"), verbose=False)
    left = torch.load(output / "seed_42/sequential/task_2/checkpoint.pt", weights_only=True)
    right = torch.load(reference / "seed_42/sequential/task_2/checkpoint.pt", weights_only=True)
    for name, value in left["network"].items():
        torch.testing.assert_close(value, right["network"][name], rtol=0, atol=0)


@pytest.mark.parametrize("selector,weighting", [("rotating", "uniform"), ("basis", "interference")])
def test_study_preset_executes_and_exports_publication_plots(tmp_path, selector, weighting):
    from pathlib import Path

    import matplotlib.pyplot as plt

    from tbtrl.experiments.config import load_config
    from tbtrl.experiments.dqn_study import configure_k_replay, plot_replay_allocation, plot_study
    from tbtrl.experiments.figure_style import FigureStyle
    from tbtrl.experiments.runner import run_experiment

    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/tbtrl_gridworldmaze_dqn.json").smoke()
    base.goals.append([6, 3])
    config = configure_k_replay(
        base,
        max_tasks=1,
        sample_budget=5,
        candidate_capacity=3,
        refresh_updates=2,
        selector=selector,
        weighting=weighting,
        capacity=32,
        recent_capacity=16,
        max_groups=2,
    )
    rows = run_experiment(config, tmp_path / selector, verbose=False)
    assert rows[-1]["mode"] == "recovery"
    style = FigureStyle(dpi=60, formats=("png", "pdf", "svg"))
    fig = plot_study({selector: tmp_path / selector}, tmp_path / "figures", style=style)
    assert len(fig.axes) >= 4
    plt.close(fig)
    fig = plot_replay_allocation(tmp_path / selector, style=style)
    assert len(fig.axes) >= 4
    plt.close(fig)
    assert (tmp_path / "figures/k-replay-study.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / selector / "figures/replay-allocation.svg").exists()


@pytest.mark.parametrize(
    "change",
    [
        dict(max_tasks=True),
        dict(candidate_capacity=0),
        dict(refresh_updates=0),
        dict(weight_temperature=0),
        dict(max_tasks=0, sample_budget=2),
    ],
)
def test_invalid_bounded_options_fail_before_training(change):
    with pytest.raises(ValueError):
        KReplayOptions(**change)


def test_configs_reject_hidden_growth_and_sac_bounded_replay():
    from pathlib import Path

    from tbtrl.experiments.config import load_config
    from tbtrl.experiments.dqn_study import configure_k_replay

    root = Path(__file__).resolve().parents[1]
    config = configure_k_replay(load_config(root / "configs/tbtrl_gridworldmaze_dqn.json"))
    config.replay_schedule = "task_count"
    with pytest.raises(ValueError, match="constant"):
        config.validate()
    config.replay_schedule = "constant"
    config.recovery["replay_ratio"] = 3
    with pytest.raises(ValueError, match="replay_ratio"):
        config.validate()
    config = load_config(root / "configs/tbtrl_gridworldmaze_sac.json")
    with pytest.raises(ValueError, match="DQN"):
        configure_k_replay(config)


@pytest.mark.parametrize(
    "vectors",
    [
        {0: [0, 0], 1: [0, 0], 2: [0, 0]},
        {0: [1, 0], 1: [1, 1e-12], 2: [-1, 0]},
        {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]},
    ],
)
def test_degenerate_and_over_budget_geometry_remains_measurable(vectors):
    selector = TaskSelector(KReplayOptions(max_tasks=2, sample_budget=3))
    ids, metrics = selector.select(vectors)
    assert len(ids) == len(set(ids)) == 2
    assert all(i in vectors for i in ids)
    assert all(np.isfinite(value) for value in metrics.values())
    assert sum(selector.sample_counts(ids)) == 3


def test_candidate_pool_matches_between_selection_strategies():
    selectors = [
        TaskSelector(KReplayOptions(selector=method, max_tasks=2, candidate_capacity=4))
        for method in ("basis", "rotating", "random")
    ]
    for task in range(50):
        for selector in selectors:
            selector.observe_completed_task(task)
            selector.select({i: [1, i] for i in selector.candidate_ids})
        assert (
            selectors[0].candidate_ids == selectors[1].candidate_ids == selectors[2].candidate_ids
        )


def test_tiny_task_embeddings_do_not_create_a_geometric_direction():
    selector = TaskSelector(KReplayOptions(max_tasks=2))
    _, diagnostics = selector.select({0: [1, 0], 1: [0, 1e-10], 2: [1, 0]})
    assert diagnostics["rank"] == 1


def test_recovery_gate_uses_the_same_return_target_as_adaptation(tmp_path):
    from pathlib import Path

    from tbtrl.experiments.config import load_config, training_settings
    from tbtrl.experiments.dqn_study import configure_k_replay
    from tbtrl.experiments.runner import run_experiment

    base = load_config(
        Path(__file__).resolve().parents[1] / "configs/tbtrl_gridworldmaze_dqn.json"
    ).smoke()
    config = configure_k_replay(
        base,
        max_tasks=1,
        sample_budget=4,
        candidate_capacity=3,
        capacity=32,
        recent_capacity=16,
        max_groups=2,
    )
    # Even perfect success cannot pass this deliberately unattainable return target.
    config.recovery_return_threshold = 1.1
    assert training_settings(config, "recovery", 0)["early_stop_reward"] == 1.1
    rows = run_experiment(
        config, tmp_path / "gate", modes=("sequential", "recovery"), verbose=False
    )
    recovery = [r for r in rows if r["mode"] == "recovery"]
    assert all(r["steps"] > 0 for r in recovery)
    assert all(r["recovery_target"] == dict(metric="mean_return", value=1.1) for r in recovery)

    config.recovery_return_threshold = -100.0
    rows = run_experiment(
        config, tmp_path / "already-retained", modes=("sequential", "recovery"), verbose=False
    )
    recovery = [r for r in rows if r["mode"] == "recovery"]
    assert all(r["steps"] == 0 and r["solved"] for r in recovery)
    assert all(r["recovery_target"] == dict(metric="mean_return", value=-100.0) for r in recovery)
