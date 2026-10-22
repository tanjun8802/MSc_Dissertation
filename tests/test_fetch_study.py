import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tbtrl.evaluation import evaluate_policy_metrics
from tbtrl.experiments.config import load_config
from tbtrl.experiments.fetch_study import (
    ARCHITECTURES,
    configure_fetch_study,
    plot_baselines,
    require_baseline,
    run_baselines,
)
from tbtrl.experiments.runner import run_experiment
from tbtrl.global_replay import GlobalReplayMemory
from tbtrl.models.sac import tanh_log_jacobian

ROOT = Path(__file__).resolve().parents[1]


def base():
    return load_config(ROOT / "configs/tbtrl_fetch_pick_and_place_sac.json")


def test_saturated_log_density_has_correct_finite_gradient():
    x = torch.tensor([-70.0, -10.0, 0.0, 10.0, 70.0], requires_grad=True)
    correction = tanh_log_jacobian(x)
    assert torch.isfinite(correction).all()
    torch.testing.assert_close(torch.autograd.grad(correction.sum(), x)[0], -2 * x.tanh())
    torch.testing.assert_close(correction, tanh_log_jacobian(-x))


def test_touch_is_not_hold_or_final_success():
    class Env:
        def reset(self, seed):
            self.step_id = 0
            return [0], {}

        def step(self, action):
            self.step_id += 1
            return [0], -1, False, self.step_id == 5, {"success": self.step_id == 2}

    m = evaluate_policy_metrics(Env(), lambda _: 0, episodes=2, success_metric="hold", hold_steps=2)
    assert m["ever_success_rate"] == 1
    assert m["success_rate"] == m["final_success_rate"] == m["hold_success_rate"] == 0


def test_capacity_policy_keeps_similar_tail_until_full_and_fork_preserves_policy():
    memory = GlobalReplayMemory(30, 10, 1, 1, keep_fraction=1, max_groups=2, compression="capacity")
    for task in range(4):
        recent = memory.begin_task()
        for i in range(10):
            recent.add_transition([task * 10 + i], [0], 0, [0], False)
        memory.finish_task(task, np.ones((len(memory.representatives) + 1,) * 2))
        assert memory.stats()["archive_samples"] == min(20, (task + 1) * 10)
        assert memory.last_admission["admitted_candidates"] == 10
    assert memory.fork().compression == "capacity"


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_fetch_controls_train_and_restore(architecture, tmp_path):
    pytest.importorskip("gymnasium_robotics")
    from tbtrl.experiments.checkpoints import load_policy

    c = configure_fetch_study(base(), architecture=architecture).smoke()
    c.goals = c.goals[:1]
    rows = run_experiment(c, tmp_path / architecture, modes=("scratch",), verbose=False)
    row = rows[0]
    assert row["success_metric"] == "hold"
    assert row["initial_alpha"] == pytest.approx(row["final_alpha"])
    assert {"hold_success_rate", "ever_success_rate", "final_success_rate"} <= row["evaluation"][
        0
    ].keys()
    assert row["evaluation_details"]
    checkpoint = next((tmp_path / architecture).rglob("checkpoint.pt"))
    assert callable(load_policy(checkpoint))


def test_optimizer_moments_carry_but_phase_rates_change(tmp_path, monkeypatch):
    pytest.importorskip("gymnasium_robotics")
    from tbtrl.experiments import runner

    original = runner.train_sac
    observations = []

    def capture(**kwargs):
        result = original(**kwargs)
        observations.append((kwargs, result.optimizer_state))
        return result

    monkeypatch.setattr(runner, "train_sac", capture)
    c = configure_fetch_study(base(), architecture="tbtrl", retention="combined").smoke()
    run_experiment(c, tmp_path / "sequence", modes=("sequential", "recovery"), verbose=False)
    first, second = observations[:2]
    assert first[0]["optimizer_state"] is None
    assert second[0]["optimizer_state"] is first[1]
    assert first[1]["actor"]["param_groups"][0]["lr"] == 3e-4
    assert second[1]["actor"]["param_groups"][0]["lr"] == 1e-4
    for key in first[1]["actor"]["state"]:
        assert second[1]["actor"]["state"][key]["step"] > first[1]["actor"]["state"][key]["step"]
    assert observations[2][0]["optimizer_state"] is observations[3][0]["optimizer_state"]


def test_completed_jobs_reuse_results_and_gate_rejects_smoke(tmp_path):
    pytest.importorskip("gymnasium_robotics")
    c = base()
    c.goals = c.goals[:1]
    rows = run_baselines(c, tmp_path, explorations=("bounded",), smoke=True)
    again = run_baselines(c, tmp_path, explorations=("bounded",), smoke=True)
    assert rows == again
    with pytest.raises(ValueError):
        require_baseline(rows, exploration="bounded", architecture="plain_sac", goal_count=1)
    fig = plot_baselines(rows, tmp_path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    assert (tmp_path / "baseline-comparison.pdf").exists()
    c.training["gamma"] = 0.97
    with pytest.raises(ValueError, match="Source/config changed"):
        run_baselines(c, tmp_path, explorations=("bounded",), smoke=True)


def test_entropy_floor_and_ceiling_apply_after_updates(tmp_path):
    pytest.importorskip("gymnasium_robotics")
    c = configure_fetch_study(base()).smoke()
    c.goals = c.goals[:1]
    c.training.update(entropy_tuning_start=0, min_alpha=0.03, max_alpha=0.03, lr_ent_coef=0.1)
    rows = run_experiment(c, tmp_path / "run", modes=("scratch",), verbose=False)
    assert rows[0]["final_alpha"] == pytest.approx(0.03)
    losses = [
        json.loads(x)
        for x in next((tmp_path / "run").rglob("losses.jsonl")).read_text().splitlines()
    ]
    assert all(r["alpha"] == pytest.approx(0.03) for r in losses)


def test_plain_sac_inline_diagnostics_with_stub_camera(tmp_path, monkeypatch):
    pytest.importorskip("gymnasium_robotics")
    from tbtrl.environments.fetch import FixedStartFetch
    from tbtrl.experiments.diagnostics import NotebookDiagnostics
    from tbtrl.experiments.figure_style import FigureStyle

    monkeypatch.setattr(
        FixedStartFetch, "render", lambda self: np.full((120, 640, 3), 100, dtype=np.uint8)
    )
    c = configure_fetch_study(base()).smoke()
    c.goals = c.goals[:1]
    run_experiment(
        c,
        tmp_path / "run",
        modes=("scratch",),
        verbose=False,
        on_task_end=NotebookDiagnostics(
            display=False, rollout_episodes=1, style=FigureStyle(dpi=50, formats=("png",))
        ),
    )
    directory = tmp_path / "run/seed_42/scratch/task_0"
    assert (directory / "rollout-01.gif").exists()
    assert (directory / "learning.png").exists()
    assert not list(directory.glob("embeddings-critic*"))


def test_actor_choice_does_not_change_initial_factorised_critic():
    pytest.importorskip("gymnasium_robotics")
    from functools import partial

    from tbtrl.environments.registry import make_environment
    from tbtrl.experiments.runner import _models

    critics = []
    for architecture in ("factorised_critic", "factorised_actor_critic"):
        c = configure_fetch_study(base(), architecture=architecture).smoke()
        torch.manual_seed(123)
        factory = partial(make_environment, c.environment, **c.environment_options)
        critics.append(_models(c, factory, "cpu")[1].state_dict())
    for key in critics[0]:
        torch.testing.assert_close(critics[0][key], critics[1][key], rtol=0, atol=0)
