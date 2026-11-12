"""The Maze notebook must resume saved protocols without mixing model versions."""

import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import tbtrl
from tbtrl.continuation import atomic_json, prepare_run
from tbtrl.experiments.config import load_config
from tbtrl.experiments.dqn_study import configure_k_replay
from tbtrl.experiments.runner import run_experiment

ROOT = Path(tbtrl.__file__).resolve().parents[2]
BEFORE = "b588031e3e9569452deab0af2a370a470d697dc92e5cd81ed80c15bdd30c24f2"


@pytest.fixture
def helpers(monkeypatch):
    notebook = json.loads((ROOT / "experiments/working/TBTRL_GridworldMaze_DQN.ipynb").read_text())
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "if RUN_EXPERIMENTS:" in "".join(c["source"])
    )
    tree = ast.parse(source)
    tree.body = [
        n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    # Use a synthetic audited source identity so this migration regression does
    # not depend on future unrelated edits to the package's source tree.
    from tbtrl import continuation

    identity = continuation.runtime_identity()
    identity["source_sha256"] = "15cfbdbdb1125b0788fa7d1f4ada6e76262561962e5830ff46eef5efab3ba87c"
    monkeypatch.setattr(continuation, "runtime_identity", lambda: dict(identity))
    namespace = {}
    exec(compile(tree, "notebook_resume_helpers", "exec"), namespace)
    return namespace


def study_config():
    config = configure_k_replay(
        load_config(ROOT / "configs/tbtrl_gridworldmaze_dqn.json").smoke(),
        max_tasks=1,
        sample_budget=4,
        candidate_capacity=3,
        refresh_updates=2,
        weighting="interference",
        capacity=32,
        recent_capacity=16,
        max_groups=2,
    )
    config.model.pop("embedding_constraint")
    config.training.update(sigreg_coef=0.01, goal_separation_coef=0.04)
    return config.validate()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_audited_fix_resumes_real_boundary_and_reuses_complete_results(tmp_path, helpers, device):
    config = study_config()
    output = tmp_path / "basis-interference"

    def interrupt(**kwargs):
        if kwargs["record"]["mode"] == "sequential":
            raise RuntimeError("interrupt after first boundary")

    with pytest.raises(RuntimeError, match="interrupt after first boundary"):
        run_experiment(
            config,
            output,
            device=device,
            modes=("sequential", "recovery"),
            on_task_end=interrupt,
            verbose=False,
        )
    manifest = output / "continuation.json"
    saved = json.loads(manifest.read_text())
    saved["source_sha256"] = BEFORE
    atomic_json(manifest, saved)
    checkpoint = output / "seed_42/sequential/task_0/checkpoint.pt"
    original = checkpoint.read_bytes()
    boundary = output / "seed_42/sequential/boundary.pt"
    original_boundary = boundary.read_bytes()
    modes = ("sequential", "recovery")
    assert not helpers["study_case_complete"](output, config, modes)
    helpers["prepare_study_resume"](output, config, device, modes)
    assert json.loads((output / "continuation.before-mps-fix.json").read_text()) == saved
    assert boundary.read_bytes() == original_boundary
    helpers["prepare_study_resume"](output, config, device, modes)  # Idempotent.
    rows = run_experiment(config, output, device=device, modes=modes, resume=True, verbose=False)
    assert checkpoint.read_bytes() == original
    assert len(rows) == len(config.goals) * 2
    assert helpers["study_case_complete"](output, config, modes)
    # Completed results can be read without approving another source version.
    saved["source_sha256"] = "unrelated-historical-version"
    atomic_json(manifest, saved)
    assert helpers["study_case_complete"](output, config, modes)
    extended = deepcopy(config)
    extended.goals.append([6, 3])
    assert not helpers["study_case_complete"](output, extended, modes)
    receipt = output / f"seed_42/recovery/after_{len(config.goals)}/task_0/complete.json"
    receipt.unlink()
    assert not helpers["study_case_complete"](output, config, modes)


def manifest_for(config, output):
    from dataclasses import asdict

    output.mkdir()
    settings = asdict(config)
    settings.pop("goals")
    saved = prepare_run(
        output,
        dict(kind="rl", config=settings, device="cpu", modes=["sequential"]),
        config.goals,
        resume=False,
    )
    saved["source_sha256"] = BEFORE
    atomic_json(output / "continuation.json", saved)
    return saved


def test_restores_saved_protocol_for_existing_and_new_control(tmp_path, helpers):
    config = study_config()
    output = tmp_path / "basis-interference"
    manifest_for(config, output)
    requested = configure_k_replay(config)
    scratch = configure_k_replay(config, max_tasks=0, sample_budget=0)
    configs = {"basis-interference": requested, "scratch": scratch}
    paths = {"basis-interference": output, "scratch": tmp_path / "scratch"}
    with pytest.raises(ValueError, match="protocol"):
        helpers["restore_saved_study_configs"](configs, paths)
    restored = helpers["restore_saved_study_configs"](configs, paths, allow_saved_protocol=True)
    assert restored["basis-interference"] == config
    assert restored["scratch"].model == config.model
    assert restored["scratch"].training["goal_separation_coef"] == 0.04
    assert restored["scratch"].training["sigreg_coef"] == 0.01
    assert restored["scratch"].training["k_replay"]["max_tasks"] == 0
    assert requested.model["embedding_constraint"] == "soft"
    assert not paths["scratch"].exists()
    requested.goals = list(reversed(requested.goals))
    with pytest.raises(ValueError, match="task order changed"):
        helpers["restore_saved_study_configs"](configs, paths)


@pytest.mark.parametrize("change", ["source", "config", "runtime", "device"])
def test_migration_rejects_unverified_changes_without_rewriting(tmp_path, helpers, change):
    config = study_config()
    output = tmp_path / "case"
    saved = manifest_for(config, output)
    device = "cpu"
    if change == "source":
        saved["source_sha256"] = "unknown"
    elif change == "config":
        config.training["gamma"] = 0.9
    elif change == "runtime":
        saved["python"] = "different"
    else:
        device = "mps"
    atomic_json(output / "continuation.json", saved)
    before = (output / "continuation.json").read_bytes()
    with pytest.raises(ValueError):
        helpers["prepare_study_resume"](output, config, device, ("sequential",))
    assert (output / "continuation.json").read_bytes() == before
    assert not (output / "continuation.before-mps-fix.json").exists()
