"""Notebook continuation must not silently replace the requested study protocol."""

import ast
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from tbtrl.experiments.config import ExperimentConfig, load_config
from tbtrl.experiments.dqn_study import configure_k_replay

ROOT = Path(__file__).resolve().parents[1]


def notebook_restore():
    notebook = json.loads((ROOT / "experiments/working/TBTRL_GridworldMaze_DQN.ipynb").read_text())
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if "def restore_saved_study_configs" in "".join(c.get("source", []))
    )
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "restore_saved_study_configs"
    )
    namespace = dict(json=json, deepcopy=deepcopy, ExperimentConfig=ExperimentConfig)
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "notebook-restore", "exec"), namespace
    )
    return namespace["restore_saved_study_configs"]


def prepare(tmp_path):
    requested = configure_k_replay(load_config(ROOT / "configs/tbtrl_gridworldmaze_dqn.json"))
    historical = deepcopy(requested)
    historical.model.pop("embedding_constraint")  # historical default = hard projection
    historical.training.update(sigreg_coef=0.01, goal_separation_coef=0.04)
    config = asdict(historical)
    config.pop("goals")
    (tmp_path / "continuation.json").write_text(
        json.dumps(dict(config=config, tasks=historical.goals))
    )
    return requested, historical


def test_notebook_rejects_silent_protocol_restore(tmp_path):
    requested, _ = prepare(tmp_path)
    with pytest.raises(ValueError, match="protocol"):
        notebook_restore()({"basis": requested}, {"basis": tmp_path})


def test_notebook_can_explicitly_continue_historical_protocol(tmp_path):
    requested, _ = prepare(tmp_path)
    restored = notebook_restore()(
        {"basis": requested}, {"basis": tmp_path}, allow_saved_protocol=True
    )
    assert restored["basis"].model.get("embedding_constraint", "projected") == "projected"
    assert restored["basis"].training["sigreg_coef"] == 0.01
    assert requested.model["embedding_constraint"] == "soft"


def test_matching_protocol_allows_appended_goals(tmp_path):
    _, historical = prepare(tmp_path)
    historical.goals.append([1.0, 1.0])
    restored = notebook_restore()({"basis": historical}, {"basis": tmp_path})
    assert restored["basis"].goals == historical.goals
