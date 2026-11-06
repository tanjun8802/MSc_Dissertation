"""Execute the self-contained Maze DQN study in a fresh kernel with real smoke updates."""

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="tbtrl-dqn-study-") as directory:
    work = Path(directory)
    kernel = work / "kernels" / "tbtrl-dqn-study"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            dict(
                argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                display_name="DQN study validation",
                language="python",
                env=dict(
                    PYTHONPATH=str(root / "src"),
                    TBTRL_OUTPUT_ROOT=str(work / "runs"),
                    MPLBACKEND="Agg",
                    MPLCONFIGDIR=str(work / "mpl"),
                    IPYTHONDIR=str(work / "ipython"),
                ),
            )
        )
    )
    os.environ["JUPYTER_PATH"] = str(work) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(work / "runtime")
    path = root / "experiments/working/TBTRL_GridworldMaze_DQN.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if "parameters" in cell.metadata.get("tags", []):
            cell.source = cell.source.replace("SMOKE = False", "SMOKE = True")
            cell.source += '\nDEVICE = "cpu"'
    NotebookClient(
        notebook,
        timeout=240,
        kernel_name="tbtrl-dqn-study",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    outputs = [
        output for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
    ]
    assert any("image/png" in out.get("data", {}) for out in outputs)
    assert any("Training task=" in out.get("text", "") for out in outputs)
    summaries = list((work / "runs").rglob("summary.json"))
    assert len(summaries) == 6  # Four selectors, one scratch reference, one fine-tuning control.
    for summary in summaries:
        rows = json.loads(summary.read_text())
        assert rows
        for row in rows:
            options = row["bounded_replay"]["options"]
            for probe in row["replay_diagnostics"]:
                assert len(probe["selected_tasks"]) <= options["max_tasks"]
                assert probe["replay_samples"] <= options["sample_budget"]
                assert probe["regularised_goals"] <= options["max_tasks"] + 1
    assert list((work / "runs").rglob("replay-allocation.png"))
    print(
        "Executed Maze DQN study: six conditions, acquisition/recovery, inline figures and bounded-update audits."
    )
