"""Execute the separate fixed-budget/recovery study in a fresh smoke kernel."""

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="tbtrl-dqn-followup-") as directory:
    work = Path(directory)
    kernel = work / "kernels" / "tbtrl-followup"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            dict(
                argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                display_name="DQN follow-up validation",
                language="python",
                env=dict(
                    PYTHONPATH=str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
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
    path = root / "experiments/working/TBTRL_GridworldMaze_DQN_Followup.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if "parameters" in cell.metadata.get("tags", []):
            cell.source = cell.source.replace("SMOKE = False", "SMOKE = True")
            cell.source += '\nDEVICE = "cpu"'
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="tbtrl-followup",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    summaries = list((work / "runs").rglob("summary.json"))
    assert len(summaries) == 16  # Six acquisitions, two probes for five sequential sources.
    for summary in summaries:
        rows = json.loads(summary.read_text())
        assert len(rows) == 3
        manifest = json.loads((summary.parent / "manifest.json").read_text())
        assert manifest["config"]["training"]["sigreg_coef"] == 0.01
        for row in rows:
            if row["mode"] != "recovery":
                assert row["steps"] == 24
            elif row["recovery_performed"]:
                expected = (
                    0
                    if manifest["recovery_protocol"] == "no-replay"
                    else manifest["config"]["training"]["k_replay"]["max_tasks"]
                )
                assert all(len(d["selected_tasks"]) <= expected for d in row["replay_diagnostics"])
                assert len(row["initial_evaluation"]) == 3
    assert list((work / "runs").rglob("fixed-budget-acquisition.png"))
    assert list((work / "runs").rglob("recovery-protocols.png"))
    assert len(list((work / "runs").rglob("outcomes.csv"))) == 2
    # Analysis-only execution must not train or rewrite acquisition summaries.
    original = {p: p.read_bytes() for p in summaries}
    for cell in notebook.cells:
        if "parameters" in cell.metadata.get("tags", []):
            cell.source = cell.source.replace("RUN_EXPERIMENTS = True", "RUN_EXPERIMENTS = False")
    # Current parameters and a partially completed saved seed list must not
    # prevent analysis of the saved seed-42 results.
    for cell in notebook.cells:
        if "parameters" in cell.metadata.get("tags", []):
            cell.source += "\nSEEDS = [999]"
    detail_manifest = next((work / "runs").glob("*/smoke/acquisition/basis-k2/manifest.json"))
    saved = json.loads(detail_manifest.read_text())
    saved["config"]["seeds"].append(123)
    detail_manifest.write_text(json.dumps(saved))
    client.execute()
    assert all(p.read_bytes() == data for p, data in original.items())
    print(
        "Follow-up notebook passed: fixed budgets, two recovery protocols, exports and analysis-only reload."
    )
