"""Execute the new MADE study notebook in a fresh kernel with native offline oracles."""

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="tbtrl-made-study-") as directory:
    work = Path(directory)
    kernel = work / "kernels" / "made-study"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "MADE study validation",
                "language": "python",
                "env": {
                    "PYTHONPATH": str(root / "src"),
                    "TBTRL_OUTPUT_ROOT": str(work / "runs"),
                    "MPLCONFIGDIR": str(work / "mpl"),
                    "XDG_CACHE_HOME": str(work / "cache"),
                    "IPYTHONDIR": str(work / "ipython"),
                },
            }
        )
    )
    os.environ["JUPYTER_PATH"] = str(work) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(work / "runtime")
    notebook = nbformat.read(
        root / "experiments/working/TBTRL_MADE_Transfer_Study.ipynb", as_version=4
    )
    for cell in notebook.cells:
        if "parameters" in cell.metadata.get("tags", []):
            cell.source = cell.source.replace("SMOKE = False", "SMOKE = True")
            cell.source += '\nDEVICE = "cpu"\nRUN_RETENTION = RUN_RECOVERY = True\nfrom tbtrl.experiments.figure_style import FigureStyle\nSTYLE = FigureStyle(dpi=60, formats=("png",))'
    notebook.cells.append(
        nbformat.v4.new_code_cell("""
# A completed rerun must not construct ANY native environment or spend oracle calls.
def forbidden_factory(*args, **kwargs):
    raise AssertionError("Completed job attempted another environment/oracle call")
again = run_study(TASKS, forbidden_factory, output, config=TRAINING, seeds=SEEDS,
                 methods=METHODS, device=DEVICE, resume=True, metadata=metadata, on_task_end=diagnostics)
assert len(again) == 4
assert len(run_retention(output, forbidden_factory, campaigns=EVALUATION_CAMPAIGNS)) == 5
_, again_probes = run_recovery(output, forbidden_factory, campaigns=RECOVERY_CAMPAIGNS,
                              evaluation_campaigns=EVALUATION_CAMPAIGNS)
assert len(again_probes) == 4
assert any(v["replay"] > 0 for v in records[-1]["losses"])
print("PASS: native study, frozen retention, isolated recovery and zero-call resume")
""")
    )
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="made-study",
        resources={"metadata": {"path": str(root / "experiments/working")}},
    )
    client.execute()
    outputs = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs]
    assert any("image/png" in o.get("data", {}) for o in outputs)
    assert any("PASS: native study" in o.get("text", "") for o in outputs)
    assert not any("FIRE:" in o.get("text", "") for o in outputs)
    if os.environ.get("TBTRL_DISCOVERY_ARTIFACTS"):
        import shutil

        destination = Path(os.environ["TBTRL_DISCOVERY_ARTIFACTS"]) / "made-study"
        shutil.copytree(work / "runs", destination, dirs_exist_ok=True)
        nbformat.write(notebook, destination / "executed-study.ipynb")
    print("PASS: fresh-kernel MADE study notebook, inline figures and resume", flush=True)
