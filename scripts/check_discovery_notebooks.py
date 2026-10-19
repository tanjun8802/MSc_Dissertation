"""Execute both optional discovery notebooks in fresh kernels, using native smoke backends.

Install the optional dependencies first (docs/discovery-benchmarks.md).
No credentials, ORB model download or real Materials Project queries are used.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="tbtrl-discovery-") as directory:
    work = Path(directory)
    kernel_dir = work / "kernels" / "tbtrl-discovery"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "TBTRL discovery validation",
                "language": "python",
                "env": {
                    "TBTRL_OUTPUT_ROOT": str(work / "runs"),
                    "MPLCONFIGDIR": str(work / "mpl"),
                    "IPYTHONDIR": str(work / "ipython"),
                    "PYTHONPATH": str(root / "src"),
                },
            }
        )
    )
    os.environ["JUPYTER_PATH"] = str(work) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(work / "runtime")
    for benchmark in ("PMO", "MADE"):
        path = next((root / "experiments/working").glob(f"TBTRL_{benchmark}_*.ipynb"))
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if "parameters" in cell.metadata.get("tags", []):
                cell.source = cell.source.replace("SMOKE = False", "SMOKE = True")
                cell.source += '\nDEVICE = "cpu"\nfrom tbtrl.experiments.figure_style import FigureStyle\n_validation_style = FigureStyle(dpi=80, formats=("png",))'
            if cell.cell_type == "code":
                cell.source = cell.source.replace(
                    "NotebookDiagnostics(",
                    "NotebookDiagnostics(style=_validation_style, benchmark=",
                )
                cell.source = cell.source.replace(
                    f'plot_comparison(records, "{benchmark.lower()}", output)',
                    f'plot_comparison(records, "{benchmark.lower()}", output, style=_validation_style)',
                )
        NotebookClient(
            notebook,
            timeout=300,
            kernel_name="tbtrl-discovery",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
        assert any("Training task=" in o.get("text", "") for o in outputs)
        assert any("image/png" in o.get("data", {}) for o in outputs)
        runs = list((work / "runs").glob(f"{benchmark.lower()}-*"))
        records = json.loads((runs[0] / "summary.json").read_text())
        assert len(records) == 4 and all(r["losses"] for r in records)
        assert records[0]["history"] == records[2]["history"]
        assert any(loss["replay"] > 0 for loss in records[-1]["losses"])
        if os.environ.get("TBTRL_DISCOVERY_ARTIFACTS"):
            destination = Path(os.environ["TBTRL_DISCOVERY_ARTIFACTS"]) / benchmark.lower()
            destination.mkdir(parents=True, exist_ok=True)
            for figure in runs[0].rglob("*.png"):
                relative = figure.relative_to(runs[0])
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(figure, target)
        print(
            f"Executed {path.name}: native smoke environment, both modes, inline plots, replay updates",
            flush=True,
        )
