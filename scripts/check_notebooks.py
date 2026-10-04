"""Execute supported notebooks with fresh kernels and isolated smoke outputs."""

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="tbtrl-notebooks-") as directory:
    work = Path(directory)
    kernel_dir = work / "kernels" / "tbtrl-validation"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "TBTRL validation",
                "language": "python",
                "env": {
                    "TBTRL_OUTPUT_ROOT": str(work / "runs"),
                    "MPLCONFIGDIR": str(work / "matplotlib"),
                    "MPLBACKEND": "Agg",
                    "IPYTHONDIR": str(work / "ipython"),
                },
            }
        )
    )
    os.environ["JUPYTER_PATH"] = str(work) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(work / "runtime")
    for path in sorted((root / "experiments/working").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="tbtrl-validation",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        print(f"Executed {path.name} successfully", flush=True)
