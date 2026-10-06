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
    cases = []
    for path in sorted((root / "experiments/working").glob("*.ipynb")):
        variants = (
            ["baseline", "factorised_only", "mlp_replay", "factorised_tbtrl"]
            if "FactorisedActor" in path.stem
            else [None]
        )
        cases.extend((path, variant) for variant in variants)
    for path, variant in cases:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        # Override only the in-memory test copy; committed notebooks use full budgets.
        for cell in notebook.cells:
            if "parameters" in cell.metadata.get("tags", []):
                assert "SMOKE = False" in cell.source
                cell.source = cell.source.replace("SMOKE = False", "SMOKE = True")
                cell.source += '\nDEVICE = "cpu"'
                if variant is not None:
                    cell.source = cell.source.replace(
                        'VARIANT = "factorised_tbtrl"', f'VARIANT = "{variant}"'
                    )

        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="tbtrl-validation",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        outputs = [
            out for cell in notebook.cells if cell.cell_type == "code" for out in cell.outputs
        ]
        assert any("Training task=" in out.get("text", "") for out in outputs)
        assert any("image/png" in out.get("data", {}) for out in outputs)
        if variant is not None:
            run = next((work / "runs").glob(f"*-{variant}-*"))
            diagnostics = list(run.rglob("diagnostics.json"))
            assert len(diagnostics) == 6
            if variant.startswith("factorised"):
                assert all("actor_mu" in json.loads(p.read_text())["branches"] for p in diagnostics)
            if variant in ("mlp_replay", "factorised_tbtrl"):
                assert len(list(run.rglob("actor-losses.png"))) == 6
                assert any("actor_replay_tasks" in out.get("text", "") for out in outputs)
        else:
            assert list((work / "runs").rglob("diagnostics.json"))
        print(
            f"Executed {path.name} ({variant or 'original'}) with training logs and inline per-goal plots",
            flush=True,
        )
