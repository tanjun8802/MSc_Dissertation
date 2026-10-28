"""Execute supported notebooks with fresh kernels and isolated smoke outputs."""

import importlib.util
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
                    "PYTHONPATH": str(root / "src"),
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
        if path.stem.endswith("Baseline_Study"):
            print(f"{path.name}: use scripts/check_fetch_study.py", flush=True)
            continue
        if path.stem.startswith(("TBTRL_PMO", "TBTRL_MADE")):
            print(
                f"{path.name}: use scripts/check_discovery_notebooks.py (optional dependencies)",
                flush=True,
            )
            continue
        if path.stem.startswith("TBTRL_Fetch") and (
            importlib.util.find_spec("gymnasium_robotics") is None
            or os.environ.get("TBTRL_SKIP_ROBOTICS_RENDER") == "1"
        ):
            print(
                f"Skipped {path.name}: robotics dependencies or rendering unavailable", flush=True
            )
            continue
        variants = (
            ["baseline", "factorised_only", "mlp_replay", "factorised_tbtrl"]
            if "FactorisedActor" in path.stem
            else [None]
        )
        if path.stem.endswith("SAC_GlobalReplay"):
            cases.append((path, None, "global_replay"))
        elif path.stem.endswith("SAC_Transfer"):
            cases.extend(
                (path, None, experiment)
                for experiment in ("temperature_carry", "scaled_regularization", "policy_retention")
            )
        else:
            cases.extend((path, variant, None) for variant in variants)
    for path, variant, experiment in cases:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        # Override only the in-memory test copy; committed notebooks use full budgets.
        for cell in notebook.cells:
            if "parameters" in cell.metadata.get("tags", []):
                assert "SMOKE = False" in cell.source
                cell.source = cell.source.replace("SMOKE = False", "SMOKE = True")
                cell.source += '\nDEVICE = "cpu"'
                if (
                    path.stem.startswith("TBTRL_Fetch")
                    and os.environ.get("TBTRL_STUB_ROBOTICS_RENDER") == "1"
                ):
                    cell.source += "\nimport numpy as np\nfrom tbtrl.environments.fetch import FixedStartFetch\nFixedStartFetch.render = lambda self: np.full((120, 640, 3), 100, dtype=np.uint8)"
                cell.source += '\nfrom tbtrl.experiments.figure_style import FigureStyle\n_validation_style = FigureStyle(dpi=80, formats=("png",))'
                if experiment is not None:
                    cell.source = cell.source.replace(
                        'EXPERIMENT = "temperature_carry"', f'EXPERIMENT = "{experiment}"'
                    )
                if variant is not None:
                    cell.source = cell.source.replace(
                        'VARIANT = "factorised_tbtrl"', f'VARIANT = "{variant}"'
                    )
            if cell.cell_type == "code":
                # Full-resolution vector/raster exports have dedicated tests. Keep
                # all fresh-kernel executions light without changing notebook files.
                cell.source = (
                    cell.source.replace(
                        "NotebookDiagnostics(", "NotebookDiagnostics(style=_validation_style, "
                    )
                    .replace("plot_run(output)", "plot_run(output, style=_validation_style)")
                    .replace(
                        "plot_comparison(output)",
                        "plot_comparison(output, style=_validation_style)",
                    )
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
        if path.stem.startswith("TBTRL_Fetch"):
            assert any("image/gif" in out.get("data", {}) for out in outputs)
            assert any("fixed start verified" in out.get("text", "") for out in outputs)
        if experiment is not None:
            run = next((work / "runs").glob(f"*-{experiment}-factorised_tbtrl-*"))
            probes = list(run.rglob("transfer-probes.json"))
            assert len(probes) == 6
            assert all(json.loads(p.read_text())[0]["step"] == 0 for p in probes)
            assert len(list(run.rglob("transfer-probes.png"))) == 6
            assert len(list(run.rglob("actor-gradients.png"))) == 6
            assert any("Transfer probe task=" in out.get("text", "") for out in outputs)
        elif variant is not None:
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
            f"Executed {path.name} ({experiment or variant or 'original'}) with training logs and inline per-goal plots",
            flush=True,
        )
