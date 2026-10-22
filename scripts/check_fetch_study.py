"""Run both Fetch study notebooks in fresh kernels with real physics and tiny budgets."""

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="tbtrl-fetch-study-") as directory:
    work = Path(directory)
    kernel = work / "kernels" / "fetch-study"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            dict(
                argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                display_name="Fetch study validation",
                language="python",
                env=dict(
                    PYTHONPATH=str(root / "src"),
                    TBTRL_OUTPUT_ROOT=str(work / "runs"),
                    MPLCONFIGDIR=str(work / "mpl"),
                    MPLBACKEND="Agg",
                    IPYTHONDIR=str(work / "ipython"),
                ),
            )
        )
    )
    os.environ["JUPYTER_PATH"] = str(work) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(work / "runtime")
    for path in sorted((root / "experiments/working").glob("TBTRL_Fetch*Baseline_Study.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        # Smoke is committed as the default. Executed outputs stay outside source notebooks.
        notebook.cells.append(
            nbformat.v4.new_code_cell("""
assert SMOKE
assert len(screen) == 3 * len(BASE.goals[:2])
assert len(confirmation) == 4 * len(BASE.goals[:2])
assert all(r["post_eval_episodes"] == 1 for r in screen)
assert any("hold_success_rate" in r["evaluation"][0] for r in screen)
print("PASS: exploration controls, architecture controls and sequential/recovery jobs")
""")
        )
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="fetch-study",
            resources={"metadata": {"path": str(root / "experiments/working")}},
        ).execute()
        outputs = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs]
        assert any("PASS: exploration" in o.get("text", "") for o in outputs)
        # Agg canvases may not emit rich output; exported scientific figures are required.
        assert list((work / "runs").rglob("baseline-comparison.pdf"))
        if destination := os.environ.get("TBTRL_FETCH_STUDY_ARTIFACTS"):
            import shutil

            target = Path(destination)
            target.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, target / path.name)
            shutil.copytree(work / "runs", target / "runs", dirs_exist_ok=True)
        print(f"PASS: {path.name} in a fresh kernel", flush=True)
