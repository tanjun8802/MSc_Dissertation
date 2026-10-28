"""Atomic, trusted-local task-boundary persistence shared by experiment runners."""

import hashlib
import json
import os
import platform
import random
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "task-boundary-v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_progress(path, records, *, endpoint=None):
    """Update the saved journal without dropping other completed methods on a crash."""
    path = Path(path)
    previous = json.loads(path.read_text()) if path.exists() else []

    def key(row):
        return row["seed"], row.get("method", row["mode"]), row["task_id"]

    merged = {
        key(row): row
        for row in previous
        if endpoint is None or row["mode"] != "recovery" or row.get("after_tasks") == endpoint
    }
    merged.update({key(row): row for row in records})
    atomic_json(path, list(merged.values()))


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_boundary(path, device="cpu"):
    # Full continuation includes Python replay/state objects. Only load checkpoints
    # produced in a trusted local run directory, never an untrusted model download.
    return torch.load(path, map_location=device, weights_only=False)


@contextmanager
def run_lock(output):
    import fcntl

    with (Path(output) / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is writing this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def rng_state():
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


def runtime_identity():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    packages = {}
    for name in (
        "numpy",
        "torch",
        "gymnasium",
        "gymnasium-robotics",
        "mujoco",
        "rdkit",
        "PyTDC",
        "made",
        "orb-models",
        "pymatgen",
        "smact",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return dict(
        source_sha256=digest.hexdigest(), python=platform.python_version(), packages=packages
    )


def validate_append(old, new):
    """Only the task suffix and matching new-task data fingerprints may grow."""
    old, new = json.loads(json.dumps(old)), json.loads(json.dumps(new))
    previous, requested = old.pop("tasks"), new.pop("tasks")
    if requested[: len(previous)] != previous:
        raise ValueError(
            "Only append tasks: existing task coordinates, definitions and order must match"
        )
    # MADE reference data are fingerprinted per chemical system. Existing systems
    # must stay byte-equivalent while new systems add entries.
    for value in (old, new):
        value.setdefault("metadata", {})
    before = old["metadata"].pop("dataset_sha256", {})
    after = new["metadata"].pop("dataset_sha256", {})
    if any(after.get(k) != v for k, v in before.items()):
        raise ValueError("Existing task dataset fingerprints changed")
    added_names = {t.get("name") for t in requested[len(previous) :] if isinstance(t, dict)}
    if set(after) - set(before) - added_names:
        raise ValueError("Dataset fingerprints may only be added for appended tasks")
    if old != new:
        raise ValueError(
            "Source/config changed: run configuration differs from saved manifest (or runtime changed); use a new run directory"
        )


def prepare_run(output, specification, tasks, *, resume):
    output = Path(output)
    path = output / "continuation.json"
    spec = json.loads(
        json.dumps(dict(protocol=PROTOCOL, **runtime_identity(), **specification, tasks=tasks))
    )
    if path.exists():
        if not resume:
            raise FileExistsError("Run exists; use resume=True")
        validate_append(json.loads(path.read_text()), spec)
    elif any(p.name != ".run.lock" for p in output.iterdir()):
        raise ValueError(
            "Legacy/incomplete directory has no task-boundary manifest; start a new run"
        )
    atomic_json(path, spec)
    return spec


def show_cached(directory, *, display=True):
    """Display saved per-task figures without training, evaluation or oracle calls."""
    if not display:
        return
    from IPython.display import SVG, Image
    from IPython.display import display as show

    directory = Path(directory)
    for path in sorted(directory.glob("*.png")):
        if not path.name.startswith("rollout-"):
            show(Image(filename=str(path)))
    for path in sorted(directory.glob("*.svg")):
        if not path.with_suffix(".png").exists():
            show(SVG(filename=str(path)))
    for path in sorted(directory.glob("rollout-*.gif")):
        show(Image(filename=str(path), format="gif"))
