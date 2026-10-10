"""Regenerate publication figures from saved runs, without retraining."""

import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tbtrl.environments.registry import make_environment
from tbtrl.normalization import RunningMeanStd
from tbtrl.random import preserve_rng_state

from .config import ExperimentConfig
from .diagnostics import NotebookDiagnostics
from .figure_style import DEFAULT_STYLE, save_figure
from .plots import plot_comparison, plot_run
from .runner import _models


def _restore(checkpoint):
    config = ExperimentConfig(**checkpoint["config"]).validate()
    factory = partial(make_environment, config.environment, **config.environment_options)
    models = _models(config, factory, "cpu")
    result = SimpleNamespace()
    names = ("network", "target") if config.algorithm == "dqn" else ("actor", "critic", "target")
    for name, model in zip(names, models):
        model.load_state_dict(checkpoint[name])
        setattr(result, name, model.eval())
    if config.algorithm == "sac":
        for name in ("state_normalizer", "goal_normalizer"):
            state = checkpoint[name]
            normalizer = RunningMeanStd(tuple(state["mean"].shape), device="cpu")
            for key, value in state.items():
                getattr(normalizer, key).copy_(value)
            setattr(result, name, normalizer)
    return result, config, factory


def render_saved_run(
    output_dir, *, destination=None, style=DEFAULT_STYLE, display=False, spatial=True
):
    """Replot an existing run into publication/; leave metrics/checkpoints untouched.

    Spatial figures evaluate saved CPU models in separate seeded environments.
    They perform no training. Set spatial=False to render only saved numerical
    diagnostics. The function returns the destination path and preserves RNGs.
    """
    import matplotlib.pyplot as plt

    source = Path(output_dir)
    destination = Path(destination) if destination is not None else source / "publication"
    destination.mkdir(parents=True, exist_ok=True)
    records = json.loads((source / "summary.json").read_text())
    rendered = []
    with preserve_rng_state():
        for function, name in ((plot_run, "run-overview"), (plot_comparison, "cost-comparison")):
            fig = function(source, style=style, export=False)
            save_figure(fig, destination / name, style=style)
            if display:
                from IPython.display import display as show

                show(fig)
            plt.close(fig)
        for record in records:
            if record.get("recovery_performed") is False:
                continue
            relative = Path(f"seed_{record['seed']}") / record["mode"] / f"task_{record['task_id']}"
            directory = source / relative
            checkpoint = torch.load(
                directory / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            if checkpoint.get("format_version") != 1:
                raise ValueError("Unsupported checkpoint format.")
            if spatial:
                result, config, factory = _restore(checkpoint)
            else:
                result = SimpleNamespace()
                config, factory = ExperimentConfig(**checkpoint["config"]), None
            result.evaluations = record["learning_curve"]
            result.success_rates = record.get("success_curve", [])
            result.transfer_probes = record.get("transfer_probes", [])
            result.losses = [
                json.loads(line)
                for line in (directory / "losses.jsonl").read_text().splitlines()
                if line.strip()
            ]
            diagnostic_path = directory / "diagnostics.json"
            if diagnostic_path.exists():
                diagnostic = json.loads(diagnostic_path.read_text())
                for data in diagnostic["branches"].values():
                    for key, value in data.items():
                        if isinstance(value, list):
                            data[key] = np.asarray(value)
            else:
                ids = (
                    [r["task_id"] for r in result.transfer_probes[-1]["evaluation"]]
                    if result.transfer_probes
                    else [record["task_id"]]
                )
                diagnostic = dict(task_ids=ids, branches={}, weight_changes={})
            layout = None
            if spatial:
                env = factory(goal=config.goals[record["task_id"]])
                try:
                    if env.observation_space.shape == (2,):
                        layout = getattr(env.unwrapped, "maze", None)
                finally:
                    env.close()
            NotebookDiagnostics(
                display=display, rollout_episodes=config.eval_episodes, style=style
            ).render(
                result=result,
                config=config,
                record=record,
                diagnostic=diagnostic,
                directory=destination / relative,
                factory=factory,
                layout=None if layout is None else np.asarray(layout),
            )
            rendered.append(str(relative))
    (destination / "figure-manifest.json").write_text(
        json.dumps(
            dict(
                source=str(source.resolve()),
                style=asdict(style),
                spatial=spatial,
                rendered_tasks=rendered,
                note="Saved metrics and embeddings; spatial figures evaluate saved checkpoints on CPU without training.",
            ),
            indent=2,
        )
        + "\n"
    )
    return destination.resolve()
