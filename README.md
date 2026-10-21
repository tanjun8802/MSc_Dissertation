# TBTRL: transfer reinforcement learning

Research code for an Imperial College London MSc dissertation on transferring
factorised value representations between goal-conditioned RL tasks.

The supported experiments are **FourRooms DQN**, **Gridworld Maze DQN**, and
**Gridworld Maze SAC**, an experimental **factorised SAC actor** variant, and
five-goal **FetchReach / FetchPickAndPlace SAC** runs. They compare fresh training, sequential transfer,
retention on earlier tasks, and independent recovery from the final transferred
model. The update rules come from the working dissertation trainers.
See [the method](docs/method.md) for objectives and task-boundary behaviour.

## Install and run

Use Python 3.12 and [uv](https://docs.astral.sh/uv/) for the locked environment.

```bash
git clone https://github.com/tanjun8802/MSc_Dissertation.git
cd MSc_Dissertation
uv sync --locked --group dev

# Small CPU runs: two goals, real updates, replay, evaluation and recovery.
uv run tbtrl --config configs/tbtrl_gridworldmaze_dqn.json --output runs/dqn-smoke --smoke
uv run tbtrl --config configs/tbtrl_gridworldmaze_sac.json --output runs/sac-smoke --smoke
```

Each output directory must be new. Remove `--smoke` to use the research budget
in the configuration. Full runs can be long; SAC uses up to 500,000 environment
steps per task. Select `--device cuda` or `--device mps` explicitly if needed;
the default is CPU. Use `--modes sequential` to run only forward transfer, or
`--modes scratch sequential` to omit recovery.

For notebooks:

```bash
uv sync --locked --extra notebooks --group dev
uv run jupyter lab experiments/working
```

Select the environment's Python kernel. The eight maze/robotics notebooks default to full research training budgets;
set `SMOKE = True` only for a short execution check. They invoke the same runner as the CLI,
and use the shared trainer code. The two original Maze result notebooks are preserved as committed.

To test the actor extension, open
[`TBTRL_GridworldMaze_SAC_FactorisedActor.ipynb`](experiments/working/TBTRL_GridworldMaze_SAC_FactorisedActor.ipynb).
Its `VARIANT` selector supports `baseline`, `factorised_only`, `mlp_replay`, and
`factorised_tbtrl` (default). Logs and actor/critic diagnostic plots appear in the
training cell after each goal. Full budgets and critic settings match the original
Maze SAC configuration. See [the actor objective and experiment guide](docs/factorised-actor.md).

For the follow-up to the saved actor results, use
[`TBTRL_GridworldMaze_SAC_Transfer.ipynb`](experiments/working/TBTRL_GridworldMaze_SAC_Transfer.ipynb).
It defaults to matching initial temperature and carrying learned alpha between tasks,
with step-zero/early retention probes and actor gradient plots. Compare `matched_reset`
against `temperature_carry` first; covariance scaling and policy KL are separate optional
presets. See [the controlled experiment guide](docs/transfer-experiments.md).

For a fixed replay budget across tasks, use
[`TBTRL_GridworldMaze_SAC_GlobalReplay.ipynb`](experiments/working/TBTRL_GridworldMaze_SAC_GlobalReplay.ipynb).
It includes recent experience in the global limit, shares archive space by goal
similarity, relabels SAC rewards/termination, and reports allocated replay bytes.
See [the memory policy and matched control](docs/global-replay.md).

For robotics, use the separate
[`TBTRL_FetchReach_SAC.ipynb`](experiments/working/TBTRL_FetchReach_SAC.ipynb) and
[`TBTRL_FetchPickAndPlace_SAC.ipynb`](experiments/working/TBTRL_FetchPickAndPlace_SAC.ipynb)
notebooks. Each starts with five fixed goals, dense rewards, no HER, and the
latest factorised actor/global replay setup. The arm starts are fixed across episodes and goals; PickAndPlace samples a new
block position each episode within the native table region. Numerical plots use the same shared renderer; inline
Fetch GIFs replace maze maps. Install with
`uv sync --locked --group dev --extra notebooks --extra robotics`.
See [Fetch setup, budgets and reset semantics](docs/fetch-experiments.md).

Two additional notebooks apply candidate-action TBTRL to scientific discovery:
[`TBTRL_PMO_Molecular_Optimisation.ipynb`](experiments/working/TBTRL_PMO_Molecular_Optimisation.ipynb)
and [`TBTRL_MADE_Materials_Discovery.ipynb`](experiments/working/TBTRL_MADE_Materials_Discovery.ipynb).
The [controlled MADE study](experiments/working/TBTRL_MADE_Transfer_Study.ipynb)
adds matched baselines/ablations, campaign checkpoints, held-out retention and
independent recovery runs. See the [study guide](docs/discovery-benchmarks.md#controlled-made-study-and-resuming)
for resume limits and oracle budgets.
Each compares scratch with sequential transfer across five tasks, with inline
training diagnostics, domain-specific figures and explicit oracle budgets.
They preserve the native molecular-edit/materials-discovery mechanisms and use
optional dependencies plus a separate candidate-aware trainer. See
[installation, preserved semantics and experiment interpretation](docs/discovery-benchmarks.md).

## Supported configurations

| Configuration | Actions | Task memory |
| --- | --- | --- |
| `configs/tbtrl_fourrooms_dqn.json` | Discrete | Separate replay buffers per task |
| `configs/tbtrl_gridworldmaze_dqn.json` | Discrete | Goal-similarity buffer sharing, with reward relabelling |
| `configs/tbtrl_gridworldmaze_sac.json` | Continuous | Separate task buffers and independent twin critics |
| `configs/tbtrl_gridworldmaze_sac_factorised_actor.json` | Continuous | Factorised actor with policy replay and representation penalties |
| `configs/tbtrl_fetch_reach_sac.json` | Continuous | Factorised actor and fixed-budget global replay; five reach goals |
| `configs/tbtrl_fetch_pick_and_place_sac.json` | Continuous | Factorised actor and fixed-budget global replay; five lifting goals |

Layouts, goals, model sizes and loss settings are explicit. The three original configurations
retain the working notebooks' settings: FourRooms uses seeds 42, 123 and 456;
Maze DQN/SAC use seed 42. Trainer settings, model dimensions, layouts and goals
are checked directly against the original notebook code in CI;
scientific comparisons should use additional seeds and task orders.

## Repository layout

```text
configs/                    Experiment settings
experiments/working/        Ten notebooks (including two optional discovery benchmarks)
src/tbtrl/
  environments/             Gymnasium environments, layouts and registration
  models/                   Factorised DQN and SAC models
  training/                 DQN/SAC update loops and named results
  replay.py                 Transition buffers and task memory
  losses.py                 Representation regularisation
  evaluation.py             Seeded policy evaluation
  experiments/              Configs, orchestration, plotting and checkpoints
tests/                      Replay, algorithm and experiment regression tests
scripts/                    Notebook and migration validation
docs/                       Method, migration and environment extension guide
```

New environments enter through the [environment contract](docs/environments.md),
without adding environment-specific branches to the trainers.

## Outputs and evaluation

Each run streams training progress and saves the same messages in `training.log`,
along with `manifest.json` and `summary.json`. Notebook runs display diagnostics
immediately after every trained goal: learning/loss curves, Q and policy maps,
rollouts, raw embedding norms, cosine matrices, covariance spectra, drift and
weight changes. Numeric diagnostics and 300 dpi PNG/PDF/SVG figures are saved
beside each checkpoint. All notebooks share the same plotting conventions;
see [research figures](docs/visualisations.md) for styles, statistical labels and
regenerating publication figures from saved runs without retraining.
See [the notebook parity audit](docs/notebook-parity.md) for exact settings and
legacy behavior that required repair.

 Task directories contain
`metrics.json`, sampled `losses.jsonl`, and `checkpoint.pt`. SAC tasks already
above the recovery threshold save a skipped-recovery record without retraining.
The manifest records configuration, seeds, dependency versions, device, Git
revision and whether the checkout was modified. Checkpoints also contain each
task's effective settings.

Unsolved tasks retain `steps_to_threshold: null`; they are included in plots.
Every sequential stage is evaluated on all goals to inspect retention and
transfer to future goals. Smoke checks establish execution and update
correctness; they do not establish convergence or reproduce dissertation results.

```python
from tbtrl.experiments.checkpoints import load_policy
from tbtrl.experiments.plots import plot_run

policy = load_policy("runs/dqn-smoke/seed_42/sequential/task_1/checkpoint.pt")
action = policy([1.0, 1.0])
figure = plot_run("runs/dqn-smoke")
```

Checkpoints support evaluation and model transfer. They do **not** contain
optimiser/RNG/replay state for exact mid-task resumption. Generated artifacts
are ignored by Git.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/check_notebooks.py
uv run python -m build

# Requires the pre-refactor commit in Git history (a full clone or fetch).
uv run python scripts/verify_legacy_equivalence.py
uv run python scripts/verify_notebook_configs.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [migration notes](docs/migration.md).
Historical sandbox experiments and reference PDFs remain in Git history.
This cleanup does not select or grant an open-source licence; a licence and
attribution review are still needed before a formal open-source release.
