# ExORL and OGBench setup

## What changed

- Added `/home/runner/work/MSc_Dissertation/MSc_Dissertation/src/benchmarks/exorl.py` with ExORL dataset download, episode loading, replay-buffer conversion, and a Gymnasium-style adapter for the ExORL tasks that map directly to `dm_control` suite tasks.
- Added `/home/runner/work/MSc_Dissertation/MSc_Dissertation/src/benchmarks/ogbench.py` with lightweight wrappers around the upstream OGBench API plus conversion helpers that load OGBench datasets into the repo's `TrajectoryReplayBuffer` format.
- Added `/home/runner/work/MSc_Dissertation/MSc_Dissertation/src/benchmarks/cli.py` so both benchmarks can be smoke-tested from the command line with `python -m src.benchmarks.cli ...`.
- Added `/home/runner/work/MSc_Dissertation/MSc_Dissertation/tests/test_benchmarks.py` for synthetic benchmark-data tests that validate the repo-side conversion logic without needing large downloads.
- Updated `/home/runner/work/MSc_Dissertation/MSc_Dissertation/src/utils.py` and the existing notebooks to use the replay-buffer key name `goals` consistently.
- Added a benchmark dependency group in `/home/runner/work/MSc_Dissertation/MSc_Dissertation/pyproject.toml` so benchmark packages can be installed only when needed.

## Files to know

- `/home/runner/work/MSc_Dissertation/MSc_Dissertation/src/benchmarks/exorl.py`
- `/home/runner/work/MSc_Dissertation/MSc_Dissertation/src/benchmarks/ogbench.py`
- `/home/runner/work/MSc_Dissertation/MSc_Dissertation/src/benchmarks/cli.py`
- `/home/runner/work/MSc_Dissertation/MSc_Dissertation/docs/benchmark_setup.md`

## Install

### 1. Base repo dependencies

```bash
cd /home/runner/work/MSc_Dissertation/MSc_Dissertation
uv sync
```

### 2. Benchmark dependencies

```bash
cd /home/runner/work/MSc_Dissertation/MSc_Dissertation
uv sync --group benchmarks
```

If `dm_control` rendering complains on a headless machine, try:

```bash
export MUJOCO_GL=egl
```

If `dm_control` still fails with EGL/GL import errors, install the usual MuJoCo rendering libraries on the machine first:

```bash
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 unzip
```

## How to test OGBench

### Smoke-test the environment only

```bash
cd /home/runner/work/MSc_Dissertation/MSc_Dissertation
uv run --group benchmarks python -m src.benchmarks.cli ogbench env pointmaze-medium-navigate-v0 --task-id 1
```

### Download a dataset

```bash
cd /home/runner/work/MSc_Dissertation/MSc_Dissertation
uv run --group benchmarks python -m src.benchmarks.cli ogbench download pointmaze-medium-navigate-v0
```

### Inspect train/validation dataset shapes

```bash
cd /home/runner/work/MSc_Dissertation/MSc_Dissertation
uv run --group benchmarks python -m src.benchmarks.cli ogbench dataset pointmaze-medium-navigate-v0
```

### Use OGBench in Python

```python
from src.benchmarks import make_ogbench_env, load_ogbench_datasets

env = make_ogbench_env("pointmaze-medium-navigate-v0")
obs, info = env.reset(options={"task_id": 1})

env, train_dataset, val_dataset = load_ogbench_datasets("pointmaze-medium-navigate-v0")
```

## How to test ExORL

### Download a dataset

```bash
cd /home/runner/work/MSc_Dissertation/MSc_Dissertation
uv run python -m src.benchmarks.cli exorl download walker proto
```

This downloads the replay archive into the default repo-side benchmark cache:

- `~/.msc_dissertation/benchmarks/exorl/walker/proto/`

### Inspect the downloaded replay files

```bash
cd /home/runner/work/MSc_Dissertation/MSc_Dissertation
uv run python -m src.benchmarks.cli exorl inspect walker proto
```

### Smoke-test a suite-backed ExORL environment

These are the ExORL tasks currently wired directly in this repo:

- `cartpole_balance`
- `cartpole_balance_sparse`
- `cartpole_swingup`
- `cartpole_swingup_sparse`
- `cheetah_run`
- `quadruped_walk`
- `quadruped_run`
- `walker_stand`
- `walker_walk`
- `walker_run`

Run one with:

```bash
cd /home/runner/work/MSc_Dissertation/MSc_Dissertation
uv run --group benchmarks python -m src.benchmarks.cli exorl env walker_walk --seed 0
```

### Use ExORL in Python

```python
from src.benchmarks import download_exorl_dataset, get_exorl_replay_dir, load_exorl_dataset_into_buffer
from src.utils import TrajectoryReplayBuffer

replay_dir = download_exorl_dataset("walker", "proto")
buffer = TrajectoryReplayBuffer(capacity=100000, obs_dim=24, action_dim=6)
load_exorl_dataset_into_buffer(buffer, replay_dir, max_episodes=10)
```

## Notes and limits

- OGBench is upstream-installed through `ogbench` and `dm_control`; this repo only adds repo-friendly wrappers around those APIs.
- ExORL is not a PyPI package, so this repo integrates the public replay format directly instead of vendoring the full upstream training code.
- Direct ExORL environment support in this repo currently covers the tasks that map cleanly to `dm_control` suite tasks. The custom ExORL tasks such as `cheetah_run_backward`, `jaco_*`, and `point_mass_maze_*` still need the upstream ExORL custom task code if you want to recreate those environments exactly.
- The new repo-side tests do not download benchmark data; they only verify the local adapters and dataset-conversion logic.
