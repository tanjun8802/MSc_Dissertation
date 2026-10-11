# Fetch experiments

Open `experiments/working/TBTRL_FetchReach_SAC.ipynb` or
`experiments/working/TBTRL_FetchPickAndPlace_SAC.ipynb`. Each loads its own JSON
configuration, checks the fixed physical reset, and runs scratch, sequential
transfer and independent recovery across five ordered goals. Logs, actor/critic
diagnostics and simulator GIFs appear directly in the training cell after each
trained goal. Summary figures use the existing shared plotting functions.

## Installation and rendering

```bash
uv sync --locked --group dev --extra notebooks --extra robotics
uv run --extra notebooks --extra robotics jupyter lab experiments/working
```

Use this environment's Python kernel, restarting after installing simulator
dependencies. The robotics extra pins Gymnasium Robotics 1.4.2 and MuJoCo 3.3.x:
newer MuJoCo versions changed bindings used by this Robotics release. Core maze
experiments do not require robotics dependencies. Run manifests record both
simulator versions.

Native rendering needs a working OpenGL context. On headless Linux, install EGL
and Mesa and set `MUJOCO_GL=egl` **before importing MuJoCo**; Xvfb with GLFW is an
alternative. On macOS, use the normal local notebook session with desktop
graphics access. Physics can work even when a sandbox cannot open graphics.
The CLI can run training without graphical callbacks:

```bash
uv run --extra robotics tbtrl --config configs/tbtrl_fetch_reach_sac.json --output runs/fetch-reach-smoke --smoke
uv run --extra robotics tbtrl --config configs/tbtrl_fetch_pick_and_place_sac.json --output runs/fetch-pick-smoke --smoke
```

## Protocol and budgets

| Setting | Both notebooks |
| --- | --- |
| Tasks | Five configured 3D target positions, in metres |
| Training seeds | `[42]`, editable independently of the physical reset |
| Physical reset seed | `initial_state_seed=42` for every reset |
| Reward | Native dense reward, negative Euclidean goal distance |
| HER | Disabled; no achieved/future-goal resampling |
| Forward training | Up to 500,000 steps per goal; 10,000 warm-up steps |
| Independent recovery | Up to 100,000 steps per goal |
| Early stopping | Success rate 1.0 for three evaluations, every 10,000 steps |
| Episode horizon | 50 steps, with native continuing-task semantics |
| Evaluation episodes | Eight; one separately rendered episode per goal |
| Replay budget | 100,000 transitions including a 50,000 recent partition |
| Archive policy | Keep final 40%, similarity threshold 0.8, up to 16 groups |
| Networks | Factorised actor and twin critics, hidden width 256, latent size 16 |

`SMOKE=False` is the committed default. `SMOKE=True` deliberately reduces the
task count and budgets for execution checks, and does not measure learning.
These configs retain the latest transfer objectives, carried entropy temperature
and global replay policy. Maze-specific Q-target clipping is removed because
Fetch has different reward magnitudes. Trainer, actor and replay algorithms are
unchanged. The old SB3/sparse/HER sandbox used a different method and budget;
these notebooks are not a replication of that sandbox.

Reach uses a 10-dimensional flat state and PickAndPlace a 25-dimensional state,
with four bounded actions. The desired goal is passed separately to the existing
factorised networks. For Reach the achieved goal is gripper position; for
PickAndPlace it is object position. All five PickAndPlace targets require lifting.
Full learning performance still needs to be measured, especially for discovering
grasping without HER.

## Fixed starts and replay rewards

The adapter resets the native simulator with the configured physical seed every
time, then sets the requested target. This fixes both robot and object state
across goals, episodes, training seeds, recovery, evaluation and GIF recording.
Learning seeds still control exploration, replay and model initialisation. The
notebook verifies the entire observation after movement and across goals; tests
also compare simulator positions, velocities, controls and mocap state.

Fetch does not terminate when it reaches the goal. Its native time limit produces
truncation, and the existing trainer handles bootstrapping separately from
termination. Global replay computes rewards from raw next observations:

\[
r_g(s,a,s')=-\lVert h(s')-g\rVert_2,\qquad d_g(s')=0,
\]

where `h` extracts gripper or object position. Truncation flags remain unchanged.
Relabelling existing **configured task goals** for TBTRL replay is not HER.
The same shared memory budget applies regardless of the number of tasks. Its
actual capacity, grouping and allocated transition bytes are printed by the
notebook and saved in each task's memory metadata.

Fixed starts plus deterministic evaluation mean the eight episodes can be
identical: they are not eight independent initial-state trials. Success fractions
may therefore jump between zero and one. Use additional training seeds for
uncertainty across trained models, and report this fixed-start protocol.

## Plots and GIFs

Learning curves, losses, gradients, representations, transfer probes, retention
and cost comparisons share the maze notebooks' labels and export styles. Maze
heatmaps/arrows are replaced by native 640 × 480 RGB views with a caption strip.
GIFs run at the simulator's 25 fps and record the entire evaluation episode,
including what happens after first success. Captions report whether success was
ever reached; JSON also records final-step success and distance.

Each task saves `rollout-01.gif`, `rollout-01-final.png` and `rollouts.json` beside
its numerical plots. `render_saved_run(output, display=True)` can regenerate
these on independent environments; `spatial=False` omits simulator rendering.
Diagnostic evaluation preserves training random states and uses the same saved
policy input normalisation as normal evaluation.

Native environment specifications: [FetchReach](https://robotics.farama.org/envs/fetch/reach/),
[FetchPickAndPlace](https://robotics.farama.org/envs/fetch/pick_and_place/) and
[Robotics release notes](https://robotics.farama.org/release_notes/).
