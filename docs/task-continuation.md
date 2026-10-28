# Append-only task continuation

All maintained notebooks in `experiments/working/` now use persistent run
identifiers and checkpoints **between tasks**. Keep the same run directory,
append goals/tasks to the end of the existing list, and rerun the training cell.
Completed acquisition tasks are reused; the appended suffix is trained. The
historical sandbox notebooks under `src/experiments/` remain archival code.

## Scratch and TBTRL are separate experiments

Each seed and method has its own progress and checkpoint directory:

```
<run>/seed_42/scratch/task_0/...
<run>/seed_42/scratch/completed.json       # RL scratch progress
<run>/seed_42/sequential/task_0/...
<run>/seed_42/sequential/boundary.pt       # full continuing TBTRL state
<run>/summary.json                       # completed records for aggregate plots
```

PMO/MADE discovery uses a small scratch `boundary.pt` progress marker instead
of `completed.json`. The controlled MADE study also separates each ablation
method and uses task completion receipts. Its latest continuing method has a
full `boundary.pt`; historical `complete.pt` files retain model/diagnostic state
without duplicating the replay pool for every task.

A new scratch task still starts from an independently initialised model. It does
not load weights from the preceding scratch task; doing that would turn the
baseline into transfer learning. Sequential TBTRL restores its own actor/critic
(or Q network), target networks, replay contents/group mappings, task catalogue,
temperature, normaliser state, and the optimiser state used by its existing
carry policy. RNG state and candidate regulariser/sampling streams are retained.

Checkpointing does not introduce optimiser/normaliser carry where the original
trainer resets them at each task. DQN currently resets its optimiser per task;
SAC's `carry_optimizer_state` and `entropy_transfer` settings still govern carry.
No TD targets, actor objectives, replay admission, or regulariser equations are
changed by this feature.

The global replay restore path reconstructs views of a **single allocation**.
It does not deserialize independent copies of each task's shared buffer. The
fixed replay budget is preserved. Historical metrics, task labels, model
snapshots and figures still consume disk proportional to task count. The latest
full continuation state is replaced atomically; full replay copies are not
retained for every acquired task. Temporary writes require additional disk space.

## Usage

For maze and Fetch notebooks, append coordinates in the loaded JSON's `goals`
list. For the many-goal maze notebook, increasing `GOAL_COUNT` keeps the existing
deterministic prefix. Do not alter the other settings or change `RUN_ID`.

```python
records = run_experiment(
    config, output, device=DEVICE,
    modes=("scratch", "sequential"),
    resume=True, on_task_end=diagnostics,
)
```

The CLI also accepts `--resume`. Fetch baseline studies use a new `v3` output
root and delegate continuation to the same runner. Independent baseline jobs
are reused separately from sequential intervention jobs. Confirmation must
cover newly appended goals before the transfer gate passes.

For PMO, append oracle names to `ORACLES`. Its notebook now reserves
`GOAL_CAPACITY=128` one-hot slots before the first task, so adding an oracle does
not change earlier task vectors or the learned network's input width. Keep that
capacity unchanged. Exceeding it requires a new model/run; automatic resizing is
not implemented. The legacy `make_tasks(names)` API still uses `len(names)` slots;
use `make_tasks(names, goal_capacity=128)` for an appendable experiment.

MADE tasks use fixed 118-element indicator vectors. Append chemical systems,
provide their datasets, and preserve existing reference-data fingerprints. Both
MADE notebooks record native environment/oracle settings for compatibility.
The controlled study uses `checkpoint_campaigns=False`: an entire task is the
training commit unit. Its older campaign-resume mode remains an explicit opt-in
API feature; the notebooks do not enable it. Recovery probe snapshots are
separate diagnostic artifacts, not checkpoints for extending the main learner.

## Interruption and compatibility

A successful task commits state atomically before displaying diagnostics. A
plotting exception therefore does not require retraining the task. Rerunning
redisplays available saved per-task figures. If plotting failed before a figure
was written, the cached task remains complete and does not invent missing plots.
The existing saved-run rendering tools can regenerate figures from saved metrics
and diagnostics where those inputs are available.

If training, evaluation, or checkpoint writing is interrupted before the task's
commit, resume restores the preceding boundary and restarts that whole task.
No environment state is restored inside a task. Aborted interactions/oracle calls
are real extra work; reported records count committed tasks rather than every
aborted attempt. Concurrent writers to the same run are rejected by a process
lock; a terminated process releases its lock automatically.

The existing task definitions must be an unchanged prefix. Shrinking/reordering
or editing goals, changing model shapes, hyperparameters, seed/method sets,
recorded dependencies, device, reference data, or implementation source is
rejected. New task reference fingerprints may be added; existing ones must match.
This deliberately prevents silently continuing with a different algorithm. A
future trainer revision needs a reviewed checkpoint migration or a new run ID.
Source hashes cover the installed `tbtrl` implementation; notebooks supply the
native environment/oracle settings as manifest metadata.

These full-state checkpoints are trusted local Python/PyTorch files. Evaluation
`checkpoint.pt` and `policy.pt` artifacts remain separate and backward compatible.
An old run containing only evaluation weights cannot reconstruct missing replay
or optimiser state. Begin the new protocol once in its new directory; later
appends reuse that full state. Previous executed working notebooks are preserved
byte-for-byte in `experiments/recorded/pre_task_continuation/`.

## Incremental plots and recovery

Per-task numerical results and plots remain in their original task directories.
Training cells return saved records including the reused prefix. Aggregate plots
read `summary.json` rather than relying on a stale notebook `records` variable;
rerunning their cells includes newly committed tasks. Incremental journal writes
retain other completed methods even if an appended task later fails.

Historical checkpoints have no evaluation for goals that were not defined at the
time. Retention matrices show those entries as missing, not zero or a fabricated
retroactive measurement. New task checkpoints evaluate the current full goal
catalogue. This permits measuring retention as the catalogue grows without
retraining the historical acquisition tasks.

Recovery has a different meaning: it tests relearning from the **current final
sequential model**. After appending tasks that endpoint changes, so recovery
results live under `recovery/after_<task-count>/`. Previous endpoint results are
retained, and the latest aggregate includes only the current endpoint. Recovery
branches never overwrite the continuing TBTRL state. Completed recovery jobs at
the same endpoint are reused. In the controlled MADE study, held-out retention
also reuses previously evaluated stages.

If you want only new-task acquisition without revisiting old goals, select
`MODES=("scratch", "sequential")`, or set
`RUN_RECOVERY=False` in a study notebook. Enabling recovery after each append can
intentionally retrain earlier tasks as separate tests. Keep the acquisition method/seed selection unchanged. Recovery can be enabled or
disabled between invocations without changing the acquisition checkpoint.

## Validation scope

Tests compare uninterrupted and save/reload/append runs on CPU, including exact
network/target parameters, optimiser state, losses, normalisers, temperatures,
replay payloads and global-buffer aliasing. They cover DQN/SAC, both replay
policies, candidate learning, controlled MADE, task interruption, completed-job
reuse without oracle calls, separate scratch/TBTRL state, prefix incompatibility,
and endpoint-specific recovery. Notebook smoke checks exercise orchestration,
inline diagnostics and plots; they do not establish convergence or cross-device
bitwise reproducibility.
