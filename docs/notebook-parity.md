# Working-notebook parity audit

Reference: `0606a8f664974ecd8af2785fc575d20785f17b24`, the original
`src/experiments/working` notebooks. The notebooks now default to **full runs**.
The earlier PR defaulted to smoke mode, which reduced training to 24 steps,
recovery to 16 steps, two goals, one seed and smaller networks. That explained
the very short runs; it was not a training-speed improvement.

## Audited configurations

| Setting | FourRooms DQN | Maze DQN | Maze SAC |
| --- | --- | --- | --- |
| Seeds | 42, 123, 456 | 42 | 42 |
| Goals | 10, sampled with RNG 0 | 5, sampled with RNG 0 | Four explicit continuous goals |
| Training steps per goal | 100,000 | 100,000 | 500,000 |
| Recovery steps per goal | 100,000 | 100,000 | 100,000 |
| Warmup | 1,000 | 1,000 | 10,000 |
| Batch size | 256 | 256 | 256 |
| Hidden / embedding dimensions | 64 / 16 | 64 / 16 | 64 / 16 |
| Training frequency | 3 (recovery: 4) | 3 (recovery: 4) | 1 |
| Evaluation interval setting | 1,000 | 1,000 | 10,000 |
| Evaluation episodes | 8 | 8 | 8 |
| Early stop | Return ≥ 0.99, patience 3 | Return ≥ 0.99, patience 3 | Success = 1, patience 3 |
| Episode horizon | 100 | 100 | 200 |
| Replay capacity | 100,000 | 100,000 | 1,000,000 |
| Retained fraction | 0.4 | 0.4 | 0.4 |

DQN still evaluates only on update steps: with frequency 3 and evaluation
setting 1,000, this means every 3,000 steps. Early stopping can finish sooner
than the budget. The notebook prints its resolved seeds, goals, budget, warmup,
batch size and device before training, and selects CUDA/MPS/CPU as before.

`scripts/verify_notebook_configs.py` reads the reference notebook code directly
from Git, resolves constants and task-dependent expressions, and checks:

- Seeds, sampled/fixed goals and maze layouts.
- Network widths, embedding dimensions/radii and environment options.
- Every applicable trainer argument, including omitted arguments' defaults,
  for every goal in scratch, sequential and recovery calls.
- DQN's same-seed-per-task rule and SAC's task/recovery seed offsets.

Runner-owned model/environment/buffer objects and obsolete `lr_task_code` and
`td_steps=1` are excluded from numeric argument comparison. The retained DQN
trainer implements the original one-step target. CI runs this audit separately
from the short execution checks.

## Restored behavior

- DQN scratch keeps fresh models but accumulates seen goals for the goal
  regularizers, exactly as the original baseline loop did; replay is empty and
  its coefficient is zero. SAC scratch uses only its current goal.
- SAC scratch sets alpha to 0.1 and disables goal separation. Sequential task 0
  disables replay and goal separation; later tasks use the original values.
- Model targets are independently initialized then loaded from online weights,
  preserving the original model-initialization RNG consumption. Scratch model
  initialization is not reseeded at each goal.
- Notebook training progress is streamed immediately and persisted in
  `training.log`. Evaluation messages include returns/length, losses and replay
  counts; DQN also reports epsilon/Q bounds and SAC success/distance, both
  critics, actor/entropy and embedding diagnostics.
- Every trained goal displays learning/loss curves, Q/policy maps, rollouts,
  raw embedding norm distributions, seen-goal cosine matrices, fixed-probe
  embeddings/drift, covariance spectrum and encoder/actor weight changes.
  Both SAC critic branches are shown. Numeric arrays and PNGs are saved in
  each goal's directory before training proceeds to the next goal.

The plots are maintained implementations of these diagnostics, not copies of
the large original plotting cells. Fixed probes are consistent across goals;
PCA views explicitly use a per-goal basis. Numeric arrays are saved for other
analyses. Monitoring uses separate environments and deterministic replay
indices and is tested not to alter training weights.

## Explicit repairs and limits

The original SAC recovery cell did not run cleanly: it called the unimported
`sac_train_tbtrl` (a different, joint-critic trainer) and passed `env_id` and
`seed` to a maze factory accepting only `goal`. Recovery uses the supported
independent-critic `sac_train_tbtrl_maze` update logic, the original recovery
call's settings, alpha default 1.0, and no target clipping (matching the older
recovery trainer). This repaired path cannot be described as bit-for-bit
execution of the broken notebook cell. Scratch and sequential use the original
maze trainer and target clipping [-10, 10].

The cleanup also retains explicit environment/action/evaluation seeds, float
continuous actions, replay wraparound/alias fixes, correct truncation flags,
per-invocation covariance sketches, and final-model goal similarity computation.
These are documented corrections, so full stochastic trajectories are not
claimed to match historical notebook outputs. Optimizer objectives/order and
parameter updates of the two working main trainers are compared against their
original definitions in `scripts/verify_legacy_equivalence.py`.

Validation includes an all-penalty eight-step fixture and a 36-step fixture for
each original notebook configuration, with three evaluations interleaved. The
latter retain original network dimensions, batch size, loss settings and update
frequency while shortening warmup/budget/evaluation intervals. Original and new
online/target/actor tensors and evaluation curves match exactly under matched
environment seeds. These are regression checks, not convergence experiments;
full-budget runs have not been rerun.
