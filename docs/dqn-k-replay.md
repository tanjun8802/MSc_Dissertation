# Bounded task replay for DQN

Open `experiments/working/TBTRL_GridworldMaze_DQN.ipynb`. It contains the environment settings, six study conditions, training, independent recovery, completeness table and saved-result figures. Trainer and plot implementations remain shared Python modules so a new environment notebook does not duplicate algorithms. The recorded pre-task-continuation notebook is unchanged.

Install the locked development/notebook environment, restart the kernel, and execute the notebook from the repository. The current soft-ball v2 preset uses raw factors, squared radius-excess penalties, no goal separation and covariance off by default. Full budgets are the default. Set `SMOKE=True` for a three-task CPU-compatible execution check; those curves do not establish convergence. Change `DEVICE`, `SEEDS`, `GOAL_COUNT`, `K`, `REPLAY_SAMPLES`, `PHI_RADIUS`, `PSI_RADIUS`, norm coefficients, candidate capacity and refresh cadence in the parameters cell. Changing settings requires a new `RUN_ID`. With matching settings/source, append goals at the end and keep `RESUME=True` to continue completed task boundaries.

Set `RUN_EXPERIMENTS=False` to regenerate tables and figures from saved results. `RUN_RECOVERY` enables independent adaptation trials; each starts from a separate copy of the same final model and archive. The study preset uses acquisition exploration/update cadence for recovery, with a held-out mean-return gate to skip already retained tasks. `recovery_return_threshold` sets the same numeric return target for initial gating and adaptation stopping; the latter retains the configured patience requirement. Acquisition keeps its original return/patience criterion. Zero-step and adapted records both identify the recovery target explicitly.

## Bounded mode

`configs/tbtrl_gridworldmaze_dqn_k_replay.json` is also usable with the existing CLI. It enables `training.k_replay`; original configs and the legacy DQN path retain their behaviour. The new mode requires DQN/global replay, constant replay scheduling, `replay_ratio=0`, and no separate `replay_tasks_per_batch`. The nested options own the budgets:

| Option | Meaning |
| --- | --- |
| `max_tasks` | At most K historical task objectives, in addition to the current task |
| `sample_budget` | Total R historical transition-task pairs per update |
| `candidate_capacity` | At most M task descriptors in the selector pool |
| `refresh_updates` | Refresh selected tasks and optional weights every H optimiser updates |
| `selector` | `rotating`, `random`, or residual-greedy `basis` |
| `weighting` | `uniform` or periodic `interference` weighting |
| `seed` | Independent admission/selection RNGs |
| `weight_floor` | Uniform mixture preventing zero selected-task weight |
| `weight_temperature` | Softmax temperature for interference scores |
| `weight_smoothing` | Exponential score smoothing; zero uses the latest score |

K=0 and R=0 explicitly disable historical replay. Otherwise both are positive. R is divided across at most min(K,R,eligible candidates) tasks without exceeding its total. Historical loss is a normalised weighted sum of per-task means; the outer `replay_loss_coef` is fixed. Uniform weights reproduce a mean over selected tasks, independent of K.

Goal separation is disabled in the soft-ball preset. Task-norm penalties operate on current-plus-selected goals, at most K+1. The state-action norm penalty covers current and replayed samples, at most B+R. Both encoders remain trainable. This implementation does not add a frozen teacher, distillation or covariance rescaling.

## Candidate admission and geometry

The pool contains the newest completed task plus a uniform reservoir of older tasks. This admission policy is identical across all selectors; its RNG is separate from random selection, exploration and replay. It replaces the earlier plan's protected-basis pool so selection comparisons do not implicitly change candidate admission. The pool is an approximation: a dropped task can drift and need recovery. A full-history selector is not implemented as part of the bounded path.

Basis selection chooses actual task IDs with large residual directions outside the selected span. Near-zero/collinear vectors are handled, and unused slots are filled by deterministic rotation. This is not PCA and is not a proven barycentric spanner. Selection uses direction-normalised task vectors (the Q forward pass retains raw magnitudes); raw norms are reported separately. Diagnostic rank, residual and coefficient measurements refer to the candidate pool, not every historical task.

A deterministic common archive probe of at most 32 state-action inputs measures Q-value drift from the first refresh within each task. It does not sample training RNGs or influence selection. This probe is local coverage evidence, not a guarantee of behavioural retention. Replay still samples selected tasks' existing archive views and recomputes reward and termination for each requested goal. Tasks sharing an archive group retain distinct objectives.

Diagnostics record current/replay shared-layer gradient norms and per-refresh/cumulative extra gradient-pass counts. Interference weights use negative gradient cosine against the current task, measured on the final shared state-action encoder layer. They are detached, smoothed, and mixed with a uniform floor, and sum to one. This is a local diagnostic-based heuristic, not a proof that the combined Adam update protects an earlier task. It adds at most K+1 periodic gradient measurements; log and compare that overhead. Uniform remains the default.

## Outputs and interpretation

Each task saves `replay-diagnostics.json`, sampled `losses.jsonl`, metrics and the existing policy checkpoint. Task-boundary continuation additionally preserves candidate admission, selections, weights, selector RNGs and counters. Incompatible source/options require a new run directory. Evaluation-only policy checkpoints are not exact mid-task resumable state.

The notebook exports:

- Acquisition cost, introduced-task success, independent recovery cost and sampled refresh-update host timings.
- Selected-task weight timelines, objective counts, common-probe value drift, candidate reconstruction error and replay sample counts.
- Existing per-seed all-goal retention matrices, acquisition/recovery cost plots and detailed learning/embedding/policy maps for the chosen detail condition.

Full exports use 300 dpi PNG and editable PDF/SVG. Bands show sample SD across complete seed sequences; the table marks incomplete sequences and includes unsuccessful runs. The overview's future-goal scores are zero-update evaluations, not training inputs. Timing probes measure host work through the optimiser, exclude evaluation/Polyak overhead and do not synchronise GPU execution. Full per-task elapsed time is also saved.

Transition-array capacity and per-update objective/sample/regularisation work are bounded independently of task count. Goal catalogues, task-to-group metadata, result histories and full all-goal evaluation still grow. The claim is not constant total memory or constant end-to-end evaluation cost.

All study conditions use factorised DQN. The six conditions are rotating/random/basis uniform replay, basis interference replay, factorised scratch and factorised fine-tuning. Conventional MLP-DQN controls and individual regulariser ablations remain future experiments. The historical covariance target is off by default pending a compatible design; study it separately from selection. The previous projected v1 protocol remains reproducible from its commit and saved configs, but source changes require a new output directory.

## Validation

Run `python -m pytest tests/test_k_replay.py`, `python scripts/check_dqn_study.py`, and the normal repository checks in the configured development environment. Fresh-kernel smoke validation executes all six conditions, real TD updates, replay/recovery, inline plots and count constraints. Smoke checks establish execution and continuation correctness, not acquisition gains, retention or recovery speedups.


## Soft radii and the revised value model

The current study sets `model.embedding_constraint="soft"`. Encoder outputs and supplied goal factors are used directly in every Q-value/inference interface. Their dot product is not a cosine score. Directional normalization remains confined to task selection and archive similarity. Soft mode rejects the legacy cosine-inference flag.

For each factor z and configurable radius r, the penalty is `mean(max(0, ||z||_2-r)^2)`. It starts at r exactly, is zero throughout the interior, and does not pull all vectors onto a sphere. This replaces the legacy Smooth-L1 excess penalty starting at r-0.1. The old penalty measured Euclidean lengths already; “L1” referred to the excess penalty, not an L1 embedding norm. Squared excess grows faster for large violations, but finite coefficients cannot guarantee that factors remain inside the ball or eliminate bilinear scaling ambiguity.

`q_bound` is null in soft mode. Radius product is a reference scale, not a theorem about Q. Historical projected configs/checkpoints retain their old semantics via the default `embedding_constraint="projected"`; SAC and candidate-discovery trainers are unchanged. The soft study explicitly opts in and persists this model choice in checkpoints.

The notebook's new soft-ball figure reports phi/psi norm means/maxima, radius-exceedance fractions and weighted losses. Norm diagnostics use bounded current/replay samples and selected goals; they do not certify every state or old task. The radii can reflect problem value scale with headroom, but neither their product nor penalty coefficients are automatically optimal.

The existing covariance-to-identity implementation remains available for controlled experiments, but its coefficient defaults to zero in this revised preset. No Gaussian matching or frozen representation constraint has been added. For a future covariance study, use a compatible variance scale and an explicit current/archive sampling distribution; isotropy over arbitrary data does not establish useful features or prevent forgetting.

Finish an in-flight v1 run under its existing kernel; restart the kernel for v2 and use `maze-dqn-k-replay-soft-ball-v2` or another new run ID. Do not reload trainer modules mid-run or combine v1/v2 results as one condition.

The notebook checks requested settings against saved settings before restoring a study. A mismatch now fails with the differing fields, rather than silently restoring historical regularisers or projection. `CONTINUE_SAVED_PROTOCOL=True` explicitly retains the saved protocol; it does not bypass checkpoint/source compatibility checks. For historical analysis only, set `RUN_EXPERIMENTS=False`. A new soft-protocol run needs a new run ID. The saved v1 results used projection, SIGReg 0.01 and separation 0.04 (target cosine 0.7), despite the current preset's different defaults.

## Offline embedding geometry

The notebook's geometry figures use separate centred SVD reference planes for task and state–action factors. Each branch keeps its reference mean/axes across stages; the two branch planes are not directly comparable. Coordinates are not normalised. At the reference stage the reported energy fraction is explained variance; earlier views also include mean drift relative to that reference. True norms and pairwise task angles always use the original uncentred factors. Centred panels omit a radius circle because their origin is the reference mean.

Black rings identify the actual task IDs in the last logged replay selection, with its step and weights in the caption. Displayed factors are task-end measurements, not reconstructed selection-time vectors. These highlights show selected objectives, not PCA axes or guaranteed preservation. Exports include means, bases and selection metadata. All calculations use detached saved JSON on CPU; this path does not execute a model or touch device tensors.

The small fixed state–action probe set is illustrative. Useful-representation evidence needs balanced held-out probes, feature covariance spectra and predictive/transfer checks. Covariance-to-identity on unit-bounded d-dimensional features has a scale mismatch: population covariance trace is at most 1, while identity has trace d. Soft constraints remove that hard incompatibility but still create competing scale preferences. Any future variance/decorrelation objective should use an explicit probe distribution and a variance target compatible with the intended feature scale, then be tested as a separate ablation.

## Fixed-budget follow-up

The separate [follow-up study](dqn-followup.md) retains the historical v1 regularisers, compares basis/random across K at fixed acquisition budgets, and saves two independent recovery protocols with collateral-forgetting analysis. Open `experiments/working/TBTRL_GridworldMaze_DQN_Followup.ipynb`.
