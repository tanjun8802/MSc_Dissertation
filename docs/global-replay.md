# Fixed-budget replay across tasks

The `global_replay` experiment extends `temperature_carry`. It changes replay
ownership and enables goal relabelling; it does not add actor KL, rescale the
representation penalties, or change the SAC/TBTRL objectives. The original
`per_task` mode remains available for reproduction.

Open `experiments/working/TBTRL_GridworldMaze_SAC_GlobalReplay.ipynb`. Its default
budget is 100,000 transitions: 50,000 recent slots and 50,000 archive slots.
These are views into **one allocation**, not separate per-task allocations.
The default capacity on `ExperimentConfig` is larger (1,000,000 total and
250,000 recent); the notebook explicitly overrides it. DQN can use the same
mode by setting `replay_mode="global"` and `similarity_threshold=0.8`.

## Task-count experiment

The global-replay notebook now defaults to `GOAL_COUNT = 20`. The original four
goals remain first, in their original order. Further goals are unique free-cell
centres selected by farthest-first shortest maze-path distance from the goals
already selected, with deterministic row-major tie breaking. They are spread
across traversable corridors rather than selected by straight-line distance
through walls. All free cells must be connected. The helper
`tbtrl.experiments.goals.extend_maze_goals` also supports discrete maze coordinates.

Set `GOAL_COUNT = 4` for the original sequence. Both `global_replay` and the
`temperature_carry` memory control use the same generated sequence for a given
count. Goal construction does not consume training RNG state. Resolved goal
coordinates are stored in the existing manifest and task checkpoints, and the
notebook prints the ordered list before training.

The total/recent capacities, 16-group limit, similarity threshold, per-goal
training budgets and TBTRL updates remain unchanged. With 20 goals, the maximum
training interaction budget is five times the four-goal budget; all-goal
evaluation adds further cost. The ordinary smoke option still reduces the run
to two goals. A separate automated check executes all 20 goals with small budgets
to validate the sequence and bounded storage, not to establish performance.

Twenty tasks do not guarantee 20 groups or forced consolidation: sufficiently
similar goals can share existing groups. The notebook prints each task's nearest
similarity, admitted candidates, surviving group sizes and any consolidated pair.
Use those measurements to establish whether a run actually fills the archive or
reaches the group limit. To study group-limit pressure separately, compare an
explicitly labelled smaller `MAX_GROUPS` setting while keeping the goal sequence
and transition budget fixed. Do not interpret an absence of merges as a failure
of the memory mechanism.

## Admission and replacement

1. While training, current experience uses the recent circular partition. Old
   tasks sample archive views. Both partitions count against the global budget.
2. At each sequential task boundary, take the latest `replay_keep_fraction`
   (default 40%) of the valid recent window. This is chronological selection,
   not a success filter and not 40% of all interactions once the window wraps.
3. Recompute cosine similarities for the bounded set of group representatives
   and the current goal using the current critic. SAC averages similarities from
   its two independent goal encoders; DQN uses its single goal encoder. SAC uses
   the same goal normalisation as training, when enabled.
4. If the nearest representative has cosine `s > threshold`, share that group.
   Admit the last `(1-s)/(1-threshold)` fraction of the retained candidates.
   Effectively identical embeddings share without admitting new samples, matching
   the historical DQN rule. Otherwise create a group for the novel task.
5. At `replay_max_groups` (default 16), make room for a novel group by merging
   the most similar existing pair. Every old task keeps a group mapping.
6. Divide archive space equally among groups, redistributing unused space from
   small groups. Thin over-budget candidates at evenly spaced indices to preserve
   coverage across their stored order. Similarity controls grouping/admission;
   within-group thinning is deterministic and does not inspect rewards or TD error.

Grouping is an approximation: close goal embeddings do not prove identical
state-action distributions. Forced consolidation can lose useful coverage. No
finite pool can guarantee perfect retention for arbitrarily many unrelated tasks.
Measure all-goal retention and forward/recovery cost at several budgets and seeds.

## Relabelling and losses

Both trainers evaluate shared transitions under the requested task goal. SAC's
optional `relabel_rewards=True` path recomputes reward **and goal termination**
before constructing its soft Bellman target. Time-limit truncation is preserved;
the existing bootstrap-on-truncation option still applies. Stored reward and
termination values are ignored on this path.

The continuous maze adapter implements its existing simple/shaped rewards,
movement bonus, collision penalty and goal radius in raw observation coordinates.
It performs tensor operations on the batch device, including float32 on MPS.
Other environments must implement `relabel_transitions(batch, goal)` correctly,
including any task-independent terminal conditions. Sharing assumes compatible
dynamics and transition meaning across goals; changing environment dynamics
requires a different replay contract.

Actor replay and representation regularisation still consume replayed states and
goals. Small groups can be sampled with replacement; they are not silently skipped
when smaller than the requested batch. The preset samples at most four old tasks
per update in rotating order. Independent recovery uses a copy of the final
archive for each target goal and does not mutate the source archive.

## What is actually bounded

For total capacity `B`, observation width `d_s`, and continuous action width `d_a`,
the six float32 transition arrays occupy

`4 * B * (2*d_s + d_a + 3)` bytes.

For this maze, the notebook's 100,000 slots occupy 3.6 MB of transition arrays,
independent of the number of tasks. Task buffer objects are views, not copies.
Compaction uses bounded index arrays and one field-sized temporary. Independent
recovery also needs the source archive and one working copy; both budgets are
fixed. These are not measurements of total process/GPU memory.

Goal lists, task-to-group IDs, metric records and diagnostics still grow with
task count. Thus the claim is **constant replay transition storage**, not constant
total program memory or constant representation-regulariser computation. Sampling
remains task-balanced; multiple goals can reference the same group's samples.

`training.log`, per-task `metrics.json`, and `summary.json` report capacity,
allocated transition bytes, recent/archive occupancy, group sizes, candidate
admission, nearest similarity and merged groups. The notebook prints these beside
the usual inline training logs and per-goal diagnostic plots.

## Controlled comparison

Switch the new notebook to `EXPERIMENT="temperature_carry"` for the per-task
control. The notebook matches recent-buffer capacity and the four-task sampling
limit in both conditions. Keep actor options, interaction budgets, seeds and
task order fixed. This avoids attributing a change in the current replay window
to archive compression. The previous saved experiment used a larger current
buffer, so it is not this matched control.

Smoke runs and stress tests establish storage and update correctness. Full-budget
retention results for this replay policy have not yet been measured.
