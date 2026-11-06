# Bounded K-task replay for TBTRL DQN

Date: 2026-11-05. Implementation reference. The bounded DQN path and self-contained Maze study are now implemented; see [the usage guide](dqn-k-replay.md). Validation uses smoke budgets, not scientific convergence runs.

## Objective and accepted constraints

Keep at most K historical task objectives active in each DQN update, in addition to the current task. Use the existing fixed-capacity global replay store. Select representative past task directions to preserve useful shared representations, while measuring both immediate retention and steps needed to recover degraded tasks. K is a resource constraint, not a guarantee that all encountered task embeddings can be spanned exactly.

The user accepted embedding-based task selection as a first candidate, an externally fixed K with approximate coverage, and separate retention/recovery outcomes. The interview is closed. Algorithm details below are implementation recommendations, not established findings.

## What currently grows

- train_dqn already accepts replay_tasks_per_batch, but uses a rotating slice of all eligible task IDs, rebuilding that list every update.
- training_settings can set replay_ratio to task_id + 1. Capping objectives alone therefore leaves historical sample processing growing with task count.
- Goal separation and goal-norm penalties still encode every seen goal; separation builds an all-goal cosine matrix.
- GlobalReplayMemory bounds transition allocation and group count. Task descriptors, task-to-group metadata, results and evaluation costs can still grow. The bounded mode must claim bounded training-update work and replay transition storage, not constant total process memory.

## Resource contract

For each optimiser update in the new bounded mode:

- One current-task TD objective and at most K historical TD objectives.
- B current transitions and at most R historical transition-task pairs processed. R is fixed independently of task count; set m = min(K, eligible candidate count, R).
- Divide R across m tasks using quotient/remainder allocation, so the total never exceeds R. Require positive R and K when historical replay is enabled. No historical tasks means zero historical TD work.
- Compute goal-separation and goal-norm penalties on the unique current-plus-selected goals only, at most K + 1. This is a deliberate sampled regularisation objective, not an exact reproduction of the all-goal loss.
- Select tasks from a bounded pool of M candidates, M >= K. Refresh their embeddings and selection every H optimiser updates, under no_grad; hold selected IDs between refreshes. Validate and persist K, R, M and H.
- Keep global transition capacity and archive grouping unchanged. Count selector work, goal-regularisation work and replay samples explicitly.

K excludes the current task. Stored task directions count as task objectives only when their loss participates in an update. K=0 should be supported as an explicit no-historical-replay ablation with R=0; do not silently change the existing replay_tasks_per_batch validation contract.

## Configuration and compatibility

Introduce an optional DQN-only k_replay options object, parsed and validated by a small KReplayOptions helper. Suggested fields: max_tasks, sample_budget, selector, candidate_capacity, refresh_updates, seed, rank_tolerance and diversity_ridge. All numeric values are explicit in new experiment configs; initial implementation development values are K=4, R=256, M=32 and H=100, not tuned scientific recommendations.

Selectors: rotating, random and basis. All three use the same candidate pool, sample budget, regularisation scope and refresh cadence. Keep the existing rotating replay_tasks_per_batch path unchanged when k_replay is absent.

Reject bounded mode with replay_schedule=task_count. Require replay_mode=global for the first supported implementation. In bounded mode, R owns the historical sample budget; reject ambiguous simultaneous replay_ratio settings rather than silently allowing it to grow or become ignored. Resolve old phase overrides explicitly in the new presets. SAC, discovery trainers, old configurations and legacy equivalence paths are outside this change.

## Selector module

Add src/tbtrl/training/task_selection.py with a compact interface:

- observe_completed_task(task_id): admit a newly completed original task ID into the bounded candidate pool.
- refresh(network, task_goals, archive): re-encode at most M available original task descriptors; choose at most K task IDs; return IDs and bounded diagnostics.
- active_tasks(): return cached IDs without walking all historical task mappings.
- state_dict()/load_state_dict(): preserve pool IDs, chosen IDs, dedicated RNG state, counters and options for task-boundary continuation.

Separate candidate admission from the choice of active objectives. Maintain a protected current basis plus a bounded reservoir of alternative original task IDs. Give each newly completed task a candidate opportunity at the next boundary refresh, using a temporary pool of at most M + 1, then reduce to M. Update the reservoir incrementally using its own RNG; no all-history traversal is required. Document that excluded candidates can become stale or be missed: pool admission is an approximation and needs a later ablation.

Task IDs are not archive group IDs. Multiple tasks can share one archive view while having different goals and relabelled TD objectives. Resolve selected IDs through the current memory mapping after archive compaction; never cache obsolete buffer views. Filter absent or empty archive views within the bounded pool. Admit only completed tasks; the current task cannot simultaneously become a historical objective.

## First basis selector

Use projected task embeddings psi(g), computed with the current online goal encoder and detached from autograd. Normalize nonzero vectors for the selection score, so the initial question is directional coverage; record raw norms separately. Zero or nearly zero embeddings are flagged and do not manufacture a direction.

1. Greedily choose actual task vectors with the largest residual outside the selected span, using reorthogonalised Gram-Schmidt and deterministic task-ID tie breaks.
2. Stop adding geometric directions once the residual falls below rank_tolerance. If fewer than K distinct directions exist, fill remaining slots from the alternatives by a deterministic rotation, rather than repeatedly choosing numerically redundant tasks.
3. Apply a bounded number of improving swaps using regularised log-determinant diversity, if needed to improve conditioning. Include this within the fixed M/K refresh budget; avoid a new external linear-algebra dependency.
4. Record effective rank, residual coverage of the candidate pool, reconstruction coefficient norms and a ridge-conditioned Gram statistic. Coverage is over candidates, not all historical tasks.

Call this an approximate basis selector. Top principal components are not actual tasks, and this greedy procedure is not a proven barycentric spanner. Do not assert bounded reconstruction coefficients or whole-task performance guarantees without measuring/proving them. Start with residual-greedy selection; make swap refinement a separate incremental improvement if diagnostics justify it.

## DQN update integration

In src/tbtrl/training/dqn.py, branch once between the existing replay path and bounded replay. The bounded path obtains cached selected IDs, draws their existing archive views using the fixed R allocation, and reuses the current task_loss and reward/termination relabelling.

Keep historical loss as the mean of per-task mean TD losses, multiplied by the existing replay_loss_coef. Consequently, changing K does not silently multiply historical loss weight. Record both objective count and sample count. Use current-plus-selected goals for goal penalties; retain current-state covariance and phi-norm behaviour initially. Do not introduce frozen encoders, value distillation, new covariance scales or automatic learning-rate changes in the selector change.

Task-conditioned replay batches retain the current sampling contract in the first implementation. The spanning argument requires common state-action coverage, which task-specific replay does not guarantee. Use a small fixed common archive probe batch for diagnostic comparisons at refreshes; any proposal to train on common relabelled states should be an independently matched experiment.

Task selection receives no held-out evaluation scores or future task descriptors. Its RNG must not perturb exploration, replay sampling or evaluation streams. Diagnostic probes must preserve training RNG state and run without gradients.

## Runner, continuation and artifacts

At each completed task, integrate candidate admission after memory.finish_task has updated archive mappings. Pass selector state into the next train_dqn call and return its updated state in DQNResult. Persist it in task-boundary checkpoints using the existing continuation structure. Inspect result/checkpoint schemas before editing; add explicit schema compatibility rather than rebuilding a different selector silently on resume.

New selector options and state are part of continuation compatibility. Existing completed results cannot be relabelled as bounded-K runs. Maintain separate output directories and record the effective budgets/selection strategy in manifests and task metrics.

Persist bounded per-refresh diagnostics and sampled update counts: selected IDs, candidate IDs, active objectives, actual historical samples, goal-penalty goal count, rank/residual/norm diagnostics, refresh time, update time and transition bytes. Full all-task evaluation is an explicit reporting cost outside the bounded update claim.

## Implementation sequence

1. Add options validation and resource-contract tests. Keep all old configs valid. Add an isolated bounded-K preset using existing global replay and constant budgets.
2. Implement candidate state, deterministic rotating/random controls, task-boundary checkpoint round-trip and resume. Integrate bounded samples and selected-goal penalties into real DQN updates.
3. Add the residual-greedy basis selector and geometry diagnostics, sharing the same training path and budgets as controls.
4. Add a small runner study comparing selectors over identical goals/seeds, with retention matrices and independent recovery endpoints. Extend existing Maze/FourRooms clients rather than duplicate trainers.
5. Run regression checks and a many-task small-budget stress test. Only after these pass, screen full acquisition/retention/recovery runs; tune K and candidate settings on development sequences, then freeze settings for evaluation.

## Required verification

- Across more than K and M tasks, no update exceeds K historical objectives, R historical samples, K+1 regularised goals or M stored selector candidates; boundary candidate refresh is at most M+1.
- No full seen-goal forward pass or full task-map scan inside a bounded-mode gradient update.
- Known synthetic directions: independent, duplicate, opposite, nearly collinear, low-norm, and rank greater than K. Selection remains finite and deterministic; selected IDs always refer to real completed tasks.
- Shared archive groups preserve distinct objective identities and correct goal-specific reward/terminal relabelling. Archive merges do not invalidate selection.
- Real optimiser steps with at least two selected old tasks; loss weighting and target isolation remain correct.
- Dedicated selector RNG does not alter exploration/replay streams. Checkpoint round-trip preserves subsequent selections under the same continuation conditions.
- Legacy trainer tests, config parity and applicable legacy equivalence checks continue to pass with bounded mode disabled.
- Measure actual selector/gradient timings as task count grows, alongside invariant counts. Timing trends support practical efficiency; tests do not assert fragile absolute time thresholds.

## Scientific evaluation

First isolate selection: same global archive, candidate admission, R, K, model, penalties, seeds and task order; change only rotating/random/basis selection. Add a full-history rotating selector as a separately costed control to expose candidate-pool effects. Compare K values including K=0 and a feasible all-available-task reference.

Measure acquisition learning curves/cost, immediate old-task retention, and independent recovery to a common preregistered performance target. Report unsuccessful recovery as censored/failed rather than dropping it. Also report savings relative to original acquisition and equal-budget scratch recovery, with identical recovery exploration/update schedules. Include selector cost and gradient updates as well as environment steps.

The first result sought is whether basis selection improves retention or recovery under the same K/R/storage budgets. Preventing covariance collapse or covering embedding directions alone is not evidence of that result. The existing covariance-scale tension remains a separate follow-up ablation; changing it simultaneously would obscure the selector's effect.


## Optional experiment: dynamic per-task replay weights

Added 2026-11-05 following the user's suggestion. This is a research extension, not an assumption that unequal weights outperform the equal-weight control.

Separate a fixed total replay coefficient lambda_R from a normalized allocation w_i over selected tasks: L = L_current + lambda_R * sum_i w_i L_replay,i + L_regularisation, with w_i >= 0 and sum_i w_i = 1. Equal weights w_i = 1/m exactly recover the planned per-task mean. The case m=0 omits replay. Keep K, R and lambda_R fixed when isolating weight allocation.

Add replay_weighting = uniform | interference as an optional strategy after the bounded path and basis selector work. An initial interference hypothesis uses the negative cosine between current-task and each selected replay-task gradient on a fixed shared state-action encoder layer. A conflicting replay gradient predicts loss increase under an infinitesimal current-task-only SGD step; this is not a guarantee for the actual Adam/regularised combined update. Refresh these scores periodically on at most K tasks, detach scores/weights, reuse bounded batches and account for extra autograd passes. Never compute all-history task gradients.

Map clipped finite scores through a temperature-controlled softmax and mix with a uniform floor: w_i = rho/m + (1-rho) * softmax(score/temperature)_i. Validate 0 < rho <= 1 and temperature > 0; use uniform fallback for invalid/zero-norm signals and define m=1 explicitly. Keep weights detached so the learner cannot reduce the objective by differentiating through the allocation rule. Log gradient norms as well as weights: normalized coefficients do not ensure normalized gradient influence. Smooth scores within bounded candidate state, reset stale scores when the selected set changes, and checkpoint that state.

Raw TD error is not the initial weighting signal: reward scale, bootstrapping, noise and coverage can inflate it without demonstrating behavioural forgetting. Geometry-based reconstruction importance is another possible weighting ablation, but it should be separate from interference weighting.

Evaluate a small factorial comparison: rotating versus basis selection, each with uniform versus adaptive weights, under the same K/R/storage and fixed total lambda_R. Report acquisition, retention, recovery, selector/weighting overhead and actual gradient work. Periodic per-task gradient measurement has additional compute even with fixed K; report this cost and include a matched-gradient-work sensitivity comparison. Keep uniform as the default until adaptive weighting earns its overhead.

Relevant primary sources: GradNorm (https://proceedings.mlr.press/v80/chen18a.html) for adaptive gradient-magnitude balancing; Gradient Surgery (https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) for gradient conflict; Maximally Interfered Retrieval (https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html) for interference-aware replay selection. These motivate comparisons; none establishes this proposed weighting rule's efficacy in TBTRL.


## Implementation choices

- Candidate admission is newest-task plus an independent uniform reservoir, shared across selection strategies. This replaces the proposed protected basis plus alternatives to avoid confounding selector comparisons with admission policy.
- Residual-greedy basis selection is implemented; optional improving swaps and full-history controls are deferred. Exact-spanner guarantees are not claimed.
- Interference weights are optional and periodic. Uniform is the default.
- The notebook compares the four selection/weighting conditions plus factorised scratch and factorised fine-tuning; conventional MLP architecture controls remain outside this trainer improvement.


## Review corrections

- Raw task vectors below the norm tolerance are zeroed before directional scoring/rank calculations.
- The new DQN study uses a mean-return recovery target consistently for the zero-step gate and adapted stopping; the existing success-rate gate remains available for legacy/SAC conditions.
- Adaptive-weight diagnostics include current/replay shared-layer norms and per-refresh/cumulative extra gradient-pass counts.
