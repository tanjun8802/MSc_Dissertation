# Maze DQN follow-up: fixed budgets and recovery

Open `experiments/working/TBTRL_GridworldMaze_DQN_Followup.ipynb`. It is separate from the completed v1 notebook and owns its environment configuration, runs, analysis and saved figures.

## Accepted protocol

The base protocol is pinned in `configs/tbtrl_gridworldmaze_dqn_followup.json`. Retain the completed experiment's regularisation: projected factors with radii 1, covariance SIGReg 0.01, goal separation 0.04 with target cosine 0.7, and raw norm coefficients 0.0002. Both encoders remain trainable. Soft-ball and regulariser ablations are explicitly deferred.

Compare random and residual-greedy basis replay with uniform weights at K = 1, 2, 4, 8, 11. Default goals/seeds match the twelve-goal v1 study. Keep total historical sample budget R=256, candidate capacity 32, global slots 100000 and recent slots 50000. K excludes the current objective. More K divides the same R and changes sampled goal-regularisation scope; it is not a pure geometry intervention. K=11 covers all available historical tasks in this sequence. Candidate eviction is not tested here.

Acquisition always consumes 100000 interactions per task; early stopping is disabled for every condition, including scratch and fine-tuning. The DQN trainer records the first patience-qualified threshold crossing even when training continues. The saved `solved` flag denotes that crossing, not endpoint performance. Failure costs remain the full budget. Actual elapsed time includes all training after crossing. Per-task endpoint evaluations determine retention.

## Two independent recovery procedures

Each probe starts from the same final acquisition boundary for its condition. Each revisited task receives a separate copy: recovery is not one joint retraining sequence.

- `no-replay`: fresh current buffer, no historical transitions/objectives, fresh disabled selector, same exploration/learning rates/stopping procedure across starting models. Both phi and psi adapt. Coefficients stay unchanged; goal regularisation uses the current task only because no old tasks participate.
- `method-replay`: fork the final archive and selector, exclude the revisited goal from historical objectives, and keep the acquisition method's replay rule. Fresh current-task storage, optimisers and target reset match the trainer's existing semantics.

Both use acquisition exploration/update cadence, up to 100000 interactions, and the original return target 0.99 with patience 3. The held-out pre-update evaluation can skip a retained goal at zero cost. Tables and plots separate those skips from conditional adaptation costs. Non-target success before/after each adapted trial exposes collateral forgetting. Conditional means average within each seed before summarising across seeds; counts remain in the CSV.

## Persistence and notebook use

The default new root is `runs/maze-dqn-fixed-budget-recovery-v1/full`. Acquisition lives under `acquisition/<condition>`; probes live under `recovery/<condition>/<protocol>`. Sources remain unmodified. Probe manifests record SHA-256 fingerprints of final task-boundary files and the acquisition manifest. Changed source endpoints/configs/runtime cannot silently resume old probes. Completed receipts are written last; interrupted trials restart from the same final source.

Existing acquisition continuation validation remains strict. Use a new run ID for changed source/settings; no historical manifest migration is introduced. Analysis-only mode (`RUN_EXPERIMENTS=False`) reads saved configs and results. Missing conditions are printed; incomplete sequences remain in outcome tables and are excluded from aggregates. Analysis exports only figures/tables, not training records.

Set `SMOKE=True` first for a three-goal, one-seed, K=1/2 execution check. Default full budgets are intentionally substantial: twelve conditions including scratch and fine-tuning, with two probes per sequential condition. The notebook prints maximum interactions before execution. `RUN_RECOVERY=False` permits acquisition-only execution; enable it later with unchanged settings to add probes. Detailed online diagnostics are limited to `DETAIL_CASE`.

Exports include per-task metrics, learning/loss histories, trained checkpoints, recovery receipts, manifests, training logs, per-seed outcome CSV/JSON, K-sensitivity and recovery panels, and the detailed condition's per-seed retention matrices. Default plots use 300 dpi PNG plus PDF/SVG. Bands/error bars show seed SD, not confidence intervals. Scratch is factorised DQN, not an MLP control. Endpoints from independent recoveries must never be presented as one model retaining all skills.

## Validation

Run `PYTHONPATH=src python -m pytest tests/test_dqn_followup.py tests/test_k_replay.py` on CPU and native MPS. Real updates verify fixed budgets with threshold reporting, historical replay in method recovery, zero replay in standard recovery, source isolation, resume fingerprints, zero-step skips, failed trials and plotting. Run `python scripts/check_dqn_followup.py` for fresh-kernel execution and analysis-only regeneration. Smoke checks do not establish scientific improvements.
