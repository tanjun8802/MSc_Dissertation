# Fetch baseline and retention study

Use `experiments/working/TBTRL_FetchReach_Baseline_Study.ipynb` or
`TBTRL_FetchPickAndPlace_Baseline_Study.ipynb`. Each loads its existing
`configs/tbtrl_fetch_*_sac.json`. The JSON is the source of experiment settings;
`configure_fetch_study` contains no numerical training presets. This applies to
both FetchReach and FetchPickAndPlace. Their ordinary `*_SAC.ipynb` notebooks
use the same runner and phase resolution, without selecting study cases.
Both study notebooks import this checkout's `src` first, so an older installed
package cannot silently supply a different runner; restart the kernel after pulling.

Diagnostic rollout/GIF counts follow `eval_episodes` in both study notebooks,
just as in the ordinary SAC notebooks. These visualisations are separate from
the `post_eval_episodes` held-out evaluation. Legacy FetchPush/TD3 sandbox
notebooks under `src/experiments/` are not part of this maintained framework.

## Configuration precedence

1. The base JSON supplies `training`, model/environment/replay settings,
   `eval_episodes`, `post_eval_episodes`, and the phase dictionaries.
2. Selecting a study case applies only its explicit JSON entries, in order:
   `study_overrides.exploration`, `study_overrides.architecture`, then
   `study_overrides.retention`.
3. The runner applies JSON `scratch`, `first_task` (sequential task zero), or
   `recovery` settings over the resolved shared `training` settings.
4. `SMOKE=True` deliberately reduces budgets, model sizes, episode counts and
   evaluation intervals. Smoke tests are execution checks, not learning evidence.

Overrides merge nested dictionaries; an empty dictionary explicitly clears a
field, and `null` replaces it with `None`. For example, MLP architecture cases
clear `actor_model` and disable factor penalties, including phase-level factor
penalties. TBTRL's control entry is empty: it inherits the base configuration.
Changing a base coefficient will therefore affect TBTRL unless a selected
retention case or phase explicitly overrides that coefficient in JSON.

No helper resets learning rates, warm-up, step budgets, batch size, stopping
criteria, evaluation counts, normalisation or TBTRL regularisers. Unknown
resolved training keys fail validation. Every study job prints the resolved
static phase settings and held-out evaluation count before running or reusing
results. The job manifest records the resolved config; each task checkpoint
also records its effective training settings.

Runtime semantics still apply: `entropy_transfer="carry"` uses the previous
sequential policy's learned temperature and logs it. Global replay uses the
JSON `recent_replay_capacity` for the current buffer, and requires reward
relabelling for SAC. These are algorithm choices, not independent numeric
study presets. The notebook explicitly selects seed sets and study cases.

## Editing settings

For either Fetch study, edit `training.eval_freq` in the loaded JSON. The checked-in
value is **1,000 environment steps**. If a phase or selected JSON case explicitly
specifies `eval_freq`, edit that more specific entry instead.

Existing base values have been preserved, including learning rates, alpha,
regularisers, and **8 training evaluation episodes**. The helper previously
silently replaced that count with 32. If you want 32, set `eval_episodes` to 32
in JSON. `post_eval_episodes` is explicitly 100 for the held-out baseline check.
Holding success, Huber loss, Q-magnitude diagnostics, raw inputs and independent
model initialisation are now explicit JSON settings. These values can be edited
there; they are no longer injected by the study helper.

For temperature cases, edit `study_overrides.exploration`:

- `auto`: tune immediately without floor/ceiling.
- `bounded`: delay adaptation and use the bounds declared in this entry.
- `fixed`: use its explicit numeric `training.ent_coef` and reset transfer mode.
  This coefficient is separate from `training.initial_alpha` for auto tuning.

The architecture cases compare conventional SAC, factorised critics, factorised
actor+critics without representation penalties, and TBTRL. Parameter counts
are not matched. Retention entries define `critic_replay`, `policy_anchor`,
`capacity_archive`, `gentle_transition` and `combined`. Gentle/combined acquisition
phase learning rates are explicitly listed in those JSON entries; update them
there if desired. The KL teacher is the previous task-boundary policy and cannot
restore a skill already lost at that boundary.

## Run order and restarting

1. Run all cells with `SMOKE=True` to check execution.
2. Set `SMOKE=False`, rerun setup, then screen exploration cases on one seed.
3. Choose `SELECTED_EXPLORATION` and set `RUN_CONFIRMATION=True`; confirmation
   compares architectures over seeds 42, 43 and 44.
4. Set `RUN_TRANSFER=True` after both conventional SAC and TBTRL pass the
   confirmation gate for every goal/seed. That gate requires solved training
   and at least 90% held-out holding success on at least 100 episodes. It is
   separate from the training early-stopping criterion configured in JSON.

After a kernel restart, rerun setup and confirmation with `RUN_CONFIRMATION=True`
before transfer. Matching completed jobs are reused; missing jobs will train.
Exploration-screen results alone do not satisfy confirmation. Eight rollout GIFs
are rendered inline after each full job; that display count does not determine
how many episodes are evaluated numerically.

Version 3 adds task-boundary continuation in a new output directory. Version 2 corrected JSON precedence. Historical version 1 runs used hidden
Python presets (including evaluation every 5,000 steps), so they must not be
relabelled as the corrected experiment. The user's committed FetchReach notebook
and outputs are preserved byte-for-byte under
`experiments/recorded/fetch_study_v1/`. Working notebooks start clean in smoke mode.

Completed tasks are reused when effective config, source, dependencies and device
match. Appending goals is permitted; other JSON/source changes require a fresh
directory. Interrupted tasks restart from the preceding completed boundary.
Scratch and TBTRL have separate progress. See [task continuation](task-continuation.md). Review resolved settings before launching long
runs, and compare all methods under the same corrected evaluation protocol.

## Interpreting steps

The recorded stopping step includes warm-up and the patience window. Success is
observed only at scheduled evaluations; a coarse interval can make different
learning speeds appear identical. `steps_to_threshold` currently records the
confirmed stopping step, not the first passing evaluation. Use saved success
curves to report the first observed passing step separately. More frequent
checks also change the stopping protocol; do not combine v1 and v2 stopping
costs as if they used identical criteria.
