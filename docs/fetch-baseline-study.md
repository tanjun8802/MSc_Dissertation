# Fetch baseline and retention study

Use `experiments/working/TBTRL_FetchPickAndPlace_Baseline_Study.ipynb` or
`TBTRL_FetchReach_Baseline_Study.ipynb`. Existing executed notebooks and their
configurations are preserved. New settings are development hypotheses, not a
claim of convergence. The shared SAC density correction changes future SAC runs,
so rerun controls rather than comparing new training against historical outputs.

## Run order

1. Execute all cells with `SMOKE=True` to check the local environment. Smoke uses
   small networks and budgets and cannot pass the scientific baseline gate.
2. Set `SMOKE=False`. The exploration cell runs conventional SAC on every
   configured goal for seed 42, independently for automatic, bounded automatic,
   and fixed temperature. Each job allows 500,000 environment steps. Review the
   individual failures and plots; do not select by the best single goal.
3. Set `SELECTED_EXPLORATION` and `RUN_CONFIRMATION=True`. This compares four
   architecture/objective controls across seeds 42, 43 and 44: conventional SAC,
   factorised critics with an MLP actor, factorised actor/critics without factor
   penalties, and TBTRL with factor penalties. Parameter counts differ; these
   are not capacity-matched controls. Additional seeds are preferable for paper
   claims. A separate mean-only factorised actor remains a future ablation.
4. Enable `RUN_TRANSFER` only after conventional SAC and TBTRL scratch pass on
   every configured goal and all three seeds. The code requires solved training
   and at least 90/100 held-out episodes with sustained holding. A failing gate
   stops the transfer experiment rather than producing a misleading comparison.
   If scratch is still weak, extend the exploration search before increasing the
   transfer budget. A gate passing is empirical evidence, not a population-level
   statistical guarantee. Use fresh confirmation seeds after extensive tuning.

The default three-goal Pick-and-Place pilot is unchanged; the notebook reads and
prints the actual configured coordinates. Increase the goal count only after
the initial baseline is reliable. The arm reset remains fixed, the block is
randomised in Pick-and-Place, rewards remain native dense, and HER is disabled.

## Explicit controls

All new acquisition cases use actor/critic learning rates 3e-4, temperature rate
1e-4, initial alpha 0.03, 10k random-action warm-up, batch size 256 and Huber TD
loss. The automatic case starts tuning immediately. The bounded case holds the
initial temperature for 50k environment steps, then adapts within [0.01, 0.2].
This is a combined floor/delay intervention; neither setting is claimed optimal.
The fixed case uses alpha 0.03 throughout. Every resolved setting is saved.

Learning checks use 32 episodes every 5k steps. Early stopping requires at least
90% holding success for three consecutive checks. Holding means success on every
one of the final ten steps. Final evaluation uses 100 separate seeded episodes.
Reach, final-position and holding rates are all saved; early stopping and
retention use holding. The native environment still runs to its time limit;
success does not terminate an episode. Eight GIFs remain inline after each full
training job: these visualise eight episodes, not the entire 100-episode sample.

Both Gaussian actors now use the exact stable tanh log-Jacobian:

`2 * (log(2) - z - softplus(-2*z))`.

This avoids zero entropy gradients when float32 tanh saturates. Critic values,
targets and gradients are checked for non-finite values. The new study stops
with an explicit error if critic/target magnitude exceeds 1000. That limit is
a diagnostic guard, not value clipping or a convergence remedy. Failures retain
the training log and are not silently counted as successful/complete jobs.

Raw observations and goal coordinates remain consistent across tasks. Dynamic
normalisation is not enabled: resetting running statistics at each task would
change the meaning of inputs to transferred networks. Static coordinate scaling
and exploration curricula remain separate experiments if these controls fail.

## Retention interventions

`configure_fetch_study(..., architecture="tbtrl", retention=...)` exposes:

| Case | Change from control |
| --- | --- |
| `critic_replay` | Historical critic TD coefficient 1 instead of 0.05 |
| `policy_anchor` | KL(previous task-boundary policy || current policy), coefficient 0.01 |
| `capacity_archive` | Keep the whole selected recent tail before capacity pressure; similarity still groups tasks and quotas enforce the fixed budget |
| `gentle_transition` | Sample the transferred policy during warm-up, carry Adam moments and the target critic, reduce transfer actor/critic rates to 1e-4 |
| `combined` | All four interventions |

Scratch and the first sequential task use identical acquisition settings in all
retention cases. Recovery branches copy the same final sequential models,
optimizer states and replay independently. Carrying optimizer state restores
moments but honours the new phase's learning rates. Neither optimizer carrying
nor policy warm-up is enabled in old experiment configs.

The KL teacher is a snapshot at the start of each task, not a bank of each goal's
best historical policy. It cannot restore a skill already lost at that boundary.
The actor still uses fresh-action SAC replay. Existing TBTRL factorisation and
regularisers remain in the TBTRL control. Fetch critic soft norm targets are 8
rather than 2, allowing useful value magnitudes without immediately fighting
both penalties. Actor covariance target variance is `2 / latent_dim`, consistent
with the norm budget instead of an identity covariance with excessive trace.
These scales also need validation; simply increasing every coefficient is not
the experiment being performed.

## Restarting and reviewing

Every baseline goal/seed/case is a separate job. Rerunning a cell reuses completed
jobs only when source, configuration, device and dependency fingerprints match. A completed summary
is the commit point; missing completion receipts are treated as interrupted
jobs. Partial jobs need a new output directory and restart from scratch. Transfer
jobs reuse a completed full sequence, not an intermediate task checkpoint.
This is **not** the MADE campaign-level resume implementation and does not claim
exact mid-training continuation. Changing source or settings requires a new
study output directory. Dependencies and device are recorded in run manifests. Scratch model initialisation is seeded independently for each goal; in the study the critic has an independent random stream, so changing the actor does not also change its initial critic. Parameter counts are recorded in each task record.

Plots use shared publication styling, PNG/PDF/SVG exports and explicit unsolved
markers. Baseline points represent individual goal/seed results; horizontal lines
are descriptive means, not confidence intervals. Logs, temperature, critic loss,
all three success metrics, actor penalties and gradients remain available in
the per-job files and notebook diagnostics. Smoke execution validates plumbing,
not learning quality.
