# Controlled temperature and retention experiments

Open `experiments/working/TBTRL_GridworldMaze_SAC_Transfer.ipynb`. Install with
`uv sync --locked --extra notebooks --group dev`, select that Python kernel,
and restart the kernel after pulling changes. Run all cells. The default is
`EXPERIMENT = "temperature_carry"`, `ACTOR_VARIANT = "factorised_tbtrl"`, and
`SMOKE = False`. Logs and all plots display in the notebook's training cell.

The prior saved factorised-actor run solved all four sequential tasks but used
1.06 million steps versus 890,000 from scratch, and ended at 12.5% success on
each of the first two goals. Its temperature repeatedly restarted at 0.2 after
finishing near 0.001–0.002. Recovery restarted at 1.0. These observations motivate
the following ablations; they do not establish which change will improve learning.
The existing notebooks, saved outputs and original JSON configurations remain intact.

## Start with the temperature comparison

| EXPERIMENT | Fresh scratch / first sequential alpha | Later sequential / recovery alpha | Other changes |
| --- | --- | --- | --- |
| `historical_reset` | 0.1 / 0.2 | reset to 0.2 / 1.0 | Previous numerical settings, new probes |
| `matched_reset` | 0.1 / 0.1 | reset to 0.1 / 0.1 | Matched reset control |
| `temperature_carry` | 0.1 / 0.1 | inherit from the starting model | Default; temperature intervention only |
| `scaled_regularization` | 0.1 / 0.1 | inherit | Carry plus actor covariance-scale correction |
| `policy_retention` | 0.1 / 0.1 | inherit | Carry plus frozen-policy KL on old replay states |

Run `matched_reset` and `temperature_carry` with the same actor and seeds first.
The optional last two presets each extend `temperature_carry` independently.
They are not silently combined. `historical_reset` adds diagnostics but preserves
the previous training settings. New observations do not count toward early stopping.

The scalar learned alpha transfers between sequential tasks; Adam moments do not.
Each seed and each fresh scratch model starts independently. Every recovery task
starts from a separate copy of the final sequential model and its final alpha,
not from the preceding recovery attempt. Automatic entropy tuning remains active,
so alpha can increase if the new task's policy is too concentrated. There is no
new fixed-temperature override or arbitrary floor. Low carried temperatures may
still limit exploration; inspect early progress rather than assume transfer helps.

Model architecture, reward, critic coefficients, update frequency, replay sampling,
original stopping criterion/patience, 10,000 random warmup steps, 500,000-step task
budgets and 100,000-step recovery budgets remain unchanged. Recovery retains its
original unclipped targets; this is constant across the matched/carry comparison.
Input normalisation is disabled in these Maze configs. This experiment does not
introduce normaliser or full optimiser-state transfer.

`ACTOR_VARIANT` supports `baseline` (MLP), `factorised_only`, `mlp_replay`, and
`factorised_tbtrl`. This keeps architecture, replay and representation penalties
separable. `scaled_regularization` requires `factorised_tbtrl`; `policy_retention`
supports the two actor-replay variants. The same temperature preset applies to all
actor variants. Use additional seeds by setting `config.seeds` before running;
all runs use unique directories and save their effective settings in checkpoints.

## Early transfer and retention measurements

The new probes evaluate every seen goal, including the current goal, at environment
steps 0, 9,999 (before learning), 10,000 (first update), 10,100, 11,000 and 15,000,
then every ordinary evaluation interval. Probes use separate environments and
fixed seeds within a task. They restore Python, NumPy, CPU, CUDA and MPS RNG states
where available. They do not collect training data, update normalisers or change
the early-stopping counter. Regular evaluations remain unchanged.

- `transfer-probes.json` records environment steps, optimiser update count, alpha,
  all seen goals' success/return/length/distance, and sampled gradient diagnostics.
- `transfer-probes.png` shows temperature, all-goal retention, and the first updates.
- `actor-gradients.png` shows the pre-clipping gradient norms of each **weighted**
  actor-loss component and current/replay gradient cosine. Negative cosine suggests
  conflicting directions on that sampled batch; zero also represents a missing or
  zero gradient and should not be read as evidence of orthogonality in that case.
- `metrics.json` records effective `initial_alpha` and `final_alpha`; both are also
  recoverable from checkpoint settings/state. The notebook prints a phase summary.

Probe evaluations add wall-clock overhead but no training environment steps.
Use the same probe settings when comparing runtimes. Their evaluation seeds differ
from the legacy stopping evaluations, so success rates need not match exactly.

## Optional covariance-scale correction

Previously actor covariance was pushed toward `I` in rank 16 while factor norms
were penalised above 2. Identity covariance implies centred RMS norm 4, so these
objectives prefer conflicting scales. The optional preset sets

```text
v = phi_norm_target**2 / (2 * latent_dim) = 0.125
R_cov = ||Cov(phi) / v - I||_F**2 / latent_dim
```

This targets centred RMS norm sqrt(2), leaving some squared-norm budget for the
mean. The target is soft and does not guarantee every sample stays below 2.
This scale-normalised loss is implemented by applying the existing covariance
regulariser to `phi / sqrt(v)`. It is averaged across each mean/std action head.
Coefficients and the critic's original regularisers remain unchanged. The preset
is an explicit actor ablation, not a claim that 0.125 is optimal. The default
`covariance_target_variance = 1.0` preserves the original objective.

## Optional policy retention

At each training call with old replay buffers, freeze a copy of the incoming actor.
Add a small independent loss on selected old replay states:

```text
L_actor += 0.01 * mean_old_tasks E_s[KL(pi_previous(.|s,g) || pi_current(.|s,g))]
```

Compute the analytic diagonal Gaussian KL before tanh. Since both distributions
use the same invertible tanh transformation, this is also their exact theoretical
KL after squashing. It needs no sampled actions or inverse tanh. The snapshot gets
no gradients; the current actor's mean and standard-deviation factors both adapt.
The term is zero at the start, grows when old policies drift, and supplements the
existing SAC old-task objective. It cannot recover behaviour already forgotten by
the incoming actor or protect states absent from replay. Its coefficient 0.01 is
an untuned starting value and is off in the default temperature experiment.

## Validation and interpretation

Tests exercise all five presets, every actor variant, multiple seeds, sequential
and independent recovery temperature provenance, KL gradients, scaled covariance,
checkpoint persistence, and exact training-weight equality with probes and gradient
diagnostics enabled/disabled. Notebook smoke checks require the new inline plots
and log messages. Legacy baseline equivalence remains a separate regression check.

Full-budget convergence is not established by smoke tests. Compare steps to the
original threshold, zero-shot success, early loss of existing skills, final retention,
and runtime. A successful carry-over intervention should improve these measured
outcomes, not merely make a representation plot look different.
