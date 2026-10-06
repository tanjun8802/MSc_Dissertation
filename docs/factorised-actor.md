# Factorised actor experiment

The continuous Maze experiment can now factorise the SAC actor as well as its
critics. This is an experimental extension of TBTRL to the policy objective;
it does not apply a critic Bellman residual to actions. The original MLP actor,
critic update, environment, and training budgets remain available as a baseline.
No full-budget performance improvement has been established for this extension.

## Model

For each action component j, the Gaussian parameters are

```text
mu_j(s, g)      = dot(phi_mu,j(s), psi_pi(g))
log_std_j(s, g) = clamp(dot(phi_std,j(s), psi_pi(g)), -20, 2)
a              = tanh(mu + exp(log_std) * epsilon), epsilon ~ Normal(0, I)
```

The two state networks output `[batch, action_dim, latent_dim]`; the shared actor
goal encoder outputs `[batch, latent_dim]`. Actor and critic parameters are
independent. Sampling, reparameterisation, and the tanh log-density correction
are inherited from the original Gaussian actor. Deterministic evaluation uses
`tanh(mu)`. Inputs use the same optional state/goal normalisers as the critic.

The new config uses hidden width 64 and rank 16. State-factor output layers are
scaled by 0.01 at initialisation to limit initial saturation. This is an
initialisation choice, not a hard bound on the actor's factors. Soft excess-norm
penalties allow the mean and standard deviation to adapt beyond their targets.
The factorised architecture has a different parameter count from the MLP;
these initial ablations are not parameter-count-matched.

## Objective and replay

For each task k, define the usual SAC policy objective on replay states:

```text
J_k = mean[alpha * log pi(a | s, g_k) - min(Q1, Q2)(s, a, g_k)]
      where a is freshly sampled from the current actor

L_actor = J_current + lambda_replay * mean_k(J_old,k)
          + lambda_cov * R_cov
          + lambda_sep * R_goal_separation
          + lambda_phi * R_phi_norm
          + lambda_psi * R_psi_norm
```

Old-task replay uses each selected buffer's own goal, with fresh policy actions;
it does not regress toward stored actions. It reuses the critic's old-task batch
selection and `replay_ratio`. Each selected old task has equal weight in the
old-task mean. If no eligible buffer is available, the old-task term is zero;
`actor_replay_tasks` records how many were actually used. Scratch and first-task
training have no old-task actor objective.

Critic parameters are frozen during actor optimisation while gradients through
Q with respect to actions remain enabled. There is one actor optimiser update
per original SAC update. With actor replay enabled, automatic entropy tuning
uses the same current/old task weights, divided by `1 + lambda_replay` so adding
replay does not multiply the entropy learning rate. Without actor replay, the
original entropy update is preserved.

Representation penalties follow the critic's TBTRL approach:

- Covariance regularisation uses current and selected replay states. It computes
  the existing sketched covariance penalty separately across states for each
  mean/std action head, then averages the penalties. Differences between action
  heads cannot hide a constant state representation.
- Goal separation penalises pairwise cosine similarity above the configured
  threshold among all seen actor goal embeddings; one goal gives zero penalty.
- Norm penalties are mean linear excesses above the state/goal norm targets.
  State penalties average over both branches and all action components.

All actor coefficients live under `training.actor_tbtrl`, separately from the
critic's settings. The experimental defaults are replay coefficient 1.0,
covariance and separation coefficients 1e-5, norm coefficients 0.003, norm
targets 2.0, cosine threshold 0.85, and sketch size 16. These are **untuned starting
values**, not validated optimal settings. Setting `actor_tbtrl` to `null` disables
all actor TBTRL additions. An MLP actor supports replay alone; representation
penalties require the factorised actor.

## Notebook comparisons

Open `experiments/working/TBTRL_GridworldMaze_SAC_FactorisedActor.ipynb` after
installing the notebook dependencies. Restart an existing kernel after updating
the package. Choose `VARIANT` in the parameters cell, then run all cells:

| Variant | Actor | Old-task policy objective | Actor representation penalties |
| --- | --- | --- | --- |
| `baseline` | Original MLP | Off | Off |
| `factorised_only` | Factorised Gaussian | Off | Off |
| `mlp_replay` | Original MLP | On | Off |
| `factorised_tbtrl` | Factorised Gaussian | On | On |

`SMOKE = False` retains the original 500,000-step per-task budgets, 10,000-step
warmup and evaluation interval, 100,000-step recovery budget, goals, seed,
environment, early stopping and critic settings. `SMOKE = True` checks execution
with tiny budgets and models; it cannot assess learning performance. Each run
gets a separate output directory. The original result notebooks are unchanged.

The original scratch/transfer initial entropy coefficients (0.1 versus 0.2) and
recovery coefficient (1.0) remain for historical comparability. For a controlled
scratch-versus-transfer study, explicitly align those values across phases and
use multiple seeds; compare all actor variants with the same chosen settings.
Architecture, policy replay and representation penalties are distinct changes,
so the four variants help identify which contributes to retention or learning.

Training progress streams directly into the notebook cell. After every trained
goal the callback displays learning/loss curves, rollouts, Q/policy maps, critic
and actor embeddings, raw norms, goal cosine matrices, covariance spectra,
embedding drift and parameter changes. An additional actor-loss plot shows
current SAC loss, weighted replay, total loss and weighted representation terms.
The same diagnostics are saved next to each checkpoint.

`losses.jsonl` includes `actor_current`, `actor_replay` (unweighted),
`actor_replay_tasks`, `actor_regularization` (weighted sum), the four unweighted
`actor_sigreg` / `actor_goal_separation` / `actor_phi_norm` / `actor_psi_norm`
terms, actor factor norms and goal similarity. `actor` is the total optimised
loss. Checkpoints save actor architecture/settings and can be loaded with the
existing `load_policy` API, including normalised-input experiments.

The CLI runs the full variant from the new JSON config:

```bash
uv run tbtrl --config configs/tbtrl_gridworldmaze_sac_factorised_actor.json \
  --output runs/factorised-actor-smoke --smoke
```

## Validation scope

Regression checks cover the bilinear outputs, squashed Gaussian density,
gradients through policy actions and every actor factor, weighted replay and
regularisation losses, old-goal conditioning, checkpoint reload with differing
observation/action/goal dimensions, and diagnostics that leave trained weights
unchanged. Fresh-kernel notebook smoke checks require inline logs and plots.
The legacy comparison verifies exact baseline model/target/actor parameter
parity against the pre-refactor trainers, including old-task replay. These checks
validate implementation behaviour, not convergence. MPS execution requires an
MPS-capable runtime; CPU diagnostics also reject float64 device transfers to
guard against the previously reported MPS failure.
