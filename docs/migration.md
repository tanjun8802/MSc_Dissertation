# Migration from the dissertation workspace

The supported starting point is commit
`0606a8f664974ecd8af2785fc575d20785f17b24`, which introduced
`src/experiments/working`. Its three working notebooks now live in
`experiments/working` and invoke one package runner.

## Source mapping

| Previous implementation | Maintained implementation |
| --- | --- |
| `trainer.dqn_train_phi_psi_adjustment` | `tbtrl.training.dqn.train_dqn` |
| `trainer.sac_train_tbtrl_maze` | `tbtrl.training.sac.train_sac` |
| `FactorisedDQN_QNetwork_BallNorm` | `tbtrl.models.dqn.FactorisedQNetwork` |
| `FactorisedTwinCriticFetch` | `tbtrl.models.sac.FactorisedTwinCritic` |
| `GaussianPolicyActorSAC` | `tbtrl.models.sac.GaussianActor` |
| Notebook environment factories | `tbtrl.environments.registry` and layouts |
| Replay buffers and sharing cells | `tbtrl.replay` |
| Notebook orchestration and plotting | `tbtrl.experiments` and JSON configs |

Trainer returns are named dataclasses instead of long tuples. Configs retain
the original layouts, fixed/sampled goals, model dimensions, major budgets and
TBTRL coefficients. Scratch and recovery models are independent.

## Deliberate corrections

- Continuous actions retain floating-point values.
- Transition-only buffers remove unused, inconsistent episode-index bookkeeping.
  Tail retention and merging work after ring-buffer wraparound.
- Identical-embedding tasks receive their buffer alias; zero-fraction merges
  append nothing. Similarities are recomputed after the final update.
- DQN relabelling preserves time limits when truncation bootstrapping is disabled.
  Goal reward and termination rules move to the environment contract.
- The unused task-code learning rate and incorrect pseudo-n-step option are removed.
- Covariance sketches belong to trainer invocations instead of a global function
  attribute that retains random state across experiments.
- Goal-separation loss returns a consistent tuple for zero/one/many goals.
- Training, action-space and evaluation seeds are explicit. Scratch regularisation
  uses only its own goal, avoiding accidental information from old tasks.
- SAC goal dimensions need not equal observation dimensions.
- SAC recovery calls the supported trainer. Its old notebook referred to a
  different, unimported trainer and depended on previous kernel state.
- Budget exhaustion is not successful threshold attainment; failures remain in
  comparison plots. Checkpoints include normalisation and effective settings.

## Removed material

Sandbox notebooks outside the working folder, obsolete trainer/model variants,
unused exploration/benchmark/robotics helpers, and their setup instructions are
removed from the maintained tree. Notebook output histories and reference PDFs
are removed too. Originals remain recoverable from the parent commit; Git
history has not been rewritten.

The core installation needs NumPy, PyTorch and Gymnasium. Jupyter and plotting
are optional. Add simulator dependencies alongside a maintained adapter instead
of installing every simulator for all users. Old `from trainer import ...` and
`from utils import ...` imports are intentionally replaced by named package APIs.

Validation covers fast CPU execution, update preservation under a controlled
fixture, and packaging. It does not rerun all full-budget dissertation experiments
or establish statistical learning performance. Rerun the full configs with
multiple seeds before relying on published learning/retention comparisons.
