# TBTRL implementation contract

## DQN

`tbtrl.training.dqn.train_dqn` is extracted from the working
`dqn_train_phi_psi_adjustment` trainer. It computes
`Q(s,a,g) = phi(s,a)^T psi(g)` and projects each embedding into its configured
L2 ball. The product of the radii bounds the Q-value magnitude.

Each update minimises:

```text
TD(current)
  + replay_loss_coef * mean(TD(each selected old task))
  + sigreg_coef * covariance_penalty(phi(current))
  + goal_separation_coef * separation(psi(all seen goals))
  + phi_raw_norm_coef * smooth_L1_excess_norm(raw_phi(current))
  + psi_raw_norm_coef * smooth_L1_excess_norm(raw_psi(all seen goals))
```

Goal separation penalises squared excess cosine similarity above the threshold.
SIGReg here names the existing covariance-to-identity penalty; it is not a new
implementation of another paper's regulariser.

Replay rewards and goal termination are recomputed by
`env.relabel_transitions(batch, goal)`. Targets use the current target network,
not saved embedding targets. Time-limit flags obey `bootstrap_on_truncation`.
Only one-step transitions are supported: the unused task-code learning rate
and misleading multi-step option have been removed. State-action and goal
encoders have separate Adam optimisers and gradient clipping; target parameters
use Polyak updates and receive no gradients.

`replay_ratio` is **additional replay samples relative to the current batch**,
not a probability constrained to [0, 1]. DQN retains the working schedule
`task_id + 1`. Old task IDs rotate deterministically and losses are averaged
over selected tasks, including when two tasks share a buffer.

Maze DQN retains similarity-based sharing: below cosine 0.8, allocate a separate
buffer; above it, append a fraction `(1-s)/(1-0.8)` of the new retained data to
the nearest task's buffer. At effectively identical embeddings, share without
appending. Similarities are recomputed from the final task model. All task
aliases are recorded, including the identical-embedding case. Retaining the
last 40% preserves the buffer's capacity for subsequent merges.

## SAC

`tbtrl.training.sac.train_sac` is extracted from `sac_train_tbtrl_maze`.
Each critic has independent state-action and goal encoders. Its target is:

```text
r + gamma * bootstrap_mask * (min(Q1_target, Q2_target) - alpha * log_pi)
```

Each critic branch receives its own TD, replay, covariance, goal-separation,
and norm penalties. The parameter groups must be disjoint and cover every
trainable critic parameter. Covariance and phi-norm penalties use the combined
current and replay samples. Psi penalties use all seen goals. SAC retains the
working linear excess-norm and excess-cosine penalties; these intentionally
differ from DQN's smooth-L1 and squared-cosine penalties.

The actor optimises the entropy-regularised minimum-critic objective on the
current task. Critic parameters are temporarily frozen for this step. Entropy
temperature is learned with `ent_coef="auto"`. Target clipping, gradient clipping
and Polyak updates retain their configured behaviour. SAC task buffers keep
their original rewards and must not be shared across reward tasks without a
relabelling implementation.

## Task boundaries and evaluation

| State | Scratch | Sequential | Independent recovery |
| --- | --- | --- | --- |
| Online weights | Fresh per task | Carried to next task | Copy of the same final sequential model |
| Target weights | Copied from online | Reset from online by default | Reset from copied online by default |
| Optimisers | New | New each task | New |
| Current replay | Empty | Empty | Empty |
| Old replay | None | Retained by task memory | DQN: none; SAC: every other task |
| Known goals | Current goal | All seen goals | All training goals |
| Running normalisation | New | New each task | New |
| SAC temperature | Initially 0.1 in scratch runner | Configured initial value (0.2) | Configured initial value |

Normalisation is disabled in supplied configurations. When enabled, final
statistics are returned and saved for consistent evaluation. The runner seeds
model initialisation and each environment/action space. Task seeds use
`seed + task_id`, removing dependence on previous notebook random state.
Independent recovery does not mutate the final forward model or replay memories.
SAC recovery runs below the retention threshold (0.9); smoke mode exercises all
recovery paths.

Trainer evaluation retains legacy scheduling: evaluation occurs on update
steps divisible by `eval_freq`, after warmup. Full DQN has `train_freq=3` and
`eval_freq=1000`, giving an effective 3000-step interval. Unsuccessful runs retain
their full budget cost. The runner also evaluates all goals after every completed
task using separate seeded environments. Future-goal scores are zero-update
evaluations; subsequent training is transfer through fine-tuning.

## Validation scope

The tests exercise actual optimiser updates with replay and all regularisers,
SAC parameter separation, target isolation, float actions, replay wraparound and
sharing, and all supported configurations through scratch/transfer/recovery and
checkpoint loading. Notebook validation uses fresh kernels.

`scripts/verify_legacy_equivalence.py` compares original and new trainers over
eight environment steps (seven updates), identically seeded environments,
old-task replay and regularisation. It uses each implementation's replay class
and compares online, target and SAC actor tensors exactly. Projection-free sketch
dimensions match the working configurations. This is not full-budget learning
curve equivalence: the seeding and correctness fixes change behaviour outside
that controlled comparison.
