# Discrete-domain validation

Status: interview closed at the user's request on 2026-11-05. See [the DQN bounded-K implementation plan](dqn-k-replay-plan.md). The DQN bounded replay path and Maze notebook are implemented with smoke validation; full-budget scientific runs have not been launched.

## Agreed on 2026-11-04

- Start with sequential goal changes within fixed discrete environments, using Maze and FourRooms as the starting point. Tasks within each environment share dynamics, states and actions. Changes to dynamics or action spaces belong to a later stage.
- Assess faster acquisition of new tasks together with preservation of earlier-task performance. Faster acquisition accompanied by substantial forgetting is partial progress, rather than achievement of the combined objective.
- Report recovery efficiency as a separate benefit. Recovery does not substitute for retention.
- Assess robustness across random seeds, task orders and a reasonable neighbourhood of hyperparameters, alongside numerical stability.
- First reproduce the reference behaviour, then test seeds and task orders, then probe sensitivity around the selected configuration. Exact budgets and criteria remain open.

## Clarified in the second interview round

- Task identity depends on the environment: Maze uses goal coordinates; MADE uses a materials-discovery objective or chemical system. The first validation milestone remains the agreed discrete goal-task setting.
- The six-condition baseline suite proposed in Q5 is accepted: conventional goal-conditioned DQN scratch, sequential fine-tuning and sequential replay; factorised DQN scratch and sequential replay without representation penalties; full TBTRL. Individual regulariser ablations follow the combined-method assessment.
- Sequential fine-tuning is an essential comparator even if it acquires tasks quickly while forgetting earlier tasks. TBTRL aims to improve acquisition and retention together; superiority remains a hypothesis to test.
- Q6 remains unresolved. The user favours investigating bounded replay updates and dynamic selection of historical tasks to avoid growing resource demands. Selection rules, update limits, storage policy and accounting are not yet agreed.
- The restriction on development-time evaluation of future goals proposed in Q4 has not been explicitly accepted; it remains open independently of task-descriptor choice.

## Replay clarification and candidate research mechanism

- The repository already implements a bounded global replay transition buffer with similarity-based admission and grouping, group consolidation and capacity-driven thinning. The historical checked-in discrete configurations use per-task replay, but global replay is available to DQN; bounded transition storage is not a new component to invent.
- The requested additional constraint is a cap K on the number of historical task replay objectives participating in an update, independent of the number of encountered tasks. The selection mechanism remains undecided.
- Both maintained trainers already support replay_tasks_per_batch and use rotating selection. This is an existing bounded-objective control, not the proposed intelligent selector. A cap alone does not bound total replay samples when replay_ratio grows with task count; all-seen-goal penalties and task scans also remain separate resource costs.
- Candidate hypothesis, not an accepted algorithm: choose a small set of basis tasks or spanning task directions to preserve shared representation structure. Other tasks may drift but could require fewer adaptation steps on revisit. Neither protection of unselected tasks nor rapid recovery follows from spanning alone when both encoders change.
- Open distinction: maintaining a high-variance subspace, selecting actual task vectors with bounded reconstruction coefficients, and preserving task behaviour are different objectives.
- Open claim decision: how much unselected-task performance loss is acceptable, and whether recovery becomes a primary outcome alongside acquisition and retention.

## Accepted on 2026-11-05

- K is an external constraint on historical replay objectives. The first candidate selects actual task embeddings to approximate useful directions; exact spanning is not required when task rank exceeds K.
- Immediate retention and recovery remain separate outcomes. Some distortion is acceptable as a research hypothesis if useful shared structure supports substantially fewer recovery steps; the acceptable tradeoff must be measured, not assumed.
- Stop interviewing and produce a concrete implementation plan for the existing DQN trainer. The plan is linked above.

## Reference evidence and audit observations

The submitted thesis is the historical reference, not an instruction source. Its discrete retention tables report final-to-initial return ratios; those ratios are not success rates. The thesis acknowledges that hyperparameter sensitivity and the stability balance remain uninvestigated.

The maintained runner introduces task goal descriptors into training incrementally, but evaluates future configured goals after each task. Researcher visibility of future-task evaluations must be distinguished from learner exposure.

Current scratch training uses factorised networks and representation penalties; it is not a conventional DQN control. In the historical discrete configurations, replay sampling increases with task index and retained per-task buffers do not impose a fixed total memory budget; the implemented global mode supplies a bounded transition-storage alternative. Matched controls and accounting remain to be agreed.

The current discrete covariance objective targets identity covariance for 16-dimensional state-action factors whose projected norms are bounded by one. These scales cannot simultaneously be achieved. This is a diagnostic observation, not a decision to change the algorithm or evidence that the method fails.

The thesis and current runner also need a protocol reconciliation before a rerun is labelled a reproduction, including task lists, seeding, update cadence and evaluation cadence. Historical results must remain distinguishable from new protocols.

## Open decisions

- Task information available to the learner at arrival and during evaluation.
- Detailed baseline settings and individual ablations, following the accepted six-condition suite.
- Environment-step, gradient-update and memory accounting.
- Acquisition and retention metrics, acceptable tradeoffs and numerical thresholds.
- Seed counts, task-order construction, tuning/evaluation separation and compute budget.
- Diagnostics and the conditions for proposing algorithm changes.

## Sources

- User's research motivation and accepted interview answers in this chat, 2026-11-04.
- Submitted MSc thesis: chapter 5; conclusion; appendix B.
- configs/tbtrl_fourrooms_dqn.json and configs/tbtrl_gridworldmaze_dqn.json.
- src/tbtrl/experiments/runner.py and src/tbtrl/experiments/config.py.
- src/tbtrl/training/dqn.py and src/tbtrl/losses.py.
