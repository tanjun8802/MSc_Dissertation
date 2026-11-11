# TBTRL and successor features: research positioning

Reviewed 2026-11-10. Targeted primary-source review, not an exhaustive SOTA ranking. Proposed directions below are hypotheses; no comparative TBTRL results are established here. Running experiments and their code/configuration were not modified.

## Current implementation

TBTRL uses Q(s,a,g) = phi(s,a)^T psi(g). Both factors are learned through scalar value learning. This is a shared value basis, not an explicit policy-conditioned successor-feature Bellman model. See [method contract](method.md) and [bounded DQN guide](dqn-k-replay.md).

The bounded mode limits historical objectives to K, historical transition-task pairs to R, and selector candidates to M. The current objective is additional. Candidate vectors are direction-normalised for residual-greedy selection; forward values need not be normalised. This is not a proven barycentric spanner. Transition storage and update work are bounded; goal metadata and full evaluation still grow.

## Primary-source reading map

| Source | Relevance and boundary |
| --- | --- |
| [Barreto et al., SF, NeurIPS 2017](https://arxiv.org/html/1606.05312v2) | Reward revaluation and GPI support transfer. Sequential experiments already exist; the guarantee assumes accurate policy values, rather than guaranteeing their preservation during future training. |
| [Deep SF/GPI, ICML 2018](https://proceedings.mlr.press/v80/barreto18a.html) | Previous reward functions can supply features; newly learned policies join the reusable skill set. Handcrafted features are not a universal SF requirement. |
| [USFA, ICLR 2019](https://arxiv.org/abs/1812.07626) | Shared policy-conditioned prediction supports parametric generalisation and GPI. Separate reward and policy descriptors clarify what is being evaluated. |
| [Simple SF, NeurIPS 2024](https://arxiv.org/html/2410.22133v2) | Online learned features without pretraining; scalar TD plus reward prediction. Discrete A–B–A–B experiments are relevant, but revisit learning curves do not isolate immediate retention. |
| [Fast/Slow SF, ICML 2026](https://proceedings.mlr.press/v306/chua26a.html), [full text](https://arxiv.org/html/2605.26357) | Multi-timescale consolidation. Four Rooms combines reward reversals with drift; MuJoCo studies drifting dynamics. Count parameter copies and throughput costs. |
| [TD-JEPA](https://arxiv.org/html/2510.00739), [ICLR 2026 paper](https://openreview.net/pdf?id=SzXDuBN8M1) | Offline reward-free latent prediction for zero-shot control; also evaluates later adaptation. Prior data coverage and pretraining distinguish its protocol from TBTRL's sequential acquisition. |
| [Sparse-TD-JEPA, October 2026 preprint](https://arxiv.org/html/2610.14400) | Sparsity improves reported zero-shot results. Its normalised task representation makes the penalty different from TBTRL's soft radius-excess regularisation. |
| [ATLAS, August 2026 preprint](https://arxiv.org/html/2608.04334) | Growing spatial topology plus successor-based planning. Main baselines fail initial acquisition; preserved routing does not eliminate low-level controller forgetting. Not a fixed-storage comparator. |
| [Successor Clusters, TMLR 2025](https://openreview.net/pdf?id=UB22Tt3sfF) | Reachability-informed behavioural bases are established prior work; a basis alone is insufficient novelty. |
| [Successor Flow Features, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html) | Online successor-measure learning supplies another branch beyond offline pretraining; its evidence concerns continuous control, not this bounded task-replay protocol. |

## Proposed research claim

Investigate whether a fixed number of representative historical objectives preserves a reusable neural value basis as tasks arrive, improving acquisition, immediate retention and independent recovery under matched replay and compute budgets. Neither novelty nor superiority follows merely from bilinear factorisation, sequential arrival, learned features, or no pretraining.

## Analysis and candidate improvements

1. **Measure function-space similarity.** Raw embedding distances depend on factorisation coordinates. Invertible changes phi'=A phi, psi'=A^{-T} psi preserve Q but can change distances. Compare task Q profiles or action advantages on a bounded archive probe; task-coordinate plots alone do not establish behavioural similarity.
2. **Test span maintenance under joint drift.** If an unselected task code has bounded-coefficient reconstruction from selected codes, small errors in selected value functions may control its error only under additional stability and coverage conditions. Jointly moving encoders, changing coefficients and missing states break that simple argument. Record reconstruction error, value drift and policy agreement separately.
3. **Preserve decision-critical details.** Average value preservation can miss rare junctions. A proposed bounded anchor component could prioritise small action margins or remembered failures. Teacher targets require stored outputs or a teacher and must count toward memory; this is not implemented by current basis replay.
4. **Test where compression fails.** Sweep K and task diversity independently. Redundant goals and independent bottleneck tasks should produce different tradeoffs. Fixed latent dimension need not approximate every new optimal Q function well merely because dynamics are shared.
5. **Keep regularisation subordinate to behaviour.** Isotropy, variance and norm penalties cannot by themselves establish retention. Compare simple objectives before combining multiple additional losses.

## Evaluation requirements

Distinguish acquisition cost, unseen-task zero-update performance, old-task zero-update retention, and recovery steps. Recover every past task independently from the same final sequential checkpoint. Count unsuccessful runs and full budgets. Measure wall-clock alongside interactions; include replay, selection, gradient probes, memory and extra parameter sets.

Use conventional goal-conditioned DQN and factorised random-replay controls before attributing gains to representation learning or selection. Simple SF is a close online learned-feature comparator; Fast/Slow SF tests a stronger consolidation alternative. Match observations, task descriptors, reward information, termination semantics, exploration and replay use. Give pretrained methods an explicitly different condition with their prior data and compute disclosed.

A frozen archive is a useful retention reference but spends growing storage. A shared SF model can forget; a shared TBTRL model can forget. The study should identify a measured resource/performance tradeoff rather than claim that either factorisation prevents forgetting automatically.
