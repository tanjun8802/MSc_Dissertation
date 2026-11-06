# TBTRL research

Terminology for sequential goal learning and its discrete-domain validation.

## Language

**Sequential goal learning**:
Learning tasks one at a time, with each new goal and its reward introduced when that task begins. Previously acquired knowledge provides the starting point for later learning.

**Task descriptor**:
The environment-specific information identifying the current objective when its task begins. Maze tasks use goal coordinates; materials-discovery tasks can identify the chemical system being explored.

**Discrete goal task**:
A goal-conditioned task in an environment with discrete states and actions. In the initial validation scope, tasks within an environment share its dynamics, state space and action space.

**Acquisition**:
Learning to perform a newly introduced task. Acquisition efficiency describes the resources needed to reach a stated performance criterion.

**Retention**:
Performance on an earlier task after learning subsequent tasks, measured before additional training on that earlier task.
_Avoid_: Recovery, relearning

**Recovery**:
Additional learning on an earlier task after subsequent tasks have been learned. Recovery efficiency is distinct from retaining the ability without further training.
_Avoid_: Retention

**Robustness**:
Persistence of a measured benefit across random seeds, task orders and a reasonable neighbourhood of hyperparameter settings.
_Avoid_: Numerical stability when referring to reliability across experiments

**Numerical stability**:
Training behaviour that remains finite and avoids numerical failures. It is one part of the broader stability assessment, alongside robustness.
