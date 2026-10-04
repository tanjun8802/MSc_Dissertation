# Adding an environment

Register a factory before calling `run_experiment`:

```python
from tbtrl.environments.registry import register_environment


def make_my_environment(*, goal, max_horizon=100):
    return MyGoalEnvironment(goal=goal, max_episode_steps=max_horizon)


register_environment("my-environment", make_my_environment)
```

Select the name in `ExperimentConfig.environment`; factory options go in
`environment_options`. Registration is process-local. For CLI use, import the
registration from `tbtrl.environments.registry`, or use a small Python runner
that registers the factory first. Keep optional simulator imports inside the
factory so core imports work without those dependencies.

## Shared contract

- Follow Gymnasium's seeded reset and five-value step API.
- Expose a flat numeric Box observation and fixed-size goal vector. Observation
  and goal dimensions may differ. Adapt dictionary/image observations explicitly.
- Terminate on the goal and distinguish termination from time-limit truncation.
  Provide a finite episode horizon for evaluation.
- Put `success` or `is_success` in step info. Optionally implement
  `goal_distance(observation, goal)`; missing distances are recorded as null.
- Tasks in one sequential run must share observation/action/goal dimensions and
  transition semantics. Cross-domain transfer needs an explicit representation
  adapter and replay compatibility design.

The factory returns independent environments for training, evaluation and
model dimension discovery. The runner closes the environments it owns.

## DQN

Use `Discrete(n)` actions and provide:

```python
def relabel_transitions(self, batch, goal):
    # Return [batch_size, 1] tensors on batch.obs.device:
    # rewards recomputed for this goal, and goal-specific termination flags.
    # Leave batch.truncated intact.
    return rewards, terminated
```

This permits shared replay without putting maze coordinates or reward rules
inside the trainer. See `DiscreteGoalMaze` for an example.

## SAC

Use a one-dimensional Box action vector with bounds [-1, 1]. Wrap physical
ranges with a normalised-action adapter. Replay rewards and termination belong
to the goal under which each buffer was collected. Do not enable similarity
sharing for SAC buffers.

## Extension checks

Test seeded reset/step behaviour, termination versus time limits, goal reward
semantics, and a short real training update. For DQN, relabel identical
transitions under two goals. For SAC, test independent observation and goal
dimensions and action bounds. Keep long convergence runs separate from CPU CI.
