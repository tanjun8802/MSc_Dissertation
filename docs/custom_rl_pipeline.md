# Custom RL Pipeline (MuJoCo + robosuite)

This framework gives you a minimal training pipeline where:

1. A Gymnasium-compatible wrapper exposes robosuite as a standard env.
2. Your custom algorithm receives `observation_space` and `action_space` at init.
3. The training loop sends observations to your algorithm and returns actions to the env.
4. Transitions are passed back to your algorithm for replay/update logic.

## Files added

- `rl_pipeline/envs/robosuite_gym.py`
- `rl_pipeline/algorithms/base.py`
- `rl_pipeline/training/loop.py`
- `scripts/run_custom_pipeline.py`

## How to run the framework

From repository root:

```bash
python scripts/run_custom_pipeline.py --env Lift --robot Panda --total-steps 5000
```

## How to plug in your algorithm

1. Implement a class inheriting `BaseAlgorithm`.
2. Implement `select_action(...)` to output a valid action for `action_space`.
3. Use `observe(...)` to store transitions in your replay/trajectory buffer.
4. Use `update(...)` for gradient steps.
5. Replace `RandomPolicyAlgorithm` in `scripts/run_custom_pipeline.py` with your algorithm class.

## Observation / action flow

- `RobosuiteGymWrapper` defines:
  - `env.observation_space`
  - `env.action_space`
- These are passed into your algorithm constructor.
- At each training step:
  - loop calls `algorithm.select_action(observation, step)`
  - loop calls `env.step(action)`
  - loop sends transition to `algorithm.observe(...)`
  - loop triggers `algorithm.update(...)` after `learning_starts`

## Notes

- By default, observations are flattened into one vector for easier custom-policy integration.
- Set `flatten_observation=False` in `RobosuiteGymWrapper` if you want dict observations.
- Keep this as your main research loop; use SB3 SAC separately as a sanity baseline.
