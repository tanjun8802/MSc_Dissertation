# Complex Simulation Stack Setup (MuJoCo + robosuite + Gymnasium + Isaac Lab)

This guide prepares a future-proof robotics RL stack for dissertation experiments.

## 1) Stack roles and how they interact

- **MuJoCo**: low-level rigid-body physics simulator (contacts, dynamics, control stepping).
- **robosuite**: task/environment layer on top of MuJoCo (Franka/UR5, manipulation tasks, sensors, rewards).
- **Gymnasium wrapper**: standard RL API (`reset`, `step`, `observation`, `action_space`) so policies/trainers can be swapped easily.
- **Isaac Lab (optional)**: GPU-first simulator/training ecosystem for large-scale parallel rollout when MuJoCo+robosuite CPU rollouts become a bottleneck.

### Data/control flow

1. RL algorithm chooses action from current observation.
2. Gymnasium wrapper forwards action to robosuite env.
3. robosuite calls MuJoCo physics step.
4. New simulator state is converted into Gymnasium observation/reward/termination flags.
5. Transition is returned to the RL algorithm.

Isaac Lab can be used as a separate backend when moving to high-throughput training, while preserving the same high-level training loop ideas.

---

## 2) Recommended environment strategy

Use **separate virtual environments**:

- `venv-mujoco-robosuite`: day-to-day robotics RL prototyping.
- `venv-isaac`: optional high-throughput/GPU experiments (to avoid dependency conflicts and heavy installs in the base workflow).

---

## 3) Baseline install (MuJoCo + robosuite + Gymnasium)

From repository root:

```bash
cd /home/runner/work/MSc_Dissertation/MSc_Dissertation
python -m venv .venv-sim
source .venv-sim/bin/activate
python -m pip install --upgrade pip
pip install mujoco robosuite gymnasium numpy
```

Quick smoke test:

```bash
python - <<'PY'
import mujoco
import robosuite
print("MuJoCo:", mujoco.__version__)
print("robosuite:", robosuite.__version__)
PY
```

---

## 4) Gymnasium interface plan for robosuite

Use a thin adapter so your RL code can treat robosuite tasks like normal Gymnasium envs:

- Wrap robosuite env creation in a class exposing:
  - `reset(seed=None, options=None)`
  - `step(action)`
  - `render()`
  - `close()`
- Normalize outputs into:
  - `obs` (NumPy arrays / dict observations)
  - `reward` (float)
  - `terminated` / `truncated` flags
  - `info` dict
- Define `action_space` and `observation_space` with Gymnasium spaces.

This keeps algorithms decoupled from simulator specifics and simplifies later migration to Isaac Lab.

---

## 5) Optional Isaac Lab track (GPU-scale rollouts)

Choose Isaac Lab when you need:

- many parallel environments,
- faster wall-clock training on GPU hardware,
- larger policy/model experiments.

Operational recommendation:

- Keep Isaac Lab as a separate experiment track and environment.
- Reuse the same experiment configuration schema and logging format used for MuJoCo runs.
- Compare on common metrics (sample efficiency, final return, wall-clock training time).

---

## 6) Experiment design recommendations

- Start with MuJoCo + robosuite for correctness and iteration speed.
- Standardize observation/action preprocessing once via Gymnasium wrappers.
- Add deterministic seeds and fixed evaluation protocols.
- Move to Isaac Lab only after baselines are stable.

---

## 7) Common pitfalls

- **Dependency conflicts**: isolate Isaac Lab from base stack.
- **Observation mismatch**: lock a consistent observation format early.
- **Termination semantics**: separate `terminated` vs `truncated` correctly.
- **Control frequency mismatch**: ensure policy step rate matches simulator/control loop assumptions.

---

## 8) Suggested phased adoption

1. Install MuJoCo + robosuite + Gymnasium and validate one task rollout.
2. Integrate Gymnasium wrapper into your training interface.
3. Benchmark baseline algorithms on 1–2 canonical manipulation tasks.
4. Add Isaac Lab track only for scaling experiments.
5. Keep evaluation pipeline identical across backends for fair comparison.
