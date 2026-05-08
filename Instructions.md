# User Instructions — MSc Dissertation RL Codebase

This guide explains how to **run experiments**, **compare RL approaches**,
**switch environments**, and **modify the core mathematical equations** of each
algorithm.  No prior knowledge of the repository internals is required.

---

## Table of Contents

1. [Quick Setup](#1-quick-setup)
2. [Running Individual Experiments](#2-running-individual-experiments)
   - 2.1 [Random Baseline](#21-random-baseline)
   - 2.2 [GCRL — Contrastive RL](#22-gcrl--contrastive-rl)
   - 2.3 [RCRL — Reward-Conditioned RL](#23-rcrl--reward-conditioned-rl)
3. [Comparing All Approaches at Once](#3-comparing-all-approaches-at-once)
4. [Using the Evaluation Notebook](#4-using-the-evaluation-notebook)
5. [Changing Environments](#5-changing-environments)
   - 5.1 [Open GridWorld](#51-open-gridworld)
   - 5.2 [Four-Rooms GridWorld](#52-four-rooms-gridworld)
   - 5.3 [Windy GridWorld](#53-windy-gridworld)
   - 5.4 [Adding a Custom Environment](#54-adding-a-custom-environment)
6. [Modifying Hyperparameters via Config Files](#6-modifying-hyperparameters-via-config-files)
7. [Modifying Core Mathematical Equations](#7-modifying-core-mathematical-equations)
   - 7.1 [Random Baseline — no equations](#71-random-baseline--no-equations)
   - 7.2 [GCRL — infoNCE Contrastive Objective](#72-gcrl--infonce-contrastive-objective)
   - 7.3 [RCRL — Q-Learning with Diverse ψ Sampling](#73-rcrl--q-learning-with-diverse-ψ-sampling)
   - 7.4 [Shared Equations (Bellman, Returns, GAE)](#74-shared-equations-bellman-returns-gae)
8. [Project Structure Reference](#8-project-structure-reference)

---

## 1. Quick Setup

```bash
# Clone (if you haven't already)
git clone <repo-url>
cd MSc_Dissertation

# Install dependencies
pip install numpy PyYAML matplotlib pandas

# Smoke test — confirm the import chain works
python -c "import sys; sys.path.insert(0, '.'); from experiments.compare_approaches import parse_args; print('OK')"
```

All experiment scripts must be run **from the repository root** (the directory
that contains `agents/`, `environments/`, `experiments/`, etc.).

---

## 2. Running Individual Experiments

### 2.1 Random Baseline

Runs a uniform-random policy on the chosen environment.  No learning occurs —
this is purely an exploration baseline.

```bash
python experiments/run_baseline.py \
    --episodes 300 \
    --height 10 --width 10 \
    --seed 42 \
    --log-dir logs/random
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | 300 | Number of training episodes |
| `--height` / `--width` | 10 / 10 | Grid dimensions |
| `--max-steps` | 800 | Maximum steps per episode |
| `--seed` | 42 | Random seed |
| `--log-dir` | `logs` | Where to write `metrics.csv` |
| `--render` | off | Print final grid layout |

---

### 2.2 GCRL — Contrastive RL

Trains the single-goal contrastive agent (Liu, Tang & Eysenbach, 2024).  The
critic `C(s, a, sf)` is learned via the infoNCE objective without any reward
signal; exploration emerges from conditioning always on the hard target goal.

```bash
python experiments/run_gcrl.py \
    --episodes 500 \
    --height 10 --width 10 \
    --goal 99 \
    --seed 42 \
    --log-dir logs/gcrl
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | 1000 | Training episodes |
| `--goal` | `n_states - 1` | Target goal (flat state index) |
| `--alpha` | 0.1 | Critic step size |
| `--temperature` | 1.0 | Softmax temperature τ |
| `--contrastive-gamma` | auto | Geometric future-state sampling γ_c |
| `--n-negatives` | 10 | Negative samples per infoNCE update |
| `--logsumexp-reg` | 0.01 | LogSumExp regularisation coefficient |
| `--eval-every` | 100 | Evaluate every N episodes |
| `--log-dir` | `logs/gcrl` | Output directory |

---

### 2.3 RCRL — Reward-Conditioned RL

Trains a Q-table `Q[s, ψ-bin, a]` conditioned on a reward parameterisation ψ.
Always acts under the nominal ψ* (sparse goal reward) but trains on a mixture
of alternative ψ values to improve robustness.

```bash
python experiments/run_rcrl.py \
    --explore-episodes 400 \
    --exploit-episodes 100 \
    --height 10 --width 10 \
    --goal 99 \
    --seed 42 \
    --log-dir logs/rcrl
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--explore-episodes` | 400 | Phase 1 (Q-learning with diverse ψ) |
| `--exploit-episodes` | 100 | Phase 2 (greedy under ψ*) |
| `--n-psi-bins` | 5 | Discrete ψ bins |
| `--psi-min` | -0.1 | Most negative step-cost weight |
| `--psi-mix-alpha` | 0.5 | Fraction of nominal ψ* draws |
| `--alpha` | 0.1 | Q-learning step size |
| `--epsilon` | 1.0 | Initial ε for ε-greedy |
| `--epsilon-min` | 0.05 | Minimum ε after decay |
| `--epsilon-decay` | 0.995 | Multiplicative ε decay per episode |
| `--log-dir` | `logs/rcrl` | Output directory |

---

## 3. Comparing All Approaches at Once

`experiments/compare_approaches.py` runs **Random, GCRL, and RCRL** in sequence
on the same environment and saves structured logs for the evaluation notebook.

```bash
python experiments/compare_approaches.py \
    --env gridworld \
    --height 10 --width 10 \
    --episodes 300 \
    --max-steps 800 \
    --goal-state 99 \
    --start-state 0 \
    --eval-every 50 \
    --seed 42 \
    --log-dir logs/compare
```

Supported `--env` values:

| Value | Environment |
|---|---|
| `gridworld` | Open GridWorld (default) |
| `four_rooms` | Four-Rooms GridWorld |
| `windy` | Windy GridWorld (stochastic) |

Logs are written to `<log-dir>/random/`, `<log-dir>/gcrl/`, `<log-dir>/rcrl/`.
Each subdirectory contains:

| File | Contents |
|---|---|
| `metrics.csv` | Per-episode reward, length, mode (train/eval) |
| `coverage.csv` | Cumulative unique-state counts per episode |
| `visit_counts.npy` | Per-cell visit totals for the heatmap |
| `trajectory.csv` | Last evaluation episode trajectory |
| `q_early/mid/late.npy` | Q-value snapshots (RCRL) |
| `c_table.npy` | Full contrastive critic table (GCRL) |

---

## 4. Using the Evaluation Notebook

Open `notebooks/evaluation.ipynb` in Jupyter.

```bash
jupyter notebook notebooks/evaluation.ipynb
```

**Step-by-step guide:**

1. **Section 1 — Setup & Run**: Edit the config block at the top of the cell
   (grid size, episodes, environment name, log directory, etc.) and run the cell.
   Set `FORCE_RERUN = True` to always re-run experiments, or `False` to load
   saved logs without re-running.

2. **Section 2 — Reward over Time**: Training-reward plots and comparison overlay.

3. **Section 3 — Epsilon Decay**: Exploration schedule for RCRL (and constant
   ε = 1 for Random).

4. **Section 4 — Trajectory**: Per-approach trajectory arrow maps for the last
   evaluation episode.

5. **Section 4b — State Coverage**: Quantitative table + line chart of cumulative
   unique states discovered, plus per-approach visit-frequency **heatmaps**
   showing *where* each agent explored.

6. **Section 5 — Transfer Learning**: Zero-shot and fine-tuned goal
   generalisation evaluation.

### Changing the environment in the notebook

Edit the `ENV_NAME` variable at the top of the Section 1 cell and re-run:

```python
ENV_NAME = 'four_rooms'   # 'gridworld' | 'four_rooms' | 'windy'
GRID_HEIGHT = 11
GRID_WIDTH  = 11
```

The coverage heatmap in Section 4b automatically reads the wall layout from the
chosen environment, so walls appear correctly as grey cells.

---

## 5. Changing Environments

### 5.1 Open GridWorld

Default environment.  No walls, fully connected.

```bash
python experiments/compare_approaches.py --env gridworld --height 10 --width 10
```

Configuration (`configs/default.yaml`):
```yaml
env:
  name: GridWorld
  height: 10
  width: 10
  goal_pos: [9, 9]
  walls: []
```

### 5.2 Four-Rooms GridWorld

Classic benchmark (Sutton, Precup & Singh, 1999).  The grid is partitioned
into four rooms by two perpendicular internal walls, each with a single narrow
doorway.  Recommended minimum size: 11 × 11.

```bash
python experiments/compare_approaches.py \
    --env four_rooms --height 11 --width 11 \
    --goal-state 120 --episodes 500
```

The doorway positions are computed automatically from the grid dimensions —
you do not need to specify wall positions manually.

### 5.3 Windy GridWorld

Stochastic navigation environment (Sutton & Barto, Example 6.5).  After each
action, wind in certain columns applies an upward displacement (±1 stochastic
noise in stochastic mode).

```bash
python experiments/compare_approaches.py \
    --env windy --height 7 --width 10 \
    --goal-state 69 --episodes 300
```

Wind column strengths are defined in
`environments/windy_gridworld.py → _DEFAULT_WIND`.  Edit that dict to change
which columns are windy and by how much.

### 5.4 Adding a Custom Environment

1. Create a new file in `environments/` (e.g. `environments/maze.py`).
2. Subclass `environments.gridworld.GridWorld` (or `environments.base_env.BaseEnv`
   for non-grid environments).
3. Override `step()` if your environment has non-standard dynamics.
4. Register the new environment in `environments/__init__.py`:

```python
# environments/__init__.py
from environments.maze import MazeEnv

def make_env(name, **kwargs):
    name = name.lower().replace('-', '_')
    if name in ('maze',):
        return MazeEnv(**kwargs)
    # ... existing cases ...
```

5. Use `--env maze` when calling any experiment script.

---

## 6. Modifying Hyperparameters via Config Files

Each algorithm has a YAML config file in `configs/`:

| File | Algorithm |
|---|---|
| `configs/default.yaml` | Shared defaults / Random baseline |
| `configs/gcrl.yaml` | GCRL (Contrastive RL) |
| `configs/rcrl.yaml` | RCRL (Reward-Conditioned) |

Edit the relevant YAML file and re-run — **CLI flags always override YAML
values**, so you can also override any field on the command line without
modifying the file:

```bash
python experiments/compare_approaches.py \
    --alpha 0.05 \         # override YAML alpha
    --episodes 500         # override YAML n_episodes
```

---

## 7. Modifying Core Mathematical Equations

### 7.1 Random Baseline — no equations

The random agent samples actions uniformly from `{0, 1, 2, 3}` using NumPy's
random number generator.  There is no value function or update rule.

```
File: agents/random_agent.py → select_action()
```

---

### 7.2 GCRL — infoNCE Contrastive Objective

**What it does:** learns a critic `C(s, a, sf)` (the logit that taking action
`a` from state `s` will reach future state `sf`) using a contrastive loss.

**Where to find it:**
```
agents/goal_conditioned_agent.py → finish_episode_with_contrastive_update()
```

**The infoNCE + LogSumExp update (Eq. 3 in the paper):**

```
For each mini-batch of (s, a, sf) positive pairs + N-1 negative sf' values:

    C[s, a, sf]  += α · (1 - σ(C[s, a, sf]) - logsumexp_reg · softmax(C[s, a, ·]))
    C[s, a, sf'] -= α · (σ(C[s, a, sf']) + logsumexp_reg · softmax(C[s, a, ·]))
```

**To modify the objective**, edit the loop inside
`finish_episode_with_contrastive_update()`:

```python
# Approximate gradient of Eq. 3 (infoNCE + LogSumExp reg)
# Positive update: push C[s, a, sf] up
pos_grad = 1.0 - _sigmoid(C_pos) - self.logsumexp_reg * softmax_all[a_idx]
self.C[s, a_idx, sf] += self.alpha * pos_grad

# Negative update: push C[s, a, sf'] down
for j, sf_neg in enumerate(negatives):
    neg_grad = _sigmoid(C_neg[j]) + self.logsumexp_reg * softmax_neg[j]
    self.C[s, a_idx, sf_neg] -= self.alpha * neg_grad
```

**Key parameters:**
- `alpha` — step size for critic updates
- `temperature` — softmax temperature τ that controls policy greediness
  (`π(a|s,g) ∝ exp(C[s,a,g] / τ)`)
- `logsumexp_reg` — strength of the LogSumExp regularisation (paper default: 0.01)
- `n_negatives` — number of negative samples per infoNCE update
- `contrastive_gamma` — controls the geometric future-state sampling distribution
  (`Δ ~ Geom(1 - contrastive_gamma)`); increase for longer episodes

**Policy (action selection):**
```
File: agents/goal_conditioned_agent.py → select_action()

π(a | s, g) = softmax(C[s, :, g] / τ)
```

---

### 7.3 RCRL — Q-Learning with Diverse ψ Sampling

**What it does:** trains `Q[s, ψ-bin, a]` via standard Q-learning, but at
each update samples ψ from a mixture and recomputes the reward `r_ψ`
without new environment interaction.

**Where to find it:**
```
agents/reward_conditioned_agent.py → update()
```

**The TD update (per-step):**

```
1. Decompose transition → reward components c₁ (goal reached), c₂ (step taken)
2. Sample ψ-bin from PΨ = psi_mix_alpha · δ(ψ*) + (1−α) · Uniform(Ψ)
3. Compute parameterised reward:  r_ψ = ψ₁ · c₁ + ψ₂ · c₂   (ψ₁ = 1.0 fixed)
4. TD update:
       Q[s, ψ-bin, a] += α · (r_ψ + γ · max_a' Q[s', ψ-bin, a'] - Q[s, ψ-bin, a])
```

**To modify the reward decomposition**, edit `_decompose_reward()`:
```python
# agents/reward_conditioned_agent.py → _decompose_reward()
# c1 = 1 if goal reached (terminated), else 0
# c2 = 1 always (step indicator)
c1 = float(terminated)
c2 = 1.0
```

**To change the mixture distribution PΨ**, edit `update()`:
```python
# Sample ψ-bin: nominal (bin 0) with probability psi_mix_alpha, else uniform
if self.np_random.random() < self.psi_mix_alpha:
    psi_bin = self.nominal_psi_bin   # always bin 0
else:
    psi_bin = int(self.np_random.integers(0, self.n_psi_bins))
```

**Key parameters:**
- `alpha` — Q-learning step size
- `gamma` — discount factor
- `n_psi_bins` — number of ψ bins (more bins = finer reward variety, larger table)
- `psi_min` — most negative step-cost weight (controls the range of ψ values)
- `psi_mix_alpha` — fraction of updates using nominal ψ*

---

### 7.4 Shared Equations (Bellman, Returns, GAE)

Pure-NumPy reference implementations live in `equations/`:

| File | Contents |
|---|---|
| `equations/bellman.py` | Bellman expectation (`V^π`) and optimality (`V*`, `Q*`) equations |
| `equations/value_functions.py` | Monte-Carlo returns, TD(0) error, GAE-λ |
| `equations/policy_gradient.py` | REINFORCE, entropy bonus, PPO clip |

These are **reference implementations** used for verification and teaching —
they are not called during the tabular experiments (which implement their own
inline updates).  Modify them if you want to extend the codebase to new agents
or use them as ground-truth checks.

**Example — changing the Bellman backup:**
```python
# equations/bellman.py → bellman_optimality_q()
# Current (standard Q*):
#   Q_new[s, a] = Σ_{s'} P[s,a,s'] · (R[s,a,s'] + γ · max_{a'} Q[s', a'])

# To use Double-Q correction, replace max_{a'} Q[s', a'] with a target network:
#   Q_new[s, a] = Σ_{s'} P[s,a,s'] · (R[s,a,s'] + γ · Q_target[s', argmax Q[s']])
```

---

## 8. Project Structure Reference

```
MSc_Dissertation/
│
├── agents/                     # RL agent implementations
│   ├── base_agent.py           #   Abstract agent interface
│   ├── random_agent.py         #   Uniform-random exploration
│   ├── goal_conditioned_agent.py   # GCRL: infoNCE contrastive critic
│   └── reward_conditioned_agent.py # RCRL: Q[s, ψ, a] with diverse ψ sampling
│
├── environments/               # Grid environments
│   ├── base_env.py             #   Gymnasium-compatible abstract base
│   ├── gridworld.py            #   Open GridWorld
│   ├── four_rooms.py           #   Four-Rooms GridWorld (walls + doorways)
│   ├── windy_gridworld.py      #   Stochastic Windy GridWorld
│   └── wrappers.py             #   TimeLimit, NormaliseObservation wrappers
│
├── experiments/                # Experiment runners and CLI entry points
│   ├── base_experiment.py      #   Abstract train / eval loop
│   ├── run_baseline.py         #   Random baseline CLI
│   ├── run_gcrl.py             #   GCRL CLI
│   ├── run_rcrl.py             #   RCRL CLI
│   └── compare_approaches.py  #   Side-by-side comparison of all three
│
├── equations/                  # Pure-NumPy RL equation reference
│   ├── bellman.py              #   Bellman expectation & optimality
│   ├── value_functions.py      #   MC returns, TD error, GAE
│   └── policy_gradient.py      #   REINFORCE, entropy bonus, PPO clip
│
├── mdp/                        # MDP utilities
│   ├── tabular_mdp.py          #   Explicit P/R tables + value iteration
│   ├── reward_free_mdp.py      #   Reward-free wrapper
│   └── shortest_path.py        #   BFS optimal-baseline solver
│
├── utils/                      # Shared utilities
│   ├── config.py               #   YAML config loader
│   ├── logger.py               #   CSV + stdout logger
│   ├── metrics.py              #   EpisodeMetrics dataclass
│   ├── replay_buffer.py        #   Fixed-capacity circular buffer
│   └── transfer_eval.py        #   Zero-shot / fine-tuning transfer helpers
│
├── configs/                    # YAML hyperparameter files
│   ├── default.yaml            #   Shared defaults
│   ├── gcrl.yaml               #   GCRL hyperparameters
│   └── rcrl.yaml               #   RCRL hyperparameters
│
├── notebooks/
│   └── evaluation.ipynb        #   Interactive evaluation & visualisation
│
├── Instructions.md             #   ← You are here
└── README.md                   #   Project overview
```

---

## Common Pitfalls

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Run scripts from the **repository root**, not from inside `experiments/` |
| `KeyError: '--goal-state'` | Use `--goal-state` (not `--goal`) for `compare_approaches.py` |
| Empty heatmap in notebook | Ensure `FORCE_RERUN = True` in Section 1 and re-run the cell |
| Heatmap shows wrong walls | Make sure `ENV_NAME` in the notebook matches the `--env` used to generate logs |
| GCRL never reaches goal | Try reducing `--contrastive-gamma` (e.g. `0.9` for small grids) |
| RCRL slow to converge | Increase `--explore-episodes` or reduce `--epsilon-decay` |
