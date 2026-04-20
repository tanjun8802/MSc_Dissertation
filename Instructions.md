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
   - 2.4 [ExDM — Exploratory Diffusion Model](#24-exdm--exploratory-diffusion-model)
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
   - 7.4 [ExDM — Score-Based Diffusion Intrinsic Reward](#74-exdm--score-based-diffusion-intrinsic-reward)
   - 7.5 [Shared Equations (Bellman, Returns, GAE)](#75-shared-equations-bellman-returns-gae)
8. [PyTorch and Deep RL](#8-pytorch-and-deep-rl)
9. [Project Structure Reference](#9-project-structure-reference)

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

### 2.4 ExDM — Exploratory Diffusion Model

ExDM (Ying et al., 2025) drives reward-free exploration by maintaining a
**state diffusion score model** that fits the empirical distribution of visited
states.  The score-prediction error is used as an intrinsic reward — states
that are rarely visited are poorly modelled and therefore receive high reward.

```bash
python experiments/run_exdm.py \
    --episodes 500 \
    --height 10 --width 10 \
    --goal 99 \
    --seed 42 \
    --log-dir logs/exdm
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--episodes` | 500 | Training episodes |
| `--goal` | `n_states - 1` | Flat state index of the evaluation goal |
| `--alpha` | 0.1 | Q-learning step size (behaviour policy) |
| `--model-lr` | 0.01 | SGD learning rate for the score model W[t] |
| `--n-diffusion-steps` | 10 | DDPM diffusion timesteps T |
| `--epsilon` | 1.0 | Initial ε for ε-greedy exploration |
| `--epsilon-min` | 0.05 | Minimum ε after decay |
| `--epsilon-decay` | 0.995 | Multiplicative ε decay per episode |
| `--buffer-capacity` | 10000 | Replay buffer for the score model |
| `--batch-size` | 32 | Mini-batch size for score model updates |
| `--n-model-updates` | 5 | Score model gradient steps per environment step |
| `--reward-samples` | 10 | (ε, t) samples used to estimate R_score(s) |
| `--eval-every` | 50 | Evaluate every N episodes |
| `--log-dir` | `logs/exdm` | Output directory |

---

`experiments/compare_approaches.py` runs **Random, GCRL, RCRL, ExDM, and Optimal (BFS)** in sequence
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

Logs are written to `<log-dir>/random/`, `<log-dir>/gcrl/`, `<log-dir>/rcrl/`, `<log-dir>/exdm/`.
Each subdirectory contains:

| File | Contents |
|---|---|
| `metrics.csv` | Per-episode reward, length, mode (train/eval) |
| `coverage.csv` | Cumulative unique-state counts per episode |
| `visit_counts.npy` | Per-cell visit totals for the heatmap |
| `trajectory.csv` | Last evaluation episode trajectory |
| `q_early/mid/late.npy` | Q-value snapshots (RCRL / ExDM) |
| `q_table.npy` | Full Q-table (RCRL / ExDM) |
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
| `configs/exdm.yaml` | ExDM (Exploratory Diffusion Model) |

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

### 7.4 ExDM — Score-Based Diffusion Intrinsic Reward

**What it does:** trains a state diffusion model online on the empirical
distribution of visited states.  Its reconstruction MSE serves as an intrinsic
reward that steers the agent toward less-explored regions of the grid.

**Reference:** Ying et al. (2025), arXiv:2502.07279.

**Where to find it:**
```
agents/exdm_agent.py
```

**Forward diffusion (DDPM, linear noise schedule):**

```
s_t = √ᾱ_t · s_onehot + √(1−ᾱ_t) · ε,   ε ~ N(0, I)

where  ᾱ_t = Π_{k=1}^{t} (1 − β_k),  β_k = linspace(β_start, β_end, T)[k]
```

**Score model:**
```
# agents/exdm_agent.py → _score_model_update()
# One weight matrix W[t] ∈ R^{n_states × n_states} per diffusion timestep.

ε̂  =  W[t] @ s_t                          # predicted noise
Loss = ‖W[t] @ s_t − ε‖²                  # MSE
W[t] -= model_lr · (ε̂ − ε) ⊗ s_t         # SGD update (outer product)
```

**Score-based intrinsic reward (Eq. 8 in the paper):**
```
# agents/exdm_agent.py → _compute_intrinsic_reward()

R_score(s) = (1/K) Σ_{k=1}^{K} ‖W[t_k] @ s_{t_k} − ε_k‖²
```

A poorly visited state s has not been stored in the replay buffer often,
so W fits it poorly and MSE is high → high R_score → agent is rewarded for
exploring s.

**Behaviour policy:**
```
# agents/exdm_agent.py → update()

# Q-learning TD update with intrinsic reward (extrinsic reward ignored):
Q[s, a] += α · (R_score(s) + γ · max_{a'} Q[s', a'] − Q[s, a])
```

**Key parameters:**
- `model_lr` — SGD step size for the score model weight matrices W[t]
- `n_diffusion_steps` (T) — number of DDPM forward diffusion steps; more steps
  give a richer noise schedule but increase the size of W (T × n_states²)
- `beta_start`, `beta_end` — linear noise schedule endpoints (DDPM defaults)
- `reward_samples` (K) — Monte Carlo samples for the intrinsic reward estimate;
  more samples → lower variance but slower per-step cost
- `n_model_updates` — gradient steps per environment step; increase for faster
  score model convergence
- `alpha` — Q-learning step size for the behaviour policy

**To modify the noise schedule**, edit `_linear_beta_schedule()`:
```python
# agents/exdm_agent.py → _linear_beta_schedule()
# Current: linear β from beta_start → beta_end
# To use a cosine schedule (Ho et al., 2020):
#   t_arr = np.arange(n_steps) / n_steps
#   betas = 1 - np.cos((t_arr + 0.008) / 1.008 * np.pi / 2) ** 2 / ...
```

**To use a shared weight matrix instead of per-step matrices**, edit `__init__`:
```python
# Replace:  self.W = np.zeros((n_diffusion_steps, n_states, n_states))
# With:     self.W_shared = np.zeros((n_states, n_states + 1))
# And embed t as an additional scalar input: [s_t; t/T]
```

---

### 7.5 Shared Equations (Bellman, Returns, GAE)

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

## 8. PyTorch and Deep RL

### 8.1 Installing PyTorch

PyTorch is now listed as a dependency in `pyproject.toml`:

```bash
pip install torch            # CPU-only (default, works everywhere)
# or — for NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 8.2 Device helper and MLP backbone (`utils/torch_utils.py`)

```python
from utils.torch_utils import get_device, MLP

device = get_device()          # auto: CUDA > MPS > CPU
net = MLP(in_dim=4, out_dim=2, hidden_sizes=[128, 128]).to(device)
```

`get_device()` automatically picks the best hardware.  All future deep agents
should call this in `__init__` and store `self.device`.

### 8.3 Abstract deep agent (`BaseDeepAgent`)

`utils/torch_utils.py` also provides `BaseDeepAgent`, an extension of
`BaseAgent` that adds `self.device`, `self.network`, and `self.optimizer`.
Subclass it for any neural network agent:

```python
from utils.torch_utils import BaseDeepAgent, MLP

class DQNAgent(BaseDeepAgent):
    def __init__(self, n_obs, n_actions, **kw):
        super().__init__(n_obs, n_actions, **kw)
        self.network = MLP(n_obs, n_actions).to(self.device)
        self.target  = MLP(n_obs, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
```

### 8.4 How Deep RL networks work

A Deep RL network maps from **observation → useful quantities**:

| Algorithm | Input shape | Output shape | What the output means |
|---|---|---|---|
| **DQN / DDQN** | `(B, n_obs)` | `(B, n_actions)` | Q(s, a) for every action |
| **Actor (discrete PPO)** | `(B, n_obs)` | `(B, n_actions)` | logits → softmax → π(a\|s) |
| **Actor (continuous SAC)** | `(B, n_obs)` | `(B, 2·n_act)` | μ and log σ for Gaussian policy |
| **Critic / value head** | `(B, n_obs)` | `(B, 1)` | V(s) or Q(s, a) |

Typical training loop:
1. Collect `(obs, action, reward, next_obs, done)` with the current policy.
2. Store in a **replay buffer** (off-policy) or rollout buffer (on-policy).
3. Sample a mini-batch, compute `Q_target = r + γ · max Q_target(s')`.
4. `loss = MSE(Q_pred, Q_target)` → `loss.backward()` → `optimizer.step()`.
5. Soft-update target network: `θ_target ← τ·θ + (1-τ)·θ_target`.

### 8.5 How to extend this repo for Deep RL

Recommended layout (keep tabular agents untouched):

```
agents/
  dqn_agent.py          # subclasses BaseDeepAgent; uses MLP Q-network
  ppo_agent.py          # actor-critic using two MLP heads
  sac_agent.py          # SAC; uses separate actor / two critics
environments/
  gym_wrapper.py        # wraps any Gymnasium env to match BaseEnv interface
configs/
  dqn.yaml              # lr, batch_size, replay_buffer_size, target_update_freq
  ppo.yaml
experiments/
  run_dqn.py            # analogous to run_rcrl.py
```

Concrete steps:
1. Write `agents/dqn_agent.py` inheriting `BaseDeepAgent`.  Implement
   `select_action` (ε-greedy over the Q-network) and `update` (sample
   from replay buffer, TD loss, backward).
2. Write `environments/gym_wrapper.py` to adapt Gymnasium's `step()` /
   `reset()` to the `BaseEnv` interface.
3. Write `experiments/run_dqn.py` (CLI runner, similar to existing runners).
4. Add the new agent to `experiments/compare_approaches.py`.
5. Add `configs/dqn.yaml` and document in `Instructions.md`.

---

## 9. Project Structure Reference

```
MSc_Dissertation/
│
├── agents/                     # RL agent implementations
│   ├── base_agent.py           #   Abstract agent interface
│   ├── random_agent.py         #   Uniform-random exploration
│   ├── goal_conditioned_agent.py   # GCRL: infoNCE contrastive critic
│   ├── reward_conditioned_agent.py # RCRL: Q[s, ψ, a] with diverse ψ sampling
│   └── exdm_agent.py           #   ExDM: state diffusion score model + intrinsic reward
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
│   ├── run_exdm.py             #   ExDM CLI
│   └── compare_approaches.py  #   Side-by-side comparison of all four methods
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
│   ├── transfer_eval.py        #   Zero-shot / fine-tuning transfer helpers (RCRL, GCRL, ExDM)
│   └── torch_utils.py          #   PyTorch device helper, MLP backbone, BaseDeepAgent
│
├── configs/                    # YAML hyperparameter files
│   ├── default.yaml            #   Shared defaults
│   ├── gcrl.yaml               #   GCRL hyperparameters
│   ├── rcrl.yaml               #   RCRL hyperparameters
│   └── exdm.yaml               #   ExDM hyperparameters
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
| ExDM intrinsic reward stops changing | Buffer is full of the same states — increase `--buffer-capacity` or reduce `--n-model-updates` |
| ExDM very slow per step | Reduce `--n-model-updates` (default 5) or `--n-diffusion-steps` (default 10) |
| Transfer Section 5 `NameError: display` | Old notebook — re-pull this branch; `plot_trajectory` was fixed |
| Transfer Section 5 ExDM missing | Old notebook — re-pull this branch; ExDM is now included in all transfer cells |
| `ImportError: PyTorch not installed` | `pip install torch` (see Section 8.1) |
