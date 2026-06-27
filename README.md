# MSc Imperial — Reward-Free Reinforcement Learning (jt2525)

Research codebase for the MSc dissertation on **reward-free and unsupervised reinforcement learning** at Imperial College London.

---

## Project structure

```
MSc_Dissertation/
├── environments/       # Custom environments and wrappers
│   ├── base_env.py     #   Abstract Gymnasium-compatible base class
│   ├── gridworld.py    #   Tabular GridWorld (discrete, reward-free friendly)
│   └── wrappers.py     #   TimeLimit, NormaliseObservation
│
├── mdp/                # MDP definitions
│   ├── base_mdp.py     #   Abstract MDP interface (S, A, P, R, γ)
│   ├── tabular_mdp.py  #   Explicit P/R tables + value iteration
│   └── reward_free_mdp.py  # Reward-free wrapper with offline relabelling
│
├── agents/             # RL agent implementations
│   ├── base_agent.py   #   Abstract agent (select_action / update / reset)
│   └── random_agent.py #   Uniform-random exploration baseline
│
├── equations/          # Core RL equations (pure NumPy, mathematically documented)
│   ├── bellman.py      #   Bellman expectation & optimality operators
│   ├── value_functions.py  # MC returns, TD error, GAE-λ
│   └── policy_gradient.py  # REINFORCE, entropy bonus, PPO-clip
│
├── experiments/        # Experiment runners
│   ├── base_experiment.py  # Abstract train/eval loop
│   └── run_experiment.py   # CLI entry-point (random baseline on GridWorld)
│
├── utils/              # Shared utilities
│   ├── replay_buffer.py    # Fixed-capacity circular buffer + uniform sampling
│   ├── logger.py           # CSV + stdout logger
│   └── metrics.py          # EpisodeMetrics dataclass
│
├── configs/
│   └── default.yaml    # Default hyperparameter configuration
│
├── Papers/             # Reference papers (PDFs)
└── sandbox.py          # Quick gymnasium smoke-test
```

---

## Quick start

```bash
# 1. Install dependencies
pip install numpy gymnasium

# 2. Run the random-agent baseline on GridWorld
python experiments/run_experiment.py --episodes 50 --render
```

---

## Key design principles

| Principle | How it is realised |
|---|---|
| **Modularity** | Each concern (env, MDP, agent, equations, experiments) lives in its own package |
| **Gymnasium compatibility** | `BaseEnv` mirrors the Gymnasium API so custom envs work with SB3 / CleanRL |
| **Mathematical transparency** | `equations/` contains pure-NumPy implementations with full notation docstrings |
| **Reward-free first** | `RewardFreeMDP` and `GridWorld` (no goal) are designed for exploration-phase research |
| **Reproducibility** | Every class accepts a `seed` parameter; numpy `Generator` objects are passed explicitly |

---

## Papers

| Paper | Relevance |
|---|---|
| *Reward-Free Exploration for RL* (Jin et al., 2020) | Theoretical foundation |
| *METRA* | Metric-aware skill discovery |
| *TD-JEPA* | Latent-predictive representations |
| *Does Zero-Shot RL Exist?* | Zero-shot transfer evaluation |
| *A Single Goal is All You Need* | Goal-conditioned approaches |
| *Unsupervised RL Benchmark* | Evaluation framework |
| *Can MISL Fly?* | Mutual-information skill learning in practice |
| *Overcoming the Sim-to-Real Gap* | Transfer learning context |
| *Optimal Exploration for Model-Based RL* | Exploration in nonlinear systems |

---

## Simulation stack setup

For future robotics-focused RL experiments (MuJoCo + robosuite + Gymnasium + optional Isaac Lab), see:

- `docs/simulation_stack_setup.md`

## Benchmark helpers

ExORL and OGBench repo-side helpers now live under `src/benchmarks/`.

- Setup and usage: `docs/benchmark_setup.md`
- CLI entrypoint: `python -m src.benchmarks.cli ...`
