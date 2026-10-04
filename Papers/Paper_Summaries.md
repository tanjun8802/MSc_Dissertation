# Paper Summaries: Reward-Conditioned RL & TD-JEPA

> **A plain-English guide to two cutting-edge reinforcement learning papers, with intuition, visualisations, and equation walk-throughs.**

---

## Quick Reference

| | Reward-Conditioned RL (RCRL) | TD-JEPA |
|---|---|---|
| **Year** | 2026 (preprint) | 2025 |
| **Setting** | Standard (reward-available) RL | Reward-free / Unsupervised RL |
| **Core idea** | Train one agent on *many* reward weightings simultaneously | Learn latent representations that capture *long-term future states* across many policies |
| **Key trick** | Recompute rewards for old experience using alternative weightings | Use TD learning (like Q-learning) in latent space to avoid needing on-policy data |
| **Pay-off** | Better sample efficiency + free zero-shot adaptation | Zero-shot optimisation of *any* reward at test time, especially from pixels |

---

# Part 1 — Background: The World Every RL Paper Lives In

Before diving into either paper, you need a solid grip on the standard "MDP" setting that both papers use.

## 1.1 The Markov Decision Process (MDP)

Think of the MDP as the rulebook that describes the world an AI agent lives in.

```
┌─────────────────────────────────────────────────────────┐
│                 THE MDP WORLD                           │
│                                                         │
│   ┌────────┐  action a   ┌────────────┐                 │
│   │ Agent  │ ──────────► │Environment │                 │
│   │  (π)   │             │            │                 │
│   └────────┘             └────────────┘                 │
│       ▲                       │                         │
│       │  observation s'       │ reward r(s,a)           │
│       └───────────────────────┘                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Formal notation:** M = (S, A, P, r, ρ, γ)

| Symbol | What it means | Plain English |
|--------|--------------|---------------|
| **S** | State space | All possible situations the agent can be in (e.g. the robot's joint angles and velocities) |
| **A** | Action space | All possible moves the agent can make (e.g. torques to apply to each joint) |
| **P(s'│s, a)** | Transition dynamics | Given state *s* and action *a*, what is the probability of ending up in state *s'*? |
| **r(s, a)** | Reward function | How much "happiness" does the agent get for taking action *a* in state *s*? |
| **ρ** | Initial state distribution | Where does the episode start? (e.g. a random starting position) |
| **γ ∈ (0,1)** | Discount factor | How much do we care about *future* rewards vs *immediate* rewards? (0.99 = very patient; 0.5 = very short-sighted) |

### The Agent's Goal

The agent is a **policy** π(a│s) — a rule that says "given state *s*, choose action *a* with some probability."

The agent wants to maximise its **expected discounted return**:

```
Objective = E[ r(s₀,a₀) + γ·r(s₁,a₁) + γ²·r(s₂,a₂) + ... ]
```

**Intuition of discounting:** Imagine you could get £100 now or £100 in 10 years. You'd prefer the £100 now — that's discounting. γ < 1 encodes this preference.

### Value Functions (the tools the agent uses internally)

```
V^π(s)   = "How much total reward will I get from state s if I follow policy π?"
Q^π(s,a) = "How much total reward will I get if I take action a in state s, then follow π?"
```

The agent uses these to improve its policy: choose actions where Q is highest.

---

## 1.2 The Reward Function Problem

Most RL papers assume the reward function is *fixed and perfect*. In practice, this is almost never true:

```
The Real Problem:
┌─────────────────────────────────────────────────────────┐
│  What designer WANTS:  Robot walks naturally            │
│  What designer WRITES: r = forward_velocity - 0.001×   │
│                              joint_torque²              │
│  What robot DOES:      Weird shuffling that scores high │
│                        but looks terrible               │
└─────────────────────────────────────────────────────────┘
```

The reward function is typically a *weighted sum* of several components:
- "Move forward" component
- "Don't waste energy" component  
- "Stay balanced" component
- etc.

The weights are a guess by the engineer. Change them slightly → completely different behaviour!

---

# Part 2 — Reward-Conditioned Reinforcement Learning (RCRL)

**Paper:** *Reward-Conditioned Reinforcement Learning*, Nauman, Cygan, Abbeel (2026)

## 2.1 The Core Motivation

### The Problem RCRL Solves

Imagine you're training a humanoid robot to *run*. You fix the reward as:

```
r = 1.0 × (forward_speed) − 0.001 × (energy_use)
```

But after deployment you realise you want the robot to run a bit *slower* and use *less energy*. With standard RL you have to **retrain from scratch**. That's expensive.

**RCRL's insight:** The robot already collected millions of (state, action, next-state) tuples while training to run. You can recompute **different reward scores** for those same transitions — as if the robot had been trained under the new reward — and use those to update the policy, **all without collecting any new data**.

```
Standard RL:
  Experience → train under r★ only → rigid policy for r★ only

RCRL:
  Experience → train under r★ AND r₁ AND r₂ AND ... → 
               flexible policy that adapts to any r at test time
```

### The Key Analogy: A Student Studying Many Textbooks

- **Standard RL** = student reads only one textbook, aces that exam, fails all others.
- **RCRL** = student reads many related textbooks simultaneously using the *same* study time, does well across all exams.

The trick: when studying maths, you can reframe the same practice problems as physics problems or chemistry problems — getting extra credit for free.

---

## 2.2 MDP Setting for RCRL

RCRL uses a **standard MDP** M = (S, A, P, r, ρ, γ), but crucially:

The reward function is **composite** — a parameterised combination of components:

```
r_ψ(s,a) = f(ψ, c₁(s,a), c₂(s,a), ..., cₖ(s,a))
```

Where:
- **c₁, c₂, ..., cₖ** are the *reward components* (e.g. forward velocity, energy penalty, balance term)
- **ψ** is the *parameterisation* — the set of weights/coefficients that combine the components
- **ψ★** is the *nominal* (target) parameterisation — what the engineer actually wants
- **Ψ** is the *family* of all parameterisations we consider

**Concrete example:**

```
Components:
  c₁(s,a) = forward_speed          (how fast is the robot moving?)
  c₂(s,a) = −joint_torque²         (energy efficiency, negative = penalty)
  c₃(s,a) = height_from_ground     (is the robot upright?)

Nominal reward (ψ★ = [1.0, 0.001, 0.1]):
  r_ψ★ = 1.0·c₁ + 0.001·c₂ + 0.1·c₃

Alternative parameterisation (ψ = [0.5, 0.005, 0.2]):
  r_ψ  = 0.5·c₁ + 0.005·c₂ + 0.2·c₃   ← slower but more upright
```

The *same* (s, a) tuple can be scored differently depending on which ψ you use. RCRL stores the raw components and scores them differently at training time.

---

## 2.3 The RCRL Method: Step-by-Step

### Phase 1 — Environment Interaction (Unchanged from standard RL)

The agent always interacts with the environment under the **nominal** reward ψ★:

```
action ~ π(a | s, ψ★)   ← always conditions on the "true" reward
```

Key point: **Only ψ★ is used during data collection**. The agent's behaviour in the real world is unchanged.

After each step, the environment returns:
- **s'** — the next state
- **c₁, c₂, ..., cₖ** — the raw reward *components* (not just the scalar reward!)

These are stored in a replay buffer:

```
Replay Buffer: { (s, a, s', [c₁, c₂, ..., cₖ]) }
                                 ↑
                         Store components, not just total reward!
                         This lets us recompute rewards later.
```

### Phase 2 — Diversified Training (The RCRL Innovation)

When sampling a batch of transitions for training:

**Step 1:** Sample transitions from the buffer.

**Step 2:** For *each* transition, sample a reward parameterisation from the mixture:

```
PΨ = α · δ_ψ★  +  (1−α) · pΨ
     └────────┘   └────────┘
     ψ★ with        sample from the
     probability α  broader distribution
                    over parameterisations
```

**What does this mean?**
- With probability **α**, use the nominal reward ψ★ (standard RL update)
- With probability **1−α**, sample some other parameterisation ψ and compute `r_ψ` from stored components
- **α = 0.5** in practice (half nominal, half alternatives)
- **α = 1** recovers standard RL (no reward conditioning at all)

**Step 3:** Concatenate ψ to the state:

```
Conditioned state:  z = [s, ψ]   ← state + reward parameterisation glued together
```

The actor and critic both take **z** as input, not just **s**:

```
π_θ(a | s, ψ)    ← policy conditioned on reward
Q_θ(s, a, ψ)     ← value function conditioned on reward
```

**Step 4:** Standard RL update (actor-critic loss) using the recomputed reward `r_ψ`.

### The Full Loop Visualised

```
┌─────────────────────────────────────────────────────────────────┐
│  A. ENVIRONMENT STEP                                            │
│                                                                 │
│   Policy conditioned on ψ★:  a ~ π(a | s, ψ★)                 │
│   Environment returns:       s', [c₁,...,cₖ]                   │
│   Store:                     (s, a, s', [c₁,...,cₖ]) → Buffer  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  B. DATA SAMPLING                                               │
│                                                                 │
│   Sample batch: {(sᵢ, aᵢ, s'ᵢ, [c₁,...,cₖ]ᵢ)}                │
│   Sample ψᵢ ~ PΨ = αδ_ψ★ + (1−α)pΨ  for each i               │
│   Compute reward: rᵢ = f(ψᵢ, c₁ᵢ,...,cₖᵢ)                    │
│   Create conditioned batch: {(sᵢ, aᵢ, rᵢ, s'ᵢ, ψᵢ)}          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  C. NETWORK UPDATES                                             │
│                                                                 │
│   Conditioned state: z = [s, ψ],  z' = [s', ψ]                │
│   Critic loss:  L_Q = f(Q_θ(z,a),  r_ψ + γV(z'))              │
│   Actor  loss:  L_π = f(Q_θ(z,a),  π_θ(a|z))                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Algorithm 1 (Pseudocode Explained)

```python
# Inputs:
#   ψ★       = nominal reward parameterisation (e.g. [1.0, 0.001, 0.1])
#   Ψ        = set of possible parameterisations
#   PΨ       = distribution over Ψ

# Step 1: Collect experience under ψ★ only
action = π(state, ψ★)           # always use nominal reward for action selection

# Step 2: Get reward COMPONENTS (not just total reward!)
next_state, [c1, c2, ..., ck] = environment.step(action)

# Step 3: Store components in buffer
buffer.add(state, action, [c1,...,ck], next_state)

# Step 4-7: During training (the key RCRL part)
batch = buffer.sample(B)
for each transition in batch:
    ψ = sample from PΨ                    # sample a reward parameterisation
    r = f(ψ, c1, ..., ck)                # recompute reward under ψ
    z  = concatenate(state,  ψ)           # condition state on reward
    z' = concatenate(next_state, ψ)
    agent.update(z, action, r, z')        # standard RL update with conditioned state
```

---

## 2.4 How to Construct the Alternative Parameterisations (Ψ)?

RCRL offers two strategies:

### Strategy 1: Parameterised Reward Conditioning

Take ψ★ and *perturb* it multiplicatively:

```
ψᵢ = ψ★ᵢ × Δᵢ    where Δᵢ > 0

Example:
  ψ★ = [1.0, 0.001, 0.1]
  Δ  = [0.5, 2.0,   1.3]  ← sampled from log-uniform distribution on [0.25, 4.0]
  ψ  = [0.5, 0.002, 0.13]  ← similar but shifted reward
```

**Intuition:** You're exploring nearby reward functions. If the robot should run fast (ψ★), you also train it for "run at half speed," "run at double speed," "run efficiently," etc. All from the same collected data!

```
Space of reward parameterisations:
                  ψ★ (nominal, centre)
                   ●
         ψ₃ ●    ↗  ↖    ● ψ₁
                          
              ψ₄ ●    ● ψ₂
              
The agent is trained on all these reward functions simultaneously,
but only ever *explores* under ψ★.
```

### Strategy 2: Auxiliary Task Conditioning

Instead of perturbing ψ★, use *qualitatively different* rewards for the same robot:

```
Nominal: ψ★ = "run fast"
Auxiliary Ψ = {"stand", "walk", "crawl", "hurdle", "stair-climb"}
```

**Intuition:** If you train the robot to be good at many related tasks off-policy, it learns richer representations of what the robot body can do — representations that help it be better at the main task too.

This is like teaching a student maths *and* physics simultaneously. Doing physics problems makes you better at maths, even if you're ultimately only tested on maths.

---

## 2.5 Why Does This Help? The Intuition

**Benefit 1: Better sample efficiency on the nominal task**

When you train with diverse reward functions on the *same* data, the value function must learn to predict returns under many scenarios. This forces it to learn a more general model of the environment, which is also more accurate for ψ★.

Think of it like cross-validation in supervised learning: training on diverse inputs prevents overfitting.

**Benefit 2: Zero-shot adaptation**

At test time, just change ψ:
```
"I want slower speed" → change ψ from [1.0, ...] to [0.3, ...]
                       → policy automatically adjusts, no retraining!
```

**Benefit 3: Fast finetuning**

Because the agent has *seen* alternatives to ψ★ during training, finetuning to a new reward requires far fewer environment steps — the initialisation is much better.

---

## 2.6 Key Equations Explained

### Equation 1: Composite Reward Function

```
r_ψ(s,a) = f(ψ, c₁(s,a), ..., cₖ(s,a))
```

**Plain English:** The reward is not just one number — it's a combination (weighted sum, product, etc.) of *k* building blocks (components), where ψ sets how much each component matters.

**Why it matters:** If you store the components separately, you can recompute the reward with *any* ψ later. That's RCRL's free lunch.

### Equation 2: Mixture Distribution over Parameterisations

```
PΨ = α · δ_ψ★  +  (1−α) · pΨ
```

**Plain English:**
- `δ_ψ★` = always use the nominal reward (a spike at ψ★)
- `pΨ` = draw from the distribution of alternative rewards
- `α` controls the mix (0.5 means 50/50)

**Why it matters:** This controls how much time you spend "practising for alternatives" vs "training for the real thing." Ablations show α = 0.3–0.5 works best.

### Equation 3: Elementwise Perturbation

```
ψ = ψ★ ⊙ Δ    where Δᵢ ~ log-Uniform[0.25, 4.0]
```

**Plain English:** Multiply each weight in ψ★ by a random positive number. The log-uniform distribution means you're equally likely to sample Δᵢ = 0.25 (divide by 4) as Δᵢ = 4.0 (multiply by 4) — it's symmetric on the log scale.

**Why it matters:** This ensures the alternative rewards are "in the same neighbourhood" as ψ★ — close enough to be useful, varied enough to provide new signal.

---

## 2.7 Results and What They Mean

### Result 1: Nominally Better (Figure 3 in paper)

Even evaluated *only* under ψ★, RCRL outperforms standard RL:
- +5–15% improvement across 23 single-task benchmarks
- +20–30% improvement in multi-task settings

**Why?** Training with diverse rewards regularises the value function, leading to a more accurate critic, which leads to better policy updates.

### Result 2: Transfer (Figure 4)

When finetuning to a new reward function:
- **RCRL agent:** Reaches 90% of optimal in 250k steps
- **Standard RL agent:** Much slower to adapt

**Why?** The RCRL agent has already seen (off-policy) what the alternative reward looks like. Finetuning is just reinforcing existing knowledge.

### Result 3: Zero-shot Adaptation (Figure 5)

```
Test: Can RCRL control running speed without retraining?

  Condition policy on ψ = [0.3, ...] → robot runs slowly  ✓
  Condition policy on ψ = [1.5, ...] → robot runs fast    ✓
  Standard RL (no conditioning):      → always same speed  ✗
```

This is remarkable: the agent learns to be *steerable* without ever collecting data under the alternative speeds.

---

## 2.8 RCRL Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    RCRL IN A NUTSHELL                           │
│                                                                 │
│  TRAINING:                                                      │
│   → Explore world under ψ★                                     │
│   → Store raw reward components [c₁,...,cₖ]                   │
│   → At each update, sample ψ ~ PΨ and recompute reward r_ψ     │
│   → Train policy π(a|s,ψ) and critic Q(s,a,ψ)                 │
│                                                                 │
│  TEST TIME:                                                     │
│   → Want different behaviour? Just change ψ!                   │
│   → No retraining. No new data. Free adaptation.               │
│                                                                 │
│  COST: Nearly zero (just recomputing a linear combination)      │
│  GAIN: Better performance + free zero-shot adaptation          │
└─────────────────────────────────────────────────────────────────┘
```

---

---

# Part 3 — TD-JEPA: Latent-predictive Representations for Zero-Shot RL

**Paper:** *TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning*, Bagatella, Pirotta, Touati, Lazaric, Tirinzoni (2025, FAIR/Meta + ETH Zürich)

## 3.1 A Different Problem: Reward-Free RL

TD-JEPA lives in a fundamentally different world than RCRL: there is **no reward during training at all**.

```
Standard RL:   Reward available during training → optimise one fixed task
RCRL:          Reward available during training → optimise family of tasks
TD-JEPA:       NO reward during training        → learn representations that
                                                  can optimise ANY task at test time
```

**The dream:** An agent that explores an environment freely (like a curious baby), learns a rich model of how the world works, and then when given *any* task, immediately knows how to solve it — without further training.

### Why is this hard?

Without rewards, how does the agent know what to learn? The answer: **learn the structure of the world's dynamics** — which states lead to which other states under which actions. If you understand the dynamics well enough, you can optimise any reward.

---

## 3.2 MDP Setting for TD-JEPA

TD-JEPA uses a **reward-free MDP**: M = (S, A, P, γ)

Notice: **no reward function r**! The agent never sees a reward signal during training.

**The key object: Successor Measure**

```
M^π(X | s, a) = Σₜ₌₀^∞ γᵗ · Pr(sₜ₊₁ ∈ X | s, a, π)
```

**Plain English:** Starting from state *s*, taking action *a*, and then following policy π forever, M^π(X|s,a) counts how often the agent ends up in region X of the state space, with future visits discounted by γᵗ.

**Intuition:** Think of the successor measure as a *map of where policy π tends to go*. States that π visits frequently have high measure; states never visited have measure 0.

```
Visualisation (gridworld):
                  
  Start: s         States frequently visited by π:
                   
  ┌───────────┐    ┌───────────┐
  │ · · · · ·│    │ · · · · ·│
  │ · · π→→→→│    │ · · ░░░██│  ← High M^π
  │ · · · ↓ ·│    │ · · · ░██│
  │ · · · ↓G·│    │ · · · ░██│  ← Goal region G
  └───────────┘    └───────────┘
  Policy π (arrows)    M^π (darkness = M^π mass)
```

### Why Does the Successor Measure Matter?

The Q-function (expected return) can be written as:

```
Q^π_r(s,a) = ∫_{s+} M^π(ds+ | s,a) · r(s+)
           = E_{s+ ~ M^π(·|s,a)} [r(s+)]
```

**Plain English:** The Q-value is just the *average reward over all future states the policy will visit*, weighted by how often it visits them (successor measure).

**Critical insight:** If you know M^π and the reward r, you can compute Q^π instantly — no need to run the policy!

This means: if you learn a good approximation of the successor measure *during unsupervised exploration*, you can plug in *any* reward at test time and immediately get good Q-values.

---

## 3.3 Building Blocks: What TD-JEPA Learns

TD-JEPA trains **four components** jointly:

```
┌─────────────────────────────────────────────────────────────────┐
│              TD-JEPA's Four Learned Components                  │
│                                                                 │
│  1. State Encoder  ϕ: S → R^{dϕ}                              │
│     "Compress raw state s into a small vector"                  │
│     e.g. pixels → compact feature vector                        │
│                                                                 │
│  2. Task Encoder   ψ: S → R^{dψ}                              │
│     "Compress state into a vector defining reward SPACE"        │
│     e.g. 'the dimensions that matter for any reward'            │
│                                                                 │
│  3. Policy-conditioned Predictor  Tϕ: R^{dϕ} × A × Z → R^{dψ}│
│     "Given encoded state ϕ(s), action a, and policy-id z,      │
│      predict the FUTURE encoded state ψ(s+) in the long run"   │
│                                                                 │
│  4. Parameterised Policies  π_z: S → A  for z ∈ Z             │
│     "A family of policies, one for each latent direction z"     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Intuition for the split between ϕ and ψ:**

Imagine a robot navigating a building:
- **ϕ(s)** captures *low-level control information*: joint angles, velocity, nearby obstacles. This is what the policy needs to move.
- **ψ(s)** captures *high-level task-relevant information*: which room is the robot in, what landmarks are visible. This is what defines "which reward matters."

By splitting these, each encoder can specialise.

---

## 3.4 The JEPA Paradigm (Background)

**JEPA = Joint-Embedding Predictive Architecture** (LeCun, 2022)

The central idea of JEPA: instead of predicting raw future *pixels* (very hard), predict future *representations* in a compact space.

```
Standard predictive model:
  ϕ(s) + action → predict raw s'    ← Hard! s' is high-dimensional (images)

JEPA:
  ϕ(s) + action → predict ϕ(s')    ← Easier! ϕ(s') is low-dimensional
```

```
ONE-STEP LATENT PREDICTION:

  s ──[ϕ]──► ϕ(s) ──[Predictor T]──► predicted ϕ(s')
                                           ↑ compare ↓
  s' ──[ϕ]──────────────────────────► actual ϕ(s')

  Loss: || T(ϕ(s)) − ϕ(s') ||²
```

This is the **SPR / BYOL** style of learning in RL. TD-JEPA extends this in two critical ways:

1. **Multi-step** (not just next state, but the long-term future)
2. **Policy-conditioned** (predict where *each specific policy* will go, not just the behaviour policy)

---

## 3.5 The Key Innovation: Multi-Step Policy-Conditioned Prediction

### The Problem with One-Step Prediction

One-step prediction (standard BYOL-style) learns:
```
T(ϕ(s)) ≈ ϕ(s')    ← only predicts one step ahead
```

For zero-shot RL, you need to know *where the policy ends up in the long run*, not just the next step. Computing Q-values requires summing over an infinite future!

### The Monte-Carlo Loss (MC-JEPA) — Ideal But Impractical

The ideal version:

**Equation 5:**
```
L_{MC-JEPA}(ϕ, Tϕ) = E_{(s,a)~D, z~Z, s+~M^{π_z}(·|s,a)} [ ||Tϕ(ϕ(s), a, z) − ϕ(s+)||² ]
```

**Plain English:** Train the predictor Tϕ to match the *long-run future state distribution* (sampled from the successor measure M^{π_z}) in representation space.

**Why "Monte Carlo"?** You'd need to actually run policy π_z from state s and observe where it ends up after many steps — expensive and requires on-policy data.

**Proposition 1 (key result):** The MC loss is equivalent to:
```
L_{MC-JEPA} = E [ ||Tϕ(ϕ(s), a, z) − F^{π_z}_ϕ(s,a)||² ] + constant
```

Where **F^{π_z}_ϕ(s,a)** are the **successor features** of policy π_z with encoder ϕ.

**What are successor features?**

If you have a task encoder ψ(s) and you want Q^π for reward r(s) = ψ(s)ᵀz, then:

```
Q^π_r(s,a) = F^π_ψ(s,a)ᵀ z

where F^π_ψ(s,a) = E_{s+ ~ M^π(·|s,a)} [ψ(s+)]
```

**Plain English:** Successor features are the *expected future task-encoding* if you follow policy π from (s,a). They're like Q-values, but separated from the specific reward — just multiply by z to get the Q-value for reward ψ(s)ᵀz.

```
Successor Features Intuition:

  Reward = ψ(s)ᵀz = "I weight dimension 1 of state by z₁, dim 2 by z₂, ..."
  
  Q^π(s,a) = F^π(s,a)ᵀ z
           = (average future ψ(s+)) · z
           
  So: if you know F^π(s,a), you can compute Q for ANY reward in the ψ-span
      just by doing a dot product with z!
```

---

## 3.6 The TD-JEPA Loss: Making It Off-Policy

The MC loss requires on-policy sampling from M^{π_z}, which is expensive. The paper's key contribution is replacing MC with **temporal difference (TD) learning**.

Successor features satisfy a Bellman equation (like Q-values):

```
F^{π_z}_ϕ(s,a) = E_{s'~P(·|s,a), a'~π_z(s')} [ϕ(s') + γ · F^{π_z}_ϕ(s',a')]
                 └──────────────────────────────────────────────────────────────┘
                 "Next state's representation + discounted future successor features"
```

This is analogous to the Bellman equation for Q-values:
```
Q(s,a) = r(s,a) + γ · E[Q(s',a')]
```

So we can write a **TD version** of the latent-predictive loss:

**Equation 7:**
```
L_{TD-JEPA}(ϕ, Tϕ) = E [ ||Tϕ(ϕ(s), a, z) − ϕ(s') − γ · Tϕ(ϕ(s'), a', z)||² ]
                                                         └─────────────────────────┘
                                             "Bootstrapped" target (use Tϕ itself to estimate future)
```

**Plain English:** Train Tϕ so that its prediction for (s,a,z) equals:
- The immediate next-state representation ϕ(s'), PLUS
- γ times the predictor's own estimate of the future from (s', a', z).

This is *exactly* like TD-learning for Q-values, but in latent space instead of reward space!

```
TD Learning for Q-values:          TD-JEPA:
                                   
Q(s,a) ← r + γ·Q(s',a')          Tϕ(ϕ(s),a,z) ← ϕ(s') + γ·Tϕ(ϕ(s'),a',z)
└────────┘   └───────────┘         └────────────┘   └────────────────────────┘
Current      TD target             Current          TD target
estimate                           prediction       (bootstrapped)
```

**Why this is revolutionary:** TD learning only requires single-step transitions (s, a, s'). The replay buffer just needs tuples from *any* policy — fully off-policy! No need to run π_z in the environment.

```
MC-JEPA:   Requires on-policy data from each π_z → Expensive!
TD-JEPA:   Works from a fixed offline dataset      → Free!

Both ultimately learn the same object: F^{π_z}(s,a) ≈ Tϕ(ϕ(s),a,z)
```

---

## 3.7 Separate State and Task Encoders (Asymmetric TD-JEPA)

The full version of TD-JEPA uses *two different encoders* ϕ and ψ:

```
State encoder  ϕ: S → R^{dϕ}   "How should I encode states for control?"
Task encoder   ψ: S → R^{dψ}   "What aspects of states define the reward space?"
```

The predictor Tϕ maps from ϕ-space to ψ-space:

**Equation 9 (Asymmetric TD-JEPA):**
```
L_{TD-JEPA}(ϕ, Tϕ, ψ) = E [ ||Tϕ(ϕ(s), a, z) − ψ(s') − γ · Tϕ(ϕ(s'), a', z)||² ]
                                                   └──────┘
                                              Now predicting ψ(s'), not ϕ(s')
```

And symmetrically, a second predictor Tψ maps from ψ-space to ϕ-space:
```
L_{TD-JEPA}(ψ, Tψ, ϕ) = E [ ||Tψ(ψ(s), a, z) − ϕ(s') − γ · Tψ(ψ(s'), a', z)||² ]
```

**Why both directions?** This "cross-prediction" ensures ϕ and ψ are consistent with each other. If ϕ can predict ψ's future and ψ can predict ϕ's future, they must be capturing compatible, complementary information.

```
Relationship between ϕ and ψ:

  ϕ(s) ──[Tϕ]──► predicted ψ(s+)    (ϕ predicts ψ's future)
  ψ(s) ──[Tψ]──► predicted ϕ(s+)    (ψ predicts ϕ's future)
  
  They must agree on what "the future" looks like.
```

---

## 3.8 Zero-Shot Policy Extraction

Once ϕ, ψ, Tϕ, Tψ are trained, how do you get a policy for a new task?

**At test time:**

1. **Receive** a dataset of rewarded samples: D_rwd = {(s, r(s))}
2. **Fit** the reward to ψ via linear regression:
   ```
   z_r = argmin_z E_{(s,r)~D_rwd} [(r(s) − ψ(s)ᵀz)²]
        = [E[ψ(s)ψ(s)ᵀ]]⁻¹ · E[ψ(s)·r(s)]
   ```
   **Plain English:** Find the vector z such that ψ(s)ᵀz best approximates the reward function r(s).

3. **Extract policy:** Use π_{z_r}, the pre-trained policy for task z_r.

**Why does this work?**

The policy π_z was trained to maximise Tϕ(ϕ(s),a,z)ᵀz, which approximates F^{π_z}_ψ(s,a)ᵀz ≈ Q^{π_z}_{r_z}(s,a). So π_z is (approximately) optimal for reward r(s) = ψ(s)ᵀz.

```
Test-time procedure:
                                                           
  "Make the ant reach the red door"                       
           │                                              
           ▼                                              
  Collect 10 examples: (state, reward=1 if near red door)
           │                                              
           ▼                                              
  z_r = linear_regression(ψ(s), r(s))                   
           │                                              
           ▼                                              
  Run π_{z_r}  ← already trained! No new RL needed.      
```

---

## 3.9 The Full Algorithm (TD-JEPA Training)

```python
# Inputs: offline dataset D = {(s, a, s')}, no rewards!
# Networks: ϕ, ψ (encoders), Tϕ, Tψ (predictors), π (policy)
# Target networks: ϕ⁻, ψ⁻, Tϕ⁻, Tψ⁻ (exponential moving averages)

while not_converged:
    # Sample batch
    batch = D.sample(B)
    z = Z.sample(B)              # sample policy/task vectors
    a' = π(ϕ⁻(s'), z)           # next actions from current policies
    
    # LOSS 1: ϕ predicts ψ's future (TD loss)
    L1 = ||Tϕ(ϕ(s),a,z) − ψ⁻(s') − γ·Tϕ⁻(ϕ⁻(s'),a',z)||²
    
    # LOSS 2: ψ predicts ϕ's future (TD loss, symmetric)  
    L2 = ||Tψ(ψ(s),a,z) − ϕ⁻(s') − γ·Tψ⁻(ψ⁻(s'),a',z)||²
    
    # LOSS 3: Regularisation (prevent collapse)
    L3_ϕ = Σ_{i≠j}(ϕ(sᵢ)ᵀϕ(sⱼ))² − 1   # push representations apart
    L3_ψ = Σ_{i≠j}(ψ(sᵢ)ᵀψ(sⱼ))² − 1
    
    # LOSS 4: Policy (maximise "value" in latent space)
    L4 = −Σᵢ Tϕ(ϕ(sᵢ), âᵢ, zᵢ)ᵀzᵢ    # maximise Q ≈ Tϕ·z
    
    # Update
    update ϕ, Tϕ using L1 + λ·L3_ϕ
    update ψ, Tψ using L2 + λ·L3_ψ
    update π  using L4
    EMA-update target networks  ← stabilise TD learning
```

**Why target networks?** In TD learning, the "target" (what you're training towards) itself depends on the network you're training. This creates a moving target that can diverge. Target networks are *lagged copies* updated slowly (EMA), creating a stable training signal — exactly like in DQN.

**Why regularisation (L3)?** Without it, the encoders could *collapse* — all states map to the same representation. The regularisation pushes different states to have different representations (near-orthogonal), preventing this.

---

## 3.10 The Theoretical Guarantees

The paper proves four key results for a simplified (linear, tabular) setting:

### Theorem 1: MC-JEPA ≈ Successor Measure Loss

Minimising L_{MC-JEPA}(ϕ, Tϕ, ψ) is equivalent to minimising:
```
L_SM(ϕ, Tϕ, ψ) = E_z [ ||ϕ·Tϕ·ψᵀ − M^{π_z}||²_F ]
```

**Plain English:** The latent-predictive loss indirectly learns the best linear approximation to the successor measure. The predictor Tϕ learns to "project" M^{π_z} onto the ϕ-ψ representation space.

**Why this matters:** It connects the abstract loss (predicting latents) to the concrete object we care about (successor measures, which give us Q-values).

### Theorem 2: No Collapse

Under the TD loss, if ϕ and ψ start with *identity covariance* (well-distributed representations), they **stay well-distributed** throughout training.

**Plain English:** TD-JEPA won't degenerate to useless zero representations. The regularisation (orthonormality constraint) is sufficient to prevent collapse, even without explicit contrastive losses.

### Theorem 3: TD-JEPA ≈ Successor Measure (TD Version)

The TD-JEPA loss is related to forward and backward TD losses for approximating M^{π_z}:
```
L_fw = E_z [ ||ϕ·Tϕ·ψᵀ − P^{π_z} − γ·P^{π_z}·ϕ·Tϕ·ψᵀ||²_F ]
```

**Plain English:** Same story as Theorem 1, but for the TD (off-policy) version. The predictor learns the *Bellman fixed point* of the successor measure approximation.

### Theorem 4: Zero-Shot Policy Evaluation is Correct

The policy evaluation error of using (Tϕ, ψ) for any reward r is bounded by the successor measure approximation error:

```
max_{r : ||r|| ≤ 1} E_z [ (V^{π_z}_r(s) − ϕ(s)ᵀTϕ·ω_r)² ]  ≤  2·L_SM(ϕ, Tϕ, ψ)
```

**Plain English:** If TD-JEPA minimises its loss well (small L_SM), then for *any* reward function r, the zero-shot Q-value estimates are accurate.

**Why this matters:** This is the theoretical guarantee that TD-JEPA actually *works* for zero-shot RL. Better representations (smaller loss) → more accurate zero-shot policies.

---

## 3.11 Successor Features: The Central Concept

Successor features are so central to TD-JEPA that they deserve their own dedicated section.

**Setup:** We have a task encoder ψ(s) ∈ R^{dψ}. Any reward linear in ψ can be written as:
```
r(s) = ψ(s)ᵀz    for some vector z ∈ R^{dψ}
```

**Definition of successor features:**
```
F^π_ψ(s,a) = E_{s+~M^π(·|s,a)} [ψ(s+)]
```

**Plain English:** The successor features for policy π are the *expected future task-encoding* — averaged over all future states the policy will visit (discounted).

**The magic decomposition:**
```
Q^π_r(s,a) = F^π_ψ(s,a)ᵀ z_r

where z_r is chosen so that ψ(s)ᵀz_r ≈ r(s)
```

**Why it's magic:**

```
Traditional approach:
  - For each new reward r, run RL to learn Q^π_r from scratch
  - Cost: O(reward functions) × O(RL training time)
  
Successor features approach:
  - Learn F^π_ψ(s,a) once (no reward needed!)
  - For new reward r: find z_r via regression, then Q ≈ F^π z_r
  - Cost: O(RL training time) once + O(regression) per new reward
```

The key insight: **the dynamics of the environment (captured by F^π) are decoupled from the task (captured by z)**. Learn the dynamics once, solve any task instantly.

---

## 3.12 The TD-JEPA Architecture Visualised

```
 TRAINING PHASE (reward-free, offline data):
 
  Raw state s ──────────────────────────────────────────┐
       │                                                │
    [Encoder ϕ]                                         │
       │                                                │
  ϕ(s) = compact low-level features                    │
       │                                                │
       ▼                                                │
  [Predictor Tϕ(·, a, z)]                              │
       │ given action a and task-vector z               │
       ▼                                                │
  Tϕ(ϕ(s),a,z) ≈ ψ(s') + γ·Tϕ(ϕ(s'),a',z)           │
       │         └──── TD target (bootstrapped) ────┘  │
       ▼                                                │
  Compare with ψ⁻(s')     ◄───────────────────────────┘
                └─── target task-encoder output for next state
                
  Policy π_z tries to maximise Tϕ(ϕ(s), a, z)ᵀz
  
 
 TEST-TIME (given reward function r):
 
  Observe r on a few states
       │
  z_r = argmin_z ||r(s) − ψ(s)ᵀz||²   ← linear regression
       │
  Run π_{z_r}   ← already trained, no new RL!
```

---

## 3.13 Connection to Other Methods

| Method | How it learns ψ | How it trains policies |
|--------|----------------|----------------------|
| **FB (Forward-Backward)** | Contrastive loss (pairwise dot products) | FB decomposition of M^π |
| **HILP** | Goal-reaching: preserve temporal distance | Successor features |
| **BYOL-γ** | Multi-step latent prediction (behavioural policy) | Successor features |
| **TD-JEPA** | Multi-step latent prediction (trained policies, TD) | Successor features |

**Key difference from BYOL-γ:**
- BYOL-γ predicts where the *behavioural* (data-collection) policy goes
- TD-JEPA predicts where the *trained zero-shot policies* go

This is crucial: the trained policies cover different state regions than the data-collection policy. Modelling policy-specific dynamics makes the representations more useful for zero-shot optimisation.

**Key difference from FB:**
- FB uses a contrastive loss (needs pairwise comparisons in each batch)
- TD-JEPA uses a latent-predictive loss (self-supervised, no pairwise comparisons)
- FB learns M^{π_z} ≈ F_z · B^T (bilinear factorisation, no explicit state encoder)
- TD-JEPA learns M^{π_z} ≈ ϕ · T_z · ψ^T (explicit separate state/task encoders)

---

## 3.14 Results and What They Mean

### Benchmark Setup

TD-JEPA is evaluated on 65 tasks across 13 datasets:
- **ExoRL/DMC:** Locomotion tasks (walker, cheetah, quadruped, pointmass) from random/diverse offline data
- **OGBench:** Manipulation and navigation (ant mazes, cube stacking, puzzle solving) from low-coverage offline data

Both **proprioceptive** (joint angles/velocities) and **pixel** (raw image) observations are tested.

### Main Result: Consistent Performance

```
DMC (Locomotion, proprioception):
  TD-JEPA score: ~707 / 1000 (avg)
  Best baseline: ~648 / 1000

DMC (Locomotion, pixels):
  TD-JEPA score: ~739 / 1000 (avg)  ← Best in class!
  Best baseline: ~648 / 1000

OGBench (Navigation/Manipulation):
  TD-JEPA: competitive, especially on harder pixel-based tasks
```

**Key insight:** TD-JEPA is *especially* strong when learning from pixels. Why? Latent-predictive learning naturally handles high-dimensional observations by working entirely in compressed representation space — it never needs to reconstruct images.

### Ablation: Multi-step Matters

| Method | DMC Score |
|--------|-----------|
| BYOL* (1-step, behavioural policy) | ~468 |
| BYOL-γ* (multi-step, behavioural policy) | ~582 |
| **TD-JEPA (multi-step, trained policies)** | **~707** |

**Lesson:** Modelling *long-term* and *policy-specific* dynamics is what makes representations useful for zero-shot RL.

### Ablation: Separate Encoders Helps

Separate ϕ and ψ generally outperforms sharing a single encoder, but the symmetric (shared) variant is still competitive. The gain from asymmetry is clearest in complex pixel-based tasks.

### Fast Adaptation Bonus

Because TD-JEPA trains an explicit state encoder ϕ, the learned representations transfer easily:
- Initialise standard RL (TD3) with the pre-trained ϕ and π_{z_r}
- Fine-tune with very few steps (200k vs millions from scratch)
- Frozen ϕ often sufficient — just fine-tune the policy head

```
Performance after N steps (walker, pixels):
  From scratch:     ████░░░░░░░░░░░░░  (slow)
  TD-JEPA + finetune: ████████████░░░░  (fast!)
  TD-JEPA + frozen ϕ: ████████░░░░░░░░  (also fast!)
```

---

## 3.15 TD-JEPA Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                   TD-JEPA IN A NUTSHELL                         │
│                                                                 │
│  TRAINING (no rewards, offline data):                           │
│   → Train ϕ (state encoder) and ψ (task encoder) via           │
│     TD-based latent-predictive losses                           │
│   → Train predictor Tϕ to approximate successor features        │
│   → Train policies π_z to maximise Tϕ(ϕ(s),a,z)ᵀz             │
│   → Regularise to prevent representation collapse               │
│                                                                 │
│  TEST TIME:                                                     │
│   → Given new reward r: find z_r via linear regression on ψ    │
│   → Run pre-trained policy π_{z_r}  ← zero-shot!              │
│                                                                 │
│  WHY IT WORKS:                                                  │
│   → Tϕ ≈ successor features (theoretically proven)             │
│   → Successor features = all you need for any linear reward     │
│   → TD learning makes it work from offline data                 │
│                                                                 │
│  KEY INNOVATION:                                                │
│   → Replace MC (requires on-policy data) with TD               │
│     → enables offline, reward-free pre-training                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Part 4 — How the Two Papers Relate

These papers address different problems, but share deep conceptual connections:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONCEPTUAL MAP                              │
│                                                                 │
│  Reward-free exploration ──────────────────────────────────┐   │
│  (TD-JEPA pre-training)                                    │   │
│                                                            ▼   │
│  Successor features learned ──────────────────────────────────► │
│  (Tϕ ≈ F^π_ψ)                                            Any  │
│                                                          Reward │
│  Test-time reward specification ──────────────────────────────► │
│  (TD-JEPA: z_r via regression)                          solved │
│  (RCRL: ψ via conditioning)                              fast! │
│                                                                 │
│  Key shared insight: DECOUPLE DYNAMICS from REWARD             │
│  → Learn world model (dynamics) once                           │
│  → Adapt to any reward cheaply                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Aspect | RCRL | TD-JEPA |
|--------|------|---------|
| **When rewards are available** | During training | Only at test time |
| **Adaptation mechanism** | Condition policy on ψ (reward vector) | Linear regression to find z_r |
| **Data collection** | Nominal policy only | Any policy (offline) |
| **Key decoupling** | Reward components from reward weighting | Dynamics from reward function |
| **Zero-shot?** | Yes (change ψ at test time) | Yes (compute z_r instantly) |
| **Retraining needed?** | No (for related rewards) | No (for any reward in ψ-span) |

**Both papers are part of a broader trend:** building RL systems that are *not brittle to reward specification* — either by training on many rewards simultaneously (RCRL) or by learning reward-agnostic representations (TD-JEPA).

---

# Part 5 — Glossary of Key Terms

| Term | Plain English Definition |
|------|------------------------|
| **MDP** | The mathematical rulebook describing the agent's world: states, actions, transitions, rewards |
| **Policy π(a│s)** | The agent's decision rule: probability of taking action a in state s |
| **Value function V^π(s)** | Expected future reward from state s under policy π |
| **Q-function Q^π(s,a)** | Expected future reward from taking action a in state s, then following π |
| **Discount factor γ** | How much future rewards are worth relative to immediate ones (0=myopic, 1=far-sighted) |
| **Replay buffer** | A memory bank storing past (s,a,s',r) transitions for reuse in training |
| **Off-policy learning** | Learning from data collected by a *different* policy than the one being trained |
| **Nominal reward ψ★** | The "intended" reward function in RCRL — what the engineer actually wants to optimise |
| **Successor measure M^π** | Map of where policy π tends to visit in the state space (discounted) |
| **Successor features F^π_ψ** | Expected future task-encoding under policy π; separates dynamics from reward |
| **Latent space** | Compact lower-dimensional representation of raw observations (e.g. pixels → 256D vector) |
| **State encoder ϕ** | Neural network mapping raw states to latent vectors for control |
| **Task encoder ψ** | Neural network mapping states to latent vectors that define the reward space |
| **Predictor T** | Neural network predicting future latent states from current ones |
| **JEPA** | Joint-Embedding Predictive Architecture: predict future *representations* not raw states |
| **TD learning** | Temporal Difference learning: update estimates using bootstrapped future estimates (like Q-learning) |
| **Target network** | A slowly-updated copy of a network used to stabilise TD learning |
| **Zero-shot RL** | Ability to solve a new task without any task-specific training or fine-tuning |
| **Reward conditioning** | Feeding the reward parameterisation to the policy/critic as an extra input |
| **Contrastive loss** | A loss that pushes different representations apart and similar ones together |
| **Covariance regularisation** | Penalty that encourages representations of different states to be uncorrelated (prevents collapse) |
| **EMA (Exponential Moving Average)** | Slowly update target = 0.99×old + 0.01×new — creates stable training targets |
| **BiLinear factorisation** | Expressing a matrix as a product of two thin matrices: M ≈ A·B^T |
| **Bellman equation** | The recursive definition of value functions: Q(s,a) = r + γ·max_a'Q(s',a') |
| **Projection / oblique projection** | Mapping a vector onto a subspace (the "shadow" of the vector onto a plane) |

---

# Part 6 — Why These Papers Matter

## The Broader Picture

Both papers are addressing a fundamental limitation of current RL:

**Standard RL is brittle:**
```
Train for task A → good at task A → useless at task B
```

**The dream:**
```
Train once → good at many tasks → instantly adapt to new tasks
```

These papers take different approaches to the same dream:
- **RCRL:** "While we're training, let's *also* secretly train for many related reward functions using the same data"
- **TD-JEPA:** "Forget rewards during training. Just learn the world's dynamics so thoroughly that any reward becomes trivial to optimise at test time"

## Why TD is the Key Insight in TD-JEPA

The use of **temporal difference learning in latent space** is what makes TD-JEPA practical:

1. **MC approach** (naive): Roll out each policy, observe where it ends up → expensive, on-policy
2. **TD approach** (TD-JEPA): Use Bellman's trick — learn future predictions by bootstrapping from current predictions → works from any offline data

This is the same insight that made Q-learning revolutionary in the 1990s: you don't need to fully run the policy to evaluate it. You just need to bootstrap.

## Why Successor Features Are Central to Both Papers

Both papers implicitly rely on successor features:
- **RCRL** conditions the critic on ψ — the reward weights. The critic Q(s,a,ψ) effectively learns to predict returns for any ψ, which is equivalent to learning Q ≈ F^π · ψ.
- **TD-JEPA** explicitly trains Tϕ ≈ F^{π_z}_ψ and uses them for zero-shot policy extraction.

Successor features are the "Rosetta Stone" of RL transfer — they separate *what the world does* (dynamics) from *what the agent wants* (reward), enabling flexible, sample-efficient adaptation.

---

*Summary prepared for MSc Imperial Reward-Free RL course (jt2525)*
