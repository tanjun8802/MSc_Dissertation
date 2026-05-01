# Overcoming the Sim-to-Real Gap: A Beginner-Friendly Explainer

> **Paper:** *Overcoming the Sim-to-Real Gap: Leveraging Simulation to Learn to Explore for Real-World RL*
> **Authors:** Andrew Wagenmaker (UC Berkeley), Kevin Huang, Liyiming Ke, Byron Boots, Kevin Jamieson, Abhishek Gupta (University of Washington)

---

## Table of Contents

1. [The Big Picture — What Problem Are They Solving?](#1-the-big-picture--what-problem-are-they-solving)
2. [Background Concepts for Beginners](#2-background-concepts-for-beginners)
3. [MDP Assumptions and Problem Setting](#3-mdp-assumptions-and-problem-setting)
4. [How They Achieve Good Exploration](#4-how-they-achieve-good-exploration)
5. [Learning Multiple Policies and Picking the Best One](#5-learning-multiple-policies-and-picking-the-best-one)
6. [The Full Algorithm, Step by Step](#6-the-full-algorithm-step-by-step)
7. [Why This Works — The Key Intuitions](#7-why-this-works--the-key-intuitions)
8. [Experimental Results](#8-experimental-results)
9. [Summary Cheat Sheet](#9-summary-cheat-sheet)

---

## 1. The Big Picture — What Problem Are They Solving?

### The Challenge: Robots are expensive to train in the real world

Training a robot to do a task (like pushing a puck or hammering a nail) by trial-and-error in the real world requires thousands or millions of attempts. Each attempt takes time, can wear out hardware, and may be dangerous. This is called the **sample complexity** problem — RL needs too many real-world samples to be practical.

### The Standard Approach: Sim-to-Real Transfer

The common solution is:
1. Build a **simulator** (a virtual copy of the real environment, e.g., using MuJoCo physics).
2. Train a policy in the simulator (samples are free and fast).
3. Deploy the trained policy in the real world, hoping it works.

This is called **direct sim-to-real (sim2real) transfer**.

### The Problem: The Sim-to-Real Gap

Simulators are not perfect. The real robot may behave slightly differently — different friction, different motor response, camera noise, etc. This mismatch is called the **sim-to-real gap**. Because of this gap, a policy that works perfectly in the simulator can fail completely when deployed in the real world.

### Their Key Insight

> **"It is often easier to learn to explore than to learn to solve the goal task."**

Instead of trying to transfer a policy that *solves the task* (hard), they transfer policies that are good at *exploring the environment* (much easier). These exploration policies then help collect high-quality data in the real world, from which a final optimal policy can be learned efficiently.

---

## 2. Background Concepts for Beginners

Before diving in, here are some key terms:

| Term | What It Means |
|------|---------------|
| **MDP** | Markov Decision Process — the mathematical framework for RL. An agent takes actions in an environment, receives rewards, and transitions between states. |
| **Policy (π)** | A rule that tells the agent what action to take in each state. |
| **Value function (V)** | The expected total reward an agent will get by following a policy from a given state. |
| **Q-value (Q)** | Like V, but conditioned on also specifying what action is taken first. |
| **Horizon (H)** | The number of steps in one episode. |
| **Feature map (φ)** | A function that converts (state, action) pairs into a vector of numbers that captures their important properties. |
| **Low-rank MDP** | An MDP whose transition probabilities can be written as a dot product of feature vectors — a structured, tractable setting. |
| **Sample complexity** | How many real-world interactions the algorithm needs before it finds a good policy. |
| **Polynomial vs. Exponential** | Polynomial (e.g., T²) grows manageable. Exponential (e.g., 2^H) grows unmanageable — this is the difference between "feasible" and "infeasible." |
| **Exploration vs. Exploitation** | Exploration = trying new things to gather information. Exploitation = using what you already know to maximize reward. Good RL needs both. |

---

## 3. MDP Assumptions and Problem Setting

The paper works within a very specific mathematical framework. Understanding these assumptions tells you *exactly* when and why their method works.

### 3.1 The MDP Tuple

Both the simulator (**M_sim**) and the real world (**M_real**) are modelled as an MDP:

```
M = (S, A, {P_h}, {r_h}, s_1, H)
```

- **S** — the set of all possible states (e.g., position and velocity of the robot and object)
- **A** — the set of all possible actions (e.g., how much to move the arm); assumed to be **finite** with |A| = A
- **P_h(· | s, a)** — the transition function at step h: given state s and action a, what is the probability distribution over next states?
- **r_h(s, a)** — the reward at step h; assumed **deterministic and known** (a simplifying assumption)
- **s_1** — a fixed starting state (same in sim and real)
- **H** — the finite time horizon (length of each episode)

### 3.2 The Sim-to-Real Gap (Assumption 1)

> *The simulator is close to the real world, but not identical.*

Formally:

$$\|P^{\text{real}}_h(\cdot | s,a) - P^{\text{sim}}_h(\cdot | s,a)\|_{\text{TV}} \leq \epsilon_{\text{sim}}, \quad \forall (s,a,h)$$

- The **total variation distance** (TV) measures how different two probability distributions are (0 = identical, 1 = completely different).
- **ε_sim** is the maximum gap between the simulator's and real world's transition probabilities.
- The **state space, action space, reward function, and starting state are assumed to be the same** in sim and real. Only the *dynamics* (how actions lead to next states) differ.
- ε_sim is **not assumed to be known** by the algorithm — it just needs to exist and be small enough.

**Why this matters:** A classical result (the *Simulation Lemma*) says that if you directly transfer the best simulator policy (π^{sim,⋆}) to the real world, you can only guarantee:

$$V^{\text{real},\pi^{\text{sim},\star}}_0 \geq V^{\text{real},\star}_0 - 2H^2 \epsilon_{\text{sim}}$$

In plain English: *direct transfer can be suboptimal by up to 2H²ε_sim*. If ε_sim is even slightly large, and you want to be very precise (small ε), direct transfer is not enough.

### 3.3 The Low-Rank MDP Assumption (Assumption 2)

> *The transitions have hidden structure that makes them tractable to learn.*

Both M_sim and M_real are assumed to be **low-rank MDPs**: there exist feature maps φ(s,a) ∈ ℝ^d and measure vectors µ_h(·) such that:

$$P_h(\cdot | s,a) = \langle \phi(s,a),\, \mu_h(\cdot) \rangle$$

Think of it this way: instead of needing to tabulate the transition probability for every possible (state, action) pair separately, you can describe the dynamics using a compact **d-dimensional feature vector**. This is what makes the problem tractable without requiring a finite, enumerable state space.

**Critically:**
- In **M_sim**, the feature map **φ^s is known** (we built the simulator, so we know its structure).
- In **M_real**, the feature map **φ^r is unknown** (we do not know the true features of the real world).

This asymmetry is the heart of the paper: we can exploit the known structure of the simulator to help explore the unknown real world.

### 3.4 Reachability in the Simulator (Assumption 3)

> *Every direction in the simulator's feature space can be reached by some policy.*

$$\min_h \sup_\pi \lambda_{\min}\!\left(\mathbb{E}^{M_{\text{sim}},\pi}\!\left[\phi^s(s_h, a_h)\phi^s(s_h, a_h)^\top\right]\right) \geq \lambda^*_{\min}$$

- **λ_min** (minimum eigenvalue) measures whether a matrix spans all directions.
- This condition says: at every step h, there exists at least one policy that can reach every direction in the d-dimensional feature space in M_sim.
- Importantly, **this is only assumed for M_sim, not M_real**. The real world might be much harder to explore — that's the whole problem!
- **λ*_min** is a constant that quantifies "how reachable" the simulator is. A larger λ*_min means easier exploration.

### 3.5 Bellman Completeness (Assumption 4)

> *The function class used for regression is rich enough to represent the Q-values.*

For all functions f in the hypothesis class F_{h+1}:

$$\mathcal{T}_{h+1} f \in F_h$$

where the **Bellman operator** is:

$$\mathcal{T}_{h+1} f(s,a) := r_h(s,a) + \mathbb{E}_{s' \sim P_h(\cdot|s,a)}\left[\max_{a'} f(s', a')\right]$$

This ensures that when the algorithm does least-squares regression on data to estimate Q-values (a technique called Fitted Q-Iteration / FQI), the target values are always representable within the function class. Without this, regression errors can cascade badly.

### 3.6 Key Condition: How Small Does the Gap Need to Be? (Equation 4.2)

For the whole algorithm to work, the sim-to-real gap ε_sim must satisfy:

$$\epsilon_{\text{sim}} \leq \frac{\lambda^*_{\min}}{64 \cdot d \cdot H \cdot A^3}$$

In plain English: the gap needs to be small relative to the feature dimension (d), horizon (H), action space size (A), and how reachable the simulator is (λ*_min). **Crucially, this condition does not depend on ε** (the desired accuracy), so no matter how precise you want to be, as long as the simulator is accurate enough, the algorithm works.

### 3.7 Summary of Assumptions

| Assumption | What It Says | Why It's Needed |
|---|---|---|
| Assumption 1 (Sim-to-Real Gap) | Sim and real transitions are TV-close with gap ε_sim | Ensures sim is a useful model |
| Assumption 2 (Low-Rank MDP) | Both sim & real have low-dimensional feature structure; φ^s is known | Makes the problem tractable |
| Assumption 3 (Reachability in Sim) | Every feature direction reachable in sim by some policy | Ensures exploration is possible in sim |
| Assumption 4 (Bellman Completeness) | Function class closed under Bellman operator | Ensures regression-based policy learning works |
| Condition (4.2) | ε_sim small enough relative to d, H, A, λ*_min | Ensures sim coverage transfers to real |

---

## 4. How They Achieve Good Exploration

This is the most important contribution of the paper. The exploration strategy has two levels: *in the simulator* (before touching the real world) and *in the real world* (during deployment).

### 4.1 Why Naive Exploration Fails

First, it's important to understand why the obvious approach doesn't work.

**ζ-Greedy Exploration** is the standard approach: with probability ζ, take a random action; otherwise follow the best policy you know so far. The paper proves:

> **Proposition 1:** No matter what ζ ∈ [0,1] you choose, ζ-greedy exploration requires at least **Ω(2^{H/2}) samples** — exponential in the horizon H.

**Why?** Consider a "combination lock" environment: you must play action a_1 for H consecutive steps to reach the key state. The probability of a random exploration finding this is (ζ/2)^{H−1} — exponentially small. Standard exploration is blind.

### 4.2 Exploration in the Simulator: Learning a Covering Set of Policies

The first phase happens entirely in M_sim (free samples!). The goal is to find a **set of exploration policies Π_exp** that together cover every direction in the feature space of M_sim.

**The Coverage Condition:** The set Π_exp is "good" if:

$$\lambda_{\min}\!\left(\frac{1}{|\Pi^h_{\text{exp}}|} \sum_{\pi \in \Pi^h_{\text{exp}}} \mathbb{E}^{M_{\text{sim}},\pi}\!\left[\phi^s(s_h, a_h)\phi^s(s_h, a_h)^\top\right]\right) \gtrsim \lambda^*_{\min}$$

**In plain English:** If you average the "state coverage matrices" across all policies in Π_exp, the average should have all eigenvalues bounded away from zero. This means: collectively, the policies visit every important direction in the feature space. No part of the environment is left unexplored.

**How?** The paper uses an algorithm called **DynamicOED** (Dynamic Optimal Experiment Design), which iteratively adds policies that cover the least-explored directions in feature space. It's like trying to design a set of experiments that collectively tell you the most about the system.

### 4.3 Exploration in the Real World: Policies + Random Tails

Once we have Π_exp from the simulator, we need to use them in M_real. But there's a subtlety: these policies were optimized for M_sim coverage, not M_real coverage. The sim-to-real gap means they might miss some regions of M_real.

**The Solution:** Each exploration policy π_exp is augmented with a **random tail**:
- Play π_exp for steps 1, 2, ..., h (following the simulator-trained policy).
- For steps h+1, h+2, ..., H, take actions **uniformly at random**.

The augmented policies are written **Π̃_exp**. In the real algorithm, a horizon step h is sampled uniformly from {1, ..., H}, and the corresponding augmented policy is run.

**Why does the random tail help?**

The key technical result (Lemma B.4) shows that if a policy achieves good coverage in M_sim, it can reach within a **logarithmic number of steps** of any relevant state in M_real. Since random exploration over a logarithmic-length tail costs only polynomially many samples (rather than exponentially many over the full horizon), the combined strategy is efficient.

Intuitively:
- The **sim-trained prefix** gets the agent to the right neighbourhood of the important states (using the structure learned in sim).
- The **random tail** fills in the remaining local uncertainty due to the sim-to-real gap.

> **This is the paper's core technical trick:** decompose the horizon into a structured prefix (guided by sim knowledge) and an unstructured suffix (random), and show that the suffix only needs to be short.

### 4.4 The Practical Exploration Algorithm (Algorithm 6)

In practice, the DynamicOED algorithm is replaced by a more practical approach inspired by **DIAYN** (Diversity Is All You Need) and **"One Solution Is Not All You Need" (OS)**:

**Idea:** Train an ensemble of n policies simultaneously in M_sim, with an additional **diversity reward** that encourages different policies to behave differently.

**The Diversity Reward:** A neural network discriminator d_θ(s, i) is trained to predict *which policy index i* is currently running given the observed state s. Each policy π^i_exp then receives an additional reward:

$$r_e(s,i) := \log \frac{\exp(d_\theta(s,i))}{\sum_{j=1}^{n} \exp(d_\theta(s,j))}$$

This is the **log probability that the discriminator correctly identifies which policy is running**. If policy i visits a unique region of the state space, the discriminator easily identifies it, giving a high reward. If two policies visit the same region, neither can be easily distinguished, so both receive low diversity reward. This incentivises the policies to **spread out and cover different parts of the state space**.

**Full Practical Algorithm (Algorithm 6 — "OS + SAC"):**

```
=== Phase 1: Train Exploration Policies in M_sim ===

Initialize:
  - n policies {π^1, ..., π^n} sharing weights but conditioned on latent index z ∈ {1..n}
  - Discriminator network D_ϕ(state, z) → score

For each training step i = 1 to N:
  1. Sample latent z ~ Uniform(1, n) and initial state s_0
  2. Roll out policy π_θ(·|z) in M_sim for one episode
  3. At each step t:
       - Compute discriminator score d_t = D_ϕ(s_{t+1}, z)
       - Compute diversity reward: r_e(s_{t+1}, z) = log softmax of d_t over all z'
       - If the policy is already achieving some task reward (R_π ≥ ε):
           total reward = task reward + α × diversity reward
         Else:
           total reward = task reward only
  4. Update policy π_θ with SAC to maximise total reward
  5. Update discriminator D_ϕ to maximise classification accuracy
  6. Save policies throughout training; keep those that achieve some task reward

=== Phase 2: Deploy Exploration Policies in M_real ===

While not converged:
  1. Sample z ~ Uniform(1, n), play π_θ(·|z) in M_real for one episode
  2. Add collected (state, action, reward, next_state) data to replay buffer
  3. Run one SAC update step on the real-world agent
```

The key design choice is: **the exploration reward only kicks in after the policy already achieves some task reward** (R_π ≥ ε). This prevents policies from exploring useless parts of the state space before they've learned to do anything useful.

---

## 5. Learning Multiple Policies and Picking the Best One

### 5.1 Why Multiple Policies?

There is a fundamental uncertainty the algorithm must handle: the sim-to-real gap ε_sim is **unknown**. Depending on how large ε_sim is relative to the target accuracy ε, the best strategy differs:

- **If ε_sim is small** (close to what you want): direct transfer of the simulator's optimal policy π^{sim,⋆} might already be good enough, achieving O(ε_sim)-optimality.
- **If ε_sim is large** relative to ε: direct transfer fails, and the regression-based policy learned from real-world data will be much better.

Since we don't know which case we're in, we run **both** approaches and pick the winner empirically.

### 5.2 The Two Candidate Policies

**Candidate 1 — π^{sim,⋆} (Direct Transfer):**
- The optimal policy found by solving the task in M_sim.
- Computed using any policy optimization oracle in the simulator.
- Good when ε_sim is small enough that the sim-to-real gap doesn't hurt too much.

**Candidate 2 — π^{f̂} (Regression-Based / FQI Policy):**
- Learned by running Fitted Q-Iteration (FQI) on data collected in M_real by the exploration policies.
- The regression is solved at each step h = H, H-1, ..., 1:

$$\hat{f}_h = \arg\min_{f \in F_h} \sum_{(s,a,r,s') \in D} \left(f_h(s,a) - r - \max_{a'} \hat{f}_{h+1}(s',a')\right)^2$$

- This is essentially **Fitted Q-Iteration (FQI)**: iteratively estimate Q-values backward from the final step, using least-squares regression on observed transitions.
- Good when the exploration policies have collected diverse, high-quality data in M_real.

### 5.3 Selecting the Final Policy

The algorithm evaluates both candidates by running them in M_real and measuring their actual return:

```
Algorithm 1: sim2real Exploration Policy Transfer

Step 1: Learn Π_exp in M_sim using LearnExpPolicies (covers the feature space)
Step 2: Augment to Π̃_exp (add random tails to each policy)
Step 3: Explore in M_real — play π_exp ~ Uniform(Π̃_exp) for T/2 episodes, collect data D
Step 4: Run FQI on D to get π^{f̂}
Step 5: Compute π^{sim,⋆} in M_sim
Step 6: Evaluate both candidates empirically in M_real (T/4 episodes each)
Step 7: Return π̂ = argmax_{π ∈ {π^{f̂}, π^{sim,⋆}}} V̂^{real,π}
```

The **empirical evaluation** in Step 6 is what makes this safe: rather than guessing which case we're in, we let the real world tell us which policy is actually better.

### 5.4 Why the Ensemble of n Policies (Practical Version)

In the practical algorithm, the "multiple policies" are the n members of the exploration ensemble {π^1_exp, ..., π^n_exp}. Their role is:

1. **During data collection in M_real:** randomly sample one of the n exploration policies at the start of each episode. This ensures the data covers many different parts of the real-world state space.

2. **Implicitly as candidates:** different policies may be better adapted to different regions of M_real. By collecting data from all of them and running SAC on the combined replay buffer, the final policy can leverage all of their experiences.

In the real-world Franka experiment, n = 15 exploration policies were trained. Some may succeed in certain parts of the task, others in different parts — together they provide rich coverage.

### 5.5 When Does Each Candidate Win?

The paper's **Theorem 3** (the two-case theorem) formalises this:

| Case | Condition | Which Candidate Wins |
|---|---|---|
| Case 1: ε ≥ ε_sim / 16H² | Target accuracy is loose relative to gap | π^{sim,⋆} (direct transfer) |
| Case 2: ε ≪ ε_sim | Target accuracy is tighter than gap | π^{f̂} (regression on real data) |

In both cases, Algorithm 1 automatically selects the right candidate by empirical evaluation.

---

## 6. The Full Algorithm, Step by Step

Here is the complete picture, end-to-end:

```
╔══════════════════════════════════════════════════════╗
║              PHASE 1: IN THE SIMULATOR               ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Input: M_sim (free samples!), budget T_sim          ║
║                                                      ║
║  1. Run exploration policy training (OS + SAC):      ║
║     - Train ensemble of n diverse policies           ║
║     - Use discriminator-based diversity reward       ║
║     - Save checkpoints throughout training           ║
║     - Filter: keep policies that achieve reward ≥ ε  ║
║     → Output: Π_exp = {π¹_exp, ..., π^n_exp}        ║
║                                                      ║
║  2. Also solve the task in M_sim to get π^{sim,⋆}   ║
║                                                      ║
╠══════════════════════════════════════════════════════╣
║              PHASE 2: IN THE REAL WORLD              ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  3. Augment exploration policies with random tails:  ║
║     Π̃_exp = {π_exp followed by uniform random       ║
║               actions for remaining steps}           ║
║                                                      ║
║  4. Data collection (T/2 episodes):                  ║
║     For each episode:                                ║
║       - Sample z ~ Uniform(1, n)                    ║
║       - Run π^z_exp in M_real                        ║
║       - Add (s, a, r, s') to replay buffer           ║
║                                                      ║
║  5. Policy optimization from collected data:         ║
║     - Run SAC on replay buffer to get π^{f̂}         ║
║     (optionally initialise SAC from π^{sim,⋆})       ║
║                                                      ║
║  6. Evaluate both candidates in M_real:              ║
║     - Run π^{sim,⋆} for T/4 episodes → V̂_sim        ║
║     - Run π^{f̂} for T/4 episodes → V̂_real           ║
║                                                      ║
║  7. Return π̂ = argmax(V̂_sim, V̂_real)               ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

---

## 7. Why This Works — The Key Intuitions

### 7.1 Exploration is Easier Than Solving the Task

Solving a complex manipulation task requires a very precise sequence of actions. But getting a robot to *touch* an object, or *move to approximately the right region*, is much easier. The sim-to-real gap hurts precise task-solving much more than it hurts rough exploration.

### 7.2 Coverage in Sim Transfers to Real (Lemma B.4)

The mathematical backbone of the paper: if a set of policies achieves good feature coverage in M_sim, and the sim-to-real gap is small enough (Condition 4.2), then those policies can reach within a short distance of any relevant state in M_real.

This "short distance" is **logarithmic** in the relevant problem parameters. Since random exploration over a length-k tail costs only A^k samples (exponential in k), and k is logarithmic, the total is polynomial.

### 7.3 Decoupling Exploration from Optimization

A key design principle of the practical algorithm: **the exploration policies are fixed while the task-solving policy is being trained**. This separation means:
- The exploration policies can be optimised for diversity/coverage without worrying about the task reward.
- The task-solving policy (SAC) just needs to do offline RL on whatever data the exploration policies collected.
- This avoids the chicken-and-egg problem where you need good data to train a good policy, but you need a good policy to collect good data.

### 7.4 The Exponential Improvement Over Baselines

| Method | Sample Complexity | Practical? |
|---|---|---|
| No simulator (learn from scratch) | Ω(2^H) | No |
| Direct sim2real transfer | O(ε_sim)-optimal only | Sometimes |
| ζ-greedy + no sim | Ω(2^{H/2}) | No |
| Direct transfer + ζ-greedy | Ω(2^H) to beat O(ε_sim) | No |
| **This paper (exploration transfer)** | **Poly(d, H, ε⁻¹) · log(1/δ)** | **Yes!** |

---

## 8. Experimental Results

### 8.1 Combination Lock (Toy Example)

A 2-state MDP where the optimal action in real (a_1) is the opposite of the optimal action in sim (a_2). You must take a_1 for H−1 consecutive steps to reach the key state.

- **Direct transfer:** Fails completely — always plays a_2.
- **ζ-Greedy exploration:** Fails — probability of finding the key sequence = (ζ/2)^{H−1}, exponentially small.
- **Exploration transfer (this paper):** Quickly finds the optimal policy. ✓

### 8.2 TychoEnv Robotics Simulator (Reaching Task)

7-DOF robotic arm must touch a ball with a chopstick end-effector. Sim and real differ in action bounds and control frequency. Sparse reward (only non-zero on contact).

- **Direct transfer:** Robot gets no reward at all in real world. SAC cannot learn with zero signal.
- **Exploration transfer:** Exploration policies occasionally make contact, providing enough signal for SAC to eventually learn. ✓

### 8.3 Franka Hammering (Sim-to-Sim)

Franka robot arm hammering a nail. Real environment has nail position and stiffness outside the range of sim training.

- **Direct transfer:** Learns eventually, but slowly and reaches lower final performance.
- **Exploration transfer:** Learns significantly faster and achieves higher success rate. ✓

### 8.4 Real Franka Puck Pushing (Sim-to-Real, Physical Robot)

The flagship experiment. Physical Franka robot pushing a puck from center to edge. 6 real-world runs per method.

| Method | Success Rate | Convergence |
|---|---|---|
| Direct sim2real + finetuning | **0 / 6** runs | Stuck suboptimal |
| **Exploration policy transfer** | **6 / 6** runs | Converges quickly ✓ |
| Training from scratch in real | — | Much slower |

---

## 9. Summary Cheat Sheet

### What They Assume

1. **Low-rank MDP:** both sim and real have compact feature representations; sim features are known.
2. **Bounded sim-to-real gap:** transitions differ by at most ε_sim in total variation.
3. **Reachable simulator:** every feature direction can be activated by some policy in sim.
4. **Bellman completeness:** the regression function class can represent Q-values.
5. **Gap small enough:** ε_sim ≤ λ*_min / (64·d·H·A³) for coverage to transfer.

### How They Explore

| Where | How | Purpose |
|---|---|---|
| **In sim** | Train ensemble of n diverse policies with discriminator-based diversity reward (DIAYN/OS) | Cover all feature directions in M_sim |
| **In real** | Randomly sample one of the n exploration policies per episode | Collect diverse real-world data |
| **In real** | Add random-action tails to each exploration policy | Fill coverage gaps caused by sim-to-real gap |

### How They Learn Multiple Policies

1. Train **n exploration policies** in sim using diversity rewards → produces Π_exp = {π¹, ..., π^n}.
2. Use Π_exp to collect data in M_real.
3. Train **task-solving policy π^{f̂}** via FQI / SAC on collected real data.
4. Keep the **simulator's optimal policy π^{sim,⋆}** as a fallback.
5. Evaluate both π^{f̂} and π^{sim,⋆} empirically in M_real.
6. Return whichever achieves higher real-world reward.

### The Core Theorem (Theorem 1)

If all assumptions hold and ε_sim satisfies Condition (4.2), then Algorithm 1 needs at most:

$$T \geq c \cdot \frac{d^2 H^{16}}{\epsilon^8} \cdot \log \frac{H|F|}{\delta}$$

real-world samples to find an ε-optimal policy with probability ≥ 1−δ. This is **polynomial** in all parameters — an exponential improvement over any method that doesn't use exploration policy transfer.
