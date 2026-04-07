"""
transfer_eval.py
================
Transfer-goal evaluation utilities — zero-shot and fine-tuning modes.

After training, both the GCRL C-table and the RCRL Q-table are persisted to
disk.  This module provides helpers to load those artefacts and evaluate a
**new goal state** using one of two strategies:

Zero-shot
~~~~~~~~~
The saved policy is applied directly to a new-goal environment without any
additional training.  GCRL can often succeed here because the C-table
``C[s, a, sf]`` covers *all* future states sf visited during training.  RCRL
almost always fails because its Q-table was optimised for the original goal.

Fine-tuning (downstream training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The learned policy (Q-table or C-table) is used as a **warm start** and a
short burst of further training episodes is run in an environment whose
terminal state is the *new* goal.  Both RCRL and GCRL benefit here:

* **RCRL**: Q-learning quickly adapts the conditioned Q-table to the new
  reward signal because the ψ-conditioned representation already encodes
  diverse reward shapes; only a few episodes are needed to re-point the
  nominal Q[s, ψ*, a] slice at the new goal.

* **GCRL**: The contrastive critic is updated to reinforce paths that reach
  the new goal.  Because the C-table already encodes reachability structure
  from training, convergence is much faster than training from scratch.

Public API
----------
* :class:`TransferResult`  — dataclass holding per-episode metrics.
* :func:`run_greedy_episode` — shared episode runner given
  ``action_fn(state_int) -> action_int``.
* :func:`rcrl_action_fn`   — builds a greedy fn from a Q-slice.
* :func:`gcrl_action_fn`   — builds a greedy fn from a C-table + goal.
* :func:`rcrl_finetune`    — warm-start Q-table fine-tuning for a new goal.
* :func:`gcrl_finetune`    — warm-start C-table fine-tuning for a new goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TransferResult:
    """Metrics from one transfer evaluation episode.

    Attributes
    ----------
    total_reward :
        Sum of rewards collected in the episode.
    length :
        Number of environment steps taken.
    reached_goal :
        Whether the episode terminated by reaching the new goal.
    trajectory :
        List of ``(step, state, action, reward)`` tuples. The terminal state
        is appended with ``action=-1`` when the goal is reached so that
        visualisation arrows end on the goal cell.
    """

    total_reward: float
    length: int
    reached_goal: bool
    trajectory: list[tuple[int, int, int, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Action-function factories
# ---------------------------------------------------------------------------


def rcrl_action_fn(q_nominal: np.ndarray) -> Callable[[int], int]:
    """Return a greedy action function wrapping the RCRL nominal Q-slice.

    Parameters
    ----------
    q_nominal :
        Q-values under the nominal ψ* parameterisation.  Shape
        ``(n_states, n_actions)`` — i.e. ``Q[:, nominal_psi_bin, :]`` or the
        ``q_late.npy`` array saved by :class:`experiments.run_rcrl.RCRLExperiment`.

    Returns
    -------
    action_fn :
        Callable ``action_fn(state: int) -> int`` that returns
        ``argmax Q[state, :]``.
    """

    def _fn(state: int) -> int:
        return int(np.argmax(q_nominal[state]))

    return _fn


def gcrl_action_fn(c_table: np.ndarray, goal: int) -> Callable[[int], int]:
    """Return a greedy action function wrapping the GCRL contrastive critic.

    Parameters
    ----------
    c_table :
        Full contrastive critic table.  Shape
        ``(n_states, n_actions, n_states)`` — i.e. the ``c_table.npy`` array
        saved by :class:`experiments.run_gcrl.GCRLExperiment`.
    goal :
        Flat state index of the desired goal (may differ from the training
        goal).

    Returns
    -------
    action_fn :
        Callable ``action_fn(state: int) -> int`` that returns
        ``argmax C[state, :, goal]``.
    """

    def _fn(state: int) -> int:
        return int(np.argmax(c_table[state, :, goal]))

    return _fn


# ---------------------------------------------------------------------------
# Fine-tuning helpers
# ---------------------------------------------------------------------------


def rcrl_finetune(
    q_table: np.ndarray,
    new_goal_env,
    *,
    nominal_psi_bin: int = 0,
    n_episodes: int = 20,
    alpha: float = 0.1,
    gamma: float = 0.99,
    psi_min: float = -0.1,
    psi_mix_alpha: float = 0.5,
    epsilon: float = 0.2,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.85,
    seed: int | None = None,
) -> Tuple[Callable[[int], int], np.ndarray]:
    """Warm-start fine-tuning of the RCRL Q-table for a new goal.

    Starting from the pre-trained ``q_table``, runs ``n_episodes`` of
    Q-learning in ``new_goal_env`` (whose terminal state is the new goal).
    Because the ψ-conditioned representation was trained on diverse reward
    shapes, only a small number of episodes are typically needed for the
    nominal Q-slice to redirect toward the new goal.

    Parameters
    ----------
    q_table :
        Full pre-trained Q-table.  Shape ``(n_states, n_psi_bins, n_actions)``
        — the ``q_table.npy`` artefact saved by
        :class:`experiments.run_rcrl.RCRLExperiment`.
    new_goal_env :
        A :class:`environments.gridworld.GridWorld` instance configured with
        the *new* goal position (``terminated=True`` when that goal is reached).
    nominal_psi_bin :
        Index of the nominal ψ* bin inside the Q-table (default 0).
    n_episodes :
        Number of fine-tuning episodes to run.
    alpha :
        Q-learning step size for fine-tuning.
    gamma :
        Discount factor.
    psi_min :
        Most negative ψ₂ value for the sampled reward parameterisations.
        Should match the value used during the original training.
    psi_mix_alpha :
        Fraction of nominal ψ* draws in the fine-tuning mixture PΨ.
    epsilon :
        Initial ε for ε-greedy exploration during fine-tuning.  Set lower
        than the original training epsilon to exploit the warm-started policy.
    epsilon_min :
        Minimum ε.
    epsilon_decay :
        Multiplicative ε decay applied once per episode.  A steeper decay
        (e.g. 0.85) speeds up exploitation during the short fine-tuning run.
    seed :
        Optional random seed.

    Returns
    -------
    action_fn :
        Greedy callable ``action_fn(state: int) -> int`` using the updated
        nominal Q-slice after fine-tuning.
    updated_q_table :
        The full Q-table after fine-tuning (shape unchanged).
    """
    from agents.reward_conditioned_agent import RewardConditionedAgent

    n_states, n_psi_bins, n_actions = q_table.shape

    agent = RewardConditionedAgent(
        n_states=n_states,
        n_actions=n_actions,
        n_psi_bins=n_psi_bins,
        psi_min=psi_min,
        psi_mix_alpha=psi_mix_alpha,
        gamma=gamma,
        alpha=alpha,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    # Warm-start from the pre-trained Q-table
    agent.Q = q_table.copy()

    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        obs, _ = new_goal_env.reset(seed=int(rng.integers(0, 2**31)))
        agent.reset()
        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = new_goal_env.step(action)
            agent.update(obs, action, reward, next_obs, terminated, truncated, info)
            obs = next_obs
            if terminated or truncated:
                break
        agent.finish_episode()

    updated_q = agent.Q.copy()
    q_nominal = updated_q[:, nominal_psi_bin, :]
    return rcrl_action_fn(q_nominal), updated_q


def gcrl_finetune(
    c_table: np.ndarray,
    new_goal: int,
    new_goal_env,
    *,
    n_episodes: int = 20,
    alpha: float = 0.1,
    gamma: float = 0.99,
    contrastive_gamma: float | None = None,
    temperature: float = 1.0,
    n_negatives: int = 16,
    logsumexp_reg: float = 0.01,
    n_critic_updates: int = 10,
    seed: int | None = None,
) -> Tuple[Callable[[int], int], np.ndarray]:
    """Warm-start fine-tuning of the GCRL C-table for a new goal.

    Starting from the pre-trained ``c_table``, runs ``n_episodes`` of
    contrastive-RL episodes always targeting ``new_goal``.  The contrastive
    objective reinforces paths that reach the new goal, and because the full
    reachability structure is already encoded in the C-table, convergence is
    much faster than training from scratch.

    Parameters
    ----------
    c_table :
        Full pre-trained contrastive critic.  Shape
        ``(n_states, n_actions, n_states)`` — the ``c_table.npy`` artefact
        saved by :class:`experiments.run_gcrl.GCRLExperiment`.
    new_goal :
        Flat state index of the new target goal.
    new_goal_env :
        A :class:`environments.gridworld.GridWorld` instance configured with
        the new goal as its terminal state so that episodes naturally end at
        ``new_goal`` and (s, a, sf=new_goal) pairs are generated.
    n_episodes :
        Number of fine-tuning episodes to run.
    alpha :
        Contrastive critic step size for fine-tuning.
    gamma :
        MDP discount factor.
    contrastive_gamma :
        Geometric future-state sampling parameter (Δ ~ Geom(1-cγ)-1).
        If ``None``, auto-scales to ``min_path / (min_path + 1)`` using the
        grid dimensions inferred from ``n_states``.
    temperature :
        Softmax temperature τ for action selection.
    n_negatives :
        Number of negative examples per infoNCE mini-batch update.
    logsumexp_reg :
        Coefficient of the LogSumExp regularisation term.
    n_critic_updates :
        Number of infoNCE mini-batch passes per episode.
    seed :
        Optional random seed.

    Returns
    -------
    action_fn :
        Callable ``action_fn(state: int) -> int`` using the fine-tuned
        C-table conditioned on ``new_goal``.
    updated_c_table :
        The full C-table after fine-tuning (shape unchanged).
    """
    from agents.goal_conditioned_agent import GoalConditionedAgent

    n_states, n_actions, _ = c_table.shape

    # Auto-scale contrastive_gamma if not provided.
    # Infer grid side length assuming a square(-ish) grid via sqrt(n_states).
    if contrastive_gamma is None:
        side = int(round(np.sqrt(n_states)))
        _min_path = (side - 1) * 2  # minimum corner-to-corner Manhattan distance
        contrastive_gamma = _min_path / (_min_path + 1) if _min_path > 0 else 0.9

    agent = GoalConditionedAgent(
        n_states=n_states,
        n_actions=n_actions,
        gamma=gamma,
        contrastive_gamma=contrastive_gamma,
        alpha=alpha,
        temperature=temperature,
        n_negatives=n_negatives,
        logsumexp_reg=logsumexp_reg,
        n_critic_updates=n_critic_updates,
        seed=seed,
    )
    # Warm-start from the pre-trained C-table
    agent.C = c_table.copy()
    agent.set_goal(new_goal)

    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        obs, _ = new_goal_env.reset(seed=int(rng.integers(0, 2**31)))
        agent.reset()
        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = new_goal_env.step(action)
            agent.update(obs, action, reward, next_obs, terminated, truncated, info)
            obs = next_obs
            if terminated or truncated:
                break
        agent.finish_episode_with_contrastive_update()

    updated_c = agent.C.copy()
    return gcrl_action_fn(updated_c, new_goal), updated_c


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_greedy_episode(
    env,
    action_fn: Callable[[int], int],
    seed: int | None = None,
) -> TransferResult:
    """Run a single deterministic episode with ``action_fn`` and return results.

    Parameters
    ----------
    env :
        A :class:`environments.gridworld.GridWorld` instance configured with
        the *new* goal position so that ``reward=1`` when that goal is reached.
    action_fn :
        ``state_int → action_int``.  Typically produced by
        :func:`rcrl_action_fn` or :func:`gcrl_action_fn`.
    seed :
        Optional seed forwarded to ``env.reset()``.

    Returns
    -------
    TransferResult
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    trajectory: list[tuple[int, int, int, float]] = []

    while True:
        state = int(np.asarray(obs).flat[0])
        action = action_fn(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += float(reward)
        steps += 1
        trajectory.append((steps, state, action, float(reward)))
        obs = next_obs

        if terminated or truncated:
            # Append the arrival state as a terminal marker (action=-1) so
            # trajectory visualisations draw the final arrow to the goal cell.
            if terminated:
                next_state = int(np.asarray(next_obs).flat[0])
                trajectory.append((steps + 1, next_state, -1, 0.0))
            break

    return TransferResult(
        total_reward=total_reward,
        length=steps,
        reached_goal=bool(terminated),
        trajectory=trajectory,
    )
