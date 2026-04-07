"""
transfer_eval.py
================
Zero-shot transfer-goal evaluation utilities.

After training, both the GCRL C-table and the RCRL Q-table are persisted to
disk.  This module provides lightweight helpers to load those artefacts and
run a **single greedy episode** in an environment with a *new* goal state —
no additional training.

The key contrast this exposes in the dissertation:

* **RCRL** (reward-conditioned): the Q-table Q[s, ψ*, a] was trained to
  maximise reward at the *original* goal.  When evaluated against a *new*
  goal the policy still navigates toward the original goal → 0 reward.

* **GCRL** (contrastive RL): the C-table C[s, a, sf] captures reachability
  for *all* future states sf, not just the training goal.  By querying
  C[s, :, new_goal] at eval time the agent can zero-shot navigate to any
  goal that appeared as a future state during training.

Public API
----------
* :func:`run_greedy_episode` — shared episode runner given a callable
  ``action_fn(state_int) -> action_int``.
* :func:`rcrl_action_fn`     — wraps the saved RCRL Q slice.
* :func:`gcrl_action_fn`     — wraps the saved GCRL C table for a given goal.
* :func:`TransferResult`     — lightweight dataclass holding episode metrics
  and the recorded trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TransferResult:
    """Metrics from one zero-shot transfer evaluation episode.

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
