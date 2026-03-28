"""
value_functions.py
==================
Sample-based value estimation: Monte-Carlo returns, TD errors, and GAE.

These functions consume *sequences* of experience collected by rolling out
a policy in an environment and return the corresponding value targets.

All inputs / outputs are 1-D numpy arrays indexed by time step t = 0, 1, …, T-1.
"""

from __future__ import annotations

import numpy as np


def monte_carlo_returns(
    rewards: np.ndarray,
    gamma: float,
    terminated: np.ndarray | None = None,
) -> np.ndarray:
    """Compute discounted Monte-Carlo returns G_t for every step t.

    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … + γ^{T-t-1}·r_{T-1}

    Episode boundaries (``terminated[t] = True``) reset the return
    accumulation so that future rewards do not bleed across episodes.

    Parameters
    ----------
    rewards : np.ndarray, shape (T,)
        Sequence of scalar rewards.
    gamma : float
        Discount factor γ ∈ [0, 1).
    terminated : np.ndarray of bool, shape (T,), optional
        Boolean mask indicating terminal time steps.  If ``None``, a single
        continuous episode is assumed.

    Returns
    -------
    returns : np.ndarray, shape (T,)
        Discounted return for each time step.
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    G = 0.0
    if terminated is None:
        terminated = np.zeros(T, dtype=bool)

    for t in reversed(range(T)):
        if terminated[t]:
            G = 0.0  # reset at episode boundary
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def td_error(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    terminated: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the TD(0) temporal-difference error δ_t.

    δ_t = r_t + γ · V(s_{t+1}) · (1 - terminated_t) - V(s_t)

    Parameters
    ----------
    rewards : np.ndarray, shape (T,)
        Sequence of scalar rewards.
    values : np.ndarray, shape (T,)
        Estimated values V(s_t) for each time step.
    next_values : np.ndarray, shape (T,)
        Estimated values V(s_{t+1}) for each time step.
    gamma : float
        Discount factor.
    terminated : np.ndarray of bool, shape (T,), optional
        Boolean termination mask; terminal transitions have V(s_{t+1}) = 0.

    Returns
    -------
    deltas : np.ndarray, shape (T,)
        TD errors δ_t.
    """
    if terminated is None:
        terminated = np.zeros(len(rewards), dtype=bool)
    mask = 1.0 - terminated.astype(np.float64)
    return rewards + gamma * next_values * mask - values


def generalised_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    lam: float,
    terminated: np.ndarray | None = None,
) -> np.ndarray:
    """Generalised Advantage Estimation (GAE-λ).

    Ã_t = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}

    where δ_t is the TD(0) error.  Setting λ=1 recovers Monte-Carlo
    advantage; λ=0 recovers single-step TD advantage.

    Reference: Schulman et al. (2016) "High-Dimensional Continuous Control
    Using Generalised Advantage Estimation."

    Parameters
    ----------
    rewards : np.ndarray, shape (T,)
    values : np.ndarray, shape (T,)
    next_values : np.ndarray, shape (T,)
    gamma : float
        Discount factor.
    lam : float
        GAE λ parameter controlling the bias-variance trade-off.
    terminated : np.ndarray of bool, shape (T,), optional

    Returns
    -------
    advantages : np.ndarray, shape (T,)
        Generalised advantage estimates.
    """
    deltas = td_error(rewards, values, next_values, gamma, terminated)
    T = len(deltas)
    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0

    if terminated is None:
        terminated = np.zeros(T, dtype=bool)

    for t in reversed(range(T)):
        if terminated[t]:
            gae = 0.0
        gae = deltas[t] + gamma * lam * gae
        advantages[t] = gae

    return advantages
