"""
policy_gradient.py
==================
Policy-gradient objective functions and related utilities.

These functions operate on arrays of log-probabilities, advantages, and
rewards collected from rollouts.  They return *scalar* loss values suitable
for automatic differentiation (PyTorch or JAX gradients not included here —
the implementations use pure NumPy so they can also serve as readable
mathematical references).

References
----------
* Williams (1992) "Simple Statistical Gradient-Following Algorithms" (REINFORCE)
* Schulman et al. (2017) "Proximal Policy Optimization Algorithms" (PPO-Clip)
* Mnih et al. (2016) "Asynchronous Methods for Deep RL" (entropy regularisation)
"""

from __future__ import annotations

import numpy as np


def reinforce_loss(
    log_probs: np.ndarray,
    returns: np.ndarray,
    baseline: np.ndarray | None = None,
) -> float:
    """REINFORCE policy-gradient loss (negated for gradient *descent*).

    L(θ) = -E_t [ log π_θ(a_t | s_t) · (G_t - b(s_t)) ]

    The negative sign converts the gradient *ascent* objective into a loss
    suitable for standard optimisers that perform gradient descent.

    Parameters
    ----------
    log_probs : np.ndarray, shape (T,)
        Log-probabilities log π_θ(a_t | s_t) for each time step.
    returns : np.ndarray, shape (T,)
        Discounted returns G_t (from :func:`~equations.value_functions.monte_carlo_returns`).
    baseline : np.ndarray, shape (T,), optional
        State-dependent baseline b(s_t) subtracted to reduce variance.
        Common choice: learned value function V(s_t).

    Returns
    -------
    loss : float
        Scalar REINFORCE loss (lower is better for gradient descent).
    """
    advantages = returns if baseline is None else returns - baseline
    return float(-np.mean(log_probs * advantages))


def entropy_bonus(
    action_probs: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Shannon entropy of a discrete action distribution.

    H(π(· | s)) = -Σ_a π(a|s) · log π(a|s)

    Used as a regularisation term to encourage exploration:

        L_total = L_PG - β · H(π)

    Parameters
    ----------
    action_probs : np.ndarray, shape (|A|,) or (T, |A|)
        Action probability distribution(s).  Each row must sum to 1.
    epsilon : float
        Small constant for numerical stability in the log.

    Returns
    -------
    entropy : float
        Mean entropy across all provided distributions.
    """
    probs = np.asarray(action_probs, dtype=np.float64)
    log_probs = np.log(np.clip(probs, epsilon, 1.0))
    H = -np.sum(probs * log_probs, axis=-1)
    return float(np.mean(H))


def ppo_clip_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    clip_epsilon: float = 0.2,
) -> float:
    """PPO clipped surrogate objective (negated for gradient descent).

    L_CLIP(θ) = -E_t [ min( r_t(θ) · Ã_t,  clip(r_t(θ), 1-ε, 1+ε) · Ã_t ) ]

    where r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)  is the probability ratio.

    Reference: Schulman et al. (2017), Equation (7).

    Parameters
    ----------
    log_probs_new : np.ndarray, shape (T,)
        Log-probabilities under the *current* policy π_θ.
    log_probs_old : np.ndarray, shape (T,)
        Log-probabilities under the *old* policy π_{θ_old} (frozen).
    advantages : np.ndarray, shape (T,)
        Advantage estimates (e.g. from GAE).
    clip_epsilon : float
        Clipping range ε.

    Returns
    -------
    loss : float
        Scalar PPO-Clip loss.
    """
    ratios = np.exp(log_probs_new - log_probs_old)  # r_t(θ) = exp(log π_new - log π_old)
    clipped_ratios = np.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = np.minimum(ratios * advantages, clipped_ratios * advantages)
    return float(-np.mean(surrogate))
