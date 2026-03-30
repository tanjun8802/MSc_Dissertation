"""
bellman.py
==========
Bellman expectation and optimality equations for tabular MDPs.

These are the *exact*, closed-form versions that operate on explicit
probability / value arrays (not stochastic estimates).  They form the
mathematical backbone of dynamic-programming solutions and provide a
ground-truth reference for verifying sample-based approximations.

Notation (follows Sutton & Barto 2018, Chapter 3)
--------------------------------------------------
  s, s'  : state indices
  a      : action index
  V[s]   : state-value function V(s)
  Q[s,a] : action-value function Q(s,a)
  π[s,a] : stochastic policy  π(a|s)
  P[s,a,s'] : transition probability P(s'|s,a)
  R[s,a,s'] : reward R(s,a,s')
  γ      : discount factor
"""

from __future__ import annotations

import numpy as np


def bellman_expectation_v(
    V: np.ndarray,
    pi: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Bellman *expectation* equation for V^π.

    Computes a single Bellman backup:

        V_new[s] = Σ_a  π(a|s) · Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V[s']]

    Parameters
    ----------
    V : np.ndarray, shape (|S|,)
        Current state-value estimates.
    pi : np.ndarray, shape (|S|, |A|)
        Policy; ``pi[s, a]`` = π(a|s).  Rows must sum to 1.
    P : np.ndarray, shape (|S|, |A|, |S|)
        Transition probabilities.
    R : np.ndarray, shape (|S|, |A|, |S|)
        Reward table.
    gamma : float
        Discount factor.

    Returns
    -------
    V_new : np.ndarray, shape (|S|,)
        One-step Bellman-updated value estimates.
    """
    # Q[s, a] = Σ_{s'} P[s,a,s'] · (R[s,a,s'] + γ · V[s'])
    Q = np.einsum("ijk,k->ij", P, gamma * V) + np.einsum("ijk,ijk->ij", P, R)
    # V_new[s] = Σ_a π(a|s) · Q[s,a]
    return np.einsum("ij,ij->i", pi, Q)


def bellman_optimality_v(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Bellman *optimality* equation for V*.

    Computes a single max-Bellman backup:

        V_new[s] = max_a  Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V[s']]

    Parameters
    ----------
    V : np.ndarray, shape (|S|,)
        Current state-value estimates.
    P : np.ndarray, shape (|S|, |A|, |S|)
        Transition probabilities.
    R : np.ndarray, shape (|S|, |A|, |S|)
        Reward table.
    gamma : float
        Discount factor.

    Returns
    -------
    V_new : np.ndarray, shape (|S|,)
        One greedy-max-Bellman-updated value estimates.
    """
    Q = np.einsum("ijk,k->ij", P, gamma * V) + np.einsum("ijk,ijk->ij", P, R)
    return Q.max(axis=1)


def bellman_optimality_q(
    Q: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Bellman *optimality* equation for Q*.

    Computes a single Bellman backup over the action-value function:

        Q_new[s, a] = Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · max_{a'} Q[s', a']]

    Parameters
    ----------
    Q : np.ndarray, shape (|S|, |A|)
        Current action-value estimates.
    P : np.ndarray, shape (|S|, |A|, |S|)
        Transition probabilities.
    R : np.ndarray, shape (|S|, |A|, |S|)
        Reward table.
    gamma : float
        Discount factor.

    Returns
    -------
    Q_new : np.ndarray, shape (|S|, |A|)
        Updated action-value estimates.
    """
    V_next = Q.max(axis=1)  # max_{a'} Q[s', a'], shape (|S|,)
    expected_R = np.einsum("ijk,ijk->ij", P, R)           # (|S|, |A|)
    expected_V = np.einsum("ijk,k->ij", P, gamma * V_next)  # (|S|, |A|)
    return expected_R + expected_V
