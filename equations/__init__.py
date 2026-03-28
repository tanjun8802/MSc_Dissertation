"""
equations/
==========
Core RL equations implemented as clean, documented, reusable functions.

Modules
-------
bellman         : Bellman optimality / expectation equations.
value_functions : Monte-Carlo and TD value estimation.
policy_gradient : REINFORCE and advantage estimation.
"""

from equations.bellman import (
    bellman_expectation_v,
    bellman_optimality_v,
    bellman_optimality_q,
)
from equations.value_functions import (
    monte_carlo_returns,
    td_error,
    generalised_advantage_estimate,
)
from equations.policy_gradient import (
    reinforce_loss,
    entropy_bonus,
    ppo_clip_loss,
)

__all__ = [
    # Bellman
    "bellman_expectation_v",
    "bellman_optimality_v",
    "bellman_optimality_q",
    # Value functions
    "monte_carlo_returns",
    "td_error",
    "generalised_advantage_estimate",
    # Policy gradient
    "reinforce_loss",
    "entropy_bonus",
    "ppo_clip_loss",
]
