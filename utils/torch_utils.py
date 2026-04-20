"""
torch_utils.py
==============
PyTorch utilities shared across all Deep RL agents.

This module provides:

1. :func:`get_device` — auto-select the best available compute device (CUDA
   GPU > Apple MPS > CPU) so that all agents can trivially offload to GPU.

2. :class:`MLP` — a configurable multi-layer perceptron used as the backbone
   for most tabular-to-neural network upgrades (DQN value head, actor/critic
   networks, etc.).

3. :class:`BaseDeepAgent` — abstract extension of
   :class:`~agents.base_agent.BaseAgent` that adds a PyTorch network, an
   Adam optimiser, and a ``device`` attribute so that concrete agents
   (DQN, SAC, PPO, …) only need to implement their forward pass and update
   logic.

Usage
-----
    from utils.torch_utils import get_device, MLP, BaseDeepAgent

    device = get_device()
    net = MLP(in_dim=4, out_dim=2, hidden_sizes=[128, 128]).to(device)

Overview of how Deep RL networks are typically structured
---------------------------------------------------------
In Deep RL the neural network maps from *observations* to *useful
quantities* (Q-values, policy logits, or value estimates):

    Input  : flat observation vector  o ∈ R^{n_obs}
             (pixel encoders add a CNN before this MLP)
    Hidden : stack of Linear → activation layers
    Output : depends on the algorithm:
      - DQN/DDQN      → Q(o, :) ∈ R^{n_actions}   (one Q-value per action)
      - Policy-gradient (actor) → logits ∈ R^{n_actions}  (discrete)
                                   or (μ, log σ) ∈ R^{2·n_actions}  (continuous)
      - Critic / value head → V(o) ∈ R¹

Typical training loop
---------------------
1. Collect a transition (o, a, r, o', done) with ε-greedy / softmax / reparameterised policy.
2. Store in a replay buffer.
3. Sample a mini-batch of size B.
4. Forward pass  → predicted values / logits.
5. Compute loss  (TD error for DQN, policy-gradient loss for PPO, etc.).
6. loss.backward() → optimizer.step() → (optionally) target-network soft update.
7. Repeat.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False  # allow non-deep agents to import without torch
    # Provide dummy base so the class definitions below don't NameError at import
    import types as _types
    nn = _types.ModuleType("nn")
    nn.Module = object
    nn.Sequential = object
    nn.Linear = object
    nn.ReLU = object

from agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def get_device(prefer_gpu: bool = True) -> "torch.device":
    """Return the best available compute device.

    Priority order:
      1. CUDA (any NVIDIA GPU)
      2. Apple MPS (Metal Performance Shaders on M-series Macs)
      3. CPU (always available)

    Parameters
    ----------
    prefer_gpu :
        Set to ``False`` to always return the CPU device (useful for
        debugging or when memory is limited).

    Returns
    -------
    torch.device
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. "
            "Install with: pip install torch"
        )
    import torch
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Configurable multi-layer perceptron.

    Parameters
    ----------
    in_dim :
        Size of the input vector (observation dimension after flattening).
    out_dim :
        Size of the output vector (e.g. n_actions for a Q-network).
    hidden_sizes :
        Width of each hidden layer.  Default: [256, 256].
    activation :
        Non-linearity applied between layers.  Default: ``nn.ReLU``.
    output_activation :
        Optional non-linearity applied to the *output* layer.  Default:
        ``None`` (identity) — let the loss function handle activation.

    Examples
    --------
    >>> net = MLP(in_dim=4, out_dim=2, hidden_sizes=[128, 128])
    >>> net(torch.zeros(1, 4)).shape
    torch.Size([1, 2])
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: type = None,
        output_activation: type | None = None,
    ) -> None:
        super().__init__()
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. pip install torch")
        import torch.nn as nn
        if activation is None:
            activation = nn.ReLU

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# ---------------------------------------------------------------------------
# Abstract deep agent base class
# ---------------------------------------------------------------------------


class BaseDeepAgent(BaseAgent):
    """Abstract base for agents that use a PyTorch network.

    Extends :class:`~agents.base_agent.BaseAgent` with:

    * ``self.device``   — compute device (from :func:`get_device`)
    * ``self.network``  — the neural network (set by subclass in ``__init__``)
    * ``self.optimizer`` — Adam optimizer (set by subclass in ``__init__``)

    Subclasses must call ``super().__init__()`` and then set
    ``self.network`` and ``self.optimizer``.

    Parameters
    ----------
    n_obs :
        Flat observation dimensionality (size of the input vector passed to
        the network at each step).
    n_actions :
        Number of discrete actions.
    gamma :
        Discount factor γ.
    lr :
        Learning rate for the Adam optimizer.
    seed :
        Random seed (controls both numpy and torch PRNG).
    prefer_gpu :
        Auto-select GPU if available (see :func:`get_device`).
    """

    def __init__(
        self,
        n_obs: int,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        seed: int | None = None,
        prefer_gpu: bool = True,
    ) -> None:
        super().__init__(n_actions=n_actions, gamma=gamma, seed=seed)
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for deep agents. pip install torch"
            )
        import torch

        self.n_obs = n_obs
        self.lr = lr
        self.device = get_device(prefer_gpu=prefer_gpu)

        # Seed PyTorch for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        # Subclasses should set these after calling super().__init__()
        self.network: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None

    # ------------------------------------------------------------------
    # Convenience: obs → tensor
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: Any) -> "torch.Tensor":
        """Convert an observation (array-like) to a float32 device tensor.

        Parameters
        ----------
        obs :
            Raw observation from the environment (numpy array, list, or int).

        Returns
        -------
        torch.Tensor
            Shape ``(1, n_obs)`` ready to pass to ``self.network``.
        """
        import torch
        arr = np.asarray(obs, dtype=np.float32).flatten()
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Abstract interface (inherited from BaseAgent)
    # ------------------------------------------------------------------

    @abstractmethod
    def select_action(self, observation: Any) -> Any:
        """Select an action given an observation."""

    @abstractmethod
    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> dict:
        """Update the agent from one experience tuple."""
