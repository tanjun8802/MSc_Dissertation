from itertools import chain

import torch
from torch import nn


class FactorisedTwinCritic(nn.Module):
    """
    Goal-conditioned factorised twin critic:

        Q_i(s, a, g) = phi_i(s, a)^T psi_i(g)

    Inputs:
        state:  [B, state_dim]
        action: [B, action_dim]
        goal:   [B, goal_dim]

    Outputs:
        q1: [B, 1]
        q2: [B, 1]

    Each Q-head has independent phi and psi encoders.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        activation_fn: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim)
        self.latent_dim = int(latent_dim)
        state_action_dim = self.state_dim + self.action_dim

        def mlp(input_dim: int, output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.phi1 = mlp(state_action_dim, self.latent_dim)
        self.psi1 = mlp(self.goal_dim, self.latent_dim)
        self.phi2 = mlp(state_action_dim, self.latent_dim)
        self.psi2 = mlp(self.goal_dim, self.latent_dim)

    def _check_inputs(self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor):
        if state.ndim != 2:
            raise ValueError(f"state must be [B, {self.state_dim}], got {tuple(state.shape)}.")
        if action.ndim != 2:
            raise ValueError(f"action must be [B, {self.action_dim}], got {tuple(action.shape)}.")
        if goal.ndim != 2:
            raise ValueError(f"goal must be [B, {self.goal_dim}], got {tuple(goal.shape)}.")
        if state.shape[0] != action.shape[0]:
            raise ValueError("State and action batch sizes differ.")
        if state.shape[0] != goal.shape[0]:
            raise ValueError("State and goal batch sizes differ.")
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {state.shape[-1]}.")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action.shape[-1]}.")
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(f"Expected goal_dim={self.goal_dim}, got {goal.shape[-1]}.")

    def phi1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.phi1(torch.cat([state, action], dim=-1))

    def psi1_forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.psi1(goal)

    def phi2_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.phi2(torch.cat([state, action], dim=-1))

    def psi2_forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.psi2(goal)

    def q1_forward(
        self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        self._check_inputs(state, action, goal)
        phi = self.phi1_forward(state, action)
        psi = self.psi1_forward(goal)
        return (phi * psi).sum(dim=-1, keepdim=True)

    def q2_forward(
        self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        self._check_inputs(state, action, goal)
        phi = self.phi2_forward(state, action)
        psi = self.psi2_forward(goal)
        return (phi * psi).sum(dim=-1, keepdim=True)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.q1_forward(state, action, goal), self.q2_forward(state, action, goal))

    @torch.no_grad()
    def embedding_diagnostics(
        self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
    ) -> dict[str, float]:
        """
        Optional monitoring only. It does not modify representations.
        """
        phi1 = self.phi1_forward(state, action)
        psi1 = self.psi1_forward(goal)
        phi2 = self.phi2_forward(state, action)
        psi2 = self.psi2_forward(goal)
        return {
            "phi1_norm": float(phi1.norm(dim=-1).mean().cpu()),
            "psi1_norm": float(psi1.norm(dim=-1).mean().cpu()),
            "phi2_norm": float(phi2.norm(dim=-1).mean().cpu()),
            "psi2_norm": float(psi2.norm(dim=-1).mean().cpu()),
            "phi1_abs": float(phi1.abs().mean().cpu()),
            "psi1_abs": float(psi1.abs().mean().cpu()),
            "phi2_abs": float(phi2.abs().mean().cpu()),
            "psi2_abs": float(psi2.abs().mean().cpu()),
        }

    def critic1_parameters(self):
        """
        Return all trainable parameters belonging to critic 1:
            phi1, psi1 (and q1 if you later add an explicit head).
        """
        return chain(self.phi1.parameters(), self.psi1.parameters())

    def critic2_parameters(self):
        """
        Return all trainable parameters belonging to critic 2:
            phi2, psi2 (and q2 if you later add an explicit head).
        """
        return chain(self.phi2.parameters(), self.psi2.parameters())


class GaussianActor(nn.Module):
    """
    Goal-conditioned squashed-Gaussian actor for SAC.

    Input:
        state: [B, state_dim]
        goal:  [B, goal_dim]

    Output:
        sampled action in [-1, 1]^action_dim;
        log-probability with the tanh change-of-variables correction;
        deterministic action tanh(mu).

    The environment action space must be [-1, 1]^action_dim.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        net_arch: tuple[int, ...] = (256, 256),
        activation_fn: type[nn.Module] = nn.ReLU,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        input_dim = self.state_dim + self.goal_dim
        modules = []
        last_dim = input_dim
        for hidden_dim in net_arch:
            modules.append(nn.Linear(last_dim, hidden_dim))
            modules.append(activation_fn())
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*modules)
        self.mu_layer = nn.Linear(last_dim, self.action_dim)
        self.log_std_layer = nn.Linear(last_dim, self.action_dim)

    def _validate_inputs(
        self, state: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError(f"state must have shape [B, state_dim]. Got {tuple(state.shape)}.")
        if goal.ndim != 2:
            raise ValueError(f"goal must have shape [B, goal_dim]. Got {tuple(goal.shape)}.")
        if state.shape[0] != goal.shape[0]:
            raise ValueError(
                f"State and goal batch dimensions must match: {state.shape[0]} vs {goal.shape[0]}."
            )
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {state.shape[-1]}.")
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(f"Expected goal dimension {self.goal_dim}, got {goal.shape[-1]}.")
        return (state, goal)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean:    [B, action_dim]
            log_std: [B, action_dim]
        """
        (state, goal) = self._validate_inputs(state, goal)
        x = torch.cat([state, goal], dim=-1)
        x = self.backbone(x)
        mean = self.mu_layer(x)
        log_std = self.log_std_layer(x).clamp(min=self.log_std_min, max=self.log_std_max)
        return (mean, log_std)

    def sample(
        self, state: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparameterized SAC sample.

        Returns:
            action:
                Tanh-squashed sampled action [B, action_dim].
            log_prob:
                Log probability of action after tanh correction [B, 1].
            deterministic_action:
                tanh(mean), for deterministic evaluation [B, action_dim].
        """
        (mean, log_std) = self.forward(state, goal)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre_tanh_action = normal.rsample()
        action = torch.tanh(pre_tanh_action)
        log_prob = normal.log_prob(pre_tanh_action)
        tanh_correction = torch.log(1.0 - action.pow(2) + 1e-06)
        log_prob = (log_prob - tanh_correction).sum(dim=-1, keepdim=True)
        deterministic_action = torch.tanh(mean)
        return (action, log_prob, deterministic_action)

    def deterministic(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Deterministic tanh(mean) action for evaluation.
        """
        (mean, _) = self.forward(state, goal)
        return torch.tanh(mean)


class FactorisedGaussianActor(GaussianActor):
    """Bilinear goal-conditioned Gaussian policy with independent actor factors.

    mu_j(s,g) = phi_mu_j(s)^T psi(g), and likewise for log_std_j.
    Both phi encoders return [batch, action_dim, latent_dim]; psi returns
    [batch, latent_dim]. There is no action input, critic parameter sharing,
    or hard embedding projection. Gaussian sampling/tanh correction are inherited.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        hidden_dim=64,
        latent_dim=16,
        log_std_min=-20.0,
        log_std_max=2.0,
        output_init_scale=0.01,
    ):
        # Reuse GaussianActor's policy interface without creating unused MLP heads.
        nn.Module.__init__(self)
        if any(
            not isinstance(d, int) or d < 1
            for d in (state_dim, action_dim, goal_dim, hidden_dim, latent_dim)
        ):
            raise ValueError("Actor dimensions must be positive integers.")
        if not 0 < output_init_scale <= 1:
            raise ValueError("output_init_scale must be in (0, 1].")
        if not log_std_min < log_std_max:
            raise ValueError("Require log_std_min < log_std_max.")
        self.state_dim, self.action_dim, self.goal_dim = state_dim, action_dim, goal_dim
        self.latent_dim = latent_dim
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

        def encoder(input_dim, output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
            )

        self.phi_mu = encoder(state_dim, action_dim * latent_dim)
        self.phi_log_std = encoder(state_dim, action_dim * latent_dim)
        self.psi = encoder(goal_dim, latent_dim)
        # Start near zero mean / unit std rather than saturating tanh or the
        # log-std clamp before the state and goal representations have learned.
        with torch.no_grad():
            for branch in (self.phi_mu, self.phi_log_std):
                branch[-1].weight.mul_(output_init_scale)
                branch[-1].bias.mul_(output_init_scale)

    def state_factors(self, state):
        shape = (state.shape[0], self.action_dim, self.latent_dim)
        return self.phi_mu(state).view(shape), self.phi_log_std(state).view(shape)

    def goal_factors(self, goal):
        return self.psi(goal)

    def forward(self, state, goal):
        state, goal = self._validate_inputs(state, goal)
        phi_mu, phi_std = self.state_factors(state)
        psi = self.goal_factors(goal).unsqueeze(1)
        mean = (phi_mu * psi).sum(-1)
        log_std = (phi_std * psi).sum(-1).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std
