import torch
import torch.nn.functional as F
from torch import nn


def project_to_l2_ball(x: torch.Tensor, max_norm: float, eps: float = 1e-08) -> torch.Tensor:
    """
    Project each final-dimension vector into an L2 ball.

    Vectors with norm <= max_norm are unchanged.
    Vectors with norm > max_norm are rescaled to max_norm.

    Args:
        x:
            Tensor with shape [..., embedding_dim].

        max_norm:
            Maximum allowed L2 norm.

        eps:
            Numerical stability constant.

    Returns:
        Tensor with the same shape as x and
        final-dimension L2 norm <= max_norm.
    """
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive.")
    norms = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / norms.clamp_min(eps), max=1.0)
    return x * scale


class FactorisedQNetwork(nn.Module):
    """
    Factorised goal-conditioned Q-network:

        Q(s, a, g)
        =
        phi(s, a)^T psi(g)

    The state-action and goal encoder outputs are not
    unit-normalised. Instead, each vector is projected
    into a bounded L2 ball:

        ||phi(s,a)|| <= phi_max_norm
        ||psi(g)||   <= psi_max_norm

    Therefore:

        |Q(s,a,g)|
        <= phi_max_norm * psi_max_norm

    This allows variable embedding magnitudes while
    retaining a provable Q-value bound.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        goal_dim: int = 2,
        hidden_dim: int = 128,
        rep_dim: int = 64,
        phi_max_norm: float = 2.0,
        psi_max_norm: float = 5.0,
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive.")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        if goal_dim <= 0:
            raise ValueError("goal_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if rep_dim <= 0:
            raise ValueError("rep_dim must be positive.")
        if phi_max_norm <= 0.0:
            raise ValueError("phi_max_norm must be positive.")
        if psi_max_norm <= 0.0:
            raise ValueError("psi_max_norm must be positive.")
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.action_dim = num_actions
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        self.rep_dim = rep_dim
        self.phi_max_norm = float(phi_max_norm)
        self.psi_max_norm = float(psi_max_norm)
        self.sa_encoder = nn.Sequential(
            nn.Linear(obs_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, rep_dim)
        )

    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        """
        Encode goal vectors into bounded task representations.

        Args:
            goal:
                Tensor with shape [B, goal_dim] or
                [goal_dim].

        Returns:
            Tensor with shape [B, rep_dim] or
            [rep_dim], matching the input batch structure.
        """
        psi_logits = self.goal_encoder(goal)
        psi = project_to_l2_ball(psi_logits, max_norm=self.psi_max_norm)
        return psi

    def encode_state_action(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        Encode state-action pairs into bounded representations.

        Args:
            obs:
                Tensor with shape [B, obs_dim].

            act:
                Tensor with shape [B, action_dim].

        Returns:
            Tensor with shape [B, rep_dim].
        """
        if obs.ndim != 2:
            raise ValueError(f"obs must have shape [B, obs_dim]. Got {tuple(obs.shape)}.")
        if act.ndim != 2:
            raise ValueError(f"act must have shape [B, action_dim]. Got {tuple(act.shape)}.")
        if obs.shape[0] != act.shape[0]:
            raise ValueError("obs and act must have the same batch size.")
        sa = torch.cat([obs, act], dim=-1)
        phi_logits = self.sa_encoder(sa)
        phi = project_to_l2_ball(phi_logits, max_norm=self.phi_max_norm)
        return phi

    def forward(self, obs: torch.Tensor, act: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s, a, g).

        Args:
            obs:
                [B, obs_dim]

            act:
                [B, action_dim]

            goal:
                [B, goal_dim]

        Returns:
            Q-values with shape [B, 1].
        """
        phi = self.encode_state_action(obs, act)
        psi = self.encode_goal(goal)
        if psi.ndim == 1:
            psi = psi.unsqueeze(0).expand(obs.shape[0], -1)
        q_values = (phi * psi).sum(dim=-1, keepdim=True)
        return q_values

    def q_val_for_argmax_action(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for every discrete action.

        Args:
            obs:
                Tensor with shape [B, obs_dim].

            goal:
                Tensor with shape [B, goal_dim] or
                [goal_dim].

        Returns:
            Tensor with shape [B, num_actions].
        """
        if obs.ndim != 2:
            raise ValueError("obs must have shape [B, obs_dim].")
        batch_size = obs.shape[0]
        num_actions = self.num_actions
        action_onehot = F.one_hot(
            torch.arange(num_actions, device=obs.device), num_classes=self.action_dim
        ).to(dtype=obs.dtype)
        action_onehot = action_onehot.unsqueeze(0).expand(batch_size, -1, -1)
        obs_rep = obs.unsqueeze(1).expand(-1, num_actions, -1)
        obs_flat = obs_rep.reshape(batch_size * num_actions, self.obs_dim)
        act_flat = action_onehot.reshape(batch_size * num_actions, self.action_dim)
        phi = self.encode_state_action(obs_flat, act_flat)
        phi = phi.reshape(batch_size, num_actions, self.rep_dim)
        if goal.ndim == 1:
            goal = goal.unsqueeze(0).expand(batch_size, -1)
        elif goal.shape[0] == 1:
            goal = goal.expand(batch_size, -1)
        elif goal.shape[0] != batch_size:
            raise ValueError("goal must have shape [goal_dim], [1, goal_dim], or [B, goal_dim].")
        psi = self.encode_goal(goal)
        psi = psi.unsqueeze(1).expand(batch_size, num_actions, -1)
        q_values = (phi * psi).sum(dim=-1)
        return q_values

    def forward_with_task_embedding(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        task_embedding: torch.Tensor,
        normalize_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Compute Q(s, a | psi) using a supplied task embedding.

        The default behaviour is to apply bounded-ball projection
        to the supplied task embedding. The argument
        normalize_embedding=True is retained only for
        backwards-compatible cosine experiments.

        Args:
            obs:
                Tensor with shape [B, obs_dim].

            act:
                Tensor with shape [B, action_dim].

            task_embedding:
                Tensor with shape [rep_dim] or [B, rep_dim].

            normalize_embedding:
                If False:
                    project task_embedding into the psi ball.

                If True:
                    apply strict L2 normalisation.

        Returns:
            Tensor with shape [B, 1].
        """
        phi = self.encode_state_action(obs, act)
        if task_embedding.ndim == 1:
            task_embedding = task_embedding.unsqueeze(0).expand(obs.shape[0], -1)
        elif task_embedding.shape[0] == 1:
            task_embedding = task_embedding.expand(obs.shape[0], -1)
        elif task_embedding.shape[0] != obs.shape[0]:
            raise ValueError(
                "task_embedding must have shape [rep_dim], [1, rep_dim], or [B, rep_dim]."
            )
        if normalize_embedding:
            psi = F.normalize(task_embedding, p=2, dim=-1, eps=1e-08)
        else:
            psi = project_to_l2_ball(task_embedding, max_norm=self.psi_max_norm)
        q_values = (phi * psi).sum(dim=-1, keepdim=True)
        return q_values

    def q_val_for_argmax_action_from_embedding(
        self, obs: torch.Tensor, task_embedding: torch.Tensor, normalize_embedding: bool = False
    ) -> torch.Tensor:
        """
        Compute Q-values for every discrete action using
        a supplied task embedding.

        Args:
            obs:
                Tensor with shape [B, obs_dim].

            task_embedding:
                Tensor with shape [rep_dim] or [B, rep_dim].

            normalize_embedding:
                If False:
                    project task_embedding into the psi ball.

                If True:
                    apply strict L2 normalisation.

        Returns:
            Tensor with shape [B, num_actions].
        """
        if obs.ndim != 2:
            raise ValueError("obs must have shape [B, obs_dim].")
        batch_size = obs.shape[0]
        num_actions = self.num_actions
        action_onehot = F.one_hot(
            torch.arange(num_actions, device=obs.device), num_classes=self.action_dim
        ).to(dtype=obs.dtype)
        action_onehot = action_onehot.unsqueeze(0).expand(batch_size, -1, -1)
        obs_rep = obs.unsqueeze(1).expand(-1, num_actions, -1)
        obs_flat = obs_rep.reshape(batch_size * num_actions, self.obs_dim)
        act_flat = action_onehot.reshape(batch_size * num_actions, self.action_dim)
        phi = self.encode_state_action(obs_flat, act_flat)
        phi = phi.reshape(batch_size, num_actions, self.rep_dim)
        if task_embedding.ndim == 1:
            task_embedding = task_embedding.unsqueeze(0).expand(batch_size, -1)
        elif task_embedding.shape[0] == 1:
            task_embedding = task_embedding.expand(batch_size, -1)
        elif task_embedding.shape[0] != batch_size:
            raise ValueError(
                "task_embedding must have shape [rep_dim], [1, rep_dim], or [B, rep_dim]."
            )
        if normalize_embedding:
            psi = F.normalize(task_embedding, p=2, dim=-1, eps=1e-08)
        else:
            psi = project_to_l2_ball(task_embedding, max_norm=self.psi_max_norm)
        psi = psi.unsqueeze(1).expand(batch_size, num_actions, -1)
        q_values = (phi * psi).sum(dim=-1)
        return q_values

    @torch.no_grad()
    def representation_norms(self, obs: torch.Tensor, act: torch.Tensor, goal: torch.Tensor):
        """
        Return phi norm, psi norm, and Q-value diagnostics.
        """
        phi = self.encode_state_action(obs, act)
        psi = self.encode_goal(goal)
        if psi.ndim == 1:
            psi = psi.unsqueeze(0).expand(obs.shape[0], -1)
        q_values = (phi * psi).sum(dim=-1)
        return {
            "phi_norm": phi.norm(dim=-1),
            "psi_norm": psi.norm(dim=-1),
            "q_values": q_values,
            "theoretical_q_bound": self.phi_max_norm * self.psi_max_norm,
        }
