# Now here is the NN to train the sampled batch data from the replay buffer, the goal is to learn a dynamics model that predicts s_next from (s,a) pairs.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence


def snapshot_named_parameters(model, prefix_filter=None):
    out = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if prefix_filter is None:
            out[name] = p.detach().clone()
        elif isinstance(prefix_filter, str):
            if name.startswith(prefix_filter):
                out[name] = p.detach().clone()
        else:
            if any(name.startswith(pref) for pref in prefix_filter):
                out[name] = p.detach().clone()
    return out

class StateActionRepresentationModel(nn.Module): # a simple MLP that takes in (s,a) and predicts s_next, use hidden_layers to control the depth of the MLP, and hidden_dim to control the width of the MLP

    def __init__(self, obs_dim, action_dim, hidden_dim=256, output_dim=64, inner_layers=2,normalise=False): # A single goal paper specifies no normalisation
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(inner_layers)])
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.normalise = normalise

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.fc_out(x)
        if self.normalise: # if normalise is True, then we normalise the output to have unit norm, this can help with training stability and convergence, especially when using MSE loss, as it prevents the model from producing arbitrarily large outputs
            x = F.normalize(x, dim=-1)
        return x
    

class GoalRepresentationModel(nn.Module): # symmetric counterpart to StateActionRepresentationModel — maps a future/goal state sf to an embedding ψ(sf)

    def __init__(self, obs_dim, hidden_dim=256, inner_layers=2, output_dim=64, normalise=False): # same architecture choices as phi to keep the embedding space consistent
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(inner_layers)])
        self.fc_out = nn.Linear(hidden_dim, output_dim) # output in the same embedding space as StateActionRepresentationModel so that the dot product phi(s,a)^T psi(sf) is well-defined
        self.normalise = normalise

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.fc_out(x)
        if self.normalise: # paper specifies no normalisation, but we keep the flag for experimentation
            x = F.normalize(x, dim=-1)
        return x

class SAC_Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))
    
class TD3_Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

class DQN_QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs):
        return self.net(obs)
    

class FactorisedDQN_QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        goal_dim: int = 2,
        hidden_dim: int = 128,
        rep_dim: int = 64,
        alpha = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.action_dim = num_actions
        self.goal_dim = goal_dim
        self.rep_dim = rep_dim

        self.sa_encoder = nn.Sequential(
            nn.Linear(obs_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )

        self.register_buffer(
            "alpha",
            torch.tensor(
                float(alpha),
                dtype=torch.float32,
            ),
        )


    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        psi = self.goal_encoder(goal)
        psi = F.normalize(psi, p=2, dim=-1, eps=1e-8)
        return psi

    def encode_state_action(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([obs, act], dim=-1)
        phi = self.sa_encoder(sa)
        phi = F.normalize(phi, p=2, dim=-1, eps=1e-8)
        return phi

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        phi_sa = self.encode_state_action(obs, act)      # [B, D]
        psi_z = self.encode_goal(goal)                   # [B, D]
        q_vals = self.alpha * (phi_sa * psi_z).sum(dim=-1, keepdim=True)  # [B, 1]
        return q_vals

    def q_val_for_argmax_action(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        A = self.num_actions

        act_onehot = F.one_hot(
            torch.arange(A, device=obs.device),
            num_classes=self.action_dim
        ).float()                                        # [A, A]
        act_onehot = act_onehot.unsqueeze(0).expand(B, -1, -1)   # [B, A, A]

        obs_rep = obs.unsqueeze(1).expand(-1, A, -1)             # [B, A, obs_dim]
        obs_flat = obs_rep.reshape(B * A, self.obs_dim)          # [B*A, obs_dim]
        act_flat = act_onehot.reshape(B * A, self.action_dim)    # [B*A, A]

        phi_sa = self.encode_state_action(obs_flat, act_flat)    # [B*A, D]
        phi_sa = phi_sa.view(B, A, self.rep_dim)                 # [B, A, D]

        psi_z = self.encode_goal(goal)                           # [B, D]
        psi_rep = psi_z.unsqueeze(1).expand(B, A, -1)           # [B, A, D]

        q_vals = self.alpha * (phi_sa * psi_rep).sum(dim=-1)                 # [B, A]
        return q_vals
    
    def forward_with_task_embedding(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        task_embedding: torch.Tensor,
        normalize_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Compute Q(obs, act | psi) directly from a provided task embedding.

        obs: [B, obs_dim]
        act: [B, action_dim]
        task_embedding: [B, rep_dim] or [rep_dim]
        """
        phi_sa = self.encode_state_action(obs, act)  # [B, D]

        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0).expand(obs.shape[0], -1)

        psi_z = (
            F.normalize(task_embedding, p=2, dim=-1, eps=1e-8)
            if normalize_embedding
            else task_embedding
        )

        q_vals = self.alpha * (phi_sa * psi_z).sum(dim=-1, keepdim=True)   # [B, 1]
        return q_vals
    
    def q_val_for_argmax_action_from_embedding(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        normalize_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Compute Q-values for all actions using a provided task embedding.

        obs: [B, obs_dim]
        task_embedding: [B, rep_dim] or [rep_dim]
        returns: [B, A]
        """
        B = obs.shape[0]
        A = self.num_actions

        act_onehot = F.one_hot(
            torch.arange(A, device=obs.device),
            num_classes=self.action_dim
        ).float()                                             # [A, A]
        act_onehot = act_onehot.unsqueeze(0).expand(B, -1, -1)  # [B, A, A]

        obs_rep = obs.unsqueeze(1).expand(-1, A, -1)          # [B, A, obs_dim]
        obs_flat = obs_rep.reshape(B * A, self.obs_dim)       # [B*A, obs_dim]
        act_flat = act_onehot.reshape(B * A, self.action_dim) # [B*A, A]

        phi_sa = self.encode_state_action(obs_flat, act_flat) # [B*A, D]
        phi_sa = phi_sa.view(B, A, self.rep_dim)              # [B, A, D]

        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0).expand(B, -1)

        psi_z = (
            F.normalize(task_embedding, p=2, dim=-1, eps=1e-8)
            if normalize_embedding
            else task_embedding
        )

        psi_rep = psi_z.unsqueeze(1).expand(B, A, -1)         # [B, A, D]
        q_vals = self.alpha * (phi_sa * psi_rep).sum(dim=-1)     # [B, A]
        return q_vals

class Factorised_TD3_Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        goal_dim: int = 2,
        hidden_dim: int = 128,
        rep_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.goal_dim = goal_dim
        self.rep_dim = rep_dim

        self.sa_encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )

    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        psi = self.goal_encoder(goal)
        psi = F.normalize(psi, p=2, dim=-1, eps=1e-8)
        return psi

    def encode_state_action(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([obs, act], dim=-1)
        phi = self.sa_encoder(sa)
        phi = F.normalize(phi, p=2, dim=-1, eps=1e-8)
        return phi

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        phi_sa = self.encode_state_action(obs, act)
        psi_g = self.encode_goal(goal)
        q_vals = (phi_sa * psi_g).sum(dim=-1, keepdim=True)
        return q_vals

    def forward_with_task_embedding(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        task_embedding: torch.Tensor,
        normalize_embedding: bool = False,
    ) -> torch.Tensor:
        phi_sa = self.encode_state_action(obs, act)

        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0).expand(obs.shape[0], -1)

        psi_g = (
            F.normalize(task_embedding, p=2, dim=-1, eps=1e-8)
            if normalize_embedding
            else task_embedding
        )

        q_vals = (phi_sa * psi_g).sum(dim=-1, keepdim=True)
        return q_vals

    def forward_from_sa_embedding(
        self,
        sa_embedding: torch.Tensor,
        goal: torch.Tensor,
        normalize_sa: bool = False,
    ) -> torch.Tensor:
        phi_sa = (
            F.normalize(sa_embedding, p=2, dim=-1, eps=1e-8)
            if normalize_sa
            else sa_embedding
        )
        psi_g = self.encode_goal(goal)
        q_vals = (phi_sa * psi_g).sum(dim=-1, keepdim=True)
        return q_vals

    def forward_from_embeddings(
        self,
        sa_embedding: torch.Tensor,
        task_embedding: torch.Tensor,
        normalize_sa: bool = False,
        normalize_task: bool = False,
    ) -> torch.Tensor:
        phi_sa = (
            F.normalize(sa_embedding, p=2, dim=-1, eps=1e-8)
            if normalize_sa
            else sa_embedding
        )
        psi_g = (
            F.normalize(task_embedding, p=2, dim=-1, eps=1e-8)
            if normalize_task
            else task_embedding
        )

        if psi_g.dim() == 1:
            psi_g = psi_g.unsqueeze(0).expand(phi_sa.shape[0], -1)

        q_vals = (phi_sa * psi_g).sum(dim=-1, keepdim=True)
        return q_vals

class DQN_Atari_CNN(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        assert len(obs_shape) == 3, f"Expected obs_shape=(C,H,W), got {obs_shape}"
        c, h, w = obs_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.features(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, obs):
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        obs = obs.float()
        if obs.max() > 1.5:
            obs = obs / 255.0
        x = self.features(obs)
        return self.head(x)
    
class FactorisedSuccessorDQN(nn.Module):
    """
    Factorised successor-feature network.

    Learned branches:

        phi(s)       = feature_encoder(s)
        Psi(s, a)    = sa_encoder(s, a)
        w(g)         = goal_encoder(g)

    Factorised Q-function:

        Q(s, a, g) = Psi(s, a)^T w(g)

    Recommended defaults:

        normalize_features=True
        normalize_sa=False
        normalize_goal=False

    The immediate features are normalized by default to improve
    representation stability. Successor features and goal/reward
    weights are not normalized by default because their magnitudes
    can carry occupancy and reward-scale information.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        goal_dim: int = 2,
        hidden_dim: int = 128,
        rep_dim: int = 64,
        normalize_features: bool = True,
        normalize_sa: bool = False,
        normalize_goal: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.action_dim = int(num_actions)
        self.goal_dim = int(goal_dim)
        self.rep_dim = int(rep_dim)

        self.normalize_features = bool(
            normalize_features
        )
        self.normalize_sa = bool(
            normalize_sa
        )
        self.normalize_goal = bool(
            normalize_goal
        )
        self.use_layer_norm = bool(
            use_layer_norm
        )

        self.feature_encoder = self._make_mlp(
            input_dim=self.obs_dim,
            hidden_dim=hidden_dim,
            output_dim=self.rep_dim,
        )

        self.sa_encoder = self._make_mlp(
            input_dim=self.obs_dim + self.action_dim,
            hidden_dim=hidden_dim,
            output_dim=self.rep_dim,
        )

        self.goal_encoder = self._make_mlp(
            input_dim=self.goal_dim,
            hidden_dim=hidden_dim,
            output_dim=self.rep_dim,
        )

    def _make_mlp(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> nn.Sequential:
        layers = [
            nn.Linear(
                input_dim,
                hidden_dim,
            )
        ]

        if self.use_layer_norm:
            layers.append(
                nn.LayerNorm(hidden_dim)
            )

        layers.extend(
            [
                nn.ReLU(),
                nn.Linear(
                    hidden_dim,
                    output_dim,
                ),
            ]
        )

        return nn.Sequential(*layers)

    def encode_features(
        self,
        obs: torch.Tensor,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """
        Encode an observation into immediate learned features.

        This represents:

            phi(s)

        Args:
            obs:
                [B, obs_dim]

            normalize:
                Optional override of self.normalize_features.

        Returns:
            [B, rep_dim]
        """
        if normalize is None:
            normalize = self.normalize_features

        features = self.feature_encoder(obs)

        if normalize:
            features = F.normalize(
                features,
                p=2,
                dim=-1,
                eps=1e-8,
            )

        return features

    def encode_state_action(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """
        Encode a state-action pair into successor features.

        This represents:

            Psi(s, a)

        Args:
            obs:
                [B, obs_dim]

            act:
                [B, num_actions], one-hot encoded

            normalize:
                Optional override of self.normalize_sa.

        Returns:
            [B, rep_dim]
        """
        if normalize is None:
            normalize = self.normalize_sa

        if obs.ndim != 2:
            raise ValueError(
                "obs must have shape [B, obs_dim], "
                f"got {tuple(obs.shape)}."
            )

        if act.ndim != 2:
            raise ValueError(
                "act must have shape [B, num_actions], "
                f"got {tuple(act.shape)}."
            )

        if obs.shape[0] != act.shape[0]:
            raise ValueError(
                "obs and act must have the same batch size."
            )

        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                "Unexpected observation dimension: "
                f"expected {self.obs_dim}, "
                f"got {obs.shape[-1]}."
            )

        if act.shape[-1] != self.action_dim:
            raise ValueError(
                "Unexpected action dimension: "
                f"expected {self.action_dim}, "
                f"got {act.shape[-1]}."
            )

        sa_input = torch.cat(
            [obs, act],
            dim=-1,
        )

        successor_features = self.sa_encoder(
            sa_input
        )

        if normalize:
            successor_features = F.normalize(
                successor_features,
                p=2,
                dim=-1,
                eps=1e-8,
            )

        return successor_features

    def encode_goal(
        self,
        goal: torch.Tensor,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """
        Encode the raw goal into reward/task weights.

        This represents:

            w(g)

        Args:
            goal:
                [B, goal_dim]

            normalize:
                Optional override of self.normalize_goal.

        Returns:
            [B, rep_dim]
        """
        if normalize is None:
            normalize = self.normalize_goal

        if goal.ndim != 2:
            raise ValueError(
                "goal must have shape [B, goal_dim], "
                f"got {tuple(goal.shape)}."
            )

        if goal.shape[-1] != self.goal_dim:
            raise ValueError(
                "Unexpected goal dimension: "
                f"expected {self.goal_dim}, "
                f"got {goal.shape[-1]}."
            )

        reward_weights = self.goal_encoder(
            goal
        )

        if normalize:
            reward_weights = F.normalize(
                reward_weights,
                p=2,
                dim=-1,
                eps=1e-8,
            )

        return reward_weights

    def q_from_embeddings(
        self,
        successor_features: torch.Tensor,
        reward_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q from the factorisation:

            Q = Psi(s, a)^T w(g)

        Args:
            successor_features:
                [B, rep_dim]

            reward_weights:
                [B, rep_dim]

        Returns:
            [B, 1]
        """
        if successor_features.shape != reward_weights.shape:
            raise ValueError(
                "successor_features and reward_weights "
                "must have identical shapes. "
                f"Got {tuple(successor_features.shape)} "
                f"and {tuple(reward_weights.shape)}."
            )

        return (
            successor_features
            * reward_weights
        ).sum(
            dim=-1,
            keepdim=True,
        )

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q(s, a, g).

        Args:
            obs:
                [B, obs_dim]

            act:
                [B, num_actions], one-hot

            goal:
                [B, goal_dim]

        Returns:
            [B, 1]
        """
        successor_features = (
            self.encode_state_action(
                obs,
                act,
            )
        )

        reward_weights = (
            self.encode_goal(goal)
        )

        return self.q_from_embeddings(
            successor_features,
            reward_weights,
        )

    def q_val_for_argmax_action(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-values for every discrete action.

        Args:
            obs:
                [B, obs_dim]

            goal:
                [B, goal_dim]

        Returns:
            [B, num_actions]
        """
        if obs.ndim != 2:
            raise ValueError(
                "obs must have shape [B, obs_dim], "
                f"got {tuple(obs.shape)}."
            )

        if goal.ndim != 2:
            raise ValueError(
                "goal must have shape [B, goal_dim], "
                f"got {tuple(goal.shape)}."
            )

        if obs.shape[0] != goal.shape[0]:
            raise ValueError(
                "obs and goal must have the same batch size."
            )

        batch_size = obs.shape[0]
        num_actions = self.num_actions

        action_table = F.one_hot(
            torch.arange(
                num_actions,
                device=obs.device,
            ),
            num_classes=num_actions,
        ).float()

        action_table = (
            action_table
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        obs_rep = (
            obs
            .unsqueeze(1)
            .expand(
                -1,
                num_actions,
                -1,
            )
        )

        obs_flat = obs_rep.reshape(
            batch_size * num_actions,
            self.obs_dim,
        )

        act_flat = action_table.reshape(
            batch_size * num_actions,
            self.action_dim,
        )

        successor_features = (
            self.encode_state_action(
                obs_flat,
                act_flat,
            )
        )

        successor_features = (
            successor_features.reshape(
                batch_size,
                num_actions,
                self.rep_dim,
            )
        )

        reward_weights = (
            self.encode_goal(goal)
        )

        reward_weights = (
            reward_weights
            .unsqueeze(1)
            .expand(
                -1,
                num_actions,
                -1,
            )
        )

        q_values = (
            successor_features
            * reward_weights
        ).sum(
            dim=-1
        )

        return q_values

    def forward_with_task_embedding(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        task_embedding: torch.Tensor,
        normalize_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Compute Q using an externally supplied task embedding.

        Args:
            obs:
                [B, obs_dim]

            act:
                [B, num_actions], one-hot

            task_embedding:
                [rep_dim] or [B, rep_dim]

            normalize_embedding:
                Whether to L2-normalize the supplied embedding.

        Returns:
            [B, 1]
        """
        successor_features = (
            self.encode_state_action(
                obs,
                act,
            )
        )

        if task_embedding.ndim == 1:
            task_embedding = (
                task_embedding.unsqueeze(0)
            )

        if task_embedding.shape[0] == 1:
            task_embedding = (
                task_embedding.expand(
                    obs.shape[0],
                    -1,
                )
            )

        if task_embedding.shape != successor_features.shape:
            raise ValueError(
                "task_embedding must have shape "
                f"[{obs.shape[0]}, {self.rep_dim}] "
                "after broadcasting. "
                f"Got {tuple(task_embedding.shape)}."
            )

        if normalize_embedding:
            task_embedding = F.normalize(
                task_embedding,
                p=2,
                dim=-1,
                eps=1e-8,
            )

        return self.q_from_embeddings(
            successor_features,
            task_embedding,
        )

    def q_val_for_argmax_action_from_embedding(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        normalize_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Compute Q-values for all actions using an external
        task/reward embedding.

        Args:
            obs:
                [B, obs_dim]

            task_embedding:
                [rep_dim] or [B, rep_dim]

            normalize_embedding:
                Whether to L2-normalize the supplied embedding.

        Returns:
            [B, num_actions]
        """
        if obs.ndim != 2:
            raise ValueError(
                "obs must have shape [B, obs_dim], "
                f"got {tuple(obs.shape)}."
            )

        batch_size = obs.shape[0]
        num_actions = self.num_actions

        action_table = F.one_hot(
            torch.arange(
                num_actions,
                device=obs.device,
            ),
            num_classes=num_actions,
        ).float()

        action_table = (
            action_table
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        obs_rep = (
            obs
            .unsqueeze(1)
            .expand(
                -1,
                num_actions,
                -1,
            )
        )

        obs_flat = obs_rep.reshape(
            batch_size * num_actions,
            self.obs_dim,
        )

        act_flat = action_table.reshape(
            batch_size * num_actions,
            self.action_dim,
        )

        successor_features = (
            self.encode_state_action(
                obs_flat,
                act_flat,
            )
        )

        successor_features = (
            successor_features.reshape(
                batch_size,
                num_actions,
                self.rep_dim,
            )
        )

        if task_embedding.ndim == 1:
            task_embedding = (
                task_embedding.unsqueeze(0)
            )

        if task_embedding.shape[0] == 1:
            task_embedding = (
                task_embedding.expand(
                    batch_size,
                    -1,
                )
            )

        expected_shape = (
            batch_size,
            self.rep_dim,
        )

        if tuple(task_embedding.shape) != expected_shape:
            raise ValueError(
                "task_embedding must have shape "
                f"{expected_shape}. "
                f"Got {tuple(task_embedding.shape)}."
            )

        if normalize_embedding:
            task_embedding = F.normalize(
                task_embedding,
                p=2,
                dim=-1,
                eps=1e-8,
            )

        reward_weights = (
            task_embedding
            .unsqueeze(1)
            .expand(
                -1,
                num_actions,
                -1,
            )
        )

        q_values = (
            successor_features
            * reward_weights
        ).sum(
            dim=-1
        )

        return q_values


def project_to_l2_ball(
    x: torch.Tensor,
    max_norm: float,
    eps: float = 1e-8,
) -> torch.Tensor:
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
        raise ValueError(
            "max_norm must be positive."
        )

    norms = torch.linalg.vector_norm(
        x,
        ord=2,
        dim=-1,
        keepdim=True,
    )

    scale = torch.clamp(
        max_norm
        / norms.clamp_min(eps),
        max=1.0,
    )

    return x * scale


class FactorisedDQN_QNetwork_BallNorm(nn.Module):
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
            raise ValueError(
                "obs_dim must be positive."
            )

        if num_actions <= 0:
            raise ValueError(
                "num_actions must be positive."
            )

        if goal_dim <= 0:
            raise ValueError(
                "goal_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if rep_dim <= 0:
            raise ValueError(
                "rep_dim must be positive."
            )

        if phi_max_norm <= 0.0:
            raise ValueError(
                "phi_max_norm must be positive."
            )

        if psi_max_norm <= 0.0:
            raise ValueError(
                "psi_max_norm must be positive."
            )

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.action_dim = num_actions
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        self.rep_dim = rep_dim

        self.phi_max_norm = float(
            phi_max_norm
        )

        self.psi_max_norm = float(
            psi_max_norm
        )

        self.sa_encoder = nn.Sequential(
            nn.Linear(
                obs_dim + self.action_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                rep_dim,
            ),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(
                goal_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                rep_dim,
            ),
        )

    # ------------------------------------------------------------
    # Encoder methods
    # ------------------------------------------------------------

    def encode_goal(
        self,
        goal: torch.Tensor,
    ) -> torch.Tensor:
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

        psi_logits = self.goal_encoder(
            goal
        )

        psi = project_to_l2_ball(
            psi_logits,
            max_norm=self.psi_max_norm,
        )

        return psi

    def encode_state_action(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
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
            raise ValueError(
                "obs must have shape [B, obs_dim]. "
                f"Got {tuple(obs.shape)}."
            )

        if act.ndim != 2:
            raise ValueError(
                "act must have shape [B, action_dim]. "
                f"Got {tuple(act.shape)}."
            )

        if obs.shape[0] != act.shape[0]:
            raise ValueError(
                "obs and act must have the same batch size."
            )

        sa = torch.cat(
            [
                obs,
                act,
            ],
            dim=-1,
        )

        phi_logits = self.sa_encoder(
            sa
        )

        phi = project_to_l2_ball(
            phi_logits,
            max_norm=self.phi_max_norm,
        )

        return phi

    # ------------------------------------------------------------
    # Single state-action Q-value
    # ------------------------------------------------------------

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
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

        phi = self.encode_state_action(
            obs,
            act,
        )

        psi = self.encode_goal(
            goal,
        )

        if psi.ndim == 1:
            psi = psi.unsqueeze(0).expand(
                obs.shape[0],
                -1,
            )

        q_values = (
            phi * psi
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return q_values

    # ------------------------------------------------------------
    # All-action Q-values using a goal input
    # ------------------------------------------------------------

    def q_val_for_argmax_action(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
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
            raise ValueError(
                "obs must have shape [B, obs_dim]."
            )

        batch_size = obs.shape[0]
        num_actions = self.num_actions

        action_onehot = F.one_hot(
            torch.arange(
                num_actions,
                device=obs.device,
            ),
            num_classes=self.action_dim,
        ).to(
            dtype=obs.dtype
        )

        action_onehot = (
            action_onehot
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        obs_rep = (
            obs
            .unsqueeze(1)
            .expand(
                -1,
                num_actions,
                -1,
            )
        )

        obs_flat = obs_rep.reshape(
            batch_size * num_actions,
            self.obs_dim,
        )

        act_flat = action_onehot.reshape(
            batch_size * num_actions,
            self.action_dim,
        )

        phi = self.encode_state_action(
            obs_flat,
            act_flat,
        )

        phi = phi.reshape(
            batch_size,
            num_actions,
            self.rep_dim,
        )

        if goal.ndim == 1:
            goal = goal.unsqueeze(0).expand(
                batch_size,
                -1,
            )

        elif goal.shape[0] == 1:
            goal = goal.expand(
                batch_size,
                -1,
            )

        elif goal.shape[0] != batch_size:
            raise ValueError(
                "goal must have shape [goal_dim], "
                "[1, goal_dim], or [B, goal_dim]."
            )

        psi = self.encode_goal(
            goal,
        )

        psi = (
            psi
            .unsqueeze(1)
            .expand(
                batch_size,
                num_actions,
                -1,
            )
        )

        q_values = (
            phi * psi
        ).sum(
            dim=-1,
        )

        return q_values

    # ------------------------------------------------------------
    # Single Q-value using a supplied task embedding
    # ------------------------------------------------------------

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

        phi = self.encode_state_action(
            obs,
            act,
        )

        if task_embedding.ndim == 1:
            task_embedding = (
                task_embedding
                .unsqueeze(0)
                .expand(
                    obs.shape[0],
                    -1,
                )
            )

        elif task_embedding.shape[0] == 1:
            task_embedding = (
                task_embedding.expand(
                    obs.shape[0],
                    -1,
                )
            )

        elif task_embedding.shape[0] != obs.shape[0]:
            raise ValueError(
                "task_embedding must have shape "
                "[rep_dim], [1, rep_dim], or "
                "[B, rep_dim]."
            )

        if normalize_embedding:
            psi = F.normalize(
                task_embedding,
                p=2,
                dim=-1,
                eps=1e-8,
            )

        else:
            psi = project_to_l2_ball(
                task_embedding,
                max_norm=self.psi_max_norm,
            )

        q_values = (
            phi * psi
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return q_values

    # ------------------------------------------------------------
    # All-action Q-values using a supplied embedding
    # ------------------------------------------------------------

    def q_val_for_argmax_action_from_embedding(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        normalize_embedding: bool = False,
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
            raise ValueError(
                "obs must have shape [B, obs_dim]."
            )

        batch_size = obs.shape[0]
        num_actions = self.num_actions

        action_onehot = F.one_hot(
            torch.arange(
                num_actions,
                device=obs.device,
            ),
            num_classes=self.action_dim,
        ).to(
            dtype=obs.dtype
        )

        action_onehot = (
            action_onehot
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        obs_rep = (
            obs
            .unsqueeze(1)
            .expand(
                -1,
                num_actions,
                -1,
            )
        )

        obs_flat = obs_rep.reshape(
            batch_size * num_actions,
            self.obs_dim,
        )

        act_flat = action_onehot.reshape(
            batch_size * num_actions,
            self.action_dim,
        )

        phi = self.encode_state_action(
            obs_flat,
            act_flat,
        )

        phi = phi.reshape(
            batch_size,
            num_actions,
            self.rep_dim,
        )

        if task_embedding.ndim == 1:
            task_embedding = (
                task_embedding
                .unsqueeze(0)
                .expand(
                    batch_size,
                    -1,
                )
            )

        elif task_embedding.shape[0] == 1:
            task_embedding = (
                task_embedding.expand(
                    batch_size,
                    -1,
                )
            )

        elif task_embedding.shape[0] != batch_size:
            raise ValueError(
                "task_embedding must have shape "
                "[rep_dim], [1, rep_dim], or "
                "[B, rep_dim]."
            )

        if normalize_embedding:
            psi = F.normalize(
                task_embedding,
                p=2,
                dim=-1,
                eps=1e-8,
            )

        else:
            psi = project_to_l2_ball(
                task_embedding,
                max_norm=self.psi_max_norm,
            )

        psi = (
            psi
            .unsqueeze(1)
            .expand(
                batch_size,
                num_actions,
                -1,
            )
        )

        q_values = (
            phi * psi
        ).sum(
            dim=-1,
        )

        return q_values

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    @torch.no_grad()
    def representation_norms(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        goal: torch.Tensor,
    ):
        """
        Return phi norm, psi norm, and Q-value diagnostics.
        """

        phi = self.encode_state_action(
            obs,
            act,
        )

        psi = self.encode_goal(
            goal,
        )

        if psi.ndim == 1:
            psi = psi.unsqueeze(0).expand(
                obs.shape[0],
                -1,
            )

        q_values = (
            phi * psi
        ).sum(
            dim=-1,
        )

        return {
            "phi_norm": phi.norm(
                dim=-1
            ),
            "psi_norm": psi.norm(
                dim=-1
            ),
            "q_values": q_values,
            "theoretical_q_bound": (
                self.phi_max_norm
                * self.psi_max_norm
            ),
        }

class FactorisedDQN_QNetwork_Atari(nn.Module):
    """
    Pixel-input factorised goal-conditioned Q-network.

    Q(s, a, g) = phi(s, a)^T psi(g)

    The observation encoder receives image observations and produces a
    flattened convolutional representation. The public methods mirror the
    vector-input implementation so the existing trainer can be reused.

    Accepted observation layouts:
        [B, C, H, W]
        [B, H, W, C]
        [C, H, W]
        [H, W, C]

    Pixel values are converted to floating point and, by default, scaled
    from [0, 255] to [0, 1] when the input is not floating point or has values
    larger than one.
    """

    def __init__(
        self,
        obs_shape,
        num_actions: int,
        goal_dim: int = 2,
        hidden_dim: int = 128,
        rep_dim: int = 64,
        phi_max_norm: float = 2.0,
        psi_max_norm: float = 5.0,
        channels_last: Optional[bool] = None,
        normalize_pixels: bool = True,
    ):
        super().__init__()

        if len(obs_shape) != 3:
            raise ValueError(
                "obs_shape must contain three dimensions: "
                "(C,H,W) or (H,W,C)."
            )
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

        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.num_actions = int(num_actions)
        self.action_dim = self.num_actions
        self.goal_dim = int(goal_dim)
        self.hidden_dim = int(hidden_dim)
        self.rep_dim = int(rep_dim)
        self.phi_max_norm = float(phi_max_norm)
        self.psi_max_norm = float(psi_max_norm)
        self.normalize_pixels = bool(normalize_pixels)

        inferred_channels_last = self._infer_channels_last(self.obs_shape)
        self.channels_last = (
            inferred_channels_last
            if channels_last is None
            else bool(channels_last)
        )

        if self.channels_last:
            self.in_channels = self.obs_shape[-1]
            height, width = self.obs_shape[:2]
        else:
            self.in_channels = self.obs_shape[0]
            height, width = self.obs_shape[1:]

        if self.in_channels <= 0 or height <= 0 or width <= 0:
            raise ValueError("All observation dimensions must be positive.")

        self.obs_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, height, width)
            flat_dim = self.obs_encoder(dummy).shape[-1]

        self.obs_projection = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dim),
            nn.ReLU(),
        )

        self.sa_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )

    def _infer_channels_last(self, shape):
        first, second, third = shape
        if first <= 4 and third > 4:
            return False
        if third <= 4 and first > 4:
            return True
        raise ValueError(
            "Could not infer channel layout from obs_shape. Pass "
            "channels_last=True or False explicitly."
        )

    def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if obs.ndim != 4:
            raise ValueError(
                "obs must have shape [B,C,H,W], [B,H,W,C], "
                "[C,H,W], or [H,W,C]."
            )

        obs = obs.float()
        if self.normalize_pixels and (
            not torch.is_floating_point(obs) or obs.detach().amax() > 1.0
        ):
            obs = obs / 255.0

        if self.channels_last:
            obs = obs.permute(0, 3, 1, 2).contiguous()

        expected = (self.in_channels, *self._spatial_shape())
        if tuple(obs.shape[1:]) != expected:
            raise ValueError(
                f"Expected image shape compatible with {expected}; "
                f"got {tuple(obs.shape[1:])}."
            )
        return obs

    def _spatial_shape(self):
        if self.channels_last:
            return self.obs_shape[:2]
        return self.obs_shape[1:]

    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        psi_logits = self.goal_encoder(goal.float())
        return project_to_l2_ball(
            psi_logits,
            max_norm=self.psi_max_norm,
        )

    def _encode_obs_features(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._prepare_obs(obs)
        return self.obs_projection(self.obs_encoder(obs))

    def encode_state_action(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
        if act.ndim != 2:
            raise ValueError(
                "act must have shape [B, action_dim]. "
                f"Got {tuple(act.shape)}."
            )

        obs_features = self._encode_obs_features(obs)
        if obs_features.shape[0] != act.shape[0]:
            raise ValueError("obs and act must have the same batch size.")
        if act.shape[-1] != self.action_dim:
            raise ValueError(
                f"act must have final dimension {self.action_dim}."
            )

        act_features = self.action_encoder(act.float())
        sa = torch.cat([obs_features, act_features], dim=-1)
        phi_logits = self.sa_encoder(sa)
        return project_to_l2_ball(
            phi_logits,
            max_norm=self.phi_max_norm,
        )

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        phi = self.encode_state_action(obs, act)
        psi = self.encode_goal(goal)

        if psi.ndim == 1:
            psi = psi.unsqueeze(0).expand(obs.shape[0], -1)
        elif psi.shape[0] == 1 and obs.shape[0] != 1:
            psi = psi.expand(obs.shape[0], -1)
        elif psi.shape[0] != obs.shape[0]:
            raise ValueError("goal batch size must match obs batch size.")

        return (phi * psi).sum(dim=-1, keepdim=True)

    def _expand_goal(self, goal: torch.Tensor, batch_size: int):
        if goal.ndim == 1:
            goal = goal.unsqueeze(0).expand(batch_size, -1)
        elif goal.ndim == 2 and goal.shape[0] == 1:
            goal = goal.expand(batch_size, -1)
        elif goal.ndim != 2 or goal.shape[0] != batch_size:
            raise ValueError(
                "goal must have shape [goal_dim], [1, goal_dim], "
                "or [B, goal_dim]."
            )
        return goal

    def _all_action_phi(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            batch_size = 1
        elif obs.ndim == 4:
            batch_size = obs.shape[0]
        else:
            raise ValueError("obs must be an image batch or single image.")

        obs_features = self._encode_obs_features(obs)
        if obs_features.shape[0] != batch_size:
            raise RuntimeError("Unexpected observation batch size.")

        obs_features = obs_features.unsqueeze(1).expand(
            batch_size, self.num_actions, -1
        )

        action_onehot = F.one_hot(
            torch.arange(self.num_actions, device=obs.device),
            num_classes=self.action_dim,
        ).to(dtype=obs_features.dtype)
        action_features = self.action_encoder(action_onehot)
        action_features = action_features.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        sa = torch.cat([obs_features, action_features], dim=-1)
        phi_logits = self.sa_encoder(sa.reshape(-1, 2 * self.hidden_dim))
        phi = project_to_l2_ball(
            phi_logits,
            max_norm=self.phi_max_norm,
        )
        return phi.reshape(batch_size, self.num_actions, self.rep_dim)

    def q_val_for_argmax_action(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        phi = self._all_action_phi(obs)
        batch_size = phi.shape[0]
        goal = self._expand_goal(goal, batch_size)
        psi = self.encode_goal(goal)
        psi = psi.unsqueeze(1).expand(-1, self.num_actions, -1)
        return (phi * psi).sum(dim=-1)

    def forward_with_task_embedding(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        task_embedding: torch.Tensor,
        normalize_embedding: bool = False,
    ) -> torch.Tensor:
        phi = self.encode_state_action(obs, act)
        batch_size = phi.shape[0]

        if task_embedding.ndim == 1:
            task_embedding = task_embedding.unsqueeze(0).expand(
                batch_size, -1
            )
        elif task_embedding.ndim == 2 and task_embedding.shape[0] == 1:
            task_embedding = task_embedding.expand(batch_size, -1)
        elif task_embedding.ndim != 2 or task_embedding.shape[0] != batch_size:
            raise ValueError(
                "task_embedding must have shape [rep_dim], [1, rep_dim], "
                "or [B, rep_dim]."
            )

        if task_embedding.shape[-1] != self.rep_dim:
            raise ValueError(
                f"task_embedding must have final dimension {self.rep_dim}."
            )

        if normalize_embedding:
            psi = F.normalize(task_embedding, p=2, dim=-1, eps=1e-8)
        else:
            psi = project_to_l2_ball(
                task_embedding,
                max_norm=self.psi_max_norm,
            )

        return (phi * psi).sum(dim=-1, keepdim=True)

    def q_val_for_argmax_action_from_embedding(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        normalize_embedding: bool = False,
    ) -> torch.Tensor:
        phi = self._all_action_phi(obs)
        batch_size = phi.shape[0]

        if task_embedding.ndim == 1:
            task_embedding = task_embedding.unsqueeze(0).expand(
                batch_size, -1
            )
        elif task_embedding.ndim == 2 and task_embedding.shape[0] == 1:
            task_embedding = task_embedding.expand(batch_size, -1)
        elif task_embedding.ndim != 2 or task_embedding.shape[0] != batch_size:
            raise ValueError(
                "task_embedding must have shape [rep_dim], [1, rep_dim], "
                "or [B, rep_dim]."
            )

        if task_embedding.shape[-1] != self.rep_dim:
            raise ValueError(
                f"task_embedding must have final dimension {self.rep_dim}."
            )

        if normalize_embedding:
            psi = F.normalize(task_embedding, p=2, dim=-1, eps=1e-8)
        else:
            psi = project_to_l2_ball(
                task_embedding,
                max_norm=self.psi_max_norm,
            )

        psi = psi.unsqueeze(1).expand(-1, self.num_actions, -1)
        return (phi * psi).sum(dim=-1)

    @torch.no_grad()
    def representation_norms(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        goal: torch.Tensor,
    ):
        phi = self.encode_state_action(obs, act)
        psi = self.encode_goal(goal)
        batch_size = phi.shape[0]

        if psi.ndim == 1:
            psi = psi.unsqueeze(0).expand(batch_size, -1)
        elif psi.shape[0] == 1 and batch_size != 1:
            psi = psi.expand(batch_size, -1)
        elif psi.shape[0] != batch_size:
            raise ValueError("goal batch size must match obs batch size.")

        q_values = (phi * psi).sum(dim=-1)
        return {
            "phi_norm": phi.norm(dim=-1),
            "psi_norm": psi.norm(dim=-1),
            "q_values": q_values,
            "theoretical_q_bound": (
                self.phi_max_norm * self.psi_max_norm
            ),
        }