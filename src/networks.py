# Now here is the NN to train the sampled batch data from the replay buffer, the goal is to learn a dynamics model that predicts s_next from (s,a) pairs.

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        psi = self.goal_encoder(goal)
        psi = torch.tanh(psi)
        psi = F.normalize(psi, p=2, dim=-1, eps=1e-8)
        return psi

    def encode_state_action(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([obs, act], dim=-1)
        phi = self.sa_encoder(sa)
        phi = torch.tanh(phi)
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
        q_vals = (phi_sa * psi_z).sum(dim=-1, keepdim=True)  # [B, 1]
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

        q_vals = (phi_sa * psi_rep).sum(dim=-1)                 # [B, A]
        return q_vals