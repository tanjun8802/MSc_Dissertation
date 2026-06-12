# Now here is the NN to train the sampled batch data from the replay buffer, the goal is to learn a dynamics model that predicts s_next from (s,a) pairs.

import torch
import torch.nn as nn
import torch.nn.functional as F

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