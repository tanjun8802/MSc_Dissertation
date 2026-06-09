import torch.nn as nn
import torch
import numpy as np  
import torch.nn.functional as F

MIN_STD = 1e-6
LOG_STD_MIN = float(np.log(MIN_STD))
LOG_STD_MAX =  2

class GoalConditionedActor(nn.Module): # goal-conditioned policy pi(a | s, g) — takes (s, g) and outputs a Gaussian action distribution

    def __init__(self, obs_dim, action_dim, hidden_dim=256, inner_layers=2): # note here assumes s and g are the same dim
        super().__init__()
        self.fc1 = nn.Linear(obs_dim * 2, hidden_dim)  # concatenate state s and goal g as input
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(inner_layers)])
        self.fc_mean    = nn.Linear(hidden_dim, action_dim)   # output the mean of the action distribution
        self.log_std_fc = nn.Linear(hidden_dim, action_dim)   # state-dependent log-std head

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)  # (B, obs_dim * 2)
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        mean    = self.fc_mean(x)  # (B, action_dim)
        log_std = self.log_std_fc(x).clamp(LOG_STD_MIN, LOG_STD_MAX)  # (B, action_dim)
        return mean, log_std

    def sample_action(self, obs, goal):
        mean, log_std = self.forward(obs, goal)
        std  = log_std.exp().clamp_min(MIN_STD)
        dist = torch.distributions.Normal(mean, std)

        # Reparameterised sample
        x_t    = dist.rsample()

        # Squash through tanh so actions always lie in (-1, 1)
        action = torch.tanh(x_t)

        # Correct log-prob for the tanh change-of-variables:
        # log π(a|s,g) = log N(x_t; μ, σ) - Σ log(1 - tanh²(x_t))
        log_prob = (dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)  # (B, 1)
        entropy  = dist.entropy().sum(dim=-1, keepdim=True)  # (B, 1), H(pi(.|s,g))
        return action, log_prob, entropy

class DiscreteGoalConditionedActor(nn.Module): # variant of GoalConditionedActor for discrete action spaces, outputs a categorical distribution over actions instead of Gaussian

    def __init__(self, obs_dim, num_actions = 4, hidden_dim=256, inner_layers=2):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim * 2, hidden_dim)  # concatenate state s and goal g as input
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(inner_layers)])
        self.fc_logits = nn.Linear(hidden_dim, num_actions)   # output logits for categorical distribution

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)  # (B, obs_dim * 2)
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        logits = self.fc_logits(x)  # (B, num_actions)
        return logits

    def sample_action(self, obs, goal, valid_mask=None):
        logits = self.forward(obs, goal)   # [B, 4]

        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e9)

        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()            # [B]
        log_prob = dist.log_prob(actions)  # [B]
        return actions, log_prob, logits

    def greedy_action(self, obs, goal, valid_mask=None):
        logits = self.forward(obs, goal)

        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -1e9)

        return torch.argmax(logits, dim=-1)