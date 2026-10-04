import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CovarianceRegularizer(nn.Module):
    """The working trainers' covariance-to-identity SIGReg objective.

    A random sketch, when needed, belongs to this training call rather than a
    process-global function attribute. Its projection is a registered buffer.
    """

    def __init__(self, sketch_dim=64, eps=1e-6):
        super().__init__()
        if sketch_dim < 1:
            raise ValueError("sketch_dim must be positive.")
        self.sketch_dim, self.eps = sketch_dim, eps
        self.register_buffer("projection", torch.empty(0))

    def forward(self, representation):
        batch_size, dimension = representation.shape
        if batch_size < 2:
            return representation.sum() * 0.0
        z = representation - representation.mean(dim=0, keepdim=True)
        if dimension > self.sketch_dim:
            shape = (dimension, self.sketch_dim)
            if tuple(self.projection.shape) != shape:
                self.projection = (
                    torch.randn(*shape, device=z.device, dtype=z.dtype) / dimension**0.5
                )
            z = z @ self.projection.to(device=z.device, dtype=z.dtype)
            dimension = self.sketch_dim
        covariance = z.T @ z / (batch_size - 1 + self.eps)
        identity = torch.eye(dimension, device=z.device, dtype=z.dtype)
        return (covariance - identity).pow(2).sum() / dimension


def online_goal_separation_loss(q_network, task_goals, device, target_cosine=0.85):
    task_ids = sorted(task_goals.keys())
    if len(task_ids) < 2:
        return torch.zeros((), device=device), torch.eye(len(task_ids), device=device)
    goal_batch = torch.as_tensor(
        np.asarray([task_goals[task_id] for task_id in task_ids], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    psi = q_network.encode_goal(goal_batch)
    psi = F.normalize(psi, p=2, dim=-1)
    cosine_matrix = psi @ psi.T
    pair_mask = torch.triu(torch.ones_like(cosine_matrix), diagonal=1).bool()
    pairwise_cosines = cosine_matrix[pair_mask]
    return (F.relu(pairwise_cosines - target_cosine).pow(2).mean(), cosine_matrix)


def norm_penalty_loss_l1(raw_embedding, target_norm=1.0):
    raw_norms = torch.linalg.vector_norm(raw_embedding, ord=2, dim=-1)
    excess = F.relu(raw_norms - target_norm)
    return F.smooth_l1_loss(excess, torch.zeros_like(excess))
