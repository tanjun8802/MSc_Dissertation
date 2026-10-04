import torch


class RunningMeanStd:
    """
    Running per-feature mean and variance using numerically stable
    parallel/Welford-style updates.
    """

    def __init__(self, shape, device, epsilon: float = 0.0001, dtype: torch.dtype = torch.float32):
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(epsilon, dtype=dtype, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """
        x has shape [feature_dim] or [batch_size, feature_dim].
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"RunningMeanStd.update expects [D] or [B, D], got {tuple(x.shape)}.")
        x = x.detach().to(device=self.mean.device, dtype=self.mean.dtype)
        batch_count = torch.as_tensor(
            float(x.shape[0]), dtype=self.mean.dtype, device=self.mean.device
        )
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        mean_a = self.var * self.count
        mean_b = batch_var * batch_count
        correction = delta.pow(2) * self.count * batch_count / total_count
        new_var = (mean_a + mean_b + correction) / total_count
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor, clip: float = 10.0, eps: float = 1e-08) -> torch.Tensor:
        """
        Normalize with the current running statistics.

        This function does not update statistics. Gradients still flow
        through x when x requires gradients.
        """
        x = x.to(device=self.mean.device, dtype=self.mean.dtype)
        z = (x - self.mean) / torch.sqrt(self.var + eps)
        return z.clamp(-clip, clip)
