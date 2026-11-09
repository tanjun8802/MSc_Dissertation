# Repository guidance

## Apple MPS compatibility

Apple Silicon MPS is a required backend for maintained PyTorch experiment code. Changes to models, losses, replay, trainers, evaluation, diagnostics, checkpoint loading, and notebook clients must preserve MPS execution alongside CPU and existing CUDA support.

- Use explicit, backend-supported tensor dtypes and the configured device. Keep models, inputs, replay batches, and newly created tensors on compatible devices. Convert NumPy floating-point inputs to float32 before transferring them to MPS; NumPy often defaults to float64.
- MPS does not support float64 tensors. When detached diagnostics or task selection need float64, transfer to CPU first, then cast: `gradient.detach().cpu().to(dtype=torch.float64)`. A combined `.to(device="cpu", dtype=torch.float64)` can fail while transferring from MPS. Keep differentiable training computations on the configured device; detach only quantities that intentionally do not participate in backpropagation.
- Check operator support for new tensor operations. Prefer an MPS-supported implementation. Any necessary CPU-only calculation must have an explicit device boundary, documented purpose, and accounted-for transfer/compute cost. Preserve the algorithm and experiment settings when fixing backend compatibility.

## Backend validation

For changes affecting tensor execution, add or extend regression coverage at the affected seam and run it on CPU and native MPS when available. Exercise the actual forward/backward or diagnostic path; successful imports and CPU-only tests do not establish MPS compatibility.

- Check `torch.backends.mps.is_available()` and report which backends actually ran. Mark unavailable MPS tests as skipped with a reason. If sandbox restrictions hide the GPU, use the permitted native execution route; if still unavailable, report MPS validation as unverified.
- For bounded DQN replay or weighting changes, run `PYTHONPATH=src python -m pytest tests/test_k_replay.py`. Its CPU/MPS cases cover interference weights and real DQN updates with historical replay and gradient diagnostics. A first-task-only smoke run does not exercise historical weighting.
- Exercise evaluation, notebook diagnostics, or checkpoint restoration when the changed path includes them. Check finite outputs and relevant behaviour using appropriate numerical tolerances; CPU and MPS need not produce bitwise-identical training trajectories.

For bounded DQN design and resource constraints, read [the implementation plan](docs/dqn-k-replay-plan.md) when changing replay selection, weighting, or the study protocol.
