import torch
import torch.nn.functional as F


def contrastive_loss(all_logits, temperature=1, reg_coef=0.01):

    labels = torch.arange(all_logits.size(0), device=all_logits.device)
    scaled_logits = all_logits / temperature
    infonce_loss = F.cross_entropy(scaled_logits, labels) # the first term of the loss

    logsumexp = torch.logsumexp(scaled_logits, dim=-1)
    reg_loss  = reg_coef * logsumexp.pow(2).mean()   # penalise large LogSumExp values (matches -0.01 * (...) in the maximisation objective)

    return infonce_loss + reg_loss  # minimise this to maximise the paper's contrastive objective


def repulsion_loss_to_memory(psi_new, memory, margin=1.0):

    if len(memory) == 0:
        return psi_new.new_tensor(0.0)

    mem_stack = torch.stack(
    [torch.as_tensor(m, dtype=psi_new.dtype, device=psi_new.device) for m in memory],
    dim=0
    )

    diff = psi_new.unsqueeze(1) - mem_stack.unsqueeze(0)  # [B, K, rep_dim]
    dists = diff.norm(dim=2)                              # [B, K]

    # Penalise if distance < margin
    loss_mat = torch.clamp(margin - dists, min=0.0)       # [B, K]
    loss = loss_mat.mean()                                # scalar
    return loss