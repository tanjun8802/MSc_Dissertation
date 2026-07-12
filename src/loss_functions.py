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

def sigreg_loss(representation_network, sketch_dim=32, eps=1e-6): 
    B, D = representation_network.shape
    z = representation_network - representation_network.mean(dim=0, keepdim=True)

    if D > sketch_dim:
        S = torch.randn(D, sketch_dim, device=z.device, dtype=z.dtype) / (D ** 0.5)
        z = z @ S
        D = sketch_dim

    cov = (z.T @ z) / (B - 1 + eps)
    I = torch.eye(D, device=z.device, dtype=z.dtype)
    return ((cov - I) ** 2).sum() / D


def orthogonal_loss(q_network, goal_t_single, embedding_memory, device, eps=1e-8):
    # current goal embedding: [1, D] -> [D]
    cur = q_network.encode_goal(goal_t_single).squeeze(0)

    embs = [cur]
    for e in embedding_memory:
        embs.append(torch.tensor(e, dtype=torch.float32, device=device))

    if len(embs) <= 1:
        return torch.tensor(0.0, device=device)

    M = torch.stack(embs, dim=0)  # [N, D]
    M = F.normalize(M, p=2, dim=-1, eps=eps)

    S = torch.matmul(M, M.T)      # [N, N]
    I = torch.eye(S.shape[0], device=device)
    return torch.norm(S - I, p='fro')  # scalar

def weight_regulariser_loss(model, reference_params, prefix_filter=None):
    if reference_params is None:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    loss = None

    for name, p in model.named_parameters():
        if name not in reference_params:
            continue

        if prefix_filter is not None:
            if isinstance(prefix_filter, str):
                if not name.startswith(prefix_filter):
                    continue
            else:
                if not any(name.startswith(pref) for pref in prefix_filter):
                    continue

        term = (p - reference_params[name]).pow(2).sum()
        loss = term if loss is None else loss + term

    if loss is None:
        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)

    return loss

def ewc_regulariser_loss(model, reference_params, fisher_diag, prefix_filter=None):
    if reference_params is None or fisher_diag is None:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    loss = None

    for name, p in model.named_parameters():
        if name not in reference_params or name not in fisher_diag:
            continue

        if prefix_filter is not None:
            if isinstance(prefix_filter, str):
                if not name.startswith(prefix_filter):
                    continue
            else:
                if not any(name.startswith(pref) for pref in prefix_filter):
                    continue

        term = fisher_diag[name] * (p - reference_params[name]).pow(2)
        term = term.sum()
        loss = term if loss is None else loss + term

    if loss is None:
        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)

    return loss


