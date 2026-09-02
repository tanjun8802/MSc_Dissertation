import torch
import torch.nn.functional as F
import numpy as np


def contrastive_loss(all_logits, temperature=1.0, reg_coef=0.01):

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

def sigreg_loss(representation_network, sketch_dim=64, eps=1e-6):
    B, D = representation_network.shape
    z = representation_network - representation_network.mean(dim=0, keepdim=True)

    if D > sketch_dim:
        S = torch.randn(D, sketch_dim, device=z.device, dtype=z.dtype) / (D ** 0.5)
        z = z @ S
        D = sketch_dim

    cov = (z.T @ z) / (B - 1 + eps)

    I = torch.eye(D, device=z.device, dtype=z.dtype)
    loss = ((cov - I) ** 2).sum() / D
    return loss


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



def goal_similarity_from_coords(current_goal, memory_goals, mode="euclidean", temperature=2.0):
    current_goal = np.array(current_goal, dtype=np.float32)

    if len(memory_goals) == 0:
        return np.array([], dtype=np.float32)

    memory_goals = np.array(memory_goals, dtype=np.float32)

    if mode == "euclidean":
        dists = np.linalg.norm(memory_goals - current_goal[None, :], axis=1)
    else:
        raise ValueError(f"Unknown similarity mode: {mode}")

    sims = np.exp(-dists / max(temperature, 1e-8))
    if sims.max() > 0:
        sims = sims / sims.max()
    return sims.astype(np.float32)


def goal_memory_contrastive_loss(
    current_embedding,
    embedding_memory,
    current_goal,
    memory_goals,
    similarity_mode="euclidean",
    temperature=2.0,
    pos_threshold=0.6,
    neg_threshold=0.3,
    margin=1.0,
):
    if len(embedding_memory) == 0:
        return torch.tensor(0.0, device=current_embedding.device)

    if current_embedding.dim() == 1:
        current_embedding = current_embedding.unsqueeze(0)

    mem_tensors = []
    for emb in embedding_memory:
        if isinstance(emb, np.ndarray):
            emb = torch.tensor(emb, dtype=torch.float32, device=current_embedding.device)
        else:
            emb = emb.to(current_embedding.device, dtype=torch.float32)

        if emb.dim() == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        mem_tensors.append(emb)

    mem = torch.stack(mem_tensors, dim=0)   # [M, D]
    cur = current_embedding.squeeze(0)      # [D]

    sims = goal_similarity_from_coords(
        current_goal=current_goal,
        memory_goals=memory_goals,
        mode=similarity_mode,
        temperature=temperature,
    )
    sims = torch.tensor(sims, dtype=torch.float32, device=current_embedding.device)

    dists = torch.norm(mem - cur.unsqueeze(0), dim=1)

    pos_mask = sims >= pos_threshold
    neg_mask = sims <= neg_threshold

    loss = torch.tensor(0.0, device=current_embedding.device)

    if pos_mask.any():
        pos_weights = sims[pos_mask]
        pos_dists = dists[pos_mask]
        loss = loss + (pos_weights * pos_dists.pow(2)).mean()

    if neg_mask.any():
        neg_weights = 1.0 - sims[neg_mask]
        neg_dists = dists[neg_mask]
        loss = loss + (neg_weights * F.relu(margin - neg_dists).pow(2)).mean()

    return loss

def goal_prototype_anchor_loss(
    q_network,
    goal_batch,
    prototype,
):
    """
    Forces the current goal encoder output to match
    the retrieved prototype embedding.
    """

    if prototype is None:
        return torch.zeros(
            (),
            device=goal_batch.device,
        )

    current_embedding = (
        q_network.encode_goal(
            goal_batch
        )
    )

    prototype_tensor = torch.as_tensor(
        prototype,
        dtype=current_embedding.dtype,
        device=current_embedding.device,
    )

    prototype_tensor = prototype_tensor.view(
        1,
        -1,
    ).expand_as(current_embedding)

    return F.mse_loss(
        current_embedding,
        prototype_tensor,
    )

def online_goal_separation_loss(
    q_network,
    task_goals,
    device,
    target_cosine=0.85,
):
    task_ids = sorted(
        task_goals.keys()
    )

    if len(task_ids) < 2:
        return torch.zeros(
            (),
            device=device,
        )

    goal_batch = torch.as_tensor(
        np.asarray(
            [
                task_goals[task_id]
                for task_id in task_ids
            ],
            dtype=np.float32,
        ),
        dtype=torch.float32,
        device=device,
    )

    psi = q_network.encode_goal(
        goal_batch
    )

    psi = F.normalize(
        psi,
        p=2,
        dim=-1,
    )

    cosine_matrix = (
        psi @ psi.T
    )

    pair_mask = torch.triu(
        torch.ones_like(
            cosine_matrix,
        ),
        diagonal=1,
    ).bool()

    pairwise_cosines = (
        cosine_matrix[pair_mask]
    )

    return F.relu(
        pairwise_cosines
        - target_cosine
    ).pow(2).mean(), cosine_matrix


def norm_penalty_loss_l2(
    raw_embedding,
    target_norm=1.0,
):
    raw_norms = torch.linalg.vector_norm(
        raw_embedding,
        ord=2,
        dim=-1,
    )

    return F.relu(
        raw_norms
        - target_norm
    ).pow(2).mean()


def norm_penalty_loss_l1(
    raw_embedding,
    target_norm=1.0,
):
    raw_norms = torch.linalg.vector_norm(
        raw_embedding,
        ord=2,
        dim=-1,
    )

    excess = F.relu(
        raw_norms
        - target_norm
    )

    return F.smooth_l1_loss(
        excess,
        torch.zeros_like(excess),
    )