import torch
import torch.nn.functional as F


def contrastive_loss(all_logits, temperature=1, reg_coef=0.01):

    labels = torch.arange(all_logits.size(0), device=all_logits.device)
    scaled_logits = all_logits / temperature
    infonce_loss = F.cross_entropy(scaled_logits, labels) # the first term of the loss

    logsumexp = torch.logsumexp(scaled_logits, dim=-1)
    reg_loss  = reg_coef * logsumexp.pow(2).mean()   # penalise large LogSumExp values (matches -0.01 * (...) in the maximisation objective)

    return infonce_loss + reg_loss  # minimise this to maximise the paper's contrastive objective