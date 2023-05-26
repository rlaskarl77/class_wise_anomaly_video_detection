import torch


def get_pairwise_distance(score, const=1e-0, L1=True):
    if L1:
        pair_wise_dist = torch.norm(score.unsqueeze(1) - score.unsqueeze(2), p=1, dim=-1)
    else:
        pair_wise_dist = torch.norm(score.unsqueeze(1) - score.unsqueeze(2), p=2, dim=-1)
    return torch.exp(-pair_wise_dist / const)


def get_absorbtion_time(adj, score, class_wise=False):
    sum_weights = adj.sum(-1)
    score = score.view(-1, sum_weights.size(-1))

    sum_weights += torch.clamp(score, min=0.01, max=0.99)
    adj = adj / sum_weights.unsqueeze(2)
    assert (torch.sum(adj < 0) + torch.sum(adj > 1)) == 0
    eye = torch.eye(adj.size(-1), dtype=adj.dtype).unsqueeze(0).to(torch.device('cuda'))
    try:
        f = torch.inverse((eye - adj))
    except:
        f = torch.pinverse(eye - adj)
    abs_time = f.sum(dim=-1)
    
    if class_wise:
        return abs_time, f
    return abs_time


def get_amc_score(det_score, fea, mean):
    adj_amc = get_pairwise_distance(fea, const=1e-1, L1=False)
    absorb_time = get_absorbtion_time(adj_amc, det_score)
    if mean is None:
        return -absorb_time
    elif mean == 0:
        mean = absorb_time.detach().mean()
    else:
        alpha = 0.9
        mean = alpha * mean + (1 - alpha) * absorb_time.detach().mean()
    amc_score = -(absorb_time - mean)
    return amc_score, mean