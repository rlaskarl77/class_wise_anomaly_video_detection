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

def get_pairwise_distance2(score, score_ano, const=1e-0, L1=True):
    if L1:
        pair_wise_dist = torch.norm(score.unsqueeze(1) - score.unsqueeze(2), p=1, dim=-1)
    else:
        # pair_wise_dist = torch.norm(score.unsqueeze(1) - score.unsqueeze(2), p=2, dim=-1)
        pair_wise_dist = torch.matmul(score, score)
    pair_wise_dist = torch.cat([pair_wise_dist, score_ano], dim=-1)
    return torch.exp(-pair_wise_dist / const)

def get_absorbtion_time2(adj, score, class_wise=False):
    # score_sum_weights = score.sum(-1)
    sum_weights = adj.sum(-1)
    # score_sum_weights = score_sum_weights.view(-1, sum_weights.size(-1))

    # sum_weights += torch.clamp(score_sum_weights, min=0.01, max=0.99)
    adj = adj / sum_weights.unsqueeze(2)
    R = adj[:, :, adj.size(1):]
    adj = adj[:, :, :adj.size(1)]
    assert (torch.sum(adj < 0) + torch.sum(adj > 1)) == 0
    eye = torch.eye(adj.size(-1), dtype=adj.dtype).unsqueeze(0).to(torch.device('cuda'))
    try:
        f = torch.inverse((eye - adj))
    except:
        f = torch.pinverse(eye - adj)
    abs_time = f.sum(dim=-1)

    # R = score / sum_weights.unsqueeze(2)
    abs_prob = torch.matmul(f, R)
    # import pdb;pdb.set_trace()
    
    if class_wise:
        return abs_time, abs_prob, f
    return abs_time, abs_prob


def get_amc_score2(det_score, fea, mean):
    adj_amc = get_pairwise_distance2(fea, det_score, const=1e-1, L1=False)
    absorb_time, abs_prob = get_absorbtion_time2(adj_amc, det_score)
    if mean is None:
        return -absorb_time
    elif mean == 0:
        mean = absorb_time.detach().mean()
    else:
        alpha = 0.9
        mean = alpha * mean + (1 - alpha) * absorb_time.detach().mean()
    amc_score = -(absorb_time - mean)
    return amc_score, mean, abs_prob
# class MeanValEstimator():
#     def __init__(self, num_classes)