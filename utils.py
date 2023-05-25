import torch

def get_pairwise_distance(score, const=1e-0, L1=True):
    if L1:
        pair_wise_dist = torch.norm(score.unsqueeze(1) - score.unsqueeze(2), p=1, dim=-1)
    else:
        pair_wise_dist = torch.norm(score.unsqueeze(1) - score.unsqueeze(2), p=2, dim=-1)
    return torch.exp(-pair_wise_dist / const)

def get_absorbtion_time(adj, score):
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
    return abs_time
