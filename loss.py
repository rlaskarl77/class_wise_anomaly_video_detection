import torch
import torch.nn.functional as F

def MIL(y_pred, batch_size, is_transformer=0):
    loss = torch.tensor(0.).cuda()
    loss_intra = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    if is_transformer==0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):
        anomaly_index = torch.randperm(30).cuda()
        normal_index = torch.randperm(30).cuda()

        y_anomaly = y_pred[i, :32][anomaly_index]
        y_normal  = y_pred[i, 32:][normal_index]

        y_anomaly_max = torch.max(y_anomaly) # anomaly
        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal) # normal
        y_normal_min = torch.min(y_normal)

        loss += F.relu(1.-y_anomaly_max+y_normal_max)

        sparsity += torch.sum(y_anomaly)*0.00008
        smooth += torch.sum((y_pred[i,:31] - y_pred[i,1:32])**2)*0.00008
    loss = (loss+sparsity+smooth)/batch_size

    return loss


def weaksup_intra_video_loss(amc_score, batch_size, k = 1, margin=0.5):
    # assert len(pred.size()) == 2
    pred = amc_score.view(-1, 32)
    abnorm_pred = pred[0:batch_size, :]
    abnorm_min = abnorm_pred.topk(k=k, dim=-1, largest=False)[0][:, -1]
    abnorm_max = abnorm_pred.topk(k=k, dim=-1)[0][:, -1]

    minmax_diff = (-abnorm_max + abnorm_min).view(-1)
    hinge_loss = torch.max(torch.zeros_like(minmax_diff), margin + minmax_diff).sum()
    return hinge_loss

def weaksup_intra_video_loss2(amc_score, abs_prob, anomaly_ids, batch_size, k = 1, margin=0.5):
    pred = amc_score.view(-1, 32)
    abnorm_pred = pred[0:batch_size, :]
    abnorm_abs_prob, norm_abs_prob = abs_prob.chunk(2)

    abnorm_min, abnorm_min_idx = abnorm_pred.topk(k=k, dim=-1, largest=False)
    abnorm_min = abnorm_min[:, -1]

    abnorm_max, abnorm_max_idx = abnorm_pred.topk(k=k, dim=-1)
    abnorm_max = abnorm_max[:, -1]

    minmax_diff = (-abnorm_max + abnorm_min).view(-1)
    hinge_loss = torch.max(torch.zeros_like(minmax_diff), margin + minmax_diff).sum()

    # abnorm prob cross-entropy
    num_classes = abs_prob.size(2)
    abnorm_min_idx = abnorm_min_idx.unsqueeze(2).repeat(1,1,num_classes)
    abnorm_max_idx = abnorm_max_idx.unsqueeze(2).repeat(1,1,num_classes)
    abnorm_abs_prob_min = abnorm_abs_prob.gather(1, abnorm_min_idx).squeeze()
    abnorm_abs_prob_max = abnorm_abs_prob.gather(1, abnorm_max_idx).squeeze()

    norm_abs_prob = norm_abs_prob.view(-1, num_classes)

    ace_loss = 0.1*F.nll_loss(abnorm_abs_prob_max.log(), anomaly_ids) - 0.1* entropy(abnorm_abs_prob_min)#- 0.05*entropy(norm_abs_prob)
    # entropy_diff = - entropy(abnorm_abs_prob_min, mean=False) + entropy(abnorm_abs_prob_max, mean=False)
    # ace_loss = 0.1*torch.max(torch.zeros_like(minmax_diff), margin + entropy_diff).sum() + 0.1*F.nll_loss(abnorm_abs_prob_max.log(), anomaly_ids)

    return hinge_loss + ace_loss

def entropy(p, mean=True):
    ent = - p * (p+1e-5).log()
    if not mean:
        return ent.sum(dim=1)
    return ent.sum(dim=1).mean()