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