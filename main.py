from torch.utils.data import DataLoader
from learner import Learner
from loss import *
from dataset import *
from utils import *
import os
from sklearn import metrics, preprocessing
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='GCGenome')
parser.add_argument('--mode', type=str, help='amc or noamc', default='amc')
parser.add_argument('--ckpt', type=str, help='model checkpoint', default=None)
parser.add_argument('--exp', type=int, help='Number of experiments', default=0)
parser.add_argument('--epoch', type=int, help='# of Epoch', default=150)
parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
parser.add_argument('--alpha', type=float, help='weighted sum of amc loss', default=0.1)
args = parser.parse_args()


normal_train_dataset = Normal_Loader(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)

anomaly_train_dataset = Anomaly_Loader(is_train=1)
anomaly_test_dataset = Anomaly_Loader(is_train=0)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Learner(input_dim=2048, drop_p=0.0, mode=args.mode).to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr= args.lr, weight_decay=0.0010000000474974513)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75, 100, 125], gamma=0.5)
criterion = MIL

if args.ckpt is not None:
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    mean = 0.0

    for batch_idx, ((normal_inputs, normal_ids), (anomaly_inputs, anomaly_ids)) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        input_ids = torch.cat([normal_ids, anomaly_ids], dim=0)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs, fea = model(inputs)
        loss = criterion(outputs, batch_size)

        if args.mode == 'amc':
            outputs = outputs.view(batch_size, -1, outputs.size(-1)).to(device)
            output1 = outputs[:, :32, :]
            output2 = outputs[:, 32:, :]
            outputs = torch.cat([output1, output2], dim=0)

            fea = fea.view(batch_size, -1, fea.size(-1)).to(device)
            fea1 = fea[:, :32, :]
            fea2 = fea[:, 32:, :]
            fea = torch.cat([fea1, fea2], dim=0)

            amc_score, mean = get_amc_score(outputs, fea, mean)
            amc_loss = weaksup_intra_video_loss(amc_score, batch_size, margin=0.5) * args.alpha
            loss = loss + amc_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()
    return train_loss / len(normal_train_loader)

def test_abnormal():
    model.eval()
    auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
            score, fea = model(inputs)
            if args.mode == 'amc':
                score = get_amc_score(score.unsqueeze(0), fea.unsqueeze(0), mean=None)
                score = score.squeeze().unsqueeze(1)
                # score = m(score)
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))

            for j in range(32):
                score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1

            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
            score2, fea2 = model(inputs2)
            if args.mode == 'amc':
                score2 = get_amc_score(score2.unsqueeze(0), fea2.unsqueeze(0), mean=None)
                score2 = score2.squeeze().unsqueeze(1)
                # score2 = m(score2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames2[0]//16, 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            if args.mode == 'amc':
                score_list3 = (score_list3 - np.min(score_list3)) / (np.max(score_list3) - np.min(score_list3))

            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            fpr, tpr, _ = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)
    return auc / 140

def save_model(auc):
    output_path = os.path.join('./ckpt/', str(args.exp))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    PATH = os.path.join(output_path, 'best_' + str(epoch) + '_' + str(round(auc, 3)) + '.pt')
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, PATH)

best_auc = 0
for epoch in range(0, args.epoch):
    loss = train()
    auc = test_abnormal()
    if auc > best_auc:
        best_auc = auc
        print("New Record!!")
        if epoch > 30:
            save_model(best_auc)
    print('Epoch: {}/{}  ||  loss = {},  auc = {}'.format(epoch, args.epoch, loss, auc))
print("Best AUC is {}".format(best_auc))