from pathlib import Path
import sys
import os
import numpy as np
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn import metrics, preprocessing

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from general import increment_path, select_device, LOGGER, smart_inference_mode
from learner import Learner
from loss import *
from dataset import *
from utils import *

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


@smart_inference_mode()
def run(opt, model, normal_test_loader, anomaly_test_loader):
    
    model.eval()
    auc = 0
    
    for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
        
        inputs, gts, frames = data
        inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
        
        if opt.mode=='ace':
            score, fea, _ = model(inputs)
        else:
            score,fea = model(inputs)
        
        if opt.mode=='amc' or opt.mode=='ace':
            score, _, _ = get_amc_score(score.unsqueeze(0), fea.unsqueeze(0), mean=None)
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
        
        if opt.mode=='ace':
            score2, fea2, _ = model(inputs2)
        else:
            score2, fea2 = model(inputs2)
        
        if opt.mode=='amc' or opt.mode=='ace':
            score2, _, _ = get_amc_score(score2.unsqueeze(0), fea2.unsqueeze(0), mean=None)
            score2 = score2.squeeze().unsqueeze(1)
            # score2 = m(score2)
        score2 = score2.cpu().detach().numpy()
        score_list2 = np.zeros(frames2[0])
        step2 = np.round(np.linspace(0, frames2[0]//16, 33))
        
        for kk in range(32):
            score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
        gt_list2 = np.zeros(frames2[0])
        score_list3 = np.concatenate((score_list, score_list2), axis=0)
        
        if opt.mode=='amc' or opt.mode=='ace':
            score_list3 = (score_list3 - np.min(score_list3)) / (np.max(score_list3) - np.min(score_list3))

        gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

        fpr, tpr, _ = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
        auc += metrics.auc(fpr, tpr)
    return auc / 140

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # model configurations
    parser.add_argument('--mode', type=str, help='base, amc or ace', 
                        choices=['base', 'amc', 'ace'], default='base')
    parser.add_argument('--ckpt', type=str, help='model checkpoint', default=None)
    parser.add_argument('--alpha', type=float, help='weighted sum of amc loss', default=0.1)
    parser.add_argument('--beta', type=float, help='weighted sum of classification loss', default=0.1)
    parser.add_argument('--drop-p', type=float, help='dropout possibility', default=0.0)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--classes', type=int, default=13, help='number of classes for classification')
    parser.add_argument('--classification', type=str, help='classification loss for normal clips', 
                        choices=['entropy', 'information', 'none'], default='information')
        
    # optimizer
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--epochs', type=int, help='# of Epoch', default=150)
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight-decay', type=float, help='weight decay rate', default=0.00001)
    
    
    # device setting
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='numnber or workers for dataloader')
    
    # checkpoint attributes
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    opt = parse_opt()
        
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name))
        
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with DDP training'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        
    model = Learner(input_dim=2048, drop_p=0.0, mode=opt.mode).to(device)
    assert opt.ckpt is not None, 'need checkpoint to validate'
    ckpt = torch.load(opt.ckpt)
    model.load_state_dict(ckpt['model_state_dict'])
    
    normal_test_dataset = Normal_Loader(is_train=0)
    anomaly_test_dataset = Anomaly_Loader(is_train=0)

    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)
    # validate    
    auc = run(opt, model, normal_test_loader, anomaly_test_loader)
    
    LOGGER.info(f'validate done for {opt.ckpt}\n auc = {auc}')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)