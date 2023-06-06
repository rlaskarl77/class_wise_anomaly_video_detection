from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn import metrics, preprocessing
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate
from general import de_parallel, increment_path, select_device, LOGGER, smart_DDP, smart_resume
from learner import Learner
from loss import MIL, weaksup_intra_video_loss
from dataset import Anomaly_Loader, Normal_Loader
from utils import get_amc_score, unstack_outputs

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(opt, device):
    save_dir, epochs, batch_size, workers = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.workers
        
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    normal_train_dataset = Normal_Loader(is_train=1)
    normal_test_dataset = Normal_Loader(is_train=0)

    anomaly_train_dataset = Anomaly_Loader(is_train=1)
    anomaly_test_dataset = Anomaly_Loader(is_train=0)

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True, num_workers=workers)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True, num_workers=workers)

    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True, num_workers=workers) 
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True, num_workers=workers)
    
    model = Learner(input_dim=2048, drop_p=opt.drop_p, mode=opt.mode, num_classes=opt.classes).to(device)
    
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, betas=(opt.momentum, 0.999))  # adjust beta1 to momentum
    elif opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr= opt.lr, betas=(opt.momentum, 0.999), weight_decay=0.0)
    elif opt.optimizer == 'RMSProp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr= opt.lr, momentum=opt.momentum)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr= opt.lr, momentum=opt.momentum, nesterov=True)
    elif opt.optimizer == 'Adagrad':
        optimizer = torch.optim.SGD(model.parameters(), lr= opt.lr, momentum=opt.momentum, nesterov=True)
    
    best_auc, last_epoch = 0.0, 0
    
    # resume
    if opt.ckpt is not None:
        ckpt = torch.load(opt.ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        best_auc, last_epoch, epochs = smart_resume(ckpt, optimizer, epochs)
        del ckpt
        
    cuda = device.type != 'cpu'
    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)
        
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75, 100, 125], gamma=0.5)
    scheduler.last_epoch = last_epoch - 1
    criterion = MIL
    
    nb = len(normal_train_loader)

    for epoch in range(last_epoch, epochs):
        
        pbar = enumerate(zip(normal_train_loader, anomaly_train_loader))
        
        train_loss = 0
        mean = 0.0
        
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, )
            
            mloss = 0.0
            
        for batch_idx, ((normal_inputs, normal_ids), (anomaly_inputs, anomaly_ids)) in pbar:
            
            inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
            input_ids = torch.cat([normal_ids, anomaly_ids], dim=0)
            anomaly_ids = anomaly_ids.to(device)

            batch_size = inputs.shape[0]
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            input_ids = input_ids.to(device)
            
            if opt.mode=='ace':
                outputs, fea, class_logits = model(inputs)
            else:
                outputs, fea = model(inputs)
                
            loss = criterion(outputs, batch_size)

            if opt.mode=='amc':
                outputs = unstack_outputs(outputs, batch_size)
                fea = unstack_outputs(fea, batch_size)

                amc_score, mean, _ = get_amc_score(outputs, fea, mean)
                amc_loss, _, _ = weaksup_intra_video_loss(amc_score, batch_size, margin=0.5)
                loss = loss + opt.alpha * amc_loss
                
            elif opt.mode=='ace':
                outputs = unstack_outputs(outputs, batch_size)
                fea = unstack_outputs(fea, batch_size)
                class_logits = unstack_outputs(class_logits, batch_size)

                amc_score, mean, abs_prob = get_amc_score(outputs, fea, mean, class_logits)
                
                amc_loss, abnorm_prob, norm_prob = weaksup_intra_video_loss(amc_score, batch_size, margin=0.5, abs_prob=abs_prob)
                
                ace_loss = F.nll_loss(abnorm_prob.log(), anomaly_ids)
                # normal_ent = torch.distributions.Categorical(norm_prob).entropy()
                # normal_info = (-normal_ent).exp().mean()
                # normal_ent = torch.distributions.Categorical(norm_prob).entropy().sum()
                # abnormal_ent = torch.distributions.Categorical(abnorm_prob).entropy().sum()
                
                # cls_loss = ace_loss - normal_ent + abnormal_ent
                # cls_loss = ace_loss + normal_info
                cls_loss = ace_loss
                
                loss = loss + opt.alpha * amc_loss + opt.beta * cls_loss

                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
                           
            if RANK in {-1, 0}:
                mloss = (mloss * batch_idx + loss.detach().cpu().numpy()) / (batch_idx + 1) 
                pbar.set_description(f'{epoch+1}/{epochs}, loss: {mloss:.4E}')
                 
        scheduler.step()
        epoch_loss =  train_loss / len(normal_train_loader)
    
        auc = validate.run(opt, model, normal_test_loader, anomaly_test_loader)
        
        if auc > best_auc:
            best_auc = auc
                
        if RANK in {-1, 0}:

            # Save model
            ckpt = {
                'epoch': epoch,
                'best_auc': best_auc,
                'model': deepcopy(de_parallel(model)),
                'optimizer': optimizer.state_dict(),
                'opt': vars(opt),
                'date': datetime.now().isoformat()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_auc == auc:
                torch.save(ckpt, best)
            del ckpt
                
        LOGGER.info(f'Epoch: {epoch+1}/{epochs}  ||  loss = {epoch_loss:.4E},  auc = {auc:.4E}')
    LOGGER.info("Best AUC is {}".format(best_auc))

    torch.cuda.empty_cache()
    return best_auc

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # model configurations
    parser.add_argument('--mode', type=str, help='base, amc or ace', default='base')
    parser.add_argument('--ckpt', type=str, help='model checkpoint', default=None)
    parser.add_argument('--alpha', type=float, help='weighted sum of amc loss', default=0.1)
    parser.add_argument('--beta', type=float, help='weighted sum of classification loss', default=0.1)
    parser.add_argument('--drop-p', type=float, help='dropout possibility', default=0.3)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--classes', type=int, default=13, help='number of classes for classification')
        
    # optimizer
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01)
    parser.add_argument('--epochs', type=int, help='# of Epoch', default=150)
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight-decay', type=float, help='weight decay rate', default=0.0010000000474974513)
    
    
    # device setting
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='numnber or workers for dataloader')
    
    # checkpoint attributes
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
        
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name))
        
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        
    # Train
    train(opt, device)
        
def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)