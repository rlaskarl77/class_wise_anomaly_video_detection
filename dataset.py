import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

class_ids = {
    "Abuse": 0,
    "Arrest": 1,
    "Arson": 2,
    "Assault": 3,
    "Burglary": 4,
    "Explosion": 5,
    "Fighting": 6,
    "RoadAccidents": 7,
    "Robbery": 8,
    "Shooting": 9,
    "Shoplifting": 10,
    "Stealing": 11,
    "Vandalism": 12,
    "Normal_Videos_event": 13,
}

class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path='./DATA/UCF-Crime/'):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.class_ids = class_ids
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            class_id = self.class_ids[self.data_list[idx][:-1].split("/")[0]]
            return concat_npy, class_id
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return concat_npy, gts, frames

class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path='./DATA/UCF-Crime/', version='test_anomalyv2.txt'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.class_ids = class_ids
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, version)
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            class_id = self.class_ids[self.data_list[idx][:-1].split("/")[0]]
            return concat_npy, class_id
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-1].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path+'all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            return concat_npy, gts, frames

if __name__ == '__main__':
    loader2 = Normal_Loader(is_train=0)
    print(len(loader2))
    #print(loader[1], loader2[1])
