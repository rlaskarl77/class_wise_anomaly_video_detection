from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

class Learner(nn.Module):
    def __init__(self, 
                 input_dim: int=2048, 
                 drop_p: float=0.0, 
                 mode: str='amc',
                 num_classes: Optional[int]=None):
        super(Learner, self).__init__()
        
        self.mode = mode
        self.drop_p = drop_p
        self.num_classes = num_classes
        
        self.body = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        if self.mode=='ace':
            self.cls_head = nn.Sequential(
                nn.Linear(32, 32),
                nn.SiLU(),
                nn.Linear(32, self.num_classes),
                nn.Sigmoid(),
            )
        
        self.weight_init()


    def weight_init(self):
        for layer in self.body:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                
        if self.mode=='ace':
            for layer in self.cls_head:
                if type(layer) == nn.Linear:
                    nn.init.xavier_normal_(layer.weight)


    def forward(self, x):
        x = self.body(x)
        fea = x
        x = self.classifier(x)
        
        if self.mode=='ace':
            cls_probs = self.cls_head(fea)
            return x, fea, cls_probs

        return x, fea
    