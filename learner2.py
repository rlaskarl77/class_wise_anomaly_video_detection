import torch
import torch.nn as nn
from torch.nn import functional as F

class Learner(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0, mode='amc', num_classes=13):
        super(Learner, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.drop_p = 0.6
        self.mode = mode
        self.num_classes = num_classes
        if mode == 'amc_ace':
            self.cls_ano = nn.Linear(32, self.num_classes)
        self.weight_init()
        self.vars = nn.ParameterList()

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)


    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

        if self.mode == 'amc_ace':
            nn.init.xavier_normal_(self.cls_ano.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[2], vars[3])
        x = F.dropout(x, self.drop_p, training=self.training)
        fea = x
        x = F.linear(x, vars[4], vars[5])
        if self.mode == 'amc_ace':
            # logit_ano = self.cls_ano(fea)
            # logit_ano = torch.
            logit_ano = torch.norm(fea.unsqueeze(1) - self.cls_ano.weight.unsqueeze(0), p=2, dim=-1)
            return torch.sigmoid(x), fea, logit_ano

        return torch.sigmoid(x), fea, None

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


