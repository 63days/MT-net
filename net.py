import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
import math
from torch.nn import init
from utils import truncated_normal_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2 hidden layer, relu, dim=40
# k: {5, 10, 15} query: 10

class Net(nn.Module):

    def __init__(self, transform):
        super(Net, self).__init__()
        self.transform = transform

        self.w1 = Parameter(torch.zeros(1, 40), requires_grad=True)
        self.w2 = Parameter(torch.zeros(40, 40), requires_grad=True)
        self.w3 = Parameter(torch.zeros(40, 1), requires_grad=True)

        self.b1 = Parameter(torch.zeros(40), requires_grad=True)
        self.b2 = Parameter(torch.zeros(40), requires_grad=True)
        self.b3 = Parameter(torch.zeros(1), requires_grad=True)

        if transform:
            self.t1 = Parameter(torch.eye(40), requires_grad=True)
            self.t2 = Parameter(torch.eye(40), requires_grad=True)
            self.t3 = Parameter(torch.eye(1), requires_grad=True)

        self.reset_parameters()

        self.mse = nn.MSELoss(reduction='sum')

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if name[0] == 'w':
                truncated_normal_(param, mean=0, std=1e-2)

    def forward(self, x, weights=None):
        if self.transform:
            return self._forward_withT(x, weights)
        else:
            return self._forward(x, weights)

    def _forward(self, x, weights):
        if weights is not None:
            w1, w2, w3 = weights['w1'], weights['w2'], weights['w3']
            b1, b2, b3 = weights['b1'], weights['b2'], weights['b3']
        else:
            w1, w2, w3 = self.w1, self.w2, self.w3
            b1, b2, b3 = self.b1, self.b2, self.b3

        x = F.relu(x.matmul(w1) + b1, inplace=True)
        x = F.relu(x.matmul(w2) + b2, inplace=True)
        x = x.matmul(w3) + b3

        return x

    def _forward_withT(self, x, weights):
        if weights is not None:
            w1, w2, w3 = weights['w1'], weights['w2'], weights['w3']
            b1, b2, b3 = weights['b1'], weights['b2'], weights['b3']
        else:
            w1, w2, w3 = self.w1, self.w2, self.w3
            b1, b2, b3 = self.b1, self.b2, self.b3

        x = F.relu((x.matmul(w1) + b1).matmul(self.t1), inplace=True)
        x = F.relu((x.matmul(w2) + b2).matmul(self.t2), inplace=True)
        x = (x.matmul(w3) + b3).matmul(self.t3)

        return x

    def get_loss(self, x, y, weights=None):
        pred = self.forward(x, weights)

        return F.mse_loss(pred, y)




# net = Net()
# for name, p in net.named_parameters():
#     print(name, p)