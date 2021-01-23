import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from net import Net
from torch.distributions.gumbel import Gumbel
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MAML(nn.Module):

    def __init__(self, inner_lr=1e-2, outer_lr=1e-3):
        super(MAML, self).__init__()
        self.inner_lr = inner_lr
        self.net = Net(transform=False)
        self.optimizer = optim.Adam(self.net.parameters(), lr=outer_lr)

    def forward(self, k_x, k_y, q_x, q_y):
        tasks = k_x.size(0)
        losses = 0

        for i in range(tasks):
            train_loss = self.net.get_loss(k_x[i], k_y[i])
            grad = torch.autograd.grad(train_loss, self.net.parameters())
            fast_weights = OrderedDict(
                [(name, p - self.inner_lr * g) for ((name, p), g) in zip(self.net.named_parameters(), grad)]
            )
            test_loss = self.net.get_loss(q_x[i], q_y[i], fast_weights)

            losses = losses + test_loss
        losses /= tasks

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        return losses.item()

class T_net(nn.Module):

    def __init__(self, inner_lr=1e-2, outer_lr=1e-3, mask=False, temperature=1):
        super(T_net, self).__init__()
        self.inner_lr = inner_lr
        self.net = Net(transform=True)
        self.params = self.net.parameters()
        self.mask = mask
        self.temperature = temperature
        if mask:
            self.gumbel = Gumbel(torch.tensor([0.]), torch.tensor([1.]))
            self.m1 = torch.zeros(40, requires_grad=True, device=device)
            self.m2 = torch.zeros(40, requires_grad=True, device=device)
            self.m3 = torch.zeros(1, requires_grad=True, device=device)
            truncated_normal_(self.m1, std=.1)
            truncated_normal_(self.m2, std=.1)
            truncated_normal_(self.m3, std=.1)

            self.params = list(self.net.parameters())+[self.m1, self.m2, self.m3]

        self.optimizer = optim.Adam(self.params, lr=outer_lr)

    def forward(self, k_x, k_y, q_x, q_y):
        if self.mask:
            return self._forward_with_mask(k_x, k_y, q_x, q_y)
        else:
            return self._forward(k_x, k_y, q_x, q_y)

    def _forward(self, k_x, k_y, q_x, q_y):
        tasks = k_x.size(0)
        losses = 0

        for i in range(tasks):
            train_loss = self.net.get_loss(k_x[i], k_y[i])
            weights = OrderedDict(
                [(name, p) for (name, p) in self.net.named_parameters() if name[0] != 't']
            )
            grad = torch.autograd.grad(train_loss, weights.values())
            fast_weights = OrderedDict(
                [(name, p - self.inner_lr * g) for ((name, p), g) in zip(weights.items(), grad)]
            )
            test_loss = self.net.get_loss(q_x[i], q_y[i], fast_weights)

            losses = losses + test_loss

        losses /= tasks

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        return losses.item()

    def _forward_with_mask(self, k_x, k_y, q_x, q_y):
        tasks = k_x.size(0)
        losses = 0

        m1 = gumbel_softmax(self.m1, self.temperature)[:, 0]
        m2 = gumbel_softmax(self.m2, self.temperature)[:, 0]
        m3 = gumbel_softmax(self.m3, self.temperature)[:, 0]
        mask = [m1, m2, m3, m1, m2, m3]
        for i in range(tasks):
            train_loss = self.net.get_loss(k_x[i], k_y[i])
            weights = OrderedDict(
                [(name, p) for (name, p) in self.net.named_parameters() if name[0] != 't']
            )
            grad = torch.autograd.grad(train_loss, weights.values())
            fast_weights = OrderedDict(
                [(name, p - m * self.inner_lr * g) for ((name, p), g, m) in zip(weights.items(), grad, mask)]
            )
            test_loss = self.net.get_loss(q_x[i], q_y[i], fast_weights)

            losses = losses + test_loss

        losses /= tasks

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        return losses.item()
