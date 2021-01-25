import torch
from torch import nn
from torch import optim
from torch.nn.parameter import Parameter
from collections import OrderedDict
from utils import *
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Meta(nn.Module):

    def __init__(self, use_T, use_M, inner_lr=1e-2, outer_lr=1e-3,
                 hidden_dims=[40, 40], temperature=1):
        super(Meta, self).__init__()
        self.use_T = use_T
        self.use_M = use_M
        self.inner_lr = inner_lr
        self.hidden_dims = hidden_dims
        self.temperature = temperature
        self.weights = self.construct_fc_weights()
        self.optimizer = optim.Adam(self.weights.values(), lr=outer_lr)

    def construct_fc_weights(self):
        weights = OrderedDict()
        weights['w1'] = Parameter(truncated_normal(1, self.hidden_dims[0], std=1e-2), requires_grad=True)
        weights['b1'] = Parameter(torch.zeros(self.hidden_dims[0]), requires_grad=True)
        for i in range(1, len(self.hidden_dims)):
            weights[f'w{i+1}'] = Parameter(truncated_normal(self.hidden_dims[i-1], self.hidden_dims[i], std=1e-2),
                                           requires_grad=True)
            weights[f'b{i+1}'] = Parameter(torch.zeros(self.hidden_dims[i]), requires_grad=True)
        weights[f'w{len(self.hidden_dims)+1}'] = Parameter(truncated_normal(self.hidden_dims[-1], 1, std=1e-2),
                                                           requires_grad=True)
        weights[f'b{len(self.hidden_dims)+1}'] = Parameter(torch.zeros(1), requires_grad=True)

        if self.use_T:
            for i in range(len(self.hidden_dims)):
                weights[f't{i+1}'] = Parameter(torch.eye(self.hidden_dims[i]), requires_grad=True)
            weights[f't{len(self.hidden_dims)+1}'] = Parameter(torch.eye(1), requires_grad=True)

        if self.use_M:
            for i in range(len(self.hidden_dims)):
                weights[f'm{i+1}'] = Parameter(torch.zeros(self.hidden_dims[i]), requires_grad=True)
            weights[f'm{len(self.hidden_dims)+1}'] = Parameter(torch.zeros(1), requires_grad=True)

        return weights

    def net(self, x, weights=None):
        if weights is None:
            weights = self.weights

        def _net_forward_fc(x, weights):
            hid_len = len(self.hidden_dims)

            for i in range(1, hid_len + 1):
                x = F.relu(x.matmul(weights[f'w{i}']) + weights[f'b{i}'], inplace=True)

            x = x.matmul(weights[f'w{hid_len + 1}']) + weights[f'b{hid_len + 1}']

            return x

        def _net_forward_fc_withT(x, weights):
            hid_len = len(self.hidden_dims)

            for i in range(1, hid_len + 1):
                x = F.relu((x.matmul(weights[f'w{i}']) + weights[f'b{i}']).matmul(weights[f't{i}']), inplace=True)

            x = (x.matmul(weights[f'w{hid_len + 1}']) + weights[f'b{hid_len + 1}']).matmul(weights[f't{hid_len + 1}'])

            return x

        if self.use_T:
            return _net_forward_fc_withT(x, weights)
        else:
            return _net_forward_fc(x, weights)

    def forward(self, k_x, k_y, q_x, q_y):
        tasks = k_x.size(0)
        losses = 0

        for i in range(tasks):
            pred = self.net(k_x[i])
            lossk = F.mse_loss(pred, k_y[i])
            update_keys = [k for k in self.weights.keys() if 't' not in k and 'm' not in k]
            update_weights = [self.weights[k] for k in update_keys]
            grad = dict(zip(update_keys, torch.autograd.grad(lossk, update_weights)))

            fast_weights = OrderedDict(self.weights)

            for k in self.weights.keys():
                if k not in update_keys:
                    continue
                elif self.use_M:
                    masks = dict((name[-1], gumbel_softmax(p, self.temperature)) for name, p in self.weights.items() if 'm' in name)

                    fast_weights[k] = fast_weights[k] - masks[k[-1]] * self.inner_lr * grad[k]
                else:
                    fast_weights[k] = fast_weights[k] - self.inner_lr * grad[k]

            pred = self.net(q_x[i], fast_weights)
            lossq = F.mse_loss(pred, q_y[i])

            losses = losses + lossq

        losses /= tasks

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        return losses.item()

    def test(self, xs, ys):
        if not self.use_T:
            model = 'maml'
        elif not self.use_M:
            model = 'tnet'
        else:
            model = 'mtnet'

        load_state = torch.load(f'checkpoint/{model}.ckpt', map_location=device)
        self.weights = load_state['weights']

        k = 10
        idx = np.random.choice(50, k, replace=False)
        k_idx = idx[:k]
        k_x, k_y = torch.from_numpy(xs[k_idx]), torch.from_numpy(ys[k_idx])
        xs, ys = torch.from_numpy(xs), torch.from_numpy(ys)

        pre_pred = self.net(xs, self.weights)
        pre_loss = F.mse_loss(pre_pred, ys)

        k_pred = self.net(k_x)
        loss = F.mse_loss(k_pred, k_y)

        update_keys = [k for k in self.weights.keys() if 't' not in k and 'm' not in k]
        update_weights = [self.weights[k] for k in update_keys]
        #grad = torch.autograd.grad(loss, self.weights.values())
        grad = dict(zip(update_keys, torch.autograd.grad(loss, update_weights)))

        fast_weights = OrderedDict(self.weights)

        for k in fast_weights.keys():
            if k not in update_keys:
                continue
            elif self.use_M:
                masks = dict(
                    (name[-1], gumbel_softmax(p, self.temperature)) for (name, p) in self.weights.items() if name[0] == 'm')

                fast_weights[k] = fast_weights[k] - masks[k[-1]] * self.inner_lr * grad[k]
            else:
                fast_weights[k] = fast_weights[k] - self.inner_lr * grad[k]

        post_pred = self.net(xs, fast_weights)
        post_loss = F.mse_loss(post_pred, ys)

        return pre_pred, pre_loss, post_pred, post_loss










