import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from meta import MAML, T_net


def main(args):
    maml = MAML()
    tnet = T_net()
    mtnet = T_net(mask=True)

    models = {'maml': maml,
              'tnet': tnet,
              'mtnet': mtnet
              }

    for k, v in models.items():
        load_state = torch.load(f'checkpoint/{k}.ckpt', map_location='cpu')
        v.net.load_state_dict(load_state['model_state_dict'])

    amplitude = np.random.uniform(0.1, 5.0, size=1)
    phase = np.random.uniform(0., np.pi, size=1)

    xs = np.linspace(-5, 5).reshape(-1, 1).astype('float32')
    ys = (amplitude * np.sin(xs + phase)).astype('float32')

    k = 10
    idx = np.random.choice(50, k, replace=False)
    k_idx = idx[:k]
    k_x, k_y = torch.from_numpy(xs[k_idx]), torch.from_numpy(ys[k_idx])
    xs, ys = torch.from_numpy(xs), torch.from_numpy(ys)

    pre_preds = dict(
        (name, model.net(xs)) for name, model in models.items()
    )

    pre_losses = dict(
        (name, F.mse_loss(pred, ys)) for name, pred in pre_preds.items()
    )

    step_grad = 1
    post_preds = {}
    post_losses = {}
    for name, model in models.items():
        for i in range(step_grad):
            pred = model.net(k_x)
            loss = F.mse_loss(pred, k_y)
            grad = torch.autograd.grad(loss, model.net.parameters())
            fast_weights = OrderedDict(
                (name, p - args.lr * g) for (name, p), g in zip(model.net.named_parameters(), grad)
            )
            pred = model.net(xs, fast_weights)
            post_preds[name] = pred
            post_losses[name] = F.mse_loss(pred, ys)

    for (name, pre_loss), post_loss in zip(pre_losses.items(), post_losses.values()):
        print(f'{name} pre_loss:{pre_loss.item():.3f} | post_loss:{post_loss.item():.3f}')
    xs, ys = xs.cpu().numpy(), ys.cpu().numpy()
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(xs, ys, label='GT')
    # ax[1].plot(xs, ys, label='GT')
    plt.plot(xs, ys, 'r', label='GT')
    for (name, pre_pred), post_pred in zip(pre_preds.items(), post_preds.values()):
        pre_pred, post_pred = pre_pred.cpu().detach().numpy(), post_pred.cpu().detach().numpy()
        plt.plot(xs, post_pred, '-', label=f'{name} One Grad Step')
        #plt.plot(xs, pre_pred, '-.', label=f'{name} Pre Update')



    plt.legend()
    plt.show()
    plt.savefig('graph.png')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta Learning')
    parser.add_argument(
        '--method',
        type=str,
        choices=['maml', 'tnet', 'mtnet'],
        default='tnet'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10
    )
    parser.add_argument(
        '--k_shot',
        type=int,
        default=10
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-2
    )
    args = parser.parse_args()

    main(args)