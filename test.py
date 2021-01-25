import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
#from meta import MAML, T_net
from meta2 import Meta

def main(args):
    maml = Meta(use_M=False, use_T=False)
    tnet = Meta(use_M=False, use_T=True)
    mtnet = Meta(use_M=True, use_T=True)

    models = {'maml': maml,
              'tnet': tnet,
              'mtnet': mtnet
              }

    amplitude = np.random.uniform(0.1, 5.0, size=1)
    phase = np.random.uniform(0., np.pi, size=1)

    xs = np.linspace(-5, 5).reshape(-1, 1).astype('float32')
    ys = (amplitude * np.sin(xs + phase)).astype('float32')

    pre_losses, post_losses = {}, {}
    pre_preds, post_preds = {}, {}

    for name, model in models.items():
        pre_pred, pre_loss, post_pred, post_loss = model.test(xs, ys)
        pre_preds[name], pre_losses[name] = pre_pred, pre_loss
        post_preds[name], post_losses[name] = post_pred, post_loss

    for (name, pre_loss), post_loss in zip(pre_losses.items(), post_losses.values()):
        print(f'{name} pre_loss:{pre_loss.item():.3f} | post_loss:{post_loss.item():.3f}')
    #xs, ys = xs.cpu().numpy(), ys.cpu().numpy()
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