import torch
from sinusoid import Sinusoid
from meta2 import Meta
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    print(device)
    print(args.method)
    if args.method == 'maml':
        meta = Meta(use_M=False, use_T=False)
    elif args.method == 'tnet':
        meta = Meta(use_M=False, use_T=True)
    elif args.method == 'mtnet':
        meta = Meta(use_M=True, use_T=True)

    meta.to(device)
    if not args.test:
        train_ds = Sinusoid(k_shot=args.k_shot, q_query=15, num_tasks=1000000)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        train_iter = iter(train_dl)
        best_loss = float('inf')
        losses = []
        pbar = tqdm(range(args.epochs))
        for epoch in pbar:
            k_i, k_o, q_i, q_o = next(train_iter)
            k_i, k_o, q_i, q_o = k_i.float().to(device), k_o.float().to(device), q_i.float().to(device), q_o.float().to(device)
            loss = meta(k_i, k_o, q_i, q_o)
            pbar.set_description(f'{epoch}/{args.epochs}iter | L:{loss:.4f}')

            if epoch % 100 == 0:
               losses.append(loss)

        torch.save({
            'weights': meta.weights,
            'losses': losses
        }, f'./checkpoint/{args.method}.ckpt')


        plt.plot(losses)
        plt.savefig(f'{args.method}_loss_graph.png', dpi=300)


    else:
        print('testing... k_shot:', args.k_shot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta Learning')
    parser.add_argument(
        '--method',
        type=str,
        choices=['maml', 'tnet', 'mtnet'],
        default='maml'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=70000
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
        '--test',
        action='store_true'
    )
    parser.add_argument(
        '--inner_lr',
        type=float,
        default=1e-2
    )
    parser.add_argument(
        '--outer_lr',
        type=float,
        default=1e-3
    )
    args = parser.parse_args()

    main(args)