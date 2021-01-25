import torch
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def truncated_normal(*size, mean=0, std=1):
    size = list(size)
    tensor = torch.empty(size)
    tmp = torch.empty(size+[4,]).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def gumbel_softmax(logits, temperature):
    logits = torch.stack([logits, torch.zeros(logits.shape, device=device)], 1)
    U = torch.rand(logits.shape).to(device)
    gumbel = -(torch.log(-torch.log(U + 1e-20) + 1e-20))
    y = logits + gumbel
    y = F.softmax(y / temperature, dim=1)
    y_hard = torch.eq(y, y.max(1, keepdim=True)[0]).float()

    return ((y_hard - y).detach() + y)[:, 0]

