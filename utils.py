import torch
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def gumbel_softmax(logits, temperature):
    logits = torch.stack([logits, torch.zeros(logits.shape, device=device)], 1)
    U = torch.rand(logits.shape).to(device)
    gumbel = -(torch.log(-torch.log(U + 1e-20) + 1e-20))
    y = logits + gumbel
    y = F.softmax(y / temperature)
    y_hard = torch.eq(y, y.max(1, keepdim=True)[0]).float()

    return (y_hard - y).detach() + y