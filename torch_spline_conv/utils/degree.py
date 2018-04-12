import torch

from .new import new


def node_degree(index, n, out=None):
    zero = torch.zeros(n) if out is None else out.resize_(n).fill_(0)
    one = new(zero, index.size(0)).fill_(1)
    return zero.scatter_add_(0, index, one)
