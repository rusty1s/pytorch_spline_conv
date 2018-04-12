import torch

from .new import new


def node_degree(index, n, out=None):
    if out is None:  # pragma: no cover
        zero = torch.zeros(n)
    else:
        out.resize_(n) if torch.is_tensor(out) else out.data.resize_(n)
        zero = out.fill_(0)

    one = new(zero, index.size(0)).fill_(1)
    return zero.scatter_add_(0, index, one)
