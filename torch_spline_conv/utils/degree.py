import torch

from .new import new


def node_degree(index, num_nodes, out=None):
    zero = torch.zeros(num_nodes, out=out)
    one = torch.ones(index, out=new(zero))
    return zero.scatter_add_(0, index, one)
