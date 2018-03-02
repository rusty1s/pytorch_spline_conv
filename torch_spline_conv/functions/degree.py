import torch


def node_degree(index, out=None):
    one = torch.ones(index.size(1), out)
    zero = torch.zeros(index.size(1), out)
    return zero.scatter_add_(0, index[0], one)
