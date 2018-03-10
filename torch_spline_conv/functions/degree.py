import torch


def node_degree(edge_index, n, out=None):
    zero = torch.zeros(n, out=out)
    one = torch.ones(edge_index.size(1), out=zero.new())
    return zero.scatter_add_(0, edge_index[0], one)
