import torch


def degree(index, num_nodes=None, out=None):
    num_nodes = index.max() + 1 if num_nodes is None else num_nodes
    out = index.new_empty((), dtype=torch.float) if out is None else out
    out.resize_(num_nodes).fill_(0)

    return out.scatter_add_(0, index, out.new_ones((index.size(0))))
