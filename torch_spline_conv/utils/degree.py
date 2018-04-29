import torch


def degree(index, num_nodes=None, dtype=None, device=None):
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = torch.zeros((num_nodes), dtype=dtype, device=device)
    return out.scatter_add_(0, index, out.new_ones((index.size(0))))
