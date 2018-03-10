import torch

from .degree import node_degree
from .utils import spline_basis, spline_weighting


def spline_conv(x,
                edge_index,
                pseudo,
                weight,
                kernel_size,
                is_open_spline,
                root_weight=None,
                degree=1,
                bias=None):

    n, e = x.size(0), edge_index.size(1)
    K, m_in, m_out = weight.size()

    x = x.unsqueeze(-1) if x.dim() == 1 else x

    # Get features for every target node => |E| x M_in
    output = x[edge_index[1]]

    # Get B-spline basis products and weight indices for each edge.
    basis, weight_index = spline_basis(degree, pseudo, kernel_size,
                                       is_open_spline, K)

    # Weight gathered features based on B-spline basis and trainable weights.
    output = spline_weighting(output, weight, basis, weight_index)

    # Perform the real convolution => Convert |E| x M_out to N x M_out output.
    row = edge_index[0].unsqueeze(-1).expand(e, m_out)
    zero = x.new(n, m_out).fill_(0)
    output = zero.scatter_add_(0, row, output)

    # Normalize output by node degree.
    output /= node_degree(edge_index, n, out=x.new()).clamp_(min=1)

    # Weight root node separately (if wished).
    if root_weight is not None:
        output += torch.mm(x, root_weight)

    # Add bias (if wished).
    if bias is not None:
        output += bias

    return output
