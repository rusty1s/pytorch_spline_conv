import torch
# from torch.autograd import Variable as Var

from .degree import node_degree
from .utils import spline_basis, spline_weighting


def spline_conv(x,
                index,
                pseudo,
                weight,
                kernel_size,
                is_open_spline,
                root_weight=None,
                degree=1,
                bias=None):

    x = x.unsqueeze(-1) if x.dim() == 1 else x

    # Get features for every target node => |E| x M_in
    output = x[index[1]]

    # Get B-spline basis products and weight indices for each edge.
    basis, weight_index = spline_basis(degree, pseudo, kernel_size,
                                       is_open_spline, weight.size(0))

    # Weight gathered features based on B-spline basis and trainable weights.
    output = spline_weighting(output, weight, basis, weight_index)

    # Perform the real convolution => Convert |E| x M_out to N x M_out output.
    row = index[0].unsqueeze(-1).expand(-1, output.size(1))
    # zero = x if torch.is_tensor(x) else x.data
    zero = x.new(row.size()).fill_(0)
    # row, zero = row, zero if torch.is_tensor(x) else Var(row), Var(zero)
    output = zero.scatter_add_(0, row, output)

    # Normalize output by node degree.
    output /= node_degree(index, out=x.new()).unsqueeze(-1).clamp_(min=1)

    # Weight root node separately (if wished).
    if root_weight is not None:
        output += torch.mm(x, root_weight)

    # Add bias (if wished).
    if bias is not None:
        output += bias

    return output
