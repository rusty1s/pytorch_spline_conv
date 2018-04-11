import torch

from .basis import spline_basis
from .weighting import spline_weighting

from .utils.new import new
from .utils.degree import node_degree


def spline_conv(src,
                edge_index,
                pseudo,
                weight,
                kernel_size,
                is_open_spline,
                degree=1,
                root_weight=None,
                bias=None):

    src = src.unsqueeze(-1) if src.dim() == 1 else src
    row, col = edge_index
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

    n, e, m_out = src.size(0), row.size(0), weight.size(2)

    # Weight each node.
    basis, weight_index = spline_basis(degree, pseudo, kernel_size,
                                       is_open_spline)
    output = spline_weighting(src[col], weight, basis, weight_index)

    # Perform the real convolution => Convert e x m_out to n x m_out features.
    zero = new(src, n, m_out).fill_(0)
    row_expand = row.unsqueeze(-1).expand(e, m_out)
    output = zero.scatter_add_(0, row_expand, output)

    # Normalize output by node degree.
    degree = node_degree(row, n, out=new(src))
    output /= degree.unsqueeze(-1).clamp_(min=1)

    # Weight root node separately (if wished).
    if root_weight is not None:
        output += torch.mm(src, root_weight)

    # Add bias (if wished).
    if bias is not None:
        output += bias

    return output
