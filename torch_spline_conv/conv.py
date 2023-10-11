from typing import Optional

import torch

from .basis import spline_basis
from .weighting import spline_weighting


def spline_conv(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    pseudo: torch.Tensor,
    weight: torch.Tensor,
    kernel_size: torch.Tensor,
    is_open_spline: torch.Tensor,
    degree: int = 1,
    norm: bool = True,
    root_weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Applies the spline-based convolution operator :math:`(f \star g)(i) =
    \frac{1}{|\mathcal{N}(i)|} \sum_{l=1}^{M_{in}} \sum_{j \in \mathcal{N}(i)}
    f_l(j) \cdot g_l(u(i, j))` over several node features of an input graph.
    The kernel function :math:`g_l` is defined over the weighted B-spline
    tensor product basis for a single input feature map :math:`l`.

    Args:
        x (:class:`Tensor`): Input node features of shape
            (number_of_nodes x in_channels).
        edge_index (:class:`LongTensor`): Graph edges, given by source and
            target indices, of shape (2 x number_of_edges) in the fixed
            interval [0, 1].
        pseudo (:class:`Tensor`): Edge attributes, ie. pseudo coordinates,
            of shape (number_of_edges x number_of_edge_attributes).
        weight (:class:`Tensor`): Trainable weight parameters of shape
            (kernel_size x in_channels x out_channels).
        kernel_size (:class:`LongTensor`): Number of trainable weight
            parameters in each edge dimension.
        is_open_spline (:class:`ByteTensor`): Whether to use open or closed
            B-spline bases for each dimension.
        degree (int, optional): B-spline basis degree. (default: :obj:`1`)
        norm (bool, optional): Whether to normalize output by node degree.
            (default: :obj:`True`)
        root_weight (:class:`Tensor`, optional): Additional shared trainable
            parameters for each feature of the root node of shape
            (in_channels x out_channels). (default: :obj:`None`)
        bias (:class:`Tensor`, optional): Optional bias of shape
            (out_channels). (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    x = x.unsqueeze(-1) if x.dim() == 1 else x
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

    row, col = edge_index[0], edge_index[1]
    N, E, M_out = x.size(0), row.size(0), weight.size(2)

    # Weight each node.
    basis, weight_index = spline_basis(pseudo, kernel_size, is_open_spline,
                                       degree)

    out = spline_weighting(x[col], weight, basis, weight_index)

    # Convert E x M_out to N x M_out features.
    row_expanded = row.unsqueeze(-1).expand_as(out)
    out = x.new_zeros((N, M_out)).scatter_add_(0, row_expanded, out)

    # Normalize out by node degree (if wished).
    if norm:
        ones = torch.ones(E, dtype=x.dtype, device=x.device)
        deg = out.new_zeros(N).scatter_add_(0, row, ones)
        out = out / deg.unsqueeze(-1).clamp_(min=1)

    # Weight root node separately (if wished).
    if root_weight is not None:
        out += x @ root_weight

    # Add bias (if wished).
    if bias is not None:
        out += bias

    return out
