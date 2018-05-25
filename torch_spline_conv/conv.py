import torch

from .basis import SplineBasis
from .weighting import SplineWeighting

from .utils.degree import degree as node_degree


class SplineConv(object):
    """Applies the spline-based convolution operator :math:`(f \star g)(i) =
    \frac{1}{|\mathcal{N}(i)|} \sum_{l=1}^{M_{in}} \sum_{j \in \mathcal{N}(i)}
    f_l(j) \cdot g_l(u(i, j))` over several node features of an input graph.
    The kernel function :math:`g_l` is defined over the weighted B-spline
    tensor product basis for a single input feature map :math:`l`.

    Args:
        src (:class:`Tensor`): Input node features of shape
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

    @staticmethod
    def apply(src,
              edge_index,
              pseudo,
              weight,
              kernel_size,
              is_open_spline,
              degree=1,
              norm=True,
              root_weight=None,
              bias=None):

        src = src.unsqueeze(-1) if src.dim() == 1 else src
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        row, col = edge_index
        n, m_out = src.size(0), weight.size(2)

        # Weight each node.
        data = SplineBasis.apply(pseudo, kernel_size, is_open_spline, degree)
        out = SplineWeighting.apply(src[col], weight, *data)

        # Convert e x m_out to n x m_out features.
        row_expand = row.unsqueeze(-1).expand_as(out)
        out = src.new_zeros((n, m_out)).scatter_add_(0, row_expand, out)

        # Normalize out by node degree (if wished).
        if norm:
            deg = node_degree(row, n, out.dtype, out.device)
            out = out / deg.unsqueeze(-1).clamp(min=1)

        # Weight root node separately (if wished).
        if root_weight is not None:
            out += torch.mm(src, root_weight)

        # Add bias (if wished).
        if bias is not None:
            out += bias

        return out
