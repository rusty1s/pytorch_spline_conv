import torch
from torch.autograd import Variable

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
    """Applies the spline-based convolutional operator :math:`(f \star g)(i) =
    \frac{1}{|\mathcal{N}(i)|} \sum_{l=1}^{M_{in}} \sum_{j \in \mathcal{N}(i)}
    f_l(j) \cdot g_l(u(i, j))` over several node features of an input graph.
    Here, :math:`g_l` denotes the kernel function defined over the weighted
    B-spline tensor product basis for a single input feature map :math:`l`.

    Args:
        src (Tensor or Variable): Input node features of shape
            (number_of_nodes x in_channels)
        edge_index (LongTensor): Graph edges, given by source and target
            indices, of shape (2 x number_of_edges) in the fixed interval
            [0, 1]
        pseudo (Tensor or Variable): Edge attributes, ie. pseudo coordinates,
            of shape (number_of_edges x number_of_edge_attributes)
        weight (Tensor or Variable): Trainable weight parameters of shape
            (kernel_size x in_channels x out_channels)
        kernel_size (LongTensor): Number of trainable weight parameters in each
            edge dimension
        is_open_spline (ByteTensor): Whether to use open or closed B-spline
            bases for each dimension
        degree (int): B-spline basis degree (default: :obj:`1`)
        root_weight (Tensor or Variable): Additional shared trainable
            parameters for each feature of the root node of shape
            (in_channels x out_channels) (default: :obj:`None`)
        bias (Tensor or Variable): Optional bias of shape (out_channels)
            (default: :obj:`None`)
    """

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
    row_expand = row_expand if torch.is_tensor(src) else Variable(row_expand)
    output = zero.scatter_add_(0, row_expand, output)

    # Normalize output by node degree.
    index = row if torch.is_tensor(src) else Variable(row)
    degree = node_degree(index, n, out=new(src))
    output /= degree.unsqueeze(-1).clamp(min=1)

    # Weight root node separately (if wished).
    if root_weight is not None:
        output += torch.mm(src, root_weight)

    # Add bias (if wished).
    if bias is not None:
        output += bias

    return output
