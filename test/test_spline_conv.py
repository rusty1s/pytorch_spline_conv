import pytest
import torch
from torch.autograd import Variable, gradcheck
from torch_spline_conv import spline_conv
from torch_spline_conv.functions.utils import SplineWeighting

from .utils import tensors, Tensor


@pytest.mark.parametrize('tensor', tensors)
def test_spline_conv_cpu(tensor):
    x = Tensor(tensor, [[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]])
    edge_index = torch.LongTensor([[0, 0, 0, 0], [1, 2, 3, 4]])
    pseudo = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
    pseudo = Tensor(tensor, pseudo)
    weight = torch.arange(0.5, 0.5 * 25, step=0.5, out=x.new()).view(12, 2, 1)
    kernel_size = torch.LongTensor([3, 4])
    is_open_spline = torch.ByteTensor([1, 0])
    root_weight = torch.arange(12.5, 13.5, step=0.5, out=x.new()).view(2, 1)
    bias = Tensor(tensor, [1])

    output = spline_conv(x, edge_index, pseudo, weight, kernel_size,
                         is_open_spline, root_weight, 1, bias)

    edgewise_output = [
        1 * 0.25 * (0.5 + 1.5 + 4.5 + 5.5) + 2 * 0.25 * (1 + 2 + 5 + 6),
        3 * 0.25 * (1.5 + 2.5 + 5.5 + 6.5) + 4 * 0.25 * (2 + 3 + 6 + 7),
        5 * 0.25 * (6.5 + 7.5 + 10.5 + 11.5) + 6 * 0.25 * (7 + 8 + 11 + 12),
        7 * 0.25 * (7.5 + 4.5 + 11.5 + 8.5) + 8 * 0.25 * (8 + 5 + 12 + 9),
    ]

    expected_output = [
        [1 + 12.5 * 9 + 13 * 10 + sum(edgewise_output) / 4],
        [1 + 12.5 * 1 + 13 * 2],
        [1 + 12.5 * 3 + 13 * 4],
        [1 + 12.5 * 5 + 13 * 6],
        [1 + 12.5 * 7 + 13 * 8],
    ]

    assert output.tolist() == expected_output

    x, weight, pseudo = Variable(x), Variable(weight), Variable(pseudo)
    root_weight, bias = Variable(root_weight), Variable(bias)

    output = spline_conv(x, edge_index, pseudo, weight, kernel_size,
                         is_open_spline, root_weight, 1, bias)
    assert output.data.tolist() == expected_output


def test_spline_weighting_backward_cpu():
    kernel_size = torch.LongTensor([5, 5])
    is_open_spline = torch.ByteTensor([1, 1])
    op = SplineWeighting(kernel_size, is_open_spline, 1)

    x = torch.DoubleTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    x = Variable(x, requires_grad=True)
    pseudo = [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]]
    # pseudo = Variable(torch.DoubleTensor(pseudo), requires_grad=True)
    pseudo = Variable(torch.DoubleTensor(pseudo))
    weight = torch.DoubleTensor(25, 2, 4).uniform_(-1, 1)
    weight = Variable(weight, requires_grad=True)

    assert gradcheck(op, (x, pseudo, weight), eps=1e-6, atol=1e-4) is True
