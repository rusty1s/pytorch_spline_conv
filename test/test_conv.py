from itertools import product

import pytest
import torch
from torch.autograd import gradcheck
from torch_spline_conv import spline_conv
from torch_spline_conv.utils.ffi import implemented_degrees as degrees

from .utils import dtypes, devices, tensor

tests = [{
    'src': [[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]],
    'edge_index': [[0, 0, 0, 0], [1, 2, 3, 4]],
    'pseudo': [[0.25, 0.125], [0.25, 0.375], [0.75, 0.625], [0.75, 0.875]],
    'weight': [
        [[0.5], [1]],
        [[1.5], [2]],
        [[2.5], [3]],
        [[3.5], [4]],
        [[4.5], [5]],
        [[5.5], [6]],
        [[6.5], [7]],
        [[7.5], [8]],
        [[8.5], [9]],
        [[9.5], [10]],
        [[10.5], [11]],
        [[11.5], [12]],
    ],
    'kernel_size': [3, 4],
    'is_open_spline': [1, 0],
    'root_weight': [[12.5], [13]],
    'bias': [1],
    'output': [
        [1 + 12.5 * 9 + 13 * 10 + (8.5 + 40.5 + 107.5 + 101.5) / 4],
        [1 + 12.5 * 1 + 13 * 2],
        [1 + 12.5 * 3 + 13 * 4],
        [1 + 12.5 * 5 + 13 * 6],
        [1 + 12.5 * 7 + 13 * 8],
    ]
}]


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_spline_conv_forward(test, dtype, device):
    src = tensor(test['src'], dtype, device)
    edge_index = tensor(test['edge_index'], torch.long, device)
    pseudo = tensor(test['pseudo'], dtype, device)
    weight = tensor(test['weight'], dtype, device)
    kernel_size = tensor(test['kernel_size'], torch.long, device)
    is_open_spline = tensor(test['is_open_spline'], torch.uint8, device)
    degree = torch.tensor(1)
    root_weight = tensor(test['root_weight'], dtype, device)
    bias = tensor(test['bias'], dtype, device)

    output = spline_conv(src, edge_index, pseudo, weight, kernel_size,
                         is_open_spline, degree, root_weight, bias)
    assert output.tolist() == test['output']


@pytest.mark.parametrize('degree,device', product(degrees.keys(), devices))
def test_spline_basis_backward(degree, device):
    pass
    # src = torch.DoubleTensor(3, 2).uniform_(-1, 1)
    # edge_index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    # pseudo = torch.DoubleTensor(4, 3).uniform_(0, 1)
    # weight = torch.DoubleTensor(125, 2, 4).uniform_(-1, 1)
    # kernel_size = torch.LongTensor([5, 5, 5])
    # is_open_spline = torch.ByteTensor([1, 0, 1])
    # root_weight = torch.DoubleTensor(2, 4).uniform_(-1, 1)
    # bias = torch.DoubleTensor(4).uniform_(-1, 1)

    # src = Variable(src, requires_grad=True)
    # pseudo = Variable(pseudo, requires_grad=True)
    # weight = Variable(weight, requires_grad=True)
    # root_weight = Variable(root_weight, requires_grad=True)
    # bias = Variable(bias, requires_grad=True)

    # def op(src, pseudo, weight, root_weight, bias):
    #     return spline_conv(src, edge_index, pseudo, weight, kernel_size,
    #                        is_open_spline, degree, root_weight, bias)

    # data = (src, pseudo, weight, root_weight, bias)
    # assert gradcheck(op, data, eps=1e-6, atol=1e-4) is True
