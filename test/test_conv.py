from itertools import product

import pytest
import torch
from torch.autograd import Variable, gradcheck
from torch_spline_conv import spline_conv
from torch_spline_conv.utils.ffi import implemented_degrees

from .tensor import tensors

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


@pytest.mark.parametrize('tensor,i', product(tensors, range(len(tests))))
def test_spline_conv_forward_cpu(tensor, i):
    data = tests[i]

    src = getattr(torch, tensor)(data['src'])
    edge_index = torch.LongTensor(data['edge_index'])
    pseudo = getattr(torch, tensor)(data['pseudo'])
    weight = getattr(torch, tensor)(data['weight'])
    kernel_size = torch.LongTensor(data['kernel_size'])
    is_open_spline = torch.ByteTensor(data['is_open_spline'])
    root_weight = getattr(torch, tensor)(data['root_weight'])
    bias = getattr(torch, tensor)(data['bias'])

    output = spline_conv(src, edge_index, pseudo, weight, kernel_size,
                         is_open_spline, 1, root_weight, bias)
    assert output.tolist() == data['output']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(tests))))
def test_spline_conv_forward_gpu(tensor, i):  # pragma: no cover
    data = tests[i]

    src = getattr(torch.cuda, tensor)(data['src'])
    edge_index = torch.cuda.LongTensor(data['edge_index'])
    pseudo = getattr(torch.cuda, tensor)(data['pseudo'])
    weight = getattr(torch.cuda, tensor)(data['weight'])
    kernel_size = torch.cuda.LongTensor(data['kernel_size'])
    is_open_spline = torch.cuda.ByteTensor(data['is_open_spline'])
    root_weight = getattr(torch.cuda, tensor)(data['root_weight'])
    bias = getattr(torch.cuda, tensor)(data['bias'])

    output = spline_conv(src, edge_index, pseudo, weight, kernel_size,
                         is_open_spline, 1, root_weight, bias)
    assert output.cpu().tolist() == data['output']


@pytest.mark.parametrize('degree', implemented_degrees.keys())
def test_spline_basis_backward_cpu(degree):
    src = torch.DoubleTensor(3, 2).uniform_(-1, 1)
    edge_index = torch.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    pseudo = torch.DoubleTensor(4, 3).uniform_(0, 1)
    weight = torch.DoubleTensor(125, 2, 4).uniform_(-1, 1)
    kernel_size = torch.LongTensor([5, 5, 5])
    is_open_spline = torch.ByteTensor([1, 0, 1])
    root_weight = torch.DoubleTensor(2, 4).uniform_(-1, 1)
    bias = torch.DoubleTensor(4).uniform_(-1, 1)

    src = Variable(src, requires_grad=True)
    pseudo = Variable(pseudo, requires_grad=True)
    weight = Variable(weight, requires_grad=True)
    root_weight = Variable(root_weight, requires_grad=True)
    bias = Variable(bias, requires_grad=True)

    def op(src, pseudo, weight, root_weight, bias):
        return spline_conv(src, edge_index, pseudo, weight, kernel_size,
                           is_open_spline, degree, root_weight, bias)

    data = (src, pseudo, weight, root_weight, bias)
    assert gradcheck(op, data, eps=1e-6, atol=1e-4) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('degree', [2])
def test_spline_basis_backward_gpu(degree):  # pragma: no cover
    src = torch.cuda.DoubleTensor(3, 2).uniform_(-1, 1)
    edge_index = torch.cuda.LongTensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    pseudo = torch.cuda.DoubleTensor(4, 3).uniform_(0, 1)
    weight = torch.cuda.DoubleTensor(125, 2, 4).uniform_(-1, 1)
    kernel_size = torch.cuda.LongTensor([5, 5, 5])
    is_open_spline = torch.cuda.ByteTensor([1, 0, 1])
    root_weight = torch.cuda.DoubleTensor(2, 4).uniform_(-1, 1)
    bias = torch.cuda.DoubleTensor(4).uniform_(-1, 1)

    src = Variable(src, requires_grad=False)
    pseudo = Variable(pseudo, requires_grad=True)
    weight = Variable(weight, requires_grad=False)
    root_weight = Variable(root_weight, requires_grad=False)
    bias = Variable(bias, requires_grad=False)

    def op(src, pseudo, weight, root_weight, bias):
        return spline_conv(src, edge_index, pseudo, weight, kernel_size,
                           is_open_spline, degree, root_weight, bias)

    data = (src, pseudo, weight, root_weight, bias)
    assert gradcheck(op, data, eps=1e-6, atol=1e-4) is True
