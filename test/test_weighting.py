from itertools import product

import pytest
import torch
from torch.autograd import Variable, gradcheck
from torch_spline_conv.weighting import spline_weighting, SplineWeighting
from torch_spline_conv.basis import spline_basis

from .tensor import tensors

tests = [{
    'src': [[1, 2], [3, 4]],
    'weight': [[[1], [2]], [[3], [4]], [[5], [6]], [[7], [8]]],
    'basis': [[0.5, 0, 0.5, 0], [0, 0, 0.5, 0.5]],
    'weight_index': [[0, 1, 2, 3], [0, 1, 2, 3]],
    'output': [
        [0.5 * ((1 * (1 + 5)) + (2 * (2 + 6)))],
        [0.5 * ((3 * (5 + 7)) + (4 * (6 + 8)))],
    ]
}]


@pytest.mark.parametrize('tensor,i', product(tensors, range(len(tests))))
def test_spline_weighting_forward_cpu(tensor, i):
    data = tests[i]

    src = getattr(torch, tensor)(data['src'])
    weight = getattr(torch, tensor)(data['weight'])
    basis = getattr(torch, tensor)(data['basis'])
    weight_index = torch.LongTensor(data['weight_index'])

    output = spline_weighting(src, weight, basis, weight_index)
    assert output.tolist() == data['output']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(tests))))
def test_spline_weighting_forward_gpu(tensor, i):  # pragma: no cover
    data = tests[i]

    src = getattr(torch.cuda, tensor)(data['src'])
    weight = getattr(torch.cuda, tensor)(data['weight'])
    basis = getattr(torch.cuda, tensor)(data['basis'])
    weight_index = torch.cuda.LongTensor(data['weight_index'])

    output = spline_weighting(src, weight, basis, weight_index)
    assert output.cpu().tolist() == data['output']


def test_spline_basis_backward_cpu():
    src = torch.DoubleTensor(4, 2).uniform_(0, 1)
    weight = torch.DoubleTensor(25, 2, 4).uniform_(0, 1)
    kernel_size = torch.LongTensor([5, 5])
    is_open_spline = torch.ByteTensor([1, 1])
    pseudo = torch.DoubleTensor(4, 2).uniform_(0, 1)
    basis, weight_index = spline_basis(1, pseudo, kernel_size, is_open_spline)

    src = Variable(src, requires_grad=True)
    weight = Variable(weight, requires_grad=True)
    basis = Variable(basis, requires_grad=True)
    weight_index = Variable(weight_index, requires_grad=False)

    data = (src, weight, basis, weight_index)
    assert gradcheck(SplineWeighting(), data, eps=1e-6, atol=1e-4) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_spline_basis_backward_gpu():  # pragma: no cover
    src = torch.cuda.DoubleTensor(4, 2).uniform_(0, 1)
    weight = torch.cuda.DoubleTensor(25, 2, 4).uniform_(0, 1)
    kernel_size = torch.cuda.LongTensor([5, 5])
    is_open_spline = torch.cuda.ByteTensor([1, 1])
    pseudo = torch.cuda.DoubleTensor(4, 2).uniform_(0, 1)
    basis, weight_index = spline_basis(1, pseudo, kernel_size, is_open_spline)

    src = Variable(src, requires_grad=True)
    weight = Variable(weight, requires_grad=True)
    basis = Variable(basis, requires_grad=True)
    weight_index = Variable(weight_index, requires_grad=False)

    data = (src, weight, basis, weight_index)
    assert gradcheck(SplineWeighting(), data, eps=1e-6, atol=1e-4) is True
