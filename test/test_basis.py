from itertools import product

import pytest
import torch
from torch.autograd import Variable, gradcheck
from torch_spline_conv.basis import spline_basis, SplineBasis
from torch_spline_conv.utils.ffi import implemented_degrees

from .tensor import tensors

tests = [{
    'pseudo': [[0], [0.0625], [0.25], [0.75], [0.9375], [1]],
    'kernel_size': [5],
    'is_open_spline': [1],
    'basis': [[1, 0], [0.75, 0.25], [1, 0], [1, 0], [0.25, 0.75], [1, 0]],
    'weight_index': [[0, 1], [0, 1], [1, 2], [3, 4], [3, 4], [4, 0]],
}, {
    'pseudo': [[0], [0.0625], [0.25], [0.75], [0.9375], [1]],
    'kernel_size': [4],
    'is_open_spline': [0],
    'basis': [[1, 0], [0.75, 0.25], [1, 0], [1, 0], [0.25, 0.75], [1, 0]],
    'weight_index': [[0, 1], [0, 1], [1, 2], [3, 0], [3, 0], [0, 1]],
}, {
    'pseudo': [[0.125, 0.5], [0.5, 0.5], [0.75, 0.125]],
    'kernel_size': [5, 5],
    'is_open_spline': [1, 1],
    'basis': [[0.5, 0.5, 0, 0], [1, 0, 0, 0], [0.5, 0, 0.5, 0]],
    'weight_index': [[10, 11, 15, 16], [12, 13, 17, 18], [3, 4, 8, 9]]
}]


@pytest.mark.parametrize('tensor,i', product(tensors, range(len(tests))))
def test_spline_basis_forward_cpu(tensor, i):
    data = tests[i]

    pseudo = getattr(torch, tensor)(data['pseudo'])
    kernel_size = torch.LongTensor(data['kernel_size'])
    is_open_spline = torch.ByteTensor(data['is_open_spline'])

    basis, weight_index = spline_basis(1, pseudo, kernel_size, is_open_spline)
    assert basis.tolist() == data['basis']
    assert weight_index.tolist() == data['weight_index']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(tests))))
def test_spline_basis_forward_gpu(tensor, i):  # pragma: no cover
    data = tests[i]

    pseudo = getattr(torch.cuda, tensor)(data['pseudo'])
    kernel_size = torch.cuda.LongTensor(data['kernel_size'])
    is_open_spline = torch.cuda.ByteTensor(data['is_open_spline'])

    basis, weight_index = spline_basis(1, pseudo, kernel_size, is_open_spline)
    assert basis.cpu().tolist() == data['basis']
    assert weight_index.cpu().tolist() == data['weight_index']


@pytest.mark.parametrize('degree', implemented_degrees.keys())
def test_spline_basis_backward_cpu(degree):
    kernel_size = torch.LongTensor([5, 5, 5])
    is_open_spline = torch.ByteTensor([1, 0, 1])
    pseudo = torch.DoubleTensor(4, 3).uniform_(0, 1)
    pseudo = Variable(pseudo, requires_grad=True)

    op = SplineBasis(degree, kernel_size, is_open_spline)
    assert gradcheck(op, (pseudo, ), eps=1e-6, atol=1e-4) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('degree', implemented_degrees.keys())
def test_spline_basis_backward_gpu(degree):  # pragma: no cover
    kernel_size = torch.cuda.LongTensor([5, 5, 5])
    is_open_spline = torch.cuda.ByteTensor([1, 0, 1])
    pseudo = torch.cuda.DoubleTensor(4, 3).uniform_(0, 1)
    pseudo = Variable(pseudo, requires_grad=True)

    op = SplineBasis(degree, kernel_size, is_open_spline)
    assert gradcheck(op, (pseudo, ), eps=1e-6, atol=1e-4) is True
