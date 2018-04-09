from itertools import product

import pytest
import torch
from torch_spline_conv.basis import basis_forward

from .tensor import tensors

tests = [{
    'pseudo': [0, 0.0625, 0.25, 0.75, 0.9375, 1],
    'kernel_size': [5],
    'is_open_spline': [1],
    'basis': [[1, 0], [0.75, 0.25], [1, 0], [1, 0], [0.25, 0.75], [1, 0]],
    'weight_index': [[0, 1], [0, 1], [1, 2], [3, 4], [3, 4], [4, 0]],
}, {
    'pseudo': [0, 0.0625, 0.25, 0.75, 0.9375, 1],
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
def test_basis_forward_cpu(tensor, i):
    data = tests[i]

    pseudo = getattr(torch, tensor)(data['pseudo'])
    kernel_size = torch.LongTensor(data['kernel_size'])
    is_open_spline = torch.ByteTensor(data['is_open_spline'])

    basis, weight_index = basis_forward(1, pseudo, kernel_size, is_open_spline)
    assert basis.tolist() == data['basis']
    assert weight_index.tolist() == data['weight_index']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(tests))))
def test_basis_forward_gpu(tensor, i):  # pragma: no cover
    data = tests[i]

    pseudo = getattr(torch.cuda, tensor)(data['pseudo'])
    kernel_size = torch.cuda.LongTensor(data['kernel_size'])
    is_open_spline = torch.cuda.ByteTensor(data['is_open_spline'])

    basis, weight_index = basis_forward(1, pseudo, kernel_size, is_open_spline)
    assert basis.cpu().tolist() == data['basis']
    assert weight_index.cpu().tolist() == data['weight_index']
