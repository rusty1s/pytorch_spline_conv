from itertools import product

import pytest
import torch
from torch.autograd import gradcheck
from torch_spline_conv.basis import SplineBasis
from torch_spline_conv.utils.ffi import implemented_degrees as degrees

from .utils import dtypes, devices, tensor

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


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_spline_basis_forward(test, dtype, device):
    degree = torch.tensor(1)
    pseudo = tensor(test['pseudo'], dtype, device)
    kernel_size = tensor(test['kernel_size'], torch.long, device)
    is_open_spline = tensor(test['is_open_spline'], torch.uint8, device)

    basis, weight_index = SplineBasis.apply(degree, pseudo, kernel_size,
                                            is_open_spline)
    assert basis.tolist() == test['basis']
    assert weight_index.tolist() == test['weight_index']


@pytest.mark.parametrize('degree,device', product(degrees.keys(), devices))
def test_spline_basis_backward(degree, device):
    degree = torch.tensor(degree)
    pseudo = torch.rand((4, 3), dtype=torch.double, device=device)
    pseudo.requires_grad_()
    kernel_size = tensor([5, 5, 5], torch.long, device)
    is_open_spline = tensor([1, 0, 1], torch.uint8, device)

    data = (degree, pseudo, kernel_size, is_open_spline)
    # assert gradcheck(SplineBasis.apply, data, eps=1e-6, atol=1e-4) is True
