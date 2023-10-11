from itertools import product

import pytest
import torch
from torch_spline_conv import spline_basis
from torch_spline_conv.testing import devices, dtypes, tensor

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
    if dtype == torch.bfloat16 and device == torch.device('cuda:0'):
        return

    pseudo = tensor(test['pseudo'], dtype, device)
    kernel_size = tensor(test['kernel_size'], torch.long, device)
    is_open_spline = tensor(test['is_open_spline'], torch.uint8, device)
    basis = tensor(test['basis'], dtype, device)
    weight_index = tensor(test['weight_index'], dtype, device)

    degree = 1

    out = spline_basis(pseudo, kernel_size, is_open_spline, degree)
    assert torch.allclose(out[0], basis)
    assert torch.allclose(out[1], weight_index)

    jit = torch.jit.script(spline_basis)
    jit_out = jit(pseudo, kernel_size, is_open_spline, degree)
    assert torch.allclose(jit_out[0], basis)
    assert torch.allclose(jit_out[1], weight_index)
