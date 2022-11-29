from itertools import product

import pytest
import torch
from torch.autograd import gradcheck
from torch_spline_conv import spline_conv
from torch_spline_conv.testing import devices, dtypes, tensor

degrees = [1, 2, 3]

tests = [{
    'x': [[9, 10], [1, 2], [3, 4], [5, 6], [7, 8]],
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
    'expected': [
        [1 + 12.5 * 9 + 13 * 10 + (8.5 + 40.5 + 107.5 + 101.5) / 4],
        [1 + 12.5 * 1 + 13 * 2],
        [1 + 12.5 * 3 + 13 * 4],
        [1 + 12.5 * 5 + 13 * 6],
        [1 + 12.5 * 7 + 13 * 8],
    ]
}]


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_spline_conv_forward(test, dtype, device):
    if dtype == torch.bfloat16 and device == torch.device('cuda:0'):
        return

    x = tensor(test['x'], dtype, device)
    edge_index = tensor(test['edge_index'], torch.long, device)
    pseudo = tensor(test['pseudo'], dtype, device)
    weight = tensor(test['weight'], dtype, device)
    kernel_size = tensor(test['kernel_size'], torch.long, device)
    is_open_spline = tensor(test['is_open_spline'], torch.uint8, device)
    root_weight = tensor(test['root_weight'], dtype, device)
    bias = tensor(test['bias'], dtype, device)
    expected = tensor(test['expected'], dtype, device)

    out = spline_conv(x, edge_index, pseudo, weight, kernel_size,
                      is_open_spline, 1, True, root_weight, bias)

    error = 1e-2 if dtype == torch.bfloat16 else 1e-7
    assert torch.allclose(out, expected, rtol=error, atol=error)


@pytest.mark.parametrize('degree,device', product(degrees, devices))
def test_spline_conv_backward(degree, device):
    x = torch.rand((3, 2), dtype=torch.double, device=device)
    x.requires_grad_()
    edge_index = tensor([[0, 1, 1, 2], [1, 0, 2, 1]], torch.long, device)
    pseudo = torch.rand((4, 3), dtype=torch.double, device=device)
    pseudo.requires_grad_()
    weight = torch.rand((125, 2, 4), dtype=torch.double, device=device)
    weight.requires_grad_()
    kernel_size = tensor([5, 5, 5], torch.long, device)
    is_open_spline = tensor([1, 0, 1], torch.uint8, device)
    root_weight = torch.rand((2, 4), dtype=torch.double, device=device)
    root_weight.requires_grad_()
    bias = torch.rand((4), dtype=torch.double, device=device)
    bias.requires_grad_()

    data = (x, edge_index, pseudo, weight, kernel_size, is_open_spline, degree,
            True, root_weight, bias)
    assert gradcheck(spline_conv, data, eps=1e-6, atol=1e-4) is True
