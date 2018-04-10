from itertools import product

import pytest
import torch
from torch_spline_conv.weighting import spline_weighting

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
def test_spline_basis_forward_cpu(tensor, i):
    data = tests[i]

    src = getattr(torch, tensor)(data['src'])
    weight = getattr(torch, tensor)(data['weight'])
    basis = getattr(torch, tensor)(data['basis'])
    weight_index = torch.LongTensor(data['weight_index'])

    output = spline_weighting(src, weight, basis, weight_index)
    assert output.tolist() == data['output']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.parametrize('tensor,i', product(tensors, range(len(tests))))
def test_spline_basis_forward_gpu(tensor, i):
    data = tests[i]

    src = getattr(torch.cuda, tensor)(data['src'])
    weight = getattr(torch.cuda, tensor)(data['weight'])
    basis = getattr(torch.cuda, tensor)(data['basis'])
    weight_index = torch.cuda.LongTensor(data['weight_index'])

    output = spline_weighting(src, weight, basis, weight_index)
    assert output.cpu().tolist() == data['output']
