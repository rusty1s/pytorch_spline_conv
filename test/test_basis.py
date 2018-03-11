from os import path as osp
from itertools import product

import pytest
import json
import torch
from torch_spline_conv.functions.utils import spline_basis

from .utils import tensors, Tensor

f = open(osp.join(osp.dirname(__file__), 'basis.json'), 'r')
data = json.load(f)
f.close()


@pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
def test_spline_basis_cpu(tensor, i):
    degree = data[i].get('degree')
    pseudo = Tensor(tensor, data[i]['pseudo'])
    kernel_size = torch.LongTensor(data[i]['kernel_size'])
    is_open_spline = torch.ByteTensor(data[i]['is_open_spline'])
    K = kernel_size.prod()
    expected_basis = Tensor(tensor, data[i]['expected_basis'])
    expected_index = torch.ByteTensor(data[i]['expected_index'])

    basis, index = spline_basis(degree, pseudo, kernel_size, is_open_spline, K)
    basis = [pytest.approx(x, 0.01) for x in basis.view(-1).tolist()]

    assert basis == expected_basis.view(-1).tolist()
    assert index.tolist() == expected_index.tolist()
