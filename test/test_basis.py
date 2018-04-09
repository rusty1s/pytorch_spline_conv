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
def test_basis_forward_gpu():  # pragma: no cover
    pseudo = torch.cuda.FloatTensor([0, 0.0625, 0.25, 0.75, 0.9375, 1])
    kernel_size = torch.cuda.LongTensor([5])
    is_open_spline = torch.cuda.ByteTensor([1])

    basis, weight_index = basis_forward(1, pseudo, kernel_size, is_open_spline)
    print(basis.cpu().tolist())
    print(weight_index.cpu().tolist())
    # 'basis': [[1, 0], [0.75, 0.25], [1, 0], [1, 0], [0.25, 0.75], [1, 0]],
    # 'weight_index': [[0, 1], [0, 1], [1, 2], [3, 4], [3, 4], [4, 0]],


# @pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
# def test_spline_basis_cpu(tensor, i):
#     degree = data[i].get('degree')
#     pseudo = Tensor(tensor, data[i]['pseudo'])
#     pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
#     kernel_size = torch.LongTensor(data[i]['kernel_size'])
#     is_open_spline = torch.ByteTensor(data[i]['is_open_spline'])
#     K = kernel_size.prod()
#     expected_basis = Tensor(tensor, data[i]['expected_basis'])
#     expected_index = torch.LongTensor(data[i]['expected_index'])

#     basis, index = spline_basis_forward(degree, pseudo, kernel_size,
#                                         is_open_spline, K)
#     basis = [pytest.approx(b, 0.01) for b in basis.view(-1).tolist()]

#     assert basis == expected_basis.view(-1).tolist()
#     assert index.tolist() == expected_index.tolist()

# @pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
# @pytest.mark.parametrize('tensor,i', product(tensors, range(len(data))))
# def test_spline_basis_gpu(tensor, i):  # pragma: no cover
#     degree = data[i].get('degree')
#     pseudo = Tensor(tensor, data[i]['pseudo']).cuda()
#     pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
#     kernel_size = torch.cuda.LongTensor(data[i]['kernel_size'])
#     is_open_spline = torch.cuda.ByteTensor(data[i]['is_open_spline'])
#     K = kernel_size.prod()
#     expected_basis = Tensor(tensor, data[i]['expected_basis'])
#     expected_index = torch.LongTensor(data[i]['expected_index'])

#     basis, index = spline_basis_forward(degree, pseudo, kernel_size,
#                                         is_open_spline, K)
#     basis, index = basis.cpu(), index.cpu()
#     basis = [pytest.approx(b, 0.01) for b in basis.view(-1).tolist()]

#     assert basis == expected_basis.view(-1).tolist()
#     assert index.tolist() == expected_index.tolist()
