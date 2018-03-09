import torch
from torch.autograd import Function

from .._ext import ffi

implemented_degrees = {1: 'linear', 2: 'quadratic', 3: 'cubic'}


def get_func(name, tensor):
    typename = type(tensor).__name__.replace('Tensor', '')
    cuda = 'cuda_' if tensor.is_cuda else ''
    func = getattr(ffi, 'spline_{}_{}{}'.format(name, cuda, typename))
    return func


def spline_basis(degree, pseudo, kernel_size, is_open_spline, K):
    s = (degree + 1)**kernel_size.size(0)
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
    basis = pseudo.new(pseudo.size(0), s)
    weight_index = kernel_size.new(pseudo.size(0), s)

    degree = implemented_degrees.get(degree)
    if degree is None:
        raise NotImplementedError('Basis computation not implemented for '
                                  'specified B-spline degree')

    func = get_func('basis_{}'.format(degree), pseudo)
    func(basis, weight_index, pseudo, kernel_size, is_open_spline, K)
    return basis, weight_index


def spline_weighting_forward(x, weight, basis, weight_index):
    pass


def spline_weighting_backward(x, weight, basis, weight_index):
    pass


class SplineWeighting(Function):
    def __init__(self, basis, weight_index):
        super(SplineWeighting, self).__init__()
        self.basis = basis
        self.weight_index = weight_index

    def forward(self, x, weight):
        pass

    def backward(self, grad_output):
        pass


def spline_weighting(x, weight, basis, weight_index):
    if torch.is_tensor(x):
        return spline_weighting_forward(x, weight, basis, weight_index)
    else:
        return SplineWeighting(basis, weight_index)(x, weight)
