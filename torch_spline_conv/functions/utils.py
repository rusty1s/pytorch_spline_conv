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
    assert degree is not None, (
        'Basis computation not implemented for specified B-spline degree')

    func = get_func('basis_{}'.format(degree), pseudo)
    func(basis, weight_index, pseudo, kernel_size, is_open_spline, K)
    return basis, weight_index


def spline_weighting_fw(x, weight, basis, weight_index):
    output = x.new(x.size(0), weight.size(2))
    func = get_func('weighting_fw', x)
    func(output, x, weight, basis, weight_index)
    return output


def spline_weighting_bw(grad_output, x, weight, basis, weight_index):
    grad_input = x.new(x.size(0), weight.size(1))
    grad_weight = x.new(weight)
    func = get_func('weighting_bw', x)
    func(grad_input, grad_weight, grad_output, x, weight, basis, weight_index)
    return grad_input, grad_weight


class SplineWeighting(Function):
    def __init__(self, basis, weight_index):
        super(SplineWeighting, self).__init__()
        self.basis = basis
        self.weight_index = weight_index

    def forward(self, x, weight):
        self.save_for_backward(x, weight)
        basis, weight_index = self.basis, self.weight_index
        return spline_weighting_fw(x, weight, basis, weight_index)

    def backward(self, grad_output):
        x, weight = self.saved_tensors
        basis, weight_index = self.basis, self.weight_index
        return spline_weighting_bw(grad_output, x, weight, basis, weight_index)


def spline_weighting(x, weight, basis, weight_index):
    if torch.is_tensor(x):
        return spline_weighting_fw(x, weight, basis, weight_index)
    else:
        return SplineWeighting(basis, weight_index)(x, weight)
