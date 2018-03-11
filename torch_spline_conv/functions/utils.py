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


def spline_weighting_forward(x, weight, basis, weight_index):
    output = x.new(x.size(0), weight.size(2))
    func = get_func('weighting_forward', x)
    func(output, x, weight, basis, weight_index)
    return output


def spline_weighting_backward(grad_output, x, weight, basis,
                              weight_index):  # pragma: no cover
    # grad_weight computation via `atomic_add` => Initialize with zeros.
    grad_weight = x.new(weight.size()).fill_(0)
    grad_input = x.new(x.size(0), weight.size(1))
    func = get_func('weighting_backward', x)
    func(grad_input, grad_weight, grad_output, x, weight, basis, weight_index)
    return grad_input, grad_weight


class SplineWeighting(Function):
    def __init__(self, kernel_size, is_open_spline, degree):
        super(SplineWeighting, self).__init__()
        self.kernel_size = kernel_size
        self.is_open_spline = is_open_spline
        self.degree = degree

    def forward(self, x, pseudo, weight):
        self.save_for_backward(x, weight)
        K = weight.size(0)
        basis, weight_index = spline_basis(
            self.degree, pseudo, self.kernel_size, self.is_open_spline, K)
        self.basis, self.weight_index = basis, weight_index
        return spline_weighting_forward(x, weight, basis, weight_index)

    def backward(self, grad_output):  # pragma: no cover
        x, weight = self.saved_tensors
        grad_input, grad_weight = spline_weighting_backward(
            grad_output, x, weight, self.basis, self.weight_index)
        return grad_input, None, grad_weight


def spline_weighting(x, pseudo, weight, kernel_size, is_open_spline, degree):
    if torch.is_tensor(x):
        basis, weight_index = spline_basis(degree, pseudo, kernel_size,
                                           is_open_spline, weight.size(0))
        return spline_weighting_forward(x, weight, basis, weight_index)
    else:
        op = SplineWeighting(kernel_size, is_open_spline, degree)
        return op(x, pseudo, weight)
