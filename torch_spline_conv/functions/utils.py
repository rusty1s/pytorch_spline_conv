import torch
from torch.autograd import Function

from .._ext import ffi


def get_func(name, tensor):
    typename = type(tensor).__name__.replace('Tensor', '')
    cuda = 'cuda_' if tensor.is_cuda else ''
    func = getattr(ffi, 'spline_{}_{}{}'.format(name, cuda, typename))
    return func


def spline_bases(pseudo, kernel_size, is_open_spline, degree):
    # raise NotImplementedError for degree > 3
    pass


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
