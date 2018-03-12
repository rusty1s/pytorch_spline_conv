import torch
from torch.autograd import Function

from .ffi import (spline_basis_forward, spline_weighting_forward,
                  spline_weighting_backward_input,
                  spline_weighting_backward_weight,
                  spline_weighting_backward_basis)


class SplineWeighting(Function):
    def __init__(self, kernel_size, is_open_spline, degree):
        super(SplineWeighting, self).__init__()
        self.kernel_size = kernel_size
        self.is_open_spline = is_open_spline
        self.degree = degree

    def forward(self, x, pseudo, weight):
        K = weight.size(0)
        basis, weight_index = spline_basis_forward(
            self.degree, pseudo, self.kernel_size, self.is_open_spline, K)
        output = spline_weighting_forward(x, weight, basis, weight_index)

        # self.save_for_backward(x, weight)
        # self.basis, self.weight_index = basis, weight_index

        return output

    def backward(self, grad_output):  # pragma: no cover
        pass
        # x, weight = self.saved_tensors
        # grad_input, grad_weight = spline_weighting_backward(
        #     grad_output, x, weight, self.basis, self.weight_index)
        # return grad_input, None, grad_weight


def spline_weighting(x, pseudo, weight, kernel_size, is_open_spline, degree):
    if torch.is_tensor(x):
        K = weight.size(0)
        basis, weight_index = spline_basis_forward(degree, pseudo, kernel_size,
                                                   is_open_spline, K)
        return spline_weighting_forward(x, weight, basis, weight_index)
    else:
        op = SplineWeighting(kernel_size, is_open_spline, degree)
        return op(x, pseudo, weight)
