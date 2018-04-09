import torch
from torch.autograd import Function

from .utils.ffi import basis_forward as ffi_basis_forward
from .utils.ffi import basis_backward as ffi_basis_backward


def basis_forward(degree, pseudo, kernel_size, is_open_spline):
    num_nodes, S = pseudo.size(0), (degree + 1)**kernel_size.size(0)
    basis = pseudo.new(num_nodes, S)
    weight_index = kernel_size.new(num_nodes, S)
    ffi_basis_forward(degree, basis, weight_index, pseudo, kernel_size,
                      is_open_spline)
    return basis, weight_index


def basis_backward(degree, grad_basis, pseudo, kernel_size, is_open_spline):
    grad_pseudo = pseudo.new(pseudo.size())
    ffi_basis_backward(degree, grad_pseudo, grad_basis, pseudo, kernel_size,
                       is_open_spline)
    return grad_pseudo


class SplineBasis(Function):
    def __init__(self, degree, kernel_size, is_open_spline):
        super(SplineBasis, self).__init__()
        self.degree = degree
        self.kernel_size = kernel_size
        self.is_open_spline = is_open_spline

    def forward(self, pseudo):
        self.save_for_backward(pseudo)
        return basis_forward(self.degree, pseudo, self.kernel_size,
                             self.is_open_spline)

    def backward(self, grad_basis, grad_weight_index):
        pseudo, = self.saved_tensors
        grad_pseudo = None

        if self.needs_input_grad[0]:
            grad_pseudo = basis_backward(self.degree, grad_basis, pseudo,
                                         self.kernel_size, self.is_open_spline)

        return grad_pseudo


def spline_basis(degree, pseudo, kernel_size, is_open_spline):
    if torch.is_tensor(pseudo):
        return basis_forward(degree, pseudo, kernel_size, is_open_spline)
    else:
        return SplineBasis(degree, kernel_size, is_open_spline)(pseudo)
