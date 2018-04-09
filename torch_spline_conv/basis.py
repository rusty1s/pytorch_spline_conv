import torch
from torch.autograd import Function

from .utils.ffi import basis_forward as ffi_basis_forward
from .utils.ffi import basis_backward as ffi_basis_backward


def basis_forward(degree, pseudo, kernel_size, is_open_spline):
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
    num_nodes, S = pseudo.size(0), (degree + 1)**kernel_size.size(0)
    basis = pseudo.new(num_nodes, S)
    weight_index = kernel_size.new(num_nodes, S)
    ffi_basis_forward(degree, basis, weight_index, pseudo, kernel_size,
                      is_open_spline)
    return basis, weight_index


def basis_backward(degree, grad_basis, pseudo, kernel_size, is_open_spline):
    grad_pseudo = pseudo.new(pseudo.size())
    ffi_basis_backward(degree, grad_pseudo, pseudo, kernel_size,
                       is_open_spline)


class Basis(Function):
    def __init__(self, degree, kernel_size, is_open_spline):
        super(Basis, self).__init__()
        self.degree = degree
        self.kernel_size = kernel_size
        self.is_open_spline = is_open_spline

    def forward(self, pseudo):
        self.save_for_backawrd(pseudo)
        return basis_forward(self.degree, pseudo, self.kernel_size,
                             self.is_open_spline)

    def backward(self, grad_basis, grad_weight_index):
        pass


def basis(degree, pseudo, kernel_size, is_open_spline):
    if torch.is_tensor(pseudo):
        return basis_forward(degree, pseudo, kernel_size, is_open_spline)
    else:
        return Basis(degree, kernel_size, is_open_spline)(pseudo)
