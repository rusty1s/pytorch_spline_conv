import torch
from torch.autograd import Function

from .utils.ffi import fw_basis, bw_basis


def fw(degree, pseudo, kernel_size, is_open_spline):
    num_edges, S = pseudo.size(0), (degree + 1)**kernel_size.size(0)
    basis = pseudo.new_empty((num_edges, S))
    weight_index = kernel_size.new_empty((num_edges, S))
    fw_basis(degree, basis, weight_index, pseudo, kernel_size, is_open_spline)
    return basis, weight_index


def bw(degree, grad_basis, pseudo, kernel_size, is_open_spline):
    self = torch.empty_like(pseudo)
    bw_basis(degree, self, grad_basis, pseudo, kernel_size, is_open_spline)
    return self


class SplineBasis(Function):
    @staticmethod
    def forward(ctx, degree, pseudo, kernel_size, is_open_spline):
        ctx.save_for_backward(degree, pseudo, kernel_size, is_open_spline)
        return fw(degree.item(), pseudo, kernel_size, is_open_spline)

    @staticmethod
    def backward(ctx, grad_basis, grad_weight_index):
        degree, pseudo, kernel_size, is_open_spline = ctx.saved_tensors

        grad_pseudo = None
        if ctx.needs_input_grad[1]:
            grad_pseudo = bw(degree.item(), grad_basis, pseudo, kernel_size,
                             is_open_spline)

        return None, grad_pseudo, None, None
