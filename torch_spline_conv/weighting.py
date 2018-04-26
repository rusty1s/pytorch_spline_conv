from torch.autograd import Function

from .utils.ffi import fw_weighting, bw_weighting_src
from .utils.ffi import bw_weighting_weight, bw_weighting_basis


def fw(src, weight, basis, weight_index):
    output = src.new_empty((src.size(0), weight.size(2)))
    fw_weighting(output, src, weight, basis, weight_index)
    return output


def bw_src(grad_output, weight, basis, weight_index):
    grad_src = grad_output.new_empty((grad_output.size(0), weight.size(1)))
    bw_weighting_src(grad_src, grad_output, weight, basis, weight_index)
    return grad_src


def bw_weight(grad_output, src, basis, weight_index, K):
    grad_weight = src.new_empty((K, src.size(1), grad_output.size(1)))
    bw_weighting_weight(grad_weight, grad_output, src, basis, weight_index)
    return grad_weight


def bw_basis(grad_output, src, weight, weight_index):
    grad_basis = src.new_empty(weight_index.size())
    bw_weighting_basis(grad_basis, grad_output, src, weight, weight_index)
    return grad_basis


class SplineWeighting(Function):
    @staticmethod
    def forward(ctx, src, weight, basis, weight_index):
        ctx.save_for_backward(src, weight, basis, weight_index)
        return fw(src, weight, basis, weight_index)

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_src = grad_weight = grad_basis = None
        src, weight, basis, weight_index = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_src = bw_src(grad_output, weight, basis, weight_index)

        if ctx.needs_input_grad[1]:
            K = weight.size(0)
            grad_weight = bw_weight(grad_output, src, basis, weight_index, K)

        if ctx.needs_input_grad[2]:
            grad_basis = bw_basis(grad_output, src, weight, weight_index)

        return grad_src, grad_weight, grad_basis, None
