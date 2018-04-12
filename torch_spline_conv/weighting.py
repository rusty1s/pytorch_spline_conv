import torch
from torch.autograd import Function

from .utils.ffi import weighting_forward as weighting_fw
from .utils.ffi import weighting_backward_src as weighting_bw_src
from .utils.ffi import weighting_backward_weight as weighting_bw_weight
from .utils.ffi import weighting_backward_basis as weighting_bw_basis


def weighting_forward(src, weight, basis, weight_index):
    output = src.new(src.size(0), weight.size(2))
    weighting_fw(output, src, weight, basis, weight_index)
    return output


def weighting_backward_src(grad_output, weight, basis,
                           weight_index):  # pragma: no cover
    grad_src = grad_output.new(grad_output.size(0), weight.size(1))
    weighting_bw_src(grad_src, grad_output, weight, basis, weight_index)
    return grad_src


def weighting_backward_weight(grad_output, src, basis, weight_index,
                              K):  # pragma: no cover
    grad_weight = src.new(K, src.size(1), grad_output.size(1))
    weighting_bw_weight(grad_weight, grad_output, src, basis, weight_index)
    return grad_weight


def weighting_backward_basis(grad_output, src, weight,
                             weight_index):  # pragma: no cover
    grad_basis = src.new(weight_index.size())
    weighting_bw_basis(grad_basis, grad_output, src, weight, weight_index)
    return grad_basis


class SplineWeighting(Function):
    def forward(self, src, weight, basis, weight_index):
        self.save_for_backward(src, weight, basis, weight_index)
        return weighting_forward(src, weight, basis, weight_index)

    def backward(self, grad_output):  # pragma: no cover
        grad_src = grad_weight = grad_basis = None
        src, weight, basis, weight_index = self.saved_tensors

        if self.needs_input_grad[0]:
            grad_src = weighting_backward_src(grad_output, weight, basis,
                                              weight_index)
        if self.needs_input_grad[1]:
            K = weight.size(0)
            grad_weight = weighting_backward_weight(grad_output, src, basis,
                                                    weight_index, K)

        if self.needs_input_grad[2]:
            grad_basis = weighting_backward_basis(grad_output, src, weight,
                                                  weight_index)

        return grad_src, grad_weight, grad_basis, None


def spline_weighting(src, weight, basis, weight_index):
    if torch.is_tensor(src):
        return weighting_forward(src, weight, basis, weight_index)
    else:
        return SplineWeighting()(src, weight, basis, weight_index)
