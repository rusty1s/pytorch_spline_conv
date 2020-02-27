from typing import Tuple

import torch


@torch.jit.script
def spline_basis(pseudo: torch.Tensor, kernel_size: torch.Tensor,
                 is_open_spline: torch.Tensor,
                 degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_spline_conv.spline_basis(pseudo, kernel_size,
                                                    is_open_spline, degree)


# class SplineBasis(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, pseudo, kernel_size, is_open_spline, degree):
#         ctx.save_for_backward(pseudo)
#         ctx.kernel_size = kernel_size
#         ctx.is_open_spline = is_open_spline
#         ctx.degree = degree

#         op = get_func('{}_fw'.format(implemented_degrees[degree]), pseudo)
#         basis, weight_index = op(pseudo, kernel_size, is_open_spline)

#         return basis, weight_index

#     @staticmethod
#     def backward(ctx, grad_basis, grad_weight_index):
#         pseudo, = ctx.saved_tensors
#         kernel_size, is_open_spline = ctx.kernel_size, ctx.is_open_spline
#         degree = ctx.degree
#         grad_pseudo = None

#         if ctx.needs_input_grad[0]:
#             grad_pseudo = op(grad_basis, pseudo, kernel_size, is_open_spline)

#         return grad_pseudo, None, None, None
