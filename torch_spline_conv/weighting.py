import torch
import torch_spline_conv.weighting_cpu

if torch.cuda.is_available():
    import torch_spline_conv.weighting_cuda


def get_func(name, tensor):
    if tensor.is_cuda:
        return getattr(torch_spline_conv.weighting_cuda, name)
    else:
        return getattr(torch_spline_conv.weighting_cpu, name)


class SplineWeighting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, basis, weight_index):
        ctx.weight_index = weight_index
        ctx.save_for_backward(x, weight, basis)
        op = get_func('weighting_fw', x)
        out = op(x, weight, basis, weight_index)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, basis = ctx.saved_tensors
        grad_x = grad_weight = grad_basis = None

        if ctx.needs_input_grad[0]:
            op = get_func('weighting_bw_x', x)
            grad_x = op(grad_out, weight, basis, ctx.weight_index)

        if ctx.needs_input_grad[1]:
            op = get_func('weighting_bw_w', x)
            grad_weight = op(grad_out, x, basis, ctx.weight_index,
                             weight.size(0))

        if ctx.needs_input_grad[2]:
            op = get_func('weighting_bw_b', x)
            grad_basis = op(grad_out, x, weight, ctx.weight_index)

        return grad_x, grad_weight, grad_basis, None
