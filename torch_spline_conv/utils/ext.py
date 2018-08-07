import torch
import spline_conv_cpu

if torch.cuda.is_available():
    import spline_conv_cuda


def get_func(name, tensor):
    module = spline_conv_cuda if tensor.is_cuda else spline_conv_cpu
    return getattr(module, name)
