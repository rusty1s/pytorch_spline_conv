from .._ext import ffi as ext

implemented_degrees = {1: 'linear', 2: 'quadratic', 3: 'cubic'}


def get_func(name, tensor):
    typename = type(tensor).__name__.replace('Tensor', '')
    cuda = 'cuda_' if tensor.is_cuda else ''
    func = getattr(ext, 'spline_{}_{}{}'.format(name, cuda, typename))
    return func


def spline_basis_forward(degree, pseudo, kernel_size, is_open_spline, K):
    s = (degree + 1)**kernel_size.size(0)
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
    basis = pseudo.new(pseudo.size(0), s)
    weight_index = kernel_size.new(pseudo.size(0), s)

    degree = implemented_degrees.get(degree)
    assert degree is not None, (
        'Basis computation not implemented for specified B-spline degree')

    func = get_func('{}_basis_forward'.format(degree), pseudo)
    func(basis, weight_index, pseudo, kernel_size, is_open_spline, K)
    return basis, weight_index


def spline_weighting_forward(x, weight, basis, weight_index):
    output = x.new(x.size(0), weight.size(2))
    func = get_func('weighting_forward', x)
    func(output, x, weight, basis, weight_index)
    return output


# pragma: no cover
def spline_weighting_backward_input(grad_output, weight, basis, weight_index):
    grad_input = grad_output.new(grad_output.size(0), weight.size(1))
    func = get_func('weighting_backward_input', grad_output)
    func(grad_input, grad_output, weight, basis, weight_index)
    return grad_input


# pragma: no cover
def spline_weighting_backward_basis(grad_output, x, weight, weight_index):
    grad_basis = x.new(weight_index.size())
    func = get_func('weighting_backward_basis', x)
    func(grad_basis, grad_output, x, weight, weight_index)
    return grad_basis


# pragma: no cover
def spline_weighting_backward_weight(grad_output, x, basis, weight_index, K):
    grad_weight = x.new(K, x.size(1), grad_output.size(1)).fill_(0)
    func = get_func('weighting_backward_weight', x)
    func(grad_weight, grad_output, x, basis, weight_index)
    return grad_weight
