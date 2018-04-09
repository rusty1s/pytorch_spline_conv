from .._ext import ffi

implemented_degrees = {1: 'linear', 2: 'quadratic', 3: 'cubic'}


def get_func(name, is_cuda, tensor=None):
    prefix = 'THCC' if is_cuda else 'TH'
    prefix += 'Tensor' if tensor is None else type(tensor).__name__
    return getattr(ffi, '{}_{}'.format(prefix, name))


def get_degree_str(degree):
    degree = implemented_degrees.get(degree)
    assert degree is not None, (
        'No implementation found for specified B-spline degree')
    return degree


def basis_forward(degree, basis, weight_index, pseudo, kernel_size,
                  is_open_spline):
    name = '{}BasisForward'.format(get_degree_str(degree))
    func = get_func(name, basis.is_cuda, basis)
    func(basis, weight_index, pseudo, kernel_size, is_open_spline)
