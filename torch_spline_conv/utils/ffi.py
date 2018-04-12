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


def basis_backward(degree, self, grad_basis, pseudo, kernel_size,
                   is_open_spline):  # pragma: no cover
    name = '{}BasisBackward'.format(get_degree_str(degree))
    func = get_func(name, self.is_cuda, self)
    func(self, grad_basis, pseudo, kernel_size, is_open_spline)


def weighting_forward(self, src, weight, basis, weight_index):
    func = get_func('weightingForward', self.is_cuda, self)
    func(self, src, weight, basis, weight_index)


def weighting_backward_src(self, grad_output, weight, basis,
                           weight_index):  # pragma: no cover
    func = get_func('weightingBackwardSrc', self.is_cuda, self)
    func(self, grad_output, weight, basis, weight_index)


def weighting_backward_weight(self, grad_output, src, basis,
                              weight_index):  # pragma: no cover
    func = get_func('weightingBackwardWeight', self.is_cuda, self)
    func(self, grad_output, src, basis, weight_index)


def weighting_backward_basis(self, grad_output, src, weight,
                             weight_index):  # pragma: no cover
    func = get_func('weightingBackwardBasis', self.is_cuda, self)
    func(self, grad_output, src, weight, weight_index)
