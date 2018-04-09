from .utils.ffi import basis_forward as ffi_basis_forward


def basis_forward(degree, pseudo, kernel_size, is_open_spline):
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
    num_nodes, S = pseudo.size(0), (degree + 1)**kernel_size.size(0)
    basis = pseudo.new(num_nodes, S)
    weight_index = kernel_size.new(num_nodes, S)
    ffi_basis_forward(degree, basis, weight_index, pseudo, kernel_size,
                      is_open_spline)
    return basis, weight_index
