#include <torch/extension.h>

#include "compat.h"

template <typename scalar_t> inline scalar_t linear(scalar_t v, int64_t k_mod) {
  return 1 - v - k_mod + 2 * v * k_mod;
}

template <typename scalar_t>
inline scalar_t quadratic(scalar_t v, int64_t k_mod) {
  if (k_mod == 0)
    return 0.5 * v * v - v + 0.5;
  else if (k_mod == 1)
    return -v * v + v + 0.5;
  else
    return 0.5 * v * v;
}

template <typename scalar_t> inline scalar_t cubic(scalar_t v, int64_t k_mod) {
  if (k_mod == 0)
    return (1 - v) * (1 - v) * (1 - v) / 6.0;
  else if (k_mod == 1)
    return (3 * v * v * v - 6 * v * v + 4) / 6;
  else if (k_mod == 2)
    return (-3 * v * v * v + 3 * v * v + 3 * v + 1) / 6;
  else
    return v * v * v / 6;
}

#define BASIS_FORWARD(M, PSEUDO, KERNEL_SIZE, IS_OPEN_SPLINE, FUNC)            \
  [&]() -> std::tuple<at::Tensor, at::Tensor> {                                \
    auto E = PSEUDO.size(0), D = PSEUDO.size(1);                               \
    auto S = (int64_t)(pow(M + 1, KERNEL_SIZE.size(0)) + 0.5);                 \
    auto basis = at::empty({E, S}, PSEUDO.options());                          \
    auto weight_index = at::empty({E, S}, KERNEL_SIZE.options());              \
                                                                               \
    AT_DISPATCH_FLOATING_TYPES(                                                \
        PSEUDO.scalar_type(), "basis_forward_##M", [&] {                       \
          auto pseudo_data = PSEUDO.DATA_PTR<scalar_t>();                      \
          auto kernel_size_data = KERNEL_SIZE.DATA_PTR<int64_t>();             \
          auto is_open_spline_data = IS_OPEN_SPLINE.DATA_PTR<uint8_t>();       \
          auto basis_data = basis.DATA_PTR<scalar_t>();                        \
          auto weight_index_data = weight_index.DATA_PTR<int64_t>();           \
                                                                               \
          int64_t k, wi, wi_offset;                                            \
          scalar_t b;                                                          \
                                                                               \
          for (ptrdiff_t e = 0; e < E; e++) {                                  \
            for (ptrdiff_t s = 0; s < S; s++) {                                \
              k = s;                                                           \
              wi = 0;                                                          \
              wi_offset = 1;                                                   \
              b = 1;                                                           \
              for (ptrdiff_t d = 0; d < D; d++) {                              \
                auto k_mod = k % (M + 1);                                      \
                k /= M + 1;                                                    \
                                                                               \
                auto v =                                                       \
                    pseudo_data[e * pseudo.stride(0) + d * pseudo.stride(1)];  \
                v *= kernel_size_data[d] - M * is_open_spline_data[d];         \
                                                                               \
                wi +=                                                          \
                    (((int64_t)v + k_mod) % kernel_size_data[d]) * wi_offset;  \
                wi_offset *= kernel_size_data[d];                              \
                                                                               \
                v -= floor(v);                                                 \
                v = FUNC<scalar_t>(v, k_mod);                                  \
                b *= v;                                                        \
              }                                                                \
              basis_data[e * S + s] = b;                                       \
              weight_index_data[e * S + s] = wi;                               \
            }                                                                  \
          }                                                                    \
        });                                                                    \
    return std::make_tuple(basis, weight_index);                               \
  }()

std::tuple<at::Tensor, at::Tensor> linear_fw(at::Tensor pseudo,
                                             at::Tensor kernel_size,
                                             at::Tensor is_open_spline) {
  return BASIS_FORWARD(1, pseudo, kernel_size, is_open_spline, linear);
}

std::tuple<at::Tensor, at::Tensor> quadratic_fw(at::Tensor pseudo,
                                                at::Tensor kernel_size,
                                                at::Tensor is_open_spline) {
  return BASIS_FORWARD(2, pseudo, kernel_size, is_open_spline, quadratic);
}

std::tuple<at::Tensor, at::Tensor>
cubic_fw(at::Tensor pseudo, at::Tensor kernel_size, at::Tensor is_open_spline) {
  return BASIS_FORWARD(3, pseudo, kernel_size, is_open_spline, cubic);
}

template <typename scalar_t>
inline scalar_t grad_linear(scalar_t v, int64_t k_mod) {
  return 2 * k_mod - 1;
}

template <typename scalar_t>
inline scalar_t grad_quadratic(scalar_t v, int64_t k_mod) {
  if (k_mod == 0)
    return v - 1;
  else if (k_mod == 1)
    return -2 * v + 1;
  else
    return v;
}

template <typename scalar_t>
inline scalar_t grad_cubic(scalar_t v, int64_t k_mod) {
  if (k_mod == 0)
    return (-v * v + 2 * v - 1) / 2;
  else if (k_mod == 1)
    return (3 * v * v - 4 * v) / 2;
  else if (k_mod == 2)
    return (-3 * v * v + 2 * v + 1) / 2;
  else
    return v * v / 2;
}

#define BASIS_BACKWARD(M, GRAD_BASIS, PSEUDO, KERNEL_SIZE, IS_OPEN_SPLINE,     \
                       FUNC, GRAD_FUNC)                                        \
  [&]() -> at::Tensor {                                                        \
    auto E = PSEUDO.size(0), D = PSEUDO.size(1);                               \
    auto S = GRAD_BASIS.size(1);                                               \
    auto grad_pseudo = at::empty({E, D}, PSEUDO.options());                    \
                                                                               \
    AT_DISPATCH_FLOATING_TYPES(                                                \
        PSEUDO.scalar_type(), "basis_backward_##M", [&] {                      \
          auto grad_basis_data = GRAD_BASIS.DATA_PTR<scalar_t>();              \
          auto pseudo_data = PSEUDO.DATA_PTR<scalar_t>();                      \
          auto kernel_size_data = KERNEL_SIZE.DATA_PTR<int64_t>();             \
          auto is_open_spline_data = IS_OPEN_SPLINE.DATA_PTR<uint8_t>();       \
          auto grad_pseudo_data = grad_pseudo.DATA_PTR<scalar_t>();            \
                                                                               \
          scalar_t g, tmp;                                                     \
                                                                               \
          for (ptrdiff_t e = 0; e < E; e++) {                                  \
            for (ptrdiff_t d = 0; d < D; d++) {                                \
              g = 0;                                                           \
              for (ptrdiff_t s = 0; s < S; s++) {                              \
                auto k_mod = (s / (int64_t)(pow(M + 1, d) + 0.5)) % (M + 1);   \
                auto v =                                                       \
                    pseudo_data[e * pseudo.stride(0) + d * pseudo.stride(1)];  \
                v *= kernel_size_data[d] - M * is_open_spline_data[d];         \
                v -= floor(v);                                                 \
                v = GRAD_FUNC<scalar_t>(v, k_mod);                             \
                tmp = v;                                                       \
                                                                               \
                for (ptrdiff_t d_it = 1; d_it < D; d_it++) {                   \
                  auto d_new = d_it - (d >= d_it);                             \
                  k_mod = (s / (int64_t)(pow(M + 1, d_new) + 0.5)) % (M + 1);  \
                  v = pseudo_data[e * pseudo.stride(0) +                       \
                                  d_new * pseudo.stride(1)];                   \
                  v *= kernel_size_data[d_new] -                               \
                       M * is_open_spline_data[d_new];                         \
                  v -= floor(v);                                               \
                  v = FUNC<scalar_t>(v, k_mod);                                \
                  tmp *= v;                                                    \
                }                                                              \
                g += tmp * grad_basis_data[e * grad_basis.stride(0) +          \
                                           s * grad_basis.stride(1)];          \
              }                                                                \
              g *= kernel_size_data[d] - M * is_open_spline_data[d];           \
              grad_pseudo_data[e * D + d] = g;                                 \
            }                                                                  \
          }                                                                    \
        });                                                                    \
    return grad_pseudo;                                                        \
  }()

at::Tensor linear_bw(at::Tensor grad_basis, at::Tensor pseudo,
                     at::Tensor kernel_size, at::Tensor is_open_spline) {
  return BASIS_BACKWARD(1, grad_basis, pseudo, kernel_size, is_open_spline,
                        linear, grad_linear);
}

at::Tensor quadratic_bw(at::Tensor grad_basis, at::Tensor pseudo,
                        at::Tensor kernel_size, at::Tensor is_open_spline) {
  return BASIS_BACKWARD(2, grad_basis, pseudo, kernel_size, is_open_spline,
                        quadratic, grad_quadratic);
}

at::Tensor cubic_bw(at::Tensor grad_basis, at::Tensor pseudo,
                    at::Tensor kernel_size, at::Tensor is_open_spline) {
  return BASIS_BACKWARD(3, grad_basis, pseudo, kernel_size, is_open_spline,
                        cubic, grad_cubic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_fw", &linear_fw, "Linear Basis Forward (CPU)");
  m.def("quadratic_fw", &quadratic_fw, "Quadratic Basis Forward (CPU)");
  m.def("cubic_fw", &cubic_fw, "Cubic Basis Forward (CPU)");
  m.def("linear_bw", &linear_bw, "Linear Basis Backward (CPU)");
  m.def("quadratic_bw", &quadratic_bw, "Quadratic Basis Backward (CPU)");
  m.def("cubic_bw", &cubic_bw, "Cubic Basis Backward (CPU)");
}
