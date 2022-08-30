#include "basis_cpu.h"

#include "utils.h"

template <typename scalar_t, int64_t degree> struct Basis {
  static inline scalar_t forward(scalar_t v, int64_t k_mod) {
    if (degree == 1) {
      return 1. - v - k_mod + 2. * v * k_mod;
    } else if (degree == 2) {
      if (k_mod == 0)
        return 0.5 * v * v - v + 0.5;
      else if (k_mod == 1)
        return -v * v + v + 0.5;
      else
        return 0.5 * v * v;
    } else if (degree == 3) {
      if (k_mod == 0)
        return (1. - v) * (1. - v) * (1. - v) / 6.;
      else if (k_mod == 1)
        return (3. * v * v * v - 6. * v * v + 4.) / 6.;
      else if (k_mod == 2)
        return (-3. * v * v * v + 3. * v * v + 3. * v + 1.) / 6.;
      else
        return v * v * v / 6.;
    } else {
      return (scalar_t)-1.;
    }
  }

  static inline scalar_t backward(scalar_t v, int64_t k_mod) {
    if (degree == 1) {
      return 2 * k_mod - 1;
    } else if (degree == 2) {
      if (k_mod == 0)
        return v - 1.;
      else if (k_mod == 1)
        return -2. * v + 1.;
      else
        return v;
    } else if (degree == 3) {
      if (k_mod == 0)
        return (-v * v + 2. * v - 1.) / 2.;
      else if (k_mod == 1)
        return (3. * v * v - 4. * v) / 2.;
      else if (k_mod == 2)
        return (-3. * v * v + 2. * v + 1.) / 2.;
      else
        return v * v / 2.;
    } else {
      return (scalar_t)-1.;
    }
  }
};

std::tuple<torch::Tensor, torch::Tensor>
spline_basis_fw_cpu(torch::Tensor pseudo, torch::Tensor kernel_size,
                    torch::Tensor is_open_spline, int64_t degree) {
  CHECK_CPU(pseudo);
  CHECK_CPU(kernel_size);
  CHECK_CPU(is_open_spline);

  CHECK_INPUT(kernel_size.dim() == 1);
  CHECK_INPUT(pseudo.size(1) == kernel_size.numel());
  CHECK_INPUT(is_open_spline.dim());
  CHECK_INPUT(pseudo.size(1) == is_open_spline.numel());

  auto E = pseudo.size(0);
  auto D = pseudo.size(1);
  auto S = (int64_t)(pow(degree + 1, D) + 0.5);

  auto basis = at::empty({E, S}, pseudo.options());
  auto weight_index = at::empty({E, S}, kernel_size.options());

  auto kernel_size_data = kernel_size.data_ptr<int64_t>();
  auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, pseudo.scalar_type(), "basis_fw", [&] {
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();

    AT_DISPATCH_DEGREE_TYPES(degree, [&] {
      int64_t k, wi, wi_offset;
      scalar_t b;

      for (int64_t e = 0; e < E; e++) {
        for (int64_t s = 0; s < S; s++) {
          k = s, wi = 0, wi_offset = 1, b = (scalar_t)1.;
          for (int64_t d = 0; d < D; d++) {
            int64_t k_mod = k % (DEGREE + 1);
            k /= DEGREE + 1;

            auto v = pseudo_data[e * pseudo.stride(0) + d * pseudo.stride(1)];
            v *= kernel_size_data[d] - DEGREE * is_open_spline_data[d];

            wi += (((int64_t)v + k_mod) % kernel_size_data[d]) * wi_offset;
            wi_offset *= kernel_size_data[d];

            v -= floor(v);
            v = Basis<scalar_t, DEGREE>::forward(v, k_mod);
            b *= v;
          }
          basis_data[e * S + s] = b;
          weight_index_data[e * S + s] = wi;
        }
      }
    });
  });

  return std::make_tuple(basis, weight_index);
}

torch::Tensor spline_basis_bw_cpu(torch::Tensor grad_basis,
                                  torch::Tensor pseudo,
                                  torch::Tensor kernel_size,
                                  torch::Tensor is_open_spline,
                                  int64_t degree) {
  CHECK_CPU(grad_basis);
  CHECK_CPU(pseudo);
  CHECK_CPU(kernel_size);
  CHECK_CPU(is_open_spline);

  CHECK_INPUT(grad_basis.size(0) == pseudo.size(0));
  CHECK_INPUT(kernel_size.dim() == 1);
  CHECK_INPUT(pseudo.size(1) == kernel_size.numel());
  CHECK_INPUT(is_open_spline.dim());
  CHECK_INPUT(pseudo.size(1) == is_open_spline.numel());

  auto E = pseudo.size(0);
  auto D = pseudo.size(1);
  auto S = grad_basis.size(1);

  auto grad_pseudo = at::empty({E, D}, pseudo.options());

  auto kernel_size_data = kernel_size.data_ptr<int64_t>();
  auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, pseudo.scalar_type(), "basis_bw", [&] {
    auto grad_basis_data = grad_basis.data_ptr<scalar_t>();
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto grad_pseudo_data = grad_pseudo.data_ptr<scalar_t>();

    AT_DISPATCH_DEGREE_TYPES(degree, [&] {
      scalar_t g, tmp;

      for (int64_t e = 0; e < E; e++) {
        for (int64_t d = 0; d < D; d++) {
          g = (scalar_t)0.;
          for (int64_t s = 0; s < S; s++) {
            int64_t k_mod =
                (s / (int64_t)(pow(DEGREE + 1, d) + 0.5)) % (DEGREE + 1);
            auto v = pseudo_data[e * pseudo.stride(0) + d * pseudo.stride(1)];
            v *= kernel_size_data[d] - DEGREE * is_open_spline_data[d];
            v -= floor(v);
            v = Basis<scalar_t, DEGREE>::backward(v, k_mod);
            tmp = v;

            for (int64_t d_it = 1; d_it < D; d_it++) {
              int64_t d_new = d_it - (d >= d_it);
              k_mod =
                  (s / (int64_t)(pow(DEGREE + 1, d_new) + 0.5)) % (DEGREE + 1);
              v = pseudo_data[e * pseudo.stride(0) + d_new * pseudo.stride(1)];
              v *=
                  kernel_size_data[d_new] - DEGREE * is_open_spline_data[d_new];
              v -= floor(v);
              v = Basis<scalar_t, DEGREE>::forward(v, k_mod);
              tmp *= v;
            }
            g += tmp * grad_basis_data[e * grad_basis.stride(0) +
                                       s * grad_basis.stride(1)];
          }
          g *= kernel_size_data[d] - DEGREE * is_open_spline_data[d];
          grad_pseudo_data[e * D + d] = g;
        }
      }
    });
  });

  return grad_pseudo;
}
