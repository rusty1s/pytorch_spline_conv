#include <torch/torch.h>

at::Tensor weighting_fw(at::Tensor x, at::Tensor weight, at::Tensor basis,
                        at::Tensor weight_index) {
  auto E = x.size(0), M_in = x.size(1), M_out = weight.size(2);
  auto S = basis.size(1);
  auto out = at::empty({E, M_out}, x.options());

  AT_DISPATCH_FLOATING_TYPES(out.type(), "weighting_fw", [&] {
    auto x_data = x.data<scalar_t>();
    auto weight_data = weight.data<scalar_t>();
    auto basis_data = basis.data<scalar_t>();
    auto weight_index_data = weight_index.data<int64_t>();
    auto out_data = out.data<scalar_t>();

    scalar_t v;

    for (ptrdiff_t e = 0; e < E; e++) {
      for (ptrdiff_t m_out = 0; m_out < M_out; m_out++) {
        v = 0;
        for (ptrdiff_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
          auto wi = weight_index_data[e * S + s];
          for (ptrdiff_t m_in = 0; m_in < M_in; m_in++) {
            auto tmp =
                weight_data[wi * weight.stride(0) + m_in * weight.stride(1) +
                            m_out * weight.stride(2)];
            tmp *= b * x_data[e * x.stride(0) + m_in * x.stride(1)];
            v += tmp;
          }
        }
        out_data[e * M_out + m_out] = v;
      }
    }
  });

  return out;
}

at::Tensor weighting_bw_x(at::Tensor grad_out, at::Tensor weight,
                          at::Tensor basis, at::Tensor weight_index) {
  auto E = grad_out.size(0), M_in = weight.size(1), M_out = grad_out.size(1);
  auto S = basis.size(1);
  auto grad_x = at::zeros({E, M_in}, grad_out.options());

  AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "weighting_bw_x", [&] {
    auto grad_out_data = grad_out.data<scalar_t>();
    auto weight_data = weight.data<scalar_t>();
    auto basis_data = basis.data<scalar_t>();
    auto weight_index_data = weight_index.data<int64_t>();
    auto grad_x_data = grad_x.data<scalar_t>();

    for (ptrdiff_t e = 0; e < E; e++) {
      for (ptrdiff_t m_out = 0; m_out < M_out; m_out++) {
        auto g =
            grad_out_data[e * grad_out.stride(0) + m_out * grad_out.stride(1)];
        for (ptrdiff_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
          auto wi = weight_index_data[e * S + s];
          for (ptrdiff_t m_in = 0; m_in < M_in; m_in++) {
            auto w =
                weight_data[wi * weight.stride(0) + m_in * weight.stride(1) +
                            m_out * weight.stride(2)];
            grad_x_data[e * M_in + m_in] += g * b * w;
          }
        }
      }
    }
  });

  return grad_x;
}

at::Tensor weighting_bw_w(at::Tensor grad_out, at::Tensor x, at::Tensor basis,
                          at::Tensor weight_index, int64_t K) {
  auto E = grad_out.size(0), M_in = x.size(1), M_out = grad_out.size(1);
  auto S = basis.size(1);
  auto grad_weight = at::zeros({K, M_in, M_out}, grad_out.options());

  AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "weighting_bw_w", [&] {
    auto grad_out_data = grad_out.data<scalar_t>();
    auto x_data = x.data<scalar_t>();
    auto basis_data = basis.data<scalar_t>();
    auto weight_index_data = weight_index.data<int64_t>();
    auto grad_weight_data = grad_weight.data<scalar_t>();

    for (ptrdiff_t e = 0; e < E; e++) {
      for (ptrdiff_t m_out = 0; m_out < M_out; m_out++) {
        auto g =
            grad_out_data[e * grad_out.stride(0) + m_out * grad_out.stride(1)];
        for (ptrdiff_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
          auto wi = weight_index_data[e * S + s];
          for (ptrdiff_t m_in = 0; m_in < M_in; m_in++) {
            auto v = g * b * x_data[e * x.stride(0) + m_in * x.stride(1)];
            grad_weight_data[wi * M_in * M_out + m_in * M_out + m_out] += v;
          }
        }
      }
    }
  });

  return grad_weight;
}

at::Tensor weighting_bw_b(at::Tensor grad_out, at::Tensor x, at::Tensor weight,
                          at::Tensor weight_index) {
  auto E = grad_out.size(0), M_in = x.size(1), M_out = grad_out.size(1);
  auto S = weight_index.size(1);
  auto grad_basis = at::zeros({E, S}, grad_out.options());

  AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "weighting_bw_b", [&] {
    auto grad_out_data = grad_out.data<scalar_t>();
    auto x_data = x.data<scalar_t>();
    auto weight_data = weight.data<scalar_t>();
    auto weight_index_data = weight_index.data<int64_t>();
    auto grad_basis_data = grad_basis.data<scalar_t>();

    for (ptrdiff_t e = 0; e < E; e++) {
      for (ptrdiff_t m_out = 0; m_out < M_out; m_out++) {
        auto g =
            grad_out_data[e * grad_out.stride(0) + m_out * grad_out.stride(1)];
        for (ptrdiff_t s = 0; s < S; s++) {
          scalar_t b = 0;
          auto wi = weight_index_data[e * S + s];
          for (ptrdiff_t m_in = 0; m_in < M_in; m_in++) {
            auto w =
                weight_data[wi * weight.stride(0) + m_in * weight.stride(1) +
                            m_out * weight.stride(2)];
            w *= x_data[e * x.stride(0) + m_in * x.stride(1)];
            b += w;
          }
          grad_basis_data[e * S + s] += g * b;
        }
      }
    }
  });

  return grad_basis;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weighting_fw", &weighting_fw, "Weighting Forward (CPU)");
  m.def("weighting_bw_x", &weighting_bw_x, "Weighting Backward X (CPU)");
  m.def("weighting_bw_w", &weighting_bw_w, "Weighting Backward Weight (CPU)");
  m.def("weighting_bw_b", &weighting_bw_b, "Weighting Backward Basis (CPU)");
}
