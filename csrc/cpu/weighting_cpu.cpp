#include "weighting_cpu.h"

#include "utils.h"

torch::Tensor spline_weighting_fw_cpu(torch::Tensor x, torch::Tensor weight,
                                      torch::Tensor basis,
                                      torch::Tensor weight_index) {
  CHECK_CPU(x);
  CHECK_CPU(weight);
  CHECK_CPU(basis);
  CHECK_CPU(weight_index);

  CHECK_INPUT(x.size(1) == weight.size(1));

  auto E = x.size(0);
  auto M_in = x.size(1);
  auto M_out = weight.size(2);
  auto S = basis.size(1);

  auto out = at::empty({E, M_out}, x.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, x.scalar_type(), "weighting_fw", [&] {
    auto x_data = x.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    scalar_t v;

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        v = 0;
        for (int64_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
          auto wi = weight_index_data[e * S + s];
          for (int64_t m_in = 0; m_in < M_in; m_in++) {
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

torch::Tensor spline_weighting_bw_x_cpu(torch::Tensor grad_out,
                                        torch::Tensor weight,
                                        torch::Tensor basis,
                                        torch::Tensor weight_index) {
  CHECK_CPU(grad_out);
  CHECK_CPU(weight);
  CHECK_CPU(basis);
  CHECK_CPU(weight_index);

  CHECK_INPUT(grad_out.size(1) == weight.size(2));

  auto E = grad_out.size(0);
  auto M_in = weight.size(1);
  auto M_out = grad_out.size(1);
  auto S = basis.size(1);

  auto grad_x = at::zeros({E, M_in}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_out.scalar_type(), "weighting_bw_x", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto grad_x_data = grad_x.data_ptr<scalar_t>();

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        auto g =
            grad_out_data[e * grad_out.stride(0) + m_out * grad_out.stride(1)];
        for (int64_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
          auto wi = weight_index_data[e * S + s];
          for (int64_t m_in = 0; m_in < M_in; m_in++) {
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

torch::Tensor spline_weighting_bw_weight_cpu(torch::Tensor grad_out,
                                             torch::Tensor x,
                                             torch::Tensor basis,
                                             torch::Tensor weight_index,
                                             int64_t kernel_size) {
  CHECK_CPU(grad_out);
  CHECK_CPU(x);
  CHECK_CPU(basis);
  CHECK_CPU(weight_index);

  auto E = grad_out.size(0);
  auto M_in = x.size(1);
  auto M_out = grad_out.size(1);
  auto S = basis.size(1);

  auto grad_weight = at::zeros({kernel_size, M_in, M_out}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, x.scalar_type(), "weighting_bw_weight", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto x_data = x.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto grad_weight_data = grad_weight.data_ptr<scalar_t>();

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        auto g =
            grad_out_data[e * grad_out.stride(0) + m_out * grad_out.stride(1)];
        for (int64_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
          auto wi = weight_index_data[e * S + s];
          for (int64_t m_in = 0; m_in < M_in; m_in++) {
            auto v = g * b * x_data[e * x.stride(0) + m_in * x.stride(1)];
            grad_weight_data[wi * M_in * M_out + m_in * M_out + m_out] += v;
          }
        }
      }
    }
  });

  return grad_weight;
}

torch::Tensor spline_weighting_bw_basis_cpu(torch::Tensor grad_out,
                                            torch::Tensor x,
                                            torch::Tensor weight,
                                            torch::Tensor weight_index) {
  CHECK_CPU(grad_out);
  CHECK_CPU(x);
  CHECK_CPU(weight);
  CHECK_CPU(weight_index);

  CHECK_INPUT(x.size(1) == weight.size(1));
  CHECK_INPUT(grad_out.size(1) == weight.size(2));

  auto E = grad_out.size(0);
  auto M_in = x.size(1);
  auto M_out = grad_out.size(1);
  auto S = weight_index.size(1);

  auto grad_basis = at::zeros({E, S}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, x.scalar_type(), "weighting_bw_basis", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto x_data = x.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto grad_basis_data = grad_basis.data_ptr<scalar_t>();

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        auto g =
            grad_out_data[e * grad_out.stride(0) + m_out * grad_out.stride(1)];
        for (int64_t s = 0; s < S; s++) {
          scalar_t b = 0;
          auto wi = weight_index_data[e * S + s];
          for (int64_t m_in = 0; m_in < M_in; m_in++) {
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
