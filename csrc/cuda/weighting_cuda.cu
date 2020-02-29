#include "weighting_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "atomics.cuh"
#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void
spline_weighting_fw_kernel(const scalar_t *x, const scalar_t *weight,
                           const scalar_t *basis, const int64_t *weight_index,
                           scalar_t *out, int64_t E, int64_t M_in,
                           int64_t M_out, int64_t S, int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
    scalar_t v = (scalar_t)0.;

    for (ptrdiff_t s = 0; s < S; s++) {
      const scalar_t b = basis[e * S + s];
      const int64_t wi = weight_index[e * S + s];
      for (int64_t m_in = 0; m_in < M_in; m_in++) {
        scalar_t tmp = weight[wi * M_in * M_out + m_in * M_out + m_out];
        tmp *= b * x[e * M_in + m_in];
        v += tmp;
      }
    }
    out[thread_idx] = v;
  }
}

torch::Tensor spline_weighting_fw_cuda(torch::Tensor x, torch::Tensor weight,
                                       torch::Tensor basis,
                                       torch::Tensor weight_index) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight);
  CHECK_CUDA(basis);
  CHECK_CUDA(weight_index);
  cudaSetDevice(x.get_device());

  CHECK_INPUT(x.size(1) == weight.size(1));

  auto E = x.size(0);
  auto M_in = x.size(1);
  auto M_out = weight.size(2);
  auto S = basis.size(1);

  auto out = at::empty({E, M_out}, x.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "weighting_fw", [&] {
    auto x_data = x.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    spline_weighting_fw_kernel<scalar_t>
        <<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
            x_data, weight_data, basis_data, weight_index_data, out_data, E,
            M_in, M_out, S, out.numel());
  });

  return out;
}

template <typename scalar_t>
__global__ void
spline_weighting_bw_x_kernel(const scalar_t *grad_out, const scalar_t *weight,
                             const scalar_t *basis, const int64_t *weight_index,
                             scalar_t *grad_x, int64_t E, int64_t M_in,
                             int64_t M_out, int64_t S, int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_in;
  const int64_t m_in = thread_idx % M_in;

  if (thread_idx < numel) {
    scalar_t v = (scalar_t)0.;

    for (int64_t s = 0; s < S; s++) {
      const scalar_t b = basis[e * S + s];
      const int64_t wi = weight_index[e * S + s];

      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        scalar_t tmp = weight[wi * M_out * M_in + m_out * M_in + m_in];
        tmp *= b * grad_out[e * M_out + m_out];
        v += tmp;
      }
    }
    grad_x[thread_idx] = v;
  }
}

torch::Tensor spline_weighting_bw_x_cuda(torch::Tensor grad_out,
                                         torch::Tensor weight,
                                         torch::Tensor basis,
                                         torch::Tensor weight_index) {
  CHECK_CUDA(grad_out);
  CHECK_CUDA(weight);
  CHECK_CUDA(basis);
  CHECK_CUDA(weight_index);
  cudaSetDevice(grad_out.get_device());

  CHECK_INPUT(grad_out.size(1) == weight.size(2));

  auto E = grad_out.size(0);
  auto M_in = weight.size(1);
  auto M_out = grad_out.size(1);
  auto S = basis.size(1);

  auto grad_x = at::zeros({E, M_in}, grad_out.options());
  weight = weight.transpose(1, 2).contiguous(); // Contiguous memory-access.

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "weighting_bw_x", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto grad_x_data = grad_x.data_ptr<scalar_t>();

    spline_weighting_bw_x_kernel<scalar_t>
        <<<BLOCKS(grad_x.numel()), THREADS, 0, stream>>>(
            grad_out_data, weight_data, basis_data, weight_index_data,
            grad_x_data, E, M_in, M_out, S, grad_x.numel());
  });

  return grad_x;
}

template <typename scalar_t>
__global__ void spline_weighting_bw_weight_kernel(
    const scalar_t *grad_out, const scalar_t *x, const scalar_t *basis,
    const int64_t *weight_index, scalar_t *grad_weight, int64_t E, int64_t M_in,
    int64_t M_out, int64_t S, int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
    auto g = grad_out[e * M_out + m_out];
    for (int64_t s = 0; s < S; s++) {
      const scalar_t b = basis[e * S + s];
      const int64_t wi = weight_index[e * S + s];

      for (int64_t m_in = 0; m_in < M_in; m_in++) {
        auto v = g * b * x[e * M_in + m_in];
        atomAdd(&grad_weight[wi * M_in * M_out + m_in * M_out + m_out], v);
      }
    }
  }
}

torch::Tensor spline_weighting_bw_weight_cuda(torch::Tensor grad_out,
                                              torch::Tensor x,
                                              torch::Tensor basis,
                                              torch::Tensor weight_index,
                                              int64_t kernel_size) {
  CHECK_CUDA(grad_out);
  CHECK_CUDA(x);
  CHECK_CUDA(basis);
  CHECK_CUDA(weight_index);
  cudaSetDevice(grad_out.get_device());

  auto E = grad_out.size(0);
  auto M_in = x.size(1);
  auto M_out = grad_out.size(1);
  auto S = basis.size(1);

  auto grad_weight = at::zeros({kernel_size, M_in, M_out}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "weighting_bw_weight", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto x_data = x.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto grad_weight_data = grad_weight.data_ptr<scalar_t>();

    spline_weighting_bw_weight_kernel<scalar_t>
        <<<BLOCKS(grad_out.numel()), THREADS, 0, stream>>>(
            grad_out_data, x_data, basis_data, weight_index_data,
            grad_weight_data, E, M_in, M_out, S, grad_out.numel());
  });

  return grad_weight;
}

template <typename scalar_t>
__global__ void spline_weighting_bw_basis_kernel(
    const scalar_t *grad_out, const scalar_t *x, const scalar_t *weight,
    const int64_t *weight_index, scalar_t *grad_basis, int64_t E, int64_t M_in,
    int64_t M_out, int64_t S, int64_t numel) {

  const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
    const scalar_t g = grad_out[e * M_out + m_out];

    for (int64_t s = 0; s < S; s++) {
      scalar_t v = (scalar_t)0.;
      const int64_t wi = weight_index[e * S + s];

      for (int64_t m_in = 0; m_in < M_in; m_in++) {
        const scalar_t w = weight[wi * M_in * M_out + m_in * M_out + m_out];
        v += g * w * x[e * M_in + m_in];
      }
      atomAdd(&grad_basis[e * S + s], v);
    }
  }
}

torch::Tensor spline_weighting_bw_basis_cuda(torch::Tensor grad_out,
                                             torch::Tensor x,
                                             torch::Tensor weight,
                                             torch::Tensor weight_index) {
  CHECK_CUDA(grad_out);
  CHECK_CUDA(x);
  CHECK_CUDA(weight);
  CHECK_CUDA(weight_index);
  cudaSetDevice(grad_out.get_device());

  CHECK_INPUT(x.size(1) == weight.size(1));
  CHECK_INPUT(grad_out.size(1) == weight.size(2));

  auto E = grad_out.size(0);
  auto M_in = x.size(1);
  auto M_out = grad_out.size(1);
  auto S = weight_index.size(1);

  auto grad_basis = at::zeros({E, S}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "weighting_bw_basis", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto x_data = x.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto grad_basis_data = grad_basis.data_ptr<scalar_t>();

    spline_weighting_bw_basis_kernel<scalar_t>
        <<<BLOCKS(grad_out.numel()), THREADS, 0, stream>>>(
            grad_out_data, x_data, weight_data, weight_index_data,
            grad_basis_data, E, M_in, M_out, S, grad_out.numel());
  });

  return grad_basis;
}
