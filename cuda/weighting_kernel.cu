#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "atomics.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void
weighting_fw_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> out,
                    at::cuda::detail::TensorInfo<scalar_t, int64_t> x,
                    at::cuda::detail::TensorInfo<scalar_t, int64_t> weight,
                    at::cuda::detail::TensorInfo<scalar_t, int64_t> basis,
                    at::cuda::detail::TensorInfo<int64_t, int64_t> weight_index,
                    size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = index; i < numel; i += stride) {
    int64_t e = i / out.sizes[1], m_out = i % out.sizes[1];
    auto S = basis.sizes[1];
    scalar_t v = 0;

    for (ptrdiff_t s = 0; s < S; s++) {
      auto b = basis.data[e * S + s];
      auto wi = weight_index.data[e * S + s];
      for (ptrdiff_t m_in = 0; m_in < x.sizes[1]; m_in++) {
        auto tmp =
            weight.data[wi * weight.strides[0] + m_in * weight.strides[1] +
                        m_out * weight.strides[2]];
        tmp *= b * x.data[e * x.strides[0] + m_in * x.strides[1]];
        v += tmp;
      }
    }
    out.data[i] = v;
  }
}

at::Tensor weighting_fw_cuda(at::Tensor x, at::Tensor weight, at::Tensor basis,
                             at::Tensor weight_index) {
  auto E = x.size(0), M_out = weight.size(2);
  auto out = at::empty({E, M_out}, x.options());
  AT_DISPATCH_FLOATING_TYPES(out.type(), "weighting_fw", [&] {
    weighting_fw_kernel<scalar_t><<<BLOCKS(out.numel()), THREADS>>>(
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(out),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(x),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(weight),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(basis),
        at::cuda::detail::getTensorInfo<int64_t, int64_t>(weight_index),
        out.numel());
  });
  return out;
}

template <typename scalar_t>
__global__ void weighting_bw_x_kernel(
    at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_x,
    at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_out,
    at::cuda::detail::TensorInfo<scalar_t, int64_t> weight,
    at::cuda::detail::TensorInfo<scalar_t, int64_t> basis,
    at::cuda::detail::TensorInfo<int64_t, int64_t> weight_index, size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = index; i < numel; i += stride) {
    int64_t e = i / grad_x.sizes[1], m_in = i % grad_x.sizes[1];
    auto S = basis.sizes[1];
    scalar_t v = 0;

    for (ptrdiff_t s = 0; s < S; s++) {
      auto b = basis.data[e * S + s];
      auto wi = weight_index.data[e * S + s];
      for (ptrdiff_t m_out = 0; m_out < grad_out.sizes[1]; m_out++) {
        auto tmp =
            weight.data[wi * weight.strides[0] + m_out * weight.strides[1] +
                        m_in * weight.strides[2]];
        tmp *= b *
               grad_out
                   .data[e * grad_out.strides[0] + m_out * grad_out.strides[1]];
        v += tmp;
      }
    }
    grad_x.data[i] = v;
  }
}

at::Tensor weighting_bw_x_cuda(at::Tensor grad_out, at::Tensor weight,
                               at::Tensor basis, at::Tensor weight_index) {
  auto E = grad_out.size(0), M_in = weight.size(1);
  auto grad_x = at::empty({E, M_in}, grad_out.options());
  weight = weight.transpose(1, 2).contiguous();
  AT_DISPATCH_FLOATING_TYPES(grad_x.type(), "weighting_bw_x", [&] {
    weighting_bw_x_kernel<scalar_t><<<BLOCKS(grad_x.numel()), THREADS>>>(
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_x),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_out),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(weight),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(basis),
        at::cuda::detail::getTensorInfo<int64_t, int64_t>(weight_index),
        grad_x.numel());
  });
  return grad_x;
}

template <typename scalar_t>
__global__ void weighting_bw_w_kernel(
    at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_weight,
    at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_out,
    at::cuda::detail::TensorInfo<scalar_t, int64_t> x,
    at::cuda::detail::TensorInfo<scalar_t, int64_t> basis,
    at::cuda::detail::TensorInfo<int64_t, int64_t> weight_index, size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = index; i < numel; i += stride) {
    int64_t e = i / grad_out.sizes[1], m_out = i % grad_out.sizes[1];
    int64_t S = basis.sizes[1], M_in = x.sizes[1], M_out = grad_out.sizes[1];

    auto g =
        grad_out.data[e * grad_out.strides[0] + m_out * grad_out.strides[1]];
    for (ptrdiff_t s = 0; s < S; s++) {
      auto b = basis.data[e * S + s];
      auto wi = weight_index.data[e * S + s];
      for (ptrdiff_t m_in = 0; m_in < M_in; m_in++) {
        auto v = g * b * x.data[e * x.strides[0] + m_in * x.strides[1]];
        atomicAdd(&grad_weight.data[wi * M_in * M_out + m_in * M_out + m_out],
                  v);
      }
    }
  }
}

at::Tensor weighting_bw_w_cuda(at::Tensor grad_out, at::Tensor x,
                               at::Tensor basis, at::Tensor weight_index,
                               int64_t K) {
  auto M_in = x.size(1), M_out = grad_out.size(1);
  auto grad_weight = at::zeros({K, M_in, M_out}, grad_out.options());
  AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "weighting_bw_w", [&] {
    weighting_bw_w_kernel<scalar_t><<<BLOCKS(grad_out.numel()), THREADS>>>(
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_weight),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_out),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(x),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(basis),
        at::cuda::detail::getTensorInfo<int64_t, int64_t>(weight_index),
        grad_out.numel());
  });
  return grad_weight;
}

template <typename scalar_t>
__global__ void weighting_bw_b_kernel(
    at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_basis,
    at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_out,
    at::cuda::detail::TensorInfo<scalar_t, int64_t> x,
    at::cuda::detail::TensorInfo<scalar_t, int64_t> weight,
    at::cuda::detail::TensorInfo<int64_t, int64_t> weight_index, size_t numel) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  for (ptrdiff_t i = index; i < numel; i += stride) {
    int64_t e = i / grad_out.sizes[1], m_out = i % grad_out.sizes[1];
    auto S = grad_basis.sizes[1];

    auto g =
        grad_out.data[e * grad_out.strides[0] + m_out * grad_out.strides[1]];
    for (ptrdiff_t s = 0; s < S; s++) {
      scalar_t v = 0;
      auto wi = weight_index.data[e * S + s];
      for (ptrdiff_t m_in = 0; m_in < x.sizes[1]; m_in++) {
        auto w = weight.data[wi * weight.strides[0] + m_in * weight.strides[1] +
                             m_out * weight.strides[2]];
        v += g * w * x.data[e * x.strides[0] + m_in * x.strides[1]];
      }
      atomicAdd(&grad_basis.data[e * S + s], v);
    }
  }
}

at::Tensor weighting_bw_b_cuda(at::Tensor grad_out, at::Tensor x,
                               at::Tensor weight, at::Tensor weight_index) {
  auto E = x.size(0), S = weight_index.size(1);
  auto grad_basis = at::zeros({E, S}, grad_out.options());
  AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "weighting_bw_b", [&] {
    weighting_bw_b_kernel<scalar_t><<<BLOCKS(grad_out.numel()), THREADS>>>(
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_basis),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_out),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(x),
        at::cuda::detail::getTensorInfo<scalar_t, int64_t>(weight),
        at::cuda::detail::getTensorInfo<int64_t, int64_t>(weight_index),
        grad_out.numel());
  });
  return grad_basis;
}
