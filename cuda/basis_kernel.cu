#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t> struct BasisForward {
  static inline __device__ scalar_t linear(scalar_t v, int64_t k_mod) {
    return 1 - v - k_mod + 2 * v * k_mod;
  }

  static inline __device__ scalar_t quadratic(scalar_t v, int64_t k_mod) {
    if (k_mod == 0)
      return 0.5 * v * v - v + 0.5;
    else if (k_mod == 1)
      return -v * v + v + 0.5;
    else
      return 0.5 * v * v;
  }

  static inline __device__ scalar_t cubic(scalar_t v, int64_t k_mod) {
    if (k_mod == 0)
      return (1 - v) * (1 - v) * (1 - v) / 6.0;
    else if (k_mod == 1)
      return (3 * v * v * v - 6 * v * v + 4) / 6;
    else if (k_mod == 2)
      return (-3 * v * v * v + 3 * v * v + 3 * v + 1) / 6;
    else
      return v * v * v / 6;
  }
};

#define BASIS_FORWARD(M, PSEUDO, KERNEL_SIZE, IS_OPEN_SPLINE, KERNEL_NAME)     \
  [&]() -> std::tuple<at::Tensor, at::Tensor> {                                \
    auto E = PSEUDO.size(0);                                                   \
    auto S = (int64_t)(pow(M + 1, KERNEL_SIZE.size(0)) + 0.5);                 \
    auto basis = at::empty({E, S}, PSEUDO.options());                          \
    auto weight_index = at::empty({E, S}, KERNEL_SIZE.options());              \
                                                                               \
    AT_DISPATCH_FLOATING_TYPES(PSEUDO.type(), "basis_forward_##M", [&] {       \
      KERNEL_NAME<scalar_t><<<BLOCKS(basis.numel()), THREADS>>>(               \
          at::cuda::detail::getTensorInfo<scalar_t, int64_t>(basis),           \
          at::cuda::detail::getTensorInfo<int64_t, int64_t>(weight_index),     \
          at::cuda::detail::getTensorInfo<scalar_t, int64_t>(PSEUDO),          \
          KERNEL_SIZE.data<int64_t>(), IS_OPEN_SPLINE.data<uint8_t>(),         \
          basis.numel());                                                      \
    });                                                                        \
                                                                               \
    return std::make_tuple(basis, weight_index);                               \
  }()

#define BASIS_FORWARD_KERNEL(M, BASIS, WEIGHT_INDEX, PSEUDO, KERNEL_SIZE,      \
                             IS_OPEN_SPLINE, NUMEL, CODE)                      \
  [&] {                                                                        \
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;                \
    const size_t stride = blockDim.x * gridDim.x;                              \
    for (ptrdiff_t i = index; i < NUMEL; i += stride) {                        \
      int64_t e = i / BASIS.sizes[1], s = i % BASIS.sizes[1];                  \
      int64_t k = s, wi = 0, wi_offset = 1;                                    \
      scalar_t b = 1;                                                          \
                                                                               \
      for (ptrdiff_t d = 0; d < PSEUDO.sizes[1]; d++) {                        \
        auto k_mod = k % (M + 1);                                              \
        k /= M + 1;                                                            \
                                                                               \
        auto v = PSEUDO.data[e * PSEUDO.strides[0] + d * PSEUDO.strides[1]];   \
        v *= KERNEL_SIZE[d] - M * IS_OPEN_SPLINE[d];                           \
                                                                               \
        wi += (((int64_t)v + k_mod) % KERNEL_SIZE[d]) * wi_offset;             \
        wi_offset *= KERNEL_SIZE[d];                                           \
                                                                               \
        v -= floor(v);                                                         \
        v = CODE;                                                              \
        b *= v;                                                                \
      }                                                                        \
                                                                               \
      BASIS.data[i] = b;                                                       \
      WEIGHT_INDEX.data[i] = wi;                                               \
    }                                                                          \
  }()

template <typename scalar_t>
__global__ void
linear_fw_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> basis,
                 at::cuda::detail::TensorInfo<int64_t, int64_t> weight_index,
                 at::cuda::detail::TensorInfo<scalar_t, int64_t> pseudo,
                 int64_t *kernel_size, uint8_t *is_open_spline, size_t numel) {
  BASIS_FORWARD_KERNEL(1, basis, weight_index, pseudo, kernel_size,
                       is_open_spline, numel,
                       BasisForward<scalar_t>::linear(v, k_mod));
}

std::tuple<at::Tensor, at::Tensor> linear_fw_cuda(at::Tensor pseudo,
                                                  at::Tensor kernel_size,
                                                  at::Tensor is_open_spline) {
  return BASIS_FORWARD(1, pseudo, kernel_size, is_open_spline,
                       linear_fw_kernel);
}

template <typename scalar_t>
__global__ void
quadratic_fw_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> basis,
                    at::cuda::detail::TensorInfo<int64_t, int64_t> weight_index,
                    at::cuda::detail::TensorInfo<scalar_t, int64_t> pseudo,
                    int64_t *kernel_size, uint8_t *is_open_spline,
                    size_t numel) {
  BASIS_FORWARD_KERNEL(2, basis, weight_index, pseudo, kernel_size,
                       is_open_spline, numel,
                       BasisForward<scalar_t>::quadratic(v, k_mod));
}

std::tuple<at::Tensor, at::Tensor>
quadratic_fw_cuda(at::Tensor pseudo, at::Tensor kernel_size,
                  at::Tensor is_open_spline) {
  return BASIS_FORWARD(2, pseudo, kernel_size, is_open_spline,
                       quadratic_fw_kernel);
}

template <typename scalar_t>
__global__ void
cubic_fw_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> basis,
                at::cuda::detail::TensorInfo<int64_t, int64_t> weight_index,
                at::cuda::detail::TensorInfo<scalar_t, int64_t> pseudo,
                int64_t *kernel_size, uint8_t *is_open_spline, size_t numel) {
  BASIS_FORWARD_KERNEL(3, basis, weight_index, pseudo, kernel_size,
                       is_open_spline, numel,
                       BasisForward<scalar_t>::cubic(v, k_mod));
}

std::tuple<at::Tensor, at::Tensor> cubic_fw_cuda(at::Tensor pseudo,
                                                 at::Tensor kernel_size,
                                                 at::Tensor is_open_spline) {
  return BASIS_FORWARD(3, pseudo, kernel_size, is_open_spline, cubic_fw_kernel);
}

template <typename scalar_t> struct BasisBackward {
  static inline __device__ scalar_t linear(scalar_t v, int64_t k_mod) {
    return 2 * k_mod - 1;
  }

  static inline __device__ scalar_t quadratic(scalar_t v, int64_t k_mod) {
    if (k_mod == 0)
      return v - 1;
    else if (k_mod == 1)
      return -2 * v + 1;
    else
      return v;
  }

  static inline __device__ scalar_t cubic(scalar_t v, int64_t k_mod) {
    if (k_mod == 0)
      return (-v * v + 2 * v - 1) / 2;
    else if (k_mod == 1)
      return (3 * v * v - 4 * v) / 2;
    else if (k_mod == 2)
      return (-3 * v * v + 2 * v + 1) / 2;
    else
      return v * v / 2;
  }
};

#define BASIS_BACKWARD(M, GRAD_BASIS, PSEUDO, KERNEL_SIZE, IS_OPEN_SPLINE,     \
                       KERNEL_NAME)                                            \
  [&]() -> at::Tensor {                                                        \
    auto E = PSEUDO.size(0);                                                   \
    auto D = PSEUDO.size(1);                                                   \
    auto grad_pseudo = at::empty({E, D}, PSEUDO.options());                    \
                                                                               \
    AT_DISPATCH_FLOATING_TYPES(GRAD_BASIS.type(), "basis_backward_##M", [&] {  \
      KERNEL_NAME<scalar_t><<<BLOCKS(grad_pseudo.numel()), THREADS>>>(         \
          at::cuda::detail::getTensorInfo<scalar_t, int64_t>(grad_pseudo),     \
          at::cuda::detail::getTensorInfo<scalar_t, int64_t>(GRAD_BASIS),      \
          at::cuda::detail::getTensorInfo<scalar_t, int64_t>(PSEUDO),          \
          KERNEL_SIZE.data<int64_t>(), IS_OPEN_SPLINE.data<uint8_t>(),         \
          grad_pseudo.numel());                                                \
    });                                                                        \
                                                                               \
    return grad_pseudo;                                                        \
  }()

#define BASIS_BACKWARD_KERNEL(M, GRAD_PSEUDO, GRAD_BASIS, PSEUDO, KERNEL_SIZE, \
                              IS_OPEN_SPLINE, NUMEL, CODE, GRAD_CODE)          \
  [&] {                                                                        \
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;                \
    const size_t stride = blockDim.x * gridDim.x;                              \
    for (ptrdiff_t i = index; i < NUMEL; i += stride) {                        \
      int64_t e = i / GRAD_PSEUDO.sizes[1], d = i % GRAD_PSEUDO.sizes[1];      \
      scalar_t g = 0, tmp;                                                     \
                                                                               \
      for (ptrdiff_t s = 0; s < GRAD_BASIS.sizes[1]; s++) {                    \
        auto k_mod = (s / (int64_t)(pow(M + 1, d) + 0.5)) % (M + 1);           \
        auto v = PSEUDO.data[e * PSEUDO.strides[0] + d * PSEUDO.strides[1]];   \
        v *= KERNEL_SIZE[d] - M * IS_OPEN_SPLINE[d];                           \
        v -= floor(v);                                                         \
        v = GRAD_CODE;                                                         \
        tmp = v;                                                               \
                                                                               \
        for (ptrdiff_t d_it = 1; d_it < GRAD_PSEUDO.sizes[1]; d_it++) {        \
          auto d_new = d_it - (d >= d_it);                                     \
          k_mod = (s / (int64_t)(pow(M + 1, d_new) + 0.5)) % (M + 1);          \
          v = PSEUDO.data[e * pseudo.strides[0] + d_new * PSEUDO.strides[1]];  \
          v *= KERNEL_SIZE[d_new] - M * IS_OPEN_SPLINE[d_new];                 \
          v -= floor(v);                                                       \
          v = CODE;                                                            \
          tmp *= v;                                                            \
        }                                                                      \
        g += tmp *                                                             \
             GRAD_BASIS                                                        \
                 .data[e * GRAD_BASIS.strides[0] + s * GRAD_BASIS.strides[1]]; \
      }                                                                        \
      g *= KERNEL_SIZE[d] - M * IS_OPEN_SPLINE[d];                             \
      GRAD_PSEUDO.data[i] = g;                                                 \
    }                                                                          \
  }()

template <typename scalar_t>
__global__ void
linear_bw_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_pseudo,
                 at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_basis,
                 at::cuda::detail::TensorInfo<scalar_t, int64_t> pseudo,
                 int64_t *kernel_size, uint8_t *is_open_spline, size_t numel) {
  BASIS_BACKWARD_KERNEL(1, grad_pseudo, grad_basis, pseudo, kernel_size,
                        is_open_spline, numel,
                        BasisForward<scalar_t>::linear(v, k_mod),
                        BasisBackward<scalar_t>::linear(v, k_mod));
}

at::Tensor linear_bw_cuda(at::Tensor grad_basis, at::Tensor pseudo,
                          at::Tensor kernel_size, at::Tensor is_open_spline) {
  return BASIS_BACKWARD(1, grad_basis, pseudo, kernel_size, is_open_spline,
                        linear_bw_kernel);
}

template <typename scalar_t>
__global__ void
quadratic_bw_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_pseudo,
                    at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_basis,
                    at::cuda::detail::TensorInfo<scalar_t, int64_t> pseudo,
                    int64_t *kernel_size, uint8_t *is_open_spline,
                    size_t numel) {
  BASIS_BACKWARD_KERNEL(2, grad_pseudo, grad_basis, pseudo, kernel_size,
                        is_open_spline, numel,
                        BasisForward<scalar_t>::quadratic(v, k_mod),
                        BasisBackward<scalar_t>::quadratic(v, k_mod));
}

at::Tensor quadratic_bw_cuda(at::Tensor grad_basis, at::Tensor pseudo,
                             at::Tensor kernel_size,
                             at::Tensor is_open_spline) {
  return BASIS_BACKWARD(2, grad_basis, pseudo, kernel_size, is_open_spline,
                        quadratic_bw_kernel);
}

template <typename scalar_t>
__global__ void
cubic_bw_kernel(at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_pseudo,
                at::cuda::detail::TensorInfo<scalar_t, int64_t> grad_basis,
                at::cuda::detail::TensorInfo<scalar_t, int64_t> pseudo,
                int64_t *kernel_size, uint8_t *is_open_spline, size_t numel) {
  BASIS_BACKWARD_KERNEL(3, grad_pseudo, grad_basis, pseudo, kernel_size,
                        is_open_spline, numel,
                        BasisForward<scalar_t>::cubic(v, k_mod),
                        BasisBackward<scalar_t>::cubic(v, k_mod));
}

at::Tensor cubic_bw_cuda(at::Tensor grad_basis, at::Tensor pseudo,
                         at::Tensor kernel_size, at::Tensor is_open_spline) {
  return BASIS_BACKWARD(3, grad_basis, pseudo, kernel_size, is_open_spline,
                        cubic_bw_kernel);
}
