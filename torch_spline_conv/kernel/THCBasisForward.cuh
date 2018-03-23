#define SPLINE_BASIS_FORWARD(NAME, basis, weight_index, pseudo, kernel_size, is_open_spline, K) { \
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, pseudo, kernel_size, is_open_spline)); \
\
  const int n = THCTensor_(nElement)(state, basis); \
  TensorInfo<real> basisInfo = thc_(getTensorInfo)(state, basis); \
  TensorInfo<int64_t> weightIndexInfo = thc_getTensorInfo_Long(state, weight_index); \
  TensorInfo<real> pseudoInfo = thc_(getTensorInfo)(state, pseudo); \
  int64_t *kernelSizeData = THCudaLongTensor_data(state, kernel_size); \
  uint8_t *isOpenSplineData = THCudaByteTensor_data(state, is_open_spline); \
\
  KERNEL_D_RUN(NAME, pseudoInfo.size[1], n, basisInfo, weightIndexInfo, pseudoInfo, kernelSizeData, isOpenSplineData, K) \
}

#define COMPUTE_SPLINE_BASIS_FORWARD(M, D, basis, weightIndex, pseudo, kernelSize, isOpenSpline, K, CODE) { \
  int64_t k = i % basis.size[1]; \
  int64_t pseudoOffset = ((i / basis.size[1]) % pseudo.size[0]) * pseudo.stride[0]; \
  int64_t d, k_mod, wi = 0, offset = K; Real b = 1, value; \
  for (d = 0; d < D; d++) { \
    offset /= kernelSize[d]; \
    k_mod = k % (M + 1); \
    k /= M + 1; \
    value = pseudo.data[pseudoOffset + d * pseudo.stride[1]] * (kernelSize[d] - M * isOpenSpline[d]); \
    wi += (((int64_t) value + k_mod) % kernelSize[d]) * offset; \
    value -= floor(value); \
    CODE \
    b *= value; \
  } \
  basis.data[i] = b; \
  weightIndex.data[i] = wi; \
}

template<typename Real, int D>
struct SplineBasisForward {
  static __device__ void linear(int i, const TensorInfo<Real>& basis, const TensorInfo<int64_t>& weightIndex, const TensorInfo<Real>& pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K) {
    COMPUTE_SPLINE_BASIS_FORWARD(1, D, basis, weightIndex, pseudo, kernelSize, isOpenSpline, K,
      value = 1 - value - k_mod + 2 * value * k_mod;
    )
  }
  static __device__ void quadratic(int i, const TensorInfo<Real>& basis, const TensorInfo<int64_t>& weightIndex, const TensorInfo<Real>& pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K) {
    COMPUTE_SPLINE_BASIS_FORWARD(2, D, basis, weightIndex, pseudo, kernelSize, isOpenSpline, K,
      if (k_mod == 0) value = 0.5 * value * value - value + 0.5;
      else if (k_mod == 1) value = -value * value + value + 0.5;
      else value = 0.5 * value * value;
    )
  }
  static __device__ void cubic(int i, const TensorInfo<Real>& basis, const TensorInfo<int64_t>& weightIndex, const TensorInfo<Real>& pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K) {
    COMPUTE_SPLINE_BASIS_FORWARD(3, D, basis, weightIndex, pseudo, kernelSize, isOpenSpline, K,
      if (k_mod == 0) { value = (1 - value); value = value * value * value / 6.0; }
      else if (k_mod == 1) value = (3 * value * value * value - 6 * value * value + 4) / 6;
      else if (k_mod == 2) value = (-3 * value * value * value + 3 * value * value + 3 * value + 1) / 6;
      else value = value * value * value / 6;
    )
  }
};

template<typename Real>
struct SplineBasisForward<Real, -1> {
  static __device__ void linear(int i, const TensorInfo<Real>& basis, const TensorInfo<int64_t>& weightIndex, const TensorInfo<Real>& pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K) {
    COMPUTE_SPLINE_BASIS_FORWARD(1, pseudo.size[1], basis, weightIndex, pseudo, kernelSize, isOpenSpline, K,
      value = 1 - value - k_mod + 2 * value * k_mod;
    )
  }
  static __device__ void quadratic(int i, const TensorInfo<Real>& basis, const TensorInfo<int64_t>& weightIndex, const TensorInfo<Real>& pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K) {
    COMPUTE_SPLINE_BASIS_FORWARD(2, pseudo.size[1], basis, weightIndex, pseudo, kernelSize, isOpenSpline, K,
      if (k_mod == 0) value = 0.5 * value * value - value + 0.5;
      else if (k_mod == 1) value = -value * value + value + 0.5;
      else value = 0.5 * value * value;
    )
  }
  static __device__ void cubic(int i, const TensorInfo<Real>& basis, const TensorInfo<int64_t>& weightIndex, const TensorInfo<Real>& pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K) {
    COMPUTE_SPLINE_BASIS_FORWARD(3, pseudo.size[1], basis, weightIndex, pseudo, kernelSize, isOpenSpline, K,
      if (k_mod == 0) { value = (1 - value); value = value * value * value / 6.0; }
      else if (k_mod == 1) value = (3 * value * value * value - 6 * value * value + 4) / 6;
      else if (k_mod == 2) value = (-3 * value * value * value + 3 * value * value + 3 * value + 1) / 6;
      else value = value * value * value / 6;
    )
  }
};

template<typename Real, int D>
__global__ void linearBasisForwardKernel(TensorInfo<Real> basis, TensorInfo<int64_t> weightIndex, TensorInfo<Real> pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K, int n) {
  KERNEL_LOOP(i, n) {
    SplineBasisForward<Real, D>::linear(i, basis, weightIndex, pseudo, kernelSize, isOpenSpline, K);
  }
}

template<typename Real, int D>
__global__ void quadraticBasisForwardKernel(TensorInfo<Real> basis, TensorInfo<int64_t> weightIndex, TensorInfo<Real> pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K, int n) {
  KERNEL_LOOP(i, n) {
    SplineBasisForward<Real, D>::quadratic(i, basis, weightIndex, pseudo, kernelSize, isOpenSpline, K);
  }
}

template<typename Real, int D>
__global__ void cubicBasisForwardKernel(TensorInfo<Real> basis, TensorInfo<int64_t> weightIndex, TensorInfo<Real> pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K, int n) {
  KERNEL_LOOP(i, n) {
    SplineBasisForward<Real, D>::cubic(i, basis, weightIndex, pseudo, kernelSize, isOpenSpline, K);
  }
}
