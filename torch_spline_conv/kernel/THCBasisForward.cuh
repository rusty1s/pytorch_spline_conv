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
  KERNEL_RUN(NAME, pseudoInfo.size[1], n, basisInfo, weightIndexInfo, pseudoInfo, kernelSizeData, isOpenSplineData, K) \
}

template<typename Real, int M, int D>
struct SplineBasisForward {
  static __device__ void compute(int i, const TensorInfo<Real>& basis, const TensorInfo<int64_t>& weightIndex, const TensorInfo<Real>& pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K) {
    int64_t k = i % basis.size[1];
    int64_t pseudoOffset = ((i / basis.size[1]) % pseudo.size[0]) * pseudo.stride[0];
    int64_t d, k_mod, wi = 0, offset = K; Real b = 1, value;
    for (d = 0; d < D; d++) {
      offset /= kernelSize[d];
      k_mod = k % (M + 1);
      k /= M + 1;
      value = pseudo.data[pseudoOffset + d * pseudo.stride[1]] * (kernelSize[d] - M * isOpenSpline[d]);
      wi += (((int64_t) value + k_mod) % kernelSize[d]) * offset;
      value -= floor(value);
      value = 1 - value - k_mod + 2 * value * k_mod;
      b *= value;
    }
    basis.data[i] = b;
    weightIndex.data[i] = wi;
  }
};

template<typename Real, int M>
struct SplineBasisForward<Real, M, -1> {
  static __device__ void compute(int i, const TensorInfo<Real>& basis, const TensorInfo<int64_t>& weightIndex, const TensorInfo<Real>& pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K) {
    int64_t k = i % basis.size[1];
    int64_t pseudoOffset = ((i / basis.size[1]) % pseudo.size[0]) * pseudo.stride[0];
    int64_t d, k_mod, wi = 0, offset = K; Real b = 1, value;
    for (d = 0; d < pseudo.size[1]; d++) {
      offset /= kernelSize[d];
      k_mod = k % (M + 1);
      k /= M + 1;
      value = pseudo.data[pseudoOffset + d * pseudo.stride[1]] * (kernelSize[d] - M * isOpenSpline[d]);
      wi += (((int64_t) value + k_mod) % kernelSize[d]) * offset;
      value -= floor(value);
      value = 1 - value - k_mod + 2 * value * k_mod;
      b *= value;
    }
    basis.data[i] = b;
    weightIndex.data[i] = wi;
  }
};

template<typename Real, int D>
__global__ void linearBasisForwardKernel(TensorInfo<Real> basis, TensorInfo<int64_t> weightIndex, TensorInfo<Real> pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K, int n) {
  KERNEL_LOOP(i, n) {
    SplineBasisForward<Real, 1, D>::compute(i, basis, weightIndex, pseudo, kernelSize, isOpenSpline, K);
  }
}

template<typename Real, int D>
__global__ void quadraticBasisForwardKernel(TensorInfo<Real> basis, TensorInfo<int64_t> weightIndex, TensorInfo<Real> pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K, int n) {
  KERNEL_LOOP(i, n) {
    SplineBasisForward<Real, 2, D>::compute(i, basis, weightIndex, pseudo, kernelSize, isOpenSpline, K);
  }
}

template<typename Real, int D>
__global__ void cubicBasisForwardKernel(TensorInfo<Real> basis, TensorInfo<int64_t> weightIndex, TensorInfo<Real> pseudo, int64_t *kernelSize, uint8_t *isOpenSpline, int K, int n) {
  KERNEL_LOOP(i, n) {
    SplineBasisForward<Real, 3, D>::compute(i, basis, weightIndex, pseudo, kernelSize, isOpenSpline, K);
  }
}
