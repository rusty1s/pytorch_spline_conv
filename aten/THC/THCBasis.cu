#include "THCBasis.h"

#include "common.cuh"
#include "THCNumerics.cuh"

#define THC_TENSOR_BASIS_FORWARD(NAME, state, basis, weightIndex, pseudo, kernelSize, \
                                 isOpenSpline) { \
  THCAssertSameGPU( \
    THCTensor_(checkGPU)(state, 5, basis, weightIndex, pseudo, kernelSize, isOpenSpline)); \
\
  TensorInfo<real> basisInfo = THCTensor_(getTensorInfo)(state, basis); \
  TensorInfo<int64_t> weightIndexInfo = THCudaLongTensor_getTensorInfo(state, weightIndex); \
  TensorInfo<real> pseudoInfo = THCTensor_(getTensorInfo)(state, pseudo); \
  int64_t *kernelSizeData = THCudaLongTensor_data(state, kernelSize); \
  uint8_t *isOpenSplineData = THCudaByteTensor_data(state, isOpenSpline); \
\
  KERNEL_REAL_RUN(NAME, THCTensor_(nElement)(state, basis), basisInfo, \
                  weightIndexInfo, pseudoInfo, kernelSizeData, isOpenSplineData); \
}

#define THC_TENSOR_BASIS_FORWARD_KERNEL(M, basis, weightIndex, pseudo, kernelSize, isOpenSpline, \
                                        N, CODE) { \
  KERNEL_LOOP(i, N) { \
    ptrdiff_t e = i / basis.size[1], s = i % basis.size[1], d; \
    int64_t k = s, kMod, wi = 0, wiOffset = 1; \
    T b = ScalarConvert<int, T>::to(1), v; \
\
    for (d = 0; d < pseudo.size[1]; d++) { \
      kMod = k % (M + 1); \
      k /= M + 1; \
\
      v = pseudo.data[e * pseudo.stride[0] + d * pseudo.stride[1]]; \
      v = THCNumerics<T>::mul(v, ScalarConvert<int64_t, T>::to(kernelSize[d] - M * isOpenSpline[d])); \
\
      wi += ((ScalarConvert<T, int64_t>::to(v) + kMod) % kernelSize[d]) * wiOffset; \
      wiOffset *= kernelSize[d]; \
\
      v = THCNumerics<T>::sub(v, ScalarConvert<int64_t, T>::to(ScalarConvert<T, int64_t>::to(v))); \
      CODE \
      b = THCNumerics<T>::mul(b, v); \
    } \
\
    basis.data[e * basis.stride[0] + s * basis.stride[1]] = b; \
    weightIndex.data[e * weightIndex.stride[0] + s * weightIndex.stride[1]] = wi; \
  } \
}

template<typename T>
__global__ void linearBasisForwardKernel(TensorInfo<T> basis, TensorInfo<int64_t>weightIndex,
                                         TensorInfo<T> pseudo, int64_t *kernelSize,
                                         uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_FORWARD_KERNEL(1, basis, weightIndex, pseudo, kernelSize, isOpenSpline, n,
      // 1 - v - kMod + 2 * v * kMod
      T tmp1 = THCNumerics<T>::sub(ScalarConvert<int, T>::to(1), v);
      tmp1 = THCNumerics<T>::sub(tmp1, ScalarConvert<int64_t, T>::to(kMod));
      T tmp2 = THCNumerics<T>::mul(ScalarConvert<int, T>::to(2), v);
      tmp2 = THCNumerics<T>::mul(tmp2, ScalarConvert<int64_t, T>::to(kMod));
      v = THCNumerics<T>::add(tmp1, tmp2);
  )
}

template<typename T>
__global__ void quadraticBasisForwardKernel(TensorInfo<T> basis, TensorInfo<int64_t>weightIndex,
                                            TensorInfo<T> pseudo, int64_t *kernelSize,
                                            uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_FORWARD_KERNEL(2, basis, weightIndex, pseudo, kernelSize, isOpenSpline, n,
    /* printf("DRIN"); */
  )
}

template<typename T>
__global__ void cubicBasisForwardKernel(TensorInfo<T> basis, TensorInfo<int64_t>weightIndex,
                                        TensorInfo<T> pseudo, int64_t *kernelSize,
                                        uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_FORWARD_KERNEL(3, basis, weightIndex, pseudo, kernelSize, isOpenSpline, n,
    /* printf("DRIN"); */
  )
}

#include "generic/THCBasis.cu"
#include "THC/THCGenerateFloatTypes.h"
