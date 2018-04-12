#ifndef THC_BASIS_FORWARD_INC
#define THC_BASIS_FORWARD_INC

#include "common.cuh"
#include "THCNumerics.cuh"

#define THC_TENSOR_BASIS_FORWARD(NAME, state, basis, weightIndex, pseudo, kernelSize, \
                                 isOpenSpline) { \
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 5, basis, weightIndex, pseudo, kernelSize, \
                                        isOpenSpline)); \
\
  TensorInfo<real> basisInfo = THCTensor_(getTensorInfo)(state, basis); \
  TensorInfo<int64_t> weightIndexInfo = THCudaLongTensor_getTensorInfo(state, weightIndex); \
  TensorInfo<real> pseudoInfo = THCTensor_(getTensorInfo)(state, pseudo); \
  int64_t *kernelSizeData = THCudaLongTensor_data(state, kernelSize); \
  uint8_t *isOpenSplineData = THCudaByteTensor_data(state, isOpenSpline); \
\
  KERNEL_REAL_RUN(NAME, THCTensor_(nElement)(state, basis), basisInfo, weightIndexInfo, \
                  pseudoInfo, kernelSizeData, isOpenSplineData); \
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
      v = CODE; \
      b = THCNumerics<T>::mul(b, v); \
    } \
\
    basis.data[e * basis.stride[0] + s * basis.stride[1]] = b; \
    weightIndex.data[e * weightIndex.stride[0] + s * weightIndex.stride[1]] = wi; \
  } \
}

template<typename T>
struct BasisForward {
  static inline __device__ T linear(T v, int64_t kMod) {
      // 1 - v - kMod + 2 * v * kMod
      T tmp1 = THCNumerics<T>::sub(ScalarConvert<int, T>::to(1), v);
      tmp1 = THCNumerics<T>::sub(tmp1, ScalarConvert<int64_t, T>::to(kMod));
      T tmp2 = THCNumerics<T>::mul(ScalarConvert<int, T>::to(2), v);
      tmp2 = THCNumerics<T>::mul(tmp2, ScalarConvert<int64_t, T>::to(kMod));
      return THCNumerics<T>::add(tmp1, tmp2);
  }

  static inline __device__ T quadratic(T v, int64_t kMod) {
    if (kMod == 0) {
      // 0.5 * v * v - v + 0.5
      T tmp = THCNumerics<T>::mul(THCNumerics<T>::mul(ScalarConvert<float, T>::to(0.5), v), v);
      return THCNumerics<T>::add(THCNumerics<T>::sub(tmp, v), ScalarConvert<float, T>::to(0.5));
    }
    else if (kMod == 1) {
      // -v * v + v + 0.5
      T tmp = THCNumerics<T>::mul(THCNumerics<T>::neg(v), v);
      return THCNumerics<T>::add(THCNumerics<T>::add(tmp, v), ScalarConvert<float, T>::to(0.5));
    }
    else {
      // 0.5 * v * v
      return THCNumerics<T>::mul(ScalarConvert<float, T>::to(0.5), THCNumerics<T>::mul(v, v));
    }
  }

  static inline __device__ T cubic(T v, int64_t kMod) {
    if (kMod == 0) {
      // (1 - v) * (1 -v) * (1 - v) / 6
      T tmp = THCNumerics<T>::sub(ScalarConvert<int, T>::to(1), v);
      tmp = THCNumerics<T>::mul(THCNumerics<T>::mul(tmp, tmp), tmp);
      return THCNumerics<T>::div(tmp, ScalarConvert<int, T>::to(6));
    }
    else if (kMod == 1) {
      // (3 * v * v * v - 6 * v * v + 4) / 6
      T tmp1 = THCNumerics<T>::mul(THCNumerics<T>::mul(v, v), v);
      tmp1 = THCNumerics<T>::mul(ScalarConvert<int, T>::to(3), tmp1);
      T tmp2 = THCNumerics<T>::mul(ScalarConvert<int, T>::to(6), THCNumerics<T>::mul(v, v));
      tmp1 = THCNumerics<T>::add(THCNumerics<T>::sub(tmp1, tmp2), ScalarConvert<int, T>::to(4));
      return THCNumerics<T>::div(tmp1, ScalarConvert<int, T>::to(6));
    }
    else if (kMod == 2) {
      // (-3 * v * v * v + 3 * v * v + 3 * v + 1) / 6
      T tmp1 = THCNumerics<T>::mul(THCNumerics<T>::mul(v, v), v);
      tmp1 = THCNumerics<T>::mul(ScalarConvert<int, T>::to(-3), tmp1);
      T tmp2 = THCNumerics<T>::mul(ScalarConvert<int, T>::to(3), THCNumerics<T>::mul(v, v));
      T tmp3 = THCNumerics<T>::mul(ScalarConvert<int, T>::to(3), v);
      tmp1 = THCNumerics<T>::add(THCNumerics<T>::add(tmp1, tmp2), tmp3);
      tmp1 = THCNumerics<T>::add(tmp1, ScalarConvert<int, T>::to(1));
      return THCNumerics<T>::div(tmp1, ScalarConvert<int, T>::to(6));
    }
    else {
      // v * v * v / 6
      T tmp = THCNumerics<T>::mul(THCNumerics<T>::mul(v, v), v);
      return THCNumerics<T>::div(tmp, ScalarConvert<int, T>::to(6));
    }
  }
};

#endif // THC_BASIS_FORWARD_INC
