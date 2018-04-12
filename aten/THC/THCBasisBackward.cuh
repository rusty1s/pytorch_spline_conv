#ifndef THC_BASIS_BACKWARD_INC
#define THC_BASIS_BACKWARD_INC

#include "common.cuh"
#include "THCNumerics.cuh"

#define THC_TENSOR_BASIS_BACKWARD(NAME, state, self, gradBasis, pseudo, kernelSize, \
                                  isOpenSpline) { \
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 5, self, gradBasis, pseudo, kernelSize, \
                                        isOpenSpline)); \
\
  TensorInfo<real> selfInfo = THCTensor_(getTensorInfo)(state, self); \
  TensorInfo<real> gradBasisInfo = THCTensor_(getTensorInfo)(state, gradBasis); \
  TensorInfo<real> pseudoInfo = THCTensor_(getTensorInfo)(state, pseudo); \
  int64_t *kernelSizeData = THCudaLongTensor_data(state, kernelSize); \
  uint8_t *isOpenSplineData = THCudaByteTensor_data(state, isOpenSpline); \
\
  KERNEL_REAL_RUN(NAME, THCTensor_(nElement)(state, pseudo), selfInfo, gradBasisInfo, pseudoInfo, \
                  kernelSizeData, isOpenSplineData); \
}

#define THC_TENSOR_BASIS_BACKWARD_KERNEL(M, self, gradBasis, pseudo, kernelSize, isOpenSpline, \
                                         N, CODE, GRAD_CODE) { \
  KERNEL_LOOP(i, N) { \
    ptrdiff_t e = i / self.size[1], d = i % self.size[1], s, dIt, dOther; \
    int64_t kMod; \
    T g = ScalarConvert<int, T>::to(0), v, tmp; \
    for (s = 0; s < gradBasis.size[1]; s++) { \
      kMod = (s / (ptrdiff_t) pow((float) M + 1, (float) d)) % (M + 1); \
      v = pseudo.data[e * pseudo.stride[0] + d * pseudo.stride[1]]; \
      v = THCNumerics<T>::mul(v, ScalarConvert<int64_t, T>::to(kernelSize[d] - M * isOpenSpline[d])); \
      v = THCNumerics<T>::sub(v, ScalarConvert<int64_t, T>::to(ScalarConvert<T, int64_t>::to(v))); \
      v = GRAD_CODE; \
      tmp = v; \
\
      for (dIt = 1; dIt < pseudo.size[1]; dIt++) { \
        dOther = dIt - (d >= dIt); \
        kMod = (s / (ptrdiff_t) pow((float) M + 1, (float) dOther)) % (M + 1); \
        v = pseudo.data[e * pseudo.stride[0] + dOther * pseudo.stride[1]]; \
        v = THCNumerics<T>::mul(v, ScalarConvert<int64_t, T>::to(kernelSize[dOther] - M * isOpenSpline[dOther])); \
        v = THCNumerics<T>::sub(v, ScalarConvert<int64_t, T>::to(ScalarConvert<T, int64_t>::to(v))); \
        v = CODE; \
        tmp = THCNumerics<T>::mul(tmp, v); \
      } \
\
      tmp = THCNumerics<T>::mul(tmp, gradBasis.data[e * gradBasis.stride[0] + s * gradBasis.stride[1]]); \
      g = THCNumerics<T>::add(g, tmp); \
    } \
    g = THCNumerics<T>::mul(g, ScalarConvert<int64_t, T>::to(kernelSize[d] - M * isOpenSpline[d])); \
    self.data[e * self.stride[0] + d * self.stride[1]] = g; \
  } \
}

template<typename T>
struct BasisBackward {
  static inline __device__ T linear(T v, int64_t kMod) {
      // 2 * kMod - 1
      return ScalarConvert<int64_t, T>::to(2 * kMod - 1);
  }

  static inline __device__ T quadratic(T v, int64_t kMod) {
    if (kMod == 0) {
      // v - 1
      return THCNumerics<T>::sub(v, ScalarConvert<int, T>::to(1));
    }
    else if (kMod == 1) {
      // -2 * v + 1
      T tmp = THCNumerics<T>::mul(ScalarConvert<int, T>::to(-2), v);
      return THCNumerics<T>::add(tmp, ScalarConvert<int, T>::to(1));
    }
    else return v;
  }

  static inline __device__ T cubic(T v, int64_t kMod) {
    if (kMod == 0) {
      // (-v * v + 2 * v - 1) / 2
      T tmp1 = THCNumerics<T>::mul(THCNumerics<T>::neg(v), v);
      T tmp2 = THCNumerics<T>::mul(ScalarConvert<int, T>::to(2), v);
      tmp1 = THCNumerics<T>::sub(THCNumerics<T>::add(tmp1, tmp2), ScalarConvert<int, T>::to(1));
      return THCNumerics<T>::div(tmp1, ScalarConvert<int, T>::to(2));
    }
    else if (kMod == 1) {
      // (3 * v * v - 4 * v) / 2
      T tmp = THCNumerics<T>::mul(ScalarConvert<int, T>::to(3), THCNumerics<T>::mul(v, v));
      tmp = THCNumerics<T>::sub(tmp, THCNumerics<T>::mul(ScalarConvert<int, T>::to(4), v));
      return THCNumerics<T>::div(tmp, ScalarConvert<int, T>::to(2));
    }
    else if (kMod == 2) {
      T tmp = THCNumerics<T>::mul(ScalarConvert<int, T>::to(-3), THCNumerics<T>::mul(v, v));
      tmp = THCNumerics<T>::add(tmp, THCNumerics<T>::mul(ScalarConvert<int, T>::to(2), v));
      tmp = THCNumerics<T>::add(tmp, ScalarConvert<int, T>::to(1));
      return THCNumerics<T>::div(tmp, ScalarConvert<int, T>::to(2));
    }
    else {
      // v * v / 2;
      return THCNumerics<T>::div(THCNumerics<T>::mul(v, v), ScalarConvert<int, T>::to(2));
    }
  }
};

#endif // THC_BASIS_BACKWARD_INC
