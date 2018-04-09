#include "THCBasis.h"

#include "THCBasisForward.cuh"

template<typename T>
__global__ void linearBasisForwardKernel(TensorInfo<T> basis, TensorInfo<int64_t>weightIndex,
                                         TensorInfo<T> pseudo, int64_t *kernelSize,
                                         uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_FORWARD_KERNEL(1, basis, weightIndex, pseudo, kernelSize, isOpenSpline, n,
    v = BasisForward<T>::linear(v, kMod);
  )
}

template<typename T>
__global__ void quadraticBasisForwardKernel(TensorInfo<T> basis, TensorInfo<int64_t>weightIndex,
                                            TensorInfo<T> pseudo, int64_t *kernelSize,
                                            uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_FORWARD_KERNEL(2, basis, weightIndex, pseudo, kernelSize, isOpenSpline, n,
    v = BasisForward<T>::quadratic(v, kMod);
  )
}

template<typename T>
__global__ void cubicBasisForwardKernel(TensorInfo<T> basis, TensorInfo<int64_t>weightIndex,
                                        TensorInfo<T> pseudo, int64_t *kernelSize,
                                        uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_FORWARD_KERNEL(3, basis, weightIndex, pseudo, kernelSize, isOpenSpline, n,
    v = BasisForward<T>::cubic(v, kMod);
  )
}

#include "generic/THCBasis.cu"
#include "THC/THCGenerateFloatTypes.h"
