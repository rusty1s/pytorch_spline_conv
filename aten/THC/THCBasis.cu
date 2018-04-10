#include "THCBasis.h"

#include "THCBasisForward.cuh"
#include "THCBasisBackward.cuh"

template<typename T>
__global__ void linearBasisForwardKernel(TensorInfo<T> basis, TensorInfo<int64_t>weightIndex,
                                         TensorInfo<T> pseudo, int64_t *kernelSize,
                                         uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_FORWARD_KERNEL(1, basis, weightIndex, pseudo, kernelSize, isOpenSpline, n,
                                  BasisForward<T>::linear(v, kMod))
}

template<typename T>
__global__ void quadraticBasisForwardKernel(TensorInfo<T> basis, TensorInfo<int64_t>weightIndex,
                                            TensorInfo<T> pseudo, int64_t *kernelSize,
                                            uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_FORWARD_KERNEL(2, basis, weightIndex, pseudo, kernelSize, isOpenSpline, n,
                                  BasisForward<T>::quadratic(v, kMod))
}

template<typename T>
__global__ void cubicBasisForwardKernel(TensorInfo<T> basis, TensorInfo<int64_t>weightIndex,
                                        TensorInfo<T> pseudo, int64_t *kernelSize,
                                        uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_FORWARD_KERNEL(3, basis, weightIndex, pseudo, kernelSize, isOpenSpline, n,
                                  BasisForward<T>::cubic(v, kMod))
}

template<typename T>
__global__ void linearBasisBackwardKernel(TensorInfo<T> self, TensorInfo<T>gradBasis,
                                          TensorInfo<T> pseudo, int64_t *kernelSize,
                                          uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_BACKWARD_KERNEL(1, self, gradBasis, pseudo, kernelSize, isOpenSpline, n,
                                   BasisForward<T>::linear(v, kMod),
                                   BasisBackward<T>::linear(v, kMod))
}

template<typename T>
__global__ void quadraticBasisBackwardKernel(TensorInfo<T> self, TensorInfo<T>gradBasis,
                                             TensorInfo<T> pseudo, int64_t *kernelSize,
                                             uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_BACKWARD_KERNEL(2, self, gradBasis, pseudo, kernelSize, isOpenSpline, n,
                                   BasisForward<T>::quadratic(v, kMod),
                                   BasisBackward<T>::quadratic(v, kMod))
}

template<typename T>
__global__ void cubicBasisBackwardKernel(TensorInfo<T> self, TensorInfo<T>gradBasis,
                                         TensorInfo<T> pseudo, int64_t *kernelSize,
                                         uint8_t *isOpenSpline, ptrdiff_t n) {
  THC_TENSOR_BASIS_BACKWARD_KERNEL(3, self, gradBasis, pseudo, kernelSize, isOpenSpline, n,
                                   BasisForward<T>::cubic(v, kMod),
                                   BasisBackward<T>::cubic(v, kMod))
}

#include "generic/THCBasis.cu"
#include "THC/THCGenerateFloatTypes.h"
