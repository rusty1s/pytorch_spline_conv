#include "THCWeighting.h"

#include "common.cuh"
#include "THCNumerics.cuh"
#include "THCAtomics.cuh"

template<typename T>
__global__ void weightingForwardKernel(TensorInfo<T> self, TensorInfo<T> src, TensorInfo<T> weight,
                                       TensorInfo<T> basis, TensorInfo<int64_t> weightIndex,
                                       int n) {
  KERNEL_LOOP(i, n) {
    ptrdiff_t e = i / self.size[1], mOut = i % self.size[1], s, mIn;
    T v = ScalarConvert<int, T>::to(0), b, tmp;
    int64_t wi;
    for (s = 0; s < basis.size[1]; s++) {
      b = basis.data[e * basis.stride[0] + s * basis.stride[1]];
      wi = weightIndex.data[e * weightIndex.stride[0] + s * weightIndex.stride[1]];
      for (mIn = 0; mIn < src.size[1]; mIn++) {
        tmp = weight.data[wi * weight.stride[0] + mIn * weight.stride[1] + mOut * weight.stride[2]];
        tmp = THCNumerics<T>::mul(tmp, src.data[e * src.stride[0] + mIn * src.stride[1]]);
        tmp = THCNumerics<T>::mul(tmp, b);
        v = THCNumerics<T>::add(v, tmp);
      }
    }
    self.data[e * self.stride[0] + mOut * self.stride[1]] = v;
  }
}

template<typename T>
__global__ void weightingBackwardSrcKernel(TensorInfo<T> self, TensorInfo<T> gradOutput,
                                           TensorInfo<T> weight, TensorInfo<T> basis,
                                           TensorInfo<int64_t> weightIndex, int n) {
  KERNEL_LOOP(i, n) {
    ptrdiff_t e = i / gradOutput.size[1], mOut = i % gradOutput.size[1], s, mIn;
    T v, b, tmp;
    int64_t wi;
    T g = gradOutput.data[e * gradOutput.stride[0] + mOut * gradOutput.stride[1]];
    for (mIn = 0; mIn < self.size[1]; mIn++) {
      v = ScalarConvert<int, T>::to(0);
      for (s = 0; s < basis.size[1]; s++) {
        b = basis.data[e * basis.stride[0] + s * basis.stride[1]];
        wi = weightIndex.data[e * weightIndex.stride[0] + s * weightIndex.stride[1]];
        tmp = weight.data[wi * weight.stride[0] + mIn * weight.stride[1] + mOut * weight.stride[2]];
        tmp = THCNumerics<T>::mul(tmp, b);
        tmp = THCNumerics<T>::mul(tmp, g);
        v = THCNumerics<T>::add(v, tmp);
      }
      atomicAdd(&self.data[e * self.stride[0] + mIn * self.stride[1]], v);
    }
  }
}

template<typename T>
__global__ void weightingBackwardWeightKernel(TensorInfo<T> self, TensorInfo<T> gradOutput,
                                              TensorInfo<T> src, TensorInfo<T> basis,
                                              TensorInfo<int64_t> weightIndex, int n) {
  KERNEL_LOOP(i, n) {
    ptrdiff_t e = i / gradOutput.size[1], mOut = i % gradOutput.size[1], s, mIn;
    T b, v;
    int64_t wi;
    T g = gradOutput.data[e * gradOutput.stride[0] + mOut * gradOutput.stride[1]];
    for (s = 0; s < weightIndex.size[1]; s++) {
      b = basis.data[e * basis.stride[0] + s * basis.stride[1]];
      wi = weightIndex.data[e * weightIndex.stride[0] + s * weightIndex.stride[1]];
      for (mIn = 0; mIn < src.size[1]; mIn++) {
        v = src.data[e * src.stride[0] + mIn * src.stride[1]];
        v = THCNumerics<T>::mul(THCNumerics<T>::mul(v, b), g);
        atomicAdd(&self.data[wi * self.stride[0] + mIn * self.stride[1] + mOut * self.stride[2]], v);
      }
    }
  }
}

template<typename T>
__global__ void weightingBackwardBasisKernel(TensorInfo<T> self, TensorInfo<T> gradOutput,
                                             TensorInfo<T> src, TensorInfo<T> weight,
                                             TensorInfo<int64_t> weightIndex, int n) {
  KERNEL_LOOP(i, n) {
    ptrdiff_t e = i / gradOutput.size[1], mOut = i % gradOutput.size[1], s, mIn;
    T v, tmp;
    int64_t wi;
    T g = gradOutput.data[e * gradOutput.stride[0] + mOut * gradOutput.stride[1]];
    for (s = 0; s < weightIndex.size[1]; s++) {
      v = ScalarConvert<int, T>::to(0);
      wi = weightIndex.data[e * weightIndex.stride[0] + s * weightIndex.stride[1]];
      for (mIn = 0; mIn < src.size[1]; mIn++) {
        tmp = weight.data[wi * weight.stride[0] + mIn * weight.stride[1] + mOut * weight.stride[2]];
        tmp = THCNumerics<T>::mul(tmp, src.data[e * src.stride[0] + mIn * src.stride[1]]);
        tmp = THCNumerics<T>::mul(tmp, g);
        v = THCNumerics<T>::add(v, tmp);
      }
      atomicAdd(&self.data[e * self.stride[0] + s * self.stride[1]], v);
    }
  }
}

#include "generic/THCWeighting.cu"
#include "THC/THCGenerateFloatTypes.h"
