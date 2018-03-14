#include <THC.h>

#include "kernel.h"

#include "common.cuh"
#include "THCBasisForward.cuh"

#define spline_(NAME) TH_CONCAT_4(spline_, NAME, _kernel_, Real)
#define thc_(NAME) TH_CONCAT_4(thc_, NAME, _, Real)

#include "generic/common.cu"
#include "THCGenerateAllTypes.h"

template<typename Real>
__global__ void weightingForwardKernel(TensorInfo<Real> output, TensorInfo<Real> input, TensorInfo<Real> weight, TensorInfo<Real> basis, TensorInfo<int64_t> weightIndex, int n) {
  KERNEL_LOOP(i, n) {
    int64_t edgeOffset = i / output.size[1], inputOffset = edgeOffset * input.stride[0];
    int64_t s, S = basis.size[1], m_in, M_in = input.size[1], m_out = i % output.size[1], M_out = output.size[1], weightOffset;
    Real b, value = 0;
    for (s = 0; s < S; s++) {
      b = basis.data[edgeOffset + s];
      weightOffset = weightIndex.data[edgeOffset * S + s] * M_in * M_out + m_out;
      for (m_in = 0; m_in < M_in; m_in++) {
        value += b * weight.data[weightOffset + m_in * M_out] * input.data[inputOffset + m_in * input.stride[1]];
      }
    }
    output.data[i] = value;
  }
}

#include "generic/kernel.cu"
#include "THCGenerateFloatType.h"
#include "generic/kernel.cu"
#include "THCGenerateDoubleType.h"
