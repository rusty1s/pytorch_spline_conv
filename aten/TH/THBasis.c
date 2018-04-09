#include <TH/TH.h>

#define TH_TENSOR_BASIS_FORWARD(M, basis, weightIndex, pseudo, kernelSize, isOpenSpline, CODE) { \
  real *basisData = THTensor_(data)(basis); \
  int64_t *weightIndexData = THLongTensor_data(weightIndex); \
  real *pseudoData = THTensor_(data)(pseudo); \
  int64_t *kernelSizeData = THLongTensor_data(kernelSize); \
  uint8_t *isOpenSplineData = THByteTensor_data(isOpenSpline); \
\
  ptrdiff_t e, s, d; \
  int64_t k, kMod, wi, wiOffset; \
  real b, v; \
  for (e = 0; e < THTensor_(size)(pseudo, 0); e++) { \
    for (s = 0; s < THTensor_(size)(basis, 1); s++) { \
      k = s; b = 1; wi = 0; wiOffset = 1; \
      for (d = 0; d < THTensor_(size)(pseudo, 1); d++) { \
        kMod = k % (M + 1); \
        k /= M + 1; \
\
        v = pseudoData[e * pseudo->stride[0] + d * pseudo->stride[1]]; \
        v *= kernelSizeData[d] - M * isOpenSplineData[d]; \
\
        wi += (((int64_t) v + kMod) % kernelSizeData[d]) * wiOffset; \
        wiOffset *= kernelSizeData[d]; \
\
        v -= floor(v); \
        CODE \
        b *= v; \
      } \
      basisData[e * basis->stride[0] + s * basis->stride[1]] = b; \
      weightIndexData[e * weightIndex->stride[0] + s * weightIndex->stride[1]] = wi; \
    } \
  } \
}

#include "generic/THBasis.c"
#include "THGenerateFloatTypes.h"
