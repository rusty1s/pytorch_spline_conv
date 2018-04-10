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
        v = CODE; \
        b *= v; \
      } \
      basisData[e * basis->stride[0] + s * basis->stride[1]] = b; \
      weightIndexData[e * weightIndex->stride[0] + s * weightIndex->stride[1]] = wi; \
    } \
  } \
}

#define TH_TENSOR_BASIS_BACKWARD(M, self, gradBasis, pseudo, kernelSize, isOpenSpline, CODE, \
                                 GRAD_CODE) { \
  real *selfData = THTensor_(data)(self); \
  real *gradBasisData = THTensor_(data)(gradBasis); \
  real *pseudoData = THTensor_(data)(pseudo); \
  int64_t *kernelSizeData = THLongTensor_data(kernelSize); \
  uint8_t *isOpenSplineData = THByteTensor_data(isOpenSpline); \
\
  ptrdiff_t e, d, s, dIt, dOther; \
  int64_t kMod; \
  real g, v, tmp; \
\
  for (e = 0; e < THTensor_(size)(pseudo, 0); e++) { \
    for (d = 0; d < THTensor_(size)(pseudo, 1); d++) { \
      g = 0; \
      for (s = 0; s < THTensor_(size)(gradBasis, 1); s++) { \
        kMod = (s / (ptrdiff_t) pow(M + 1, d)) % (M + 1); \
        v = pseudoData[e * pseudo->stride[0] + d * pseudo->stride[1]]; \
        v *= kernelSizeData[d] - M * isOpenSplineData[d]; \
        v -= floor(v); \
        v = GRAD_CODE; \
        tmp = v; \
\
        for (dIt = 1; dIt < THTensor_(size)(pseudo, 1); dIt++) { \
          dOther = dIt - (d >= dIt); \
          kMod = (s / (ptrdiff_t) pow(M + 1, dOther)) % (M + 1); \
          v = pseudoData[e * pseudo->stride[0] + dOther * pseudo->stride[1]]; \
          v *= kernelSizeData[dOther] - M * isOpenSplineData[dOther]; \
          v -= floor(v); \
          v = CODE; \
          tmp *= v; \
        } \
\
        g += tmp * gradBasisData[e * gradBasis->stride[0] + s * gradBasis->stride[1]]; \
      } \
      g *= kernelSizeData[d] - M * isOpenSplineData[d]; \
      selfData[e * self->stride[0] + d * self->stride[1]] = g; \
    } \
  } \
}

#include "generic/THBasis.c"
#include "THGenerateFloatTypes.h"
