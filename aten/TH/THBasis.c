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
  ptrdiff_t e, d, s; \
  int64_t k, kMod, wi, wiOffset; \
  real b, v; \
  real g; \
\
  for (e = 0; e < THTensor_(size)(pseudo, 0); e++) { \
    for (d = 0; d < THTensor_(size)(pseudo, 1); d++) { \
      real g_out = 0; \
      int64_t quotient = pow(M + 1, d); \
      for (s = 0; s < THTensor_(size)(gradBasis, 1); s++) { \
        kMod = (s / quotient) % (M + 1); \
        v = pseudoData[e * pseudo->stride[0] + d * pseudo->stride[1]]; \
        v *= kernelSizeData[d] - M * isOpenSplineData[d]; \
        v -= floor(v); \
        v = GRAD_CODE; \
        g = v; \
\
        ptrdiff_t d_it; \
        for (d_it = 0; d_it < d; d_it++) { \
          int64_t quotient2 = pow(M + 1, d_it); \
          kMod = (s / quotient2) % (M + 1); \
          v = pseudoData[e * pseudo->stride[0] + d_it * pseudo->stride[1]]; \
          v *= kernelSizeData[d_it] - M * isOpenSplineData[d_it]; \
          v -= floor(v); \
          v = CODE; \
          g *= v; \
        } \
        for (d_it = d + 1; d_it < THTensor_(size)(pseudo, 1); d_it++) { \
          int64_t quotient2 = pow(M + 1, d_it); \
          kMod = (s / quotient2) % (M + 1); \
          v = pseudoData[e * pseudo->stride[0] + d_it * pseudo->stride[1]]; \
          v *= kernelSizeData[d_it] - M * isOpenSplineData[d_it]; \
          v -= floor(v); \
          v = CODE; \
          g *= v; \
        } \
        g_out += g * gradBasisData[e * gradBasis->stride[0] + s * gradBasis->stride[1]]; \
      } \
      g_out *= kernelSizeData[d] - M * isOpenSplineData[d]; \
      selfData[e * self->stride[0] + d * self->stride[1]] = g_out; \
    } \
  } \
}

#include "generic/THBasis.c"
#include "THGenerateFloatTypes.h"
