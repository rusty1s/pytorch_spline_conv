#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBasis.c"
#else

inline real THTensor_(linear)(real v, int64_t kMod) {
  return 1 - v - kMod + 2 * v * kMod;
}

inline real THTensor_(quadratic)(real v, int64_t kMod) {
  if (kMod == 0) return 0.5 * v * v - v + 0.5;
  else if (kMod == 1) return -v * v + v + 0.5;
  else return 0.5 * v * v;
}

inline real THTensor_(cubic)(real v, int64_t kMod) {
  if (kMod == 0) { v = (1 - v); return v * v * v / 6.0; }
  else if (kMod == 1) return (3 * v * v * v - 6 * v * v + 4) / 6;
  else if (kMod == 2) return (-3 * v * v * v + 3 * v * v + 3 * v + 1) / 6;
  else return v * v * v / 6;
}

void THTensor_(linearBasisForward)(THTensor *basis, THLongTensor *weightIndex, THTensor *pseudo,
                                   THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_FORWARD(1, basis, weightIndex, pseudo, kernelSize, isOpenSpline,
    v = THTensor_(linear)(v, kMod);
  )
}

void THTensor_(quadraticBasisForward)(THTensor *basis, THLongTensor *weightIndex, THTensor *pseudo,
                                      THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_FORWARD(2, basis, weightIndex, pseudo, kernelSize, isOpenSpline,
    v = THTensor_(quadratic)(v, kMod);
  )
}

void THTensor_(cubicBasisForward)(THTensor *basis, THLongTensor *weightIndex, THTensor *pseudo,
                                  THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_FORWARD(3, basis, weightIndex, pseudo, kernelSize, isOpenSpline,
    v = THTensor_(cubic)(v, kMod);
  )
}

void THTensor_(linearBasisBackward)(THTensor *self, THTensor *gradBasis, THTensor *pseudo,
                                    THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  THTensor_(fill)(self, 0);

  real *selfData = THTensor_(data)(self);
  real *gradBasisData = THTensor_(data)(gradBasis);
  real *pseudoData = THTensor_(data)(pseudo);
  int64_t *kernelSizeData = THLongTensor_data(kernelSize);
  uint8_t *isOpenSplineData = THByteTensor_data(isOpenSpline);

  ptrdiff_t e, d, s;
  int64_t k, kMod, wi, wiOffset;
  real b, v;
  real g;

  for (e = 0; e < THTensor_(size)(pseudo, 0); e++) {
    for (d = 0; d < THTensor_(size)(pseudo, 1); d++) {
      int64_t quotient = pow(2, d);
      /* printf("e = %i, d = %i, stride0 = %i, stride1 = %i \n", e, d, pseudo->stride[0], pseudo->stride[1]); */
      for (s = 0; s < THTensor_(size)(gradBasis, 1); s++) {
        kMod = (s / quotient) % 2;
        real v = pseudoData[e * pseudo->stride[0] + d * pseudo->stride[1]];
        v *= kernelSizeData[d] - 2 * isOpenSplineData[d];
        v -= floor(v);
        v = -1 + kMod + kMod;  // grad code
        g = v;

        ptrdiff_t d_it;
        for (d_it = 0; d_it < d; d_it++) {
        }
      }
    }
    /* selfData[e * self->stride[0] + d * self->stride[1]] = 1; */
  }
}

void THTensor_(quadraticBasisBackward)(THTensor *self, THTensor *gradBasis, THTensor *pseudo,
                                       THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
}

void THTensor_(cubicBasisBackward)(THTensor *self, THTensor *gradBasis, THTensor *pseudo,
                                   THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
}

#endif // TH_GENERIC_FILE
