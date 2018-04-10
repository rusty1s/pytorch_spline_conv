#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THWeighting.c"
#else

void THTensor_(weightingForward)(THTensor *self, THTensor *src, THTensor *weight, THTensor *basis,
                                  THLongTensor *weightIndex) {
  real *selfData = THTensor_(data)(self);
  real *srcData = THTensor_(data)(src);
  real *weightData = THTensor_(data)(weight);
  real *basisData = THTensor_(data)(basis);
  int64_t *weightIndexData = THLongTensor_data(weightIndex);

  ptrdiff_t e, mOut, s, mIn;
  real v, b;
  int64_t wi;
  for (e = 0; e < THTensor_(size)(src, 0); e++) {
    for (mOut = 0; mOut < THTensor_(size)(weight, 2); mOut++) {
      v = 0;
      for (s = 0; s < THTensor_(size)(basis, 1); s++) {
        b = basisData[e * basis->stride[0] + s * basis->stride[1]];
        wi = weightIndexData[e * weightIndex->stride[0] + s * weightIndex->stride[1]];
        for (mIn = 0; mIn < THTensor_(size)(weight, 1); mIn++) {
          v += b * weightData[wi * weight->stride[0] + mIn * weight->stride[1] + mOut * weight->stride[2]] * srcData[e * src->stride[0] + mIn * src->stride[1]];
        }
      }
      selfData[e * self->stride[0] + mOut * self->stride[1]] = v;
    }
  }
}

void THTensor_(weightingBackwardSrc)(THTensor *self, THTensor *gradOutput, THTensor *weight,
                                      THTensor *basis, THLongTensor *weightIndex) {
}

void THTensor_(weightingBackwardWeight)(THTensor *self, THTensor *gradOutput, THTensor *src,
                                         THTensor *basis, THLongTensor *weightIndex) {
}

void THTensor_(weightingBackwardBasis)(THTensor *self, THTensor *gradOutput, THTensor *src,
                                        THTensor *weight, THLongTensor *weightIndex) {
}

#endif // TH_GENERIC_FILE
