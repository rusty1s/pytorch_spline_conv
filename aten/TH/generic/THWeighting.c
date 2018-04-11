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
  real v, b, tmp;
  int64_t wi;
  for (e = 0; e < THTensor_(size)(src, 0); e++) {
    for (mOut = 0; mOut < THTensor_(size)(self, 1); mOut++) {
      v = 0;
      for (s = 0; s < THTensor_(size)(basis, 1); s++) {
        b = basisData[e * basis->stride[0] + s * basis->stride[1]];
        wi = weightIndexData[e * weightIndex->stride[0] + s * weightIndex->stride[1]];
        for (mIn = 0; mIn < THTensor_(size)(src, 1); mIn++) {
          tmp = weightData[wi * weight->stride[0] + mIn * weight->stride[1] + mOut * weight->stride[2]];
          tmp *= b * srcData[e * src->stride[0] + mIn * src->stride[1]];
          v += tmp;
        }
      }
      selfData[e * self->stride[0] + mOut * self->stride[1]] = v;
    }
  }
}

void THTensor_(weightingBackwardSrc)(THTensor *self, THTensor *gradOutput, THTensor *weight,
                                      THTensor *basis, THLongTensor *weightIndex) {
  THTensor_(fill)(self, 0);

  real *selfData = THTensor_(data)(self);
  real *gradOutputData = THTensor_(data)(gradOutput);
  real *weightData = THTensor_(data)(weight);
  real *basisData = THTensor_(data)(basis);
  int64_t *weightIndexData = THLongTensor_data(weightIndex);

  ptrdiff_t e, mOut, s, mIn;
  real g, b, v;
  int64_t wi;
  for (e = 0; e < THTensor_(size)(self, 0); e++) {
    for (mOut = 0; mOut < THTensor_(size)(gradOutput, 1); mOut++) {
      g = gradOutputData[e * gradOutput->stride[0] + mOut * gradOutput->stride[1]];
      for (s = 0; s < THTensor_(size)(basis, 1); s++) {
        b = basisData[e * basis->stride[0] + s * basis->stride[1]];
        wi = weightIndexData[e * weightIndex->stride[0] + s * weightIndex->stride[1]];
        for (mIn = 0; mIn < THTensor_(size)(self, 1); mIn++) {
          v = weightData[wi * weight->stride[0] + mIn * weight->stride[1] + mOut * weight->stride[2]];
          selfData[e * self->stride[0] + mIn * self->stride[1]] += g * b * v;
        }
      }
    }
  }
}

void THTensor_(weightingBackwardWeight)(THTensor *self, THTensor *gradOutput, THTensor *src,
                                         THTensor *basis, THLongTensor *weightIndex) {
  THTensor_(fill)(self, 0);

  real *selfData = THTensor_(data)(self);
  real *gradOutputData = THTensor_(data)(gradOutput);
  real *srcData = THTensor_(data)(src);
  real *basisData = THTensor_(data)(basis);
  int64_t *weightIndexData = THLongTensor_data(weightIndex);

  ptrdiff_t e, mOut, s, mIn;
  real g, b, v;
  int64_t wi;
  for (e = 0; e < THTensor_(size)(src, 0); e++) {
    for (mOut = 0; mOut < THTensor_(size)(gradOutput, 1); mOut++) {
      g = gradOutputData[e * gradOutput->stride[0] + mOut * gradOutput->stride[1]];
      for (s = 0; s < THTensor_(size)(basis, 1); s++) {
        b = basisData[e * basis->stride[0] + s * basis->stride[1]];
        wi = weightIndexData[e * weightIndex->stride[0] + s * weightIndex->stride[1]];
        for (mIn = 0; mIn < THTensor_(size)(src, 1); mIn++) {
          v = b * g * srcData[e * src->stride[0] + mIn * src->stride[1]];
          selfData[wi * self->stride[0] + mIn * self->stride[1] + mOut * self->stride[2]] += v;
        }
      }
    }
  }
}

void THTensor_(weightingBackwardBasis)(THTensor *self, THTensor *gradOutput, THTensor *src,
                                        THTensor *weight, THLongTensor *weightIndex) {
  THTensor_(fill)(self, 0);

  real *selfData = THTensor_(data)(self);
  real *gradOutputData = THTensor_(data)(gradOutput);
  real *srcData = THTensor_(data)(src);
  real *weightData = THTensor_(data)(weight);
  int64_t *weightIndexData = THLongTensor_data(weightIndex);

  ptrdiff_t e, mOut, s, mIn;
  real g, v, tmp;
  int64_t wi;
  for (e = 0; e < THTensor_(size)(src, 0); e++) {
    for (mOut = 0; mOut < THTensor_(size)(gradOutput, 1); mOut++) {
      g = gradOutputData[e * gradOutput->stride[0] + mOut * gradOutput->stride[1]];
      for (s = 0; s < THLongTensor_size(weightIndex, 1); s++) {
        v = 0;
        wi = weightIndexData[e * weightIndex->stride[0] + s * weightIndex->stride[1]];
        for (mIn = 0; mIn < THTensor_(size)(src, 1); mIn++) {
          tmp = weightData[wi * weight->stride[0] + mIn * weight->stride[1] + mOut * weight->stride[2]];
          tmp *= srcData[e * src->stride[0] + mIn * src->stride[1]];
          v += tmp;
        }
        selfData[e * self->stride[0] + s * self->stride[1]] += g * v;
      }
    }
  }
}

#endif // TH_GENERIC_FILE
