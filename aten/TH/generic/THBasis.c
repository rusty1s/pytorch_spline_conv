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
}

void THTensor_(quadraticBasisBackward)(THTensor *self, THTensor *gradBasis, THTensor *pseudo,
                                       THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
}

void THTensor_(cubicBasisBackward)(THTensor *self, THTensor *gradBasis, THTensor *pseudo,
                                   THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
}

#endif // TH_GENERIC_FILE
