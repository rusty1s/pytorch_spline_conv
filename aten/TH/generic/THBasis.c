#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBasis.c"
#else

inline real THTensor_(linear)(real v, int64_t kMod) {
  return 1 - v - kMod + 2 * v * kMod;
}

inline real THTensor_(gradLinear)(real v, int64_t kMod) {
  return 2 * kMod - 1;
}

inline real THTensor_(quadratic)(real v, int64_t kMod) {
  if (kMod == 0) return 0.5 * v * v - v + 0.5;
  else if (kMod == 1) return -v * v + v + 0.5;
  else return 0.5 * v * v;
}

inline real THTensor_(gradQuadratic)(real v, int64_t kMod) {
  if (kMod == 0) return v - 1;
  else if (kMod == 1) return -2 * v + 1;
  else return v;
}

inline real THTensor_(cubic)(real v, int64_t kMod) {
  if (kMod == 0) { v = (1 - v); return v * v * v / 6.0; }
  else if (kMod == 1) return (3 * v * v * v - 6 * v * v + 4) / 6;
  else if (kMod == 2) return (-3 * v * v * v + 3 * v * v + 3 * v + 1) / 6;
  else return v * v * v / 6;
}

inline real THTensor_(gradCubic)(real v, int64_t kMod) {
  if (kMod == 0) return (-v * v + 2 * v - 1) / 2;
  else if (kMod == 1) return (3 * v * v - 4 * v) / 2;
  else if (kMod == 2) return (-3 * v * v + 2 * v + 1) / 2;
  else return v * v / 2;
}

void THTensor_(linearBasisForward)(THTensor *basis, THLongTensor *weightIndex, THTensor *pseudo,
                                   THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_FORWARD(1, basis, weightIndex, pseudo, kernelSize, isOpenSpline,
                          THTensor_(linear)(v, kMod))
}

void THTensor_(quadraticBasisForward)(THTensor *basis, THLongTensor *weightIndex, THTensor *pseudo,
                                      THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_FORWARD(2, basis, weightIndex, pseudo, kernelSize, isOpenSpline,
                          THTensor_(quadratic)(v, kMod))
}

void THTensor_(cubicBasisForward)(THTensor *basis, THLongTensor *weightIndex, THTensor *pseudo,
                                  THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_FORWARD(3, basis, weightIndex, pseudo, kernelSize, isOpenSpline,
                          THTensor_(cubic)(v, kMod))
}

void THTensor_(linearBasisBackward)(THTensor *self, THTensor *gradBasis, THTensor *pseudo,
                                    THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_BACKWARD(1, self, gradBasis, pseudo, kernelSize, isOpenSpline,
                           THTensor_(linear)(v, kMod), THTensor_(gradLinear)(v, kMod))
}

void THTensor_(quadraticBasisBackward)(THTensor *self, THTensor *gradBasis, THTensor *pseudo,
                                       THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_BACKWARD(2, self, gradBasis, pseudo, kernelSize, isOpenSpline,
                           THTensor_(quadratic)(v, kMod), THTensor_(gradQuadratic)(v, kMod))
}

void THTensor_(cubicBasisBackward)(THTensor *self, THTensor *gradBasis, THTensor *pseudo,
                                   THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_BACKWARD(3, self, gradBasis, pseudo, kernelSize, isOpenSpline,
                           THTensor_(cubic)(v, kMod), THTensor_(gradCubic)(v, kMod))
}

#endif // TH_GENERIC_FILE
