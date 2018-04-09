#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBasis.c"
#else

void THTensor_(linearBasisForward)(THTensor *basis, THLongTensor *weightIndex, THTensor *pseudo,
                                   THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_FORWARD(1, basis, weightIndex, pseudo, kernelSize, isOpenSpline,
    v = 1 - v - kMod + 2 * v * kMod;
  )
}

void THTensor_(quadraticBasisForward)(THTensor *basis, THLongTensor *weightIndex, THTensor *pseudo,
                                      THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_FORWARD(2, basis, weightIndex, pseudo, kernelSize, isOpenSpline,
    if (kMod == 0) v = 0.5 * v * v - v + 0.5;
    else if (kMod == 1) v = -v * v + v + 0.5;
    else v = 0.5 * v * v;
  )
}

void THTensor_(cubicBasisForward)(THTensor *basis, THLongTensor *weightIndex, THTensor *pseudo,
                                  THLongTensor *kernelSize, THByteTensor *isOpenSpline) {
  TH_TENSOR_BASIS_FORWARD(3, basis, weightIndex, pseudo, kernelSize, isOpenSpline,
    if (kMod == 0) { v = (1 - v); v = v * v * v / 6.0; }
    else if (kMod == 1) v = (3 * v * v * v - 6 * v * v + 4) / 6;
    else if (kMod == 2) v = (-3 * v * v * v + 3 * v * v + 3 * v + 1) / 6;
    else v = v * v * v / 6;
  )
}

#endif // TH_GENERIC_FILE
