#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCCBasis.c"
#else

void THCCTensor_(linearBasisForward)(THCTensor *basis, THCudaLongTensor *weightIndex,
                                     THCTensor *pseudo, THCudaLongTensor *kernelSize,
                                     THCudaByteTensor *isOpenSpline) {
  THCTensor_(linearBasisForward)(state, basis, weightIndex, pseudo, kernelSize, isOpenSpline);
}

void THCCTensor_(quadraticBasisForward)(THCTensor *basis, THCudaLongTensor *weightIndex,
                                        THCTensor *pseudo, THCudaLongTensor *kernelSize,
                                        THCudaByteTensor *isOpenSpline) {
  THCTensor_(quadraticBasisForward)(state, basis, weightIndex, pseudo, kernelSize, isOpenSpline);
}

void THCCTensor_(cubicBasisForward)(THCTensor *basis, THCudaLongTensor *weightIndex,
                                    THCTensor *pseudo, THCudaLongTensor *kernelSize,
                                    THCudaByteTensor *isOpenSpline) {
  THCTensor_(cubicBasisForward)(state, basis, weightIndex, pseudo, kernelSize, isOpenSpline);
}

void THCCTensor_(linearBasisBackward)(THCTensor *self, THCTensor *gradBasis, THCTensor *pseudo,
                                      THCudaLongTensor *kernelSize,
                                      THCudaByteTensor *isOpenSpline) {
  THCTensor_(linearBasisBackward)(state, self, gradBasis, pseudo, kernelSize, isOpenSpline);
}

void THCCTensor_(quadraticBasisBackward)(THCTensor *self, THCTensor *gradBasis, THCTensor *pseudo,
                                         THCudaLongTensor *kernelSize,
                                         THCudaByteTensor *isOpenSpline) {
  THCTensor_(quadraticBasisBackward)(state, self, gradBasis, pseudo, kernelSize, isOpenSpline);
}

void THCCTensor_(cubicBasisBackward)(THCTensor *self, THCTensor *gradBasis, THCTensor *pseudo,
                                     THCudaLongTensor *kernelSize,
                                     THCudaByteTensor *isOpenSpline) {
  THCTensor_(cubicBasisBackward)(state, self, gradBasis, pseudo, kernelSize, isOpenSpline);
}

#endif  // THC_GENERIC_FILE
