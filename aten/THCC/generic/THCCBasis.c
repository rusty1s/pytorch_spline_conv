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

#endif  // THC_GENERIC_FILE
