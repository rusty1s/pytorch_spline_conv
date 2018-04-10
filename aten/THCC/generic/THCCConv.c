#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCCConv.c"
#else

void THCCTensor_(convForward)(THCTensor *self, THCTensor *src, THCTensor *weight, THCTensor *basis,
                              THCudaLongTensor *weightIndex) {
  THCTensor_(convForward)(state, self, src, weight, basis, weightIndex);
}

void THCCTensor_(convBackwardSrc)(THCTensor *self, THCTensor *gradOutput, THCTensor *weight,
                                  THCTensor *basis, THCudaLongTensor *weightIndex) {
  THCTensor_(convBackwardSrc)(state, self, gradOutput, weight, basis, weightIndex);
}

void THCCTensor_(convBackwardBasis)(THCTensor *self, THCTensor *gradOutput, THCTensor *src,
                                    THCTensor *weight, THCudaLongTensor *weightIndex) {
  THCTensor_(convBackwardBasis)(state, self, gradOutput, src, weight, weightIndex);
}

void THCCTensor_(convBackwardWeight)(THCTensor *self, THCTensor *gradOutput, THCTensor *src,
                                     THCTensor *basis, THCudaLongTensor *weightIndex) {
  THCTensor_(convBackwardWeight)(state, self, gradOutput, src, basis, weightIndex);
}

#endif  // THC_GENERIC_FILE

