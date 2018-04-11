#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCCWeighting.c"
#else

void THCCTensor_(weightingForward)(THCTensor *self, THCTensor *src, THCTensor *weight,
                                   THCTensor *basis, THCudaLongTensor *weightIndex) {
  THCTensor_(weightingForward)(state, self, src, weight, basis, weightIndex);
}

void THCCTensor_(weightingBackwardSrc)(THCTensor *self, THCTensor *gradOutput, THCTensor *weight,
                                       THCTensor *basis, THCudaLongTensor *weightIndex) {
  THCTensor_(weightingBackwardSrc)(state, self, gradOutput, weight, basis, weightIndex);
}

void THCCTensor_(weightingBackwardWeight)(THCTensor *self, THCTensor *gradOutput, THCTensor *src,
                                          THCTensor *basis, THCudaLongTensor *weightIndex) {
  THCTensor_(weightingBackwardWeight)(state, self, gradOutput, src, basis, weightIndex);
}

void THCCTensor_(weightingBackwardBasis)(THCTensor *self, THCTensor *gradOutput, THCTensor *src,
                                         THCTensor *weight, THCudaLongTensor *weightIndex) {
  THCTensor_(weightingBackwardBasis)(state, self, gradOutput, src, weight, weightIndex);
}

void THCCTensor_(weightingBackward)(THCTensor *gradSrc, THCTensor *gradWeight,
                                    THCTensor *gradBasis, THCTensor *gradOutput, THCTensor *src,
                                    THCTensor *weight, THCTensor *basis,
                                    THCudaLongTensor *weightIndex) {
  THCTensor_(weightingBackward)(state, gradSrc, gradWeight, gradBasis, gradOutput, src, weight,
                                basis, weightIndex);
}

#endif  // THC_GENERIC_FILE
