#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCCConv.c"
#else

void THCCTensor_(convForward)(THCTensor *self, THCTensor *src, THCTensor *weight, THCTensor *basis,
                              THCudaLongTensor *weightIndex) {
}

void THCCTensor_(convBackwardSrc)(THCTensor *self, THCTensor *gradOutput, THCTensor *weight,
                                  THCTensor *basis, THCudaLongTensor *weightIndex) {
}

void THCCTensor_(convBackwardBasis)(THCTensor *self, THCTensor *gradOutput, THCTensor *src,
                                    THCTensor *weight, THCudaLongTensor *weightIndex) {
}

void THCCTensor_(convBackwardWeight)(THCTensor *self, THCTensor *gradOutput, THCTensor *src,
                                     THCTensor *basis, THCudaLongTensor *weightIndex) {
}

#endif  // THC_GENERIC_FILE

