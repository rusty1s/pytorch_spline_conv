#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCWeighting.h"
#else

void THCTensor_(weightingForward)(THCState *state, THCTensor *self, THCTensor *src,
                                  THCTensor *weight, THCTensor *basis,
                                  THCudaLongTensor *weightIndex);

void THCTensor_(weightingBackwardSrc)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                      THCTensor *weight, THCTensor *basis,
                                      THCudaLongTensor *weightIndex);

void THCTensor_(weightingBackwardWeight)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                         THCTensor *src, THCTensor *basis,
                                         THCudaLongTensor *weightIndex);

void THCTensor_(weightingBackwardBasis)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                        THCTensor *src, THCTensor *weight,
                                        THCudaLongTensor *weightIndex);

#endif // THC_GENERIC_FILE
