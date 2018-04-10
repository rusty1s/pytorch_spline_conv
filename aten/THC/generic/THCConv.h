#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCConv.h"
#else

void THCTensor_(convForward)(THCState *state, THCTensor *self, THCTensor *src, THCTensor *weight,
                             THCTensor *basis, THCudaLongTensor *weightIndex);

void THCTensor_(convBackwardSrc)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                 THCTensor *weight, THCTensor *basis,
                                 THCudaLongTensor *weightIndex);

void THCTensor_(convBackwardBasis)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                   THCTensor *src, THCTensor *weight,
                                   THCudaLongTensor *weightIndex);

void THCTensor_(convBackwardWeight)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                    THCTensor *src, THCTensor *basis,
                                    THCudaLongTensor *weightIndex);

#endif // THC_GENERIC_FILE
