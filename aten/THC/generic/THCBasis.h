#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCBasis.h"
#else

void THCTensor_(linearBasisForward)(THCState *state, THCTensor *basis,
                                    THCudaLongTensor *weightIndex, THCTensor *pseudo,
                                    THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);

void THCTensor_(quadraticBasisForward)(THCState *state, THCTensor *basis,
                                       THCudaLongTensor *weightIndex, THCTensor *pseudo,
                                       THCudaLongTensor *kernelSize,
                                       THCudaByteTensor *isOpenSpline);

void THCTensor_(cubicBasisForward)(THCState *state, THCTensor *basis,
                                   THCudaLongTensor *weightIndex, THCTensor *pseudo,
                                   THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);

void THCTensor_(linearBasisBackward)(THCState *state, THCTensor *self, THCTensor *gradBasis,
                                     THCTensor *pseudo, THCudaLongTensor *kernelSize,
                                     THCudaByteTensor *isOpenSpline);

void THCTensor_(quadraticBasisBackward)(THCState *state, THCTensor *self, THCTensor *gradBasis,
                                        THCTensor *pseudo, THCudaLongTensor *kernelSize,
                                        THCudaByteTensor *isOpenSpline);

void THCTensor_(cubicBasisBackward)(THCState *state, THCTensor *self, THCTensor *gradBasis,
                                    THCTensor *pseudo, THCudaLongTensor *kernelSize,
                                    THCudaByteTensor *isOpenSpline);

#endif // THC_GENERIC_FILE
