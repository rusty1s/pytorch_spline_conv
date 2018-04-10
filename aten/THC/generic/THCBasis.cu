#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCBasis.cu"
#else

void THCTensor_(linearBasisForward)(THCState *state, THCTensor *basis,
                                    THCudaLongTensor *weightIndex, THCTensor *pseudo,
                                    THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline) {
  THC_TENSOR_BASIS_FORWARD(linearBasisForwardKernel, state, basis, weightIndex, pseudo, kernelSize,
                           isOpenSpline)
}

void THCTensor_(quadraticBasisForward)(THCState *state, THCTensor *basis,
                                       THCudaLongTensor *weightIndex, THCTensor *pseudo,
                                       THCudaLongTensor *kernelSize,
                                       THCudaByteTensor *isOpenSpline) {
  THC_TENSOR_BASIS_FORWARD(quadraticBasisForwardKernel, state, basis, weightIndex, pseudo,
                           kernelSize, isOpenSpline)
}

void THCTensor_(cubicBasisForward)(THCState *state, THCTensor *basis,
                                   THCudaLongTensor *weightIndex, THCTensor *pseudo,
                                   THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline) {
  THC_TENSOR_BASIS_FORWARD(cubicBasisForwardKernel, state, basis, weightIndex, pseudo, kernelSize,
                           isOpenSpline)
}

void THCTensor_(linearBasisBackward)(THCState *state, THCTensor *self, THCTensor *gradBasis,
                                     THCTensor *pseudo, THCudaLongTensor *kernelSize,
                                     THCudaByteTensor *isOpenSpline) {
  THC_TENSOR_BASIS_BACKWARD(linearBasisBackwardKernel, state, self, gradBasis, pseudo, kernelSize,
                            isOpenSpline)
}

void THCTensor_(quadraticBasisBackward)(THCState *state, THCTensor *self, THCTensor *gradBasis,
                                        THCTensor *pseudo, THCudaLongTensor *kernelSize,
                                        THCudaByteTensor *isOpenSpline) {
  THC_TENSOR_BASIS_BACKWARD(quadraticBasisBackwardKernel, state, self, gradBasis, pseudo,
                            kernelSize, isOpenSpline)
}

void THCTensor_(cubicBasisBackward)(THCState *state, THCTensor *self, THCTensor *gradBasis,
                                    THCTensor *pseudo, THCudaLongTensor *kernelSize,
                                    THCudaByteTensor *isOpenSpline) {
  THC_TENSOR_BASIS_BACKWARD(cubicBasisBackwardKernel, state, self, gradBasis, pseudo, kernelSize,
                            isOpenSpline)
}

#endif // THC_GENERIC_FILE
