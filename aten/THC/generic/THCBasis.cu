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

#endif // THC_GENERIC_FILE
