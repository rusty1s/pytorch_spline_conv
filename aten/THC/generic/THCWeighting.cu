#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCWeighting.cu"
#else

void THCTensor_(weightingForward)(THCState *state, THCTensor *self, THCTensor *src,
                                  THCTensor *weight, THCTensor *basis,
                                  THCudaLongTensor *weightIndex) {
  TH_TENSOR_WEIGHTING(weightingForwardKernel, THCTensor_(nElement)(state, self), self, src, weight,
                      basis, weightIndex)
}

void THCTensor_(weightingBackwardSrc)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                      THCTensor *weight, THCTensor *basis,
                                      THCudaLongTensor *weightIndex) {
  THCTensor *tWeight = THCTensor_(newTranspose)(state, weight, 1, 2);
  weight = THCTensor_(newContiguous)(state, tWeight);

  TH_TENSOR_WEIGHTING(weightingBackwardSrcKernel, THCTensor_(nElement)(state, self), self,
                      gradOutput, weight, basis, weightIndex)

  THCTensor_(free)(state, tWeight);
  THCTensor_(free)(state, weight);
}

void THCTensor_(weightingBackwardWeight)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                         THCTensor *src, THCTensor *basis,
                                         THCudaLongTensor *weightIndex) {
  THCTensor_(fill)(state, self, ScalarConvert<int, real>::to(0));
  TH_TENSOR_WEIGHTING(weightingBackwardWeightKernel, THCTensor_(nElement)(state, gradOutput), self,
                      gradOutput, src, basis, weightIndex)
}

void THCTensor_(weightingBackwardBasis)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                        THCTensor *src, THCTensor *weight,
                                        THCudaLongTensor *weightIndex) {
  THCTensor_(fill)(state, self, ScalarConvert<int, real>::to(0));
  TH_TENSOR_WEIGHTING(weightingBackwardBasisKernel, THCTensor_(nElement)(state, gradOutput), self,
                      gradOutput, src, weight, weightIndex)
}

#endif // THC_GENERIC_FILE
