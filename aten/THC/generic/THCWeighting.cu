#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCWeighting.cu"
#else

void THCTensor_(weightingForward)(THCState *state, THCTensor *self, THCTensor *src,
                                  THCTensor *weight, THCTensor *basis,
                                  THCudaLongTensor *weightIndex) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 5, self, src, weight, basis, weightIndex));

  TensorInfo<real> selfInfo = THCTensor_(getTensorInfo)(state, self);
  TensorInfo<real> srcInfo = THCTensor_(getTensorInfo)(state, src);
  TensorInfo<real> weightInfo = THCTensor_(getTensorInfo)(state, weight);
  TensorInfo<real> basisInfo = THCTensor_(getTensorInfo)(state, basis);
  TensorInfo<int64_t> weightIndexInfo = THCudaLongTensor_getTensorInfo(state, weightIndex);

  KERNEL_REAL_RUN(weightingForwardKernel, THCTensor_(nElement)(state, self), selfInfo, srcInfo,
                  weightInfo, basisInfo, weightIndexInfo);
}

void THCTensor_(weightingBackwardSrc)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                      THCTensor *weight, THCTensor *basis,
                                      THCudaLongTensor *weightIndex) {
}

void THCTensor_(weightingBackwardWeight)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                         THCTensor *src, THCTensor *basis,
                                         THCudaLongTensor *weightIndex) {
}

void THCTensor_(weightingBackwardBasis)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                        THCTensor *src, THCTensor *weight,
                                        THCudaLongTensor *weightIndex) {
}

#endif // THC_GENERIC_FILE

