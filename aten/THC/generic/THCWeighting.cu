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
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 5, self, gradOutput, weight, basis, weightIndex));

  THCTensor_(fill)(state, self, ScalarConvert<int, real>::to(0));

  TensorInfo<real> selfInfo = THCTensor_(getTensorInfo)(state, self);
  TensorInfo<real> gradOutputInfo = THCTensor_(getTensorInfo)(state, gradOutput);
  TensorInfo<real> weightInfo = THCTensor_(getTensorInfo)(state, weight);
  TensorInfo<real> basisInfo = THCTensor_(getTensorInfo)(state, basis);
  TensorInfo<int64_t> weightIndexInfo = THCudaLongTensor_getTensorInfo(state, weightIndex);

  KERNEL_REAL_RUN(weightingBackwardSrcKernel, THCTensor_(nElement)(state, gradOutput), selfInfo,
                  gradOutputInfo, weightInfo, basisInfo, weightIndexInfo);
}

void THCTensor_(weightingBackwardWeight)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                         THCTensor *src, THCTensor *basis,
                                         THCudaLongTensor *weightIndex) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 5, self, gradOutput, src, basis, weightIndex));

  THCTensor_(fill)(state, self, ScalarConvert<int, real>::to(0));

  TensorInfo<real> selfInfo = THCTensor_(getTensorInfo)(state, self);
  TensorInfo<real> gradOutputInfo = THCTensor_(getTensorInfo)(state, gradOutput);
  TensorInfo<real> srcInfo = THCTensor_(getTensorInfo)(state, src);
  TensorInfo<real> basisInfo = THCTensor_(getTensorInfo)(state, basis);
  TensorInfo<int64_t> weightIndexInfo = THCudaLongTensor_getTensorInfo(state, weightIndex);

  KERNEL_REAL_RUN(weightingBackwardWeightKernel, THCTensor_(nElement)(state, gradOutput), selfInfo,
                  gradOutputInfo, srcInfo, basisInfo, weightIndexInfo);
}

void THCTensor_(weightingBackwardBasis)(THCState *state, THCTensor *self, THCTensor *gradOutput,
                                        THCTensor *src, THCTensor *weight,
                                        THCudaLongTensor *weightIndex) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 5, self, gradOutput, src, weight, weightIndex));

  THCTensor_(fill)(state, self, ScalarConvert<int, real>::to(0));

  TensorInfo<real> selfInfo = THCTensor_(getTensorInfo)(state, self);
  TensorInfo<real> gradOutputInfo = THCTensor_(getTensorInfo)(state, gradOutput);
  TensorInfo<real> srcInfo = THCTensor_(getTensorInfo)(state, src);
  TensorInfo<real> weightInfo = THCTensor_(getTensorInfo)(state, weight);
  TensorInfo<int64_t> weightIndexInfo = THCudaLongTensor_getTensorInfo(state, weightIndex);

  KERNEL_REAL_RUN(weightingBackwardBasisKernel, THCTensor_(nElement)(state, gradOutput), selfInfo,
                  gradOutputInfo, srcInfo, weightInfo, weightIndexInfo);
}

#endif // THC_GENERIC_FILE
