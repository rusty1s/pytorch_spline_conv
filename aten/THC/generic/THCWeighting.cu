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

  weight = THCTensor_(newTranspose)(state, weight, 1, 2);

  TensorInfo<real> selfInfo = THCTensor_(getTensorInfo)(state, self);
  TensorInfo<real> gradOutputInfo = THCTensor_(getTensorInfo)(state, gradOutput);
  TensorInfo<real> weightInfo = THCTensor_(getTensorInfo)(state, weight);
  TensorInfo<real> basisInfo = THCTensor_(getTensorInfo)(state, basis);
  TensorInfo<int64_t> weightIndexInfo = THCudaLongTensor_getTensorInfo(state, weightIndex);

  KERNEL_REAL_RUN(weightingBackwardSrcKernel, THCTensor_(nElement)(state, self), selfInfo,
                  gradOutputInfo, weightInfo, basisInfo, weightIndexInfo);

  THCTensor_(free)(state, weight);
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

void THCTensor_(weightingBackward)(THCState *state, THCTensor *gradSrc, THCTensor *gradWeight,
                                   THCTensor *gradBasis, THCTensor *gradOutput, THCTensor *src,
                                   THCTensor *weight, THCTensor *basis,
                                   THCudaLongTensor *weightIndex) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 8, gradSrc, gradWeight, gradBasis, src, weight,
                                        basis, weightIndex));

  THCTensor_(fill)(state, gradWeight, ScalarConvert<int, real>::to(0));
  THCTensor_(fill)(state, gradBasis, ScalarConvert<int, real>::to(0));

  weight = THCTensor_(newTranspose)(state, weight, 1, 2);

  TensorInfo<real> gradSrcInfo = THCTensor_(getTensorInfo)(state, gradSrc);
  TensorInfo<real> gradWeightInfo = THCTensor_(getTensorInfo)(state, gradWeight);
  TensorInfo<real> gradBasisInfo = THCTensor_(getTensorInfo)(state, gradBasis);
  TensorInfo<real> gradOutputInfo = THCTensor_(getTensorInfo)(state, gradOutput);
  TensorInfo<real> srcInfo = THCTensor_(getTensorInfo)(state, src);
  TensorInfo<real> weightInfo = THCTensor_(getTensorInfo)(state, weight);
  TensorInfo<real> basisInfo = THCTensor_(getTensorInfo)(state, basis);
  TensorInfo<int64_t> weightIndexInfo = THCudaLongTensor_getTensorInfo(state, weightIndex);

  KERNEL_REAL_RUN(weightingBackwardKernel, THCTensor_(nElement)(state, src), gradSrcInfo,
                  gradWeightInfo, gradBasisInfo, gradOutputInfo, srcInfo, weightInfo, basisInfo,
                  weightIndexInfo);

  THCTensor_(free)(state, weight);
}

#endif // THC_GENERIC_FILE
