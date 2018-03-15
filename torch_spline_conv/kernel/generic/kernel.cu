#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

void spline_(linear_basis_forward)(THCState *state, THCTensor *basis, THCudaLongTensor *weight_index, THCTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K) {
  SPLINE_BASIS_FORWARD(linearBasisForwardKernel, basis, weight_index, pseudo, kernel_size, is_open_spline, K)
}

void spline_(quadratic_basis_forward)(THCState *state, THCTensor *basis, THCudaLongTensor *weight_index, THCTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K) {
  SPLINE_BASIS_FORWARD(quadraticBasisForwardKernel, basis, weight_index, pseudo, kernel_size, is_open_spline, K)
}

void spline_(cubic_basis_forward)(THCState *state, THCTensor *basis, THCudaLongTensor *weight_index, THCTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K) {
  SPLINE_BASIS_FORWARD(cubicBasisForwardKernel, basis, weight_index, pseudo, kernel_size, is_open_spline, K)
}

void spline_(weighting_forward)(THCState *state, THCTensor *output, THCTensor *input, THCTensor *weight, THCTensor *basis, THCudaLongTensor *weight_index) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 4, input, weight, basis, weight_index));

  TensorInfo<real> outputInfo = thc_(getTensorInfo)(state, output);
  TensorInfo<real> inputInfo = thc_(getTensorInfo)(state, input);
  TensorInfo<real> weightInfo = thc_(getTensorInfo)(state, weight);
  TensorInfo<real> basisInfo = thc_(getTensorInfo)(state, basis);
  TensorInfo<int64_t> weightIndexInfo = thc_getTensorInfo_Long(state, weight_index);

  KERNEL_RUN(weightingForwardKernel, THCTensor_(nElement)(state, output), outputInfo, inputInfo, weightInfo, basisInfo, weightIndexInfo)
}

void spline_(weighting_backward_input)(THCState *state, THCTensor *grad_input, THCTensor *grad_output, THCTensor *weight, THCTensor *basis, THCudaLongTensor *weight_index) {
  TensorInfo<real> gradInputInfo = thc_(getTensorInfo)(state, grad_input);
  TensorInfo<real> gradOutputInfo = thc_(getTensorInfo)(state, grad_output);
  TensorInfo<real> weightInfo = thc_(getTensorInfo)(state, weight);
  TensorInfo<real> basisInfo = thc_(getTensorInfo)(state, basis);
  TensorInfo<int64_t> weightIndexInfo = thc_getTensorInfo_Long(state, weight_index);

  KERNEL_RUN(weightingBackwardInputKernel, THCTensor_(nElement)(state, grad_input), gradInputInfo, gradOutputInfo, weightInfo, basisInfo, weightIndexInfo)
}

void spline_(weighting_backward_weight)(THCState *state, THCTensor *grad_weight, THCTensor *grad_output, THCTensor *input, THCTensor *basis, THCudaLongTensor *weight_index) {
  TensorInfo<real> gradWeightInfo = thc_(getTensorInfo)(state, grad_weight);
  TensorInfo<real> gradOutputInfo = thc_(getTensorInfo)(state, grad_output);
  TensorInfo<real> inputInfo = thc_(getTensorInfo)(state, input);
  TensorInfo<real> basisInfo = thc_(getTensorInfo)(state, basis);
  TensorInfo<int64_t> weightIndexInfo = thc_getTensorInfo_Long(state, weight_index);

  KERNEL_RUN(weightingBackwardWeightKernel, THCTensor_(nElement)(state, grad_output), gradWeightInfo, gradOutputInfo, inputInfo, basisInfo, weightIndexInfo)
}

#endif
