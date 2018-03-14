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

#endif
