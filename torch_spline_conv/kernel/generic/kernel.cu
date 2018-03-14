#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/kernel.cu"
#else

void spline_(linear_basis_forward)(THCState *state, THCTensor *basis, THCudaLongTensor *weight_index, THCTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K) {
  printf("linear");
}

void spline_(quadratic_basis_forward)(THCState *state, THCTensor *basis, THCudaLongTensor *weight_index, THCTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K) {
  printf("quadratic");
}

void spline_(cubic_basis_forward)(THCState *state, THCTensor *basis, THCudaLongTensor *weight_index, THCTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K) {
  printf("cubic");
}

#endif
