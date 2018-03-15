#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/cuda.c"
#else

void spline_(linear_basis_forward)(THCTensor *basis, THCudaLongTensor *weight_index, THCTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K) {
  spline_kernel_(linear_basis_forward)(state, basis, weight_index, pseudo, kernel_size, is_open_spline, K);
}

void spline_(quadratic_basis_forward)(THCTensor *basis, THCudaLongTensor *weight_index, THCTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K) {
  spline_kernel_(quadratic_basis_forward)(state, basis, weight_index, pseudo, kernel_size, is_open_spline, K);
}

void spline_(cubic_basis_forward)(THCTensor *basis, THCudaLongTensor *weight_index, THCTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K) {
  spline_kernel_(cubic_basis_forward)(state, basis, weight_index, pseudo, kernel_size, is_open_spline, K);
}

void spline_(weighting_forward)(THCTensor *output, THCTensor *input, THCTensor *weight, THCTensor *basis, THCudaLongTensor *weight_index) {
  spline_kernel_(weighting_forward)(state, output, input, weight, basis, weight_index);
}

void spline_(weighting_backward_input)(THCTensor *grad_input, THCTensor *grad_output, THCTensor *weight, THCTensor *basis, THCudaLongTensor *weight_index) {
  spline_kernel_(weighting_backward_input)(state, grad_input, grad_output, weight, basis, weight_index);
}

void spline_(weighting_backward_weight)(THCTensor *grad_weight, THCTensor *grad_output, THCTensor *input, THCTensor *basis, THCudaLongTensor *weight_index) {
  spline_kernel_(weighting_backward_weight)(state, grad_weight, grad_output, input, basis, weight_index);
}

#endif
