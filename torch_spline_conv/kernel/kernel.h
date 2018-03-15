#ifdef __cplusplus
extern "C" {
#endif

void    spline_linear_basis_forward_kernel_Float (THCState *state,      THCudaTensor *basis, THCudaLongTensor *weight_index,       THCudaTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);
void    spline_linear_basis_forward_kernel_Double(THCState *state, THCudaDoubleTensor *basis, THCudaLongTensor *weight_index, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);
void spline_quadratic_basis_forward_kernel_Float (THCState *state,       THCudaTensor *basis, THCudaLongTensor *weight_index,       THCudaTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);
void spline_quadratic_basis_forward_kernel_Double(THCState *state, THCudaDoubleTensor *basis, THCudaLongTensor *weight_index, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);
void     spline_cubic_basis_forward_kernel_Float (THCState *state,       THCudaTensor *basis, THCudaLongTensor *weight_index,       THCudaTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);
void     spline_cubic_basis_forward_kernel_Double(THCState *state, THCudaDoubleTensor *basis, THCudaLongTensor *weight_index, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);

void spline_weighting_forward_kernel_Float (THCState *state,       THCudaTensor *output,       THCudaTensor *input,       THCudaTensor *weight,       THCudaTensor *basis, THCudaLongTensor *weight_index);
void spline_weighting_forward_kernel_Double(THCState *state, THCudaDoubleTensor *output, THCudaDoubleTensor *input, THCudaDoubleTensor *weight, THCudaDoubleTensor *basis, THCudaLongTensor *weight_index);

void spline_weighting_backward_input_kernel_Float (THCState *state,       THCudaTensor *grad_input,       THCudaTensor *grad_output,       THCudaTensor *weight,       THCudaTensor *basis, THCudaLongTensor *weight_index);
void spline_weighting_backward_input_kernel_Double(THCState *state, THCudaDoubleTensor *grad_input, THCudaDoubleTensor *grad_output, THCudaDoubleTensor *weight, THCudaDoubleTensor *basis, THCudaLongTensor *weight_index);

void spline_weighting_backward_weight_kernel_Float (THCState *state,       THCudaTensor *grad_weight,       THCudaTensor *grad_output,       THCudaTensor *input,       THCudaTensor *basis, THCudaLongTensor *weight_index);
void spline_weighting_backward_weight_kernel_Double(THCState *state, THCudaDoubleTensor *grad_weight, THCudaDoubleTensor *grad_output, THCudaDoubleTensor *input, THCudaDoubleTensor *basis, THCudaLongTensor *weight_index);

#ifdef __cplusplus
}
#endif
