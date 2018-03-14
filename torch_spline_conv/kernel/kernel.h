#ifdef __cplusplus
extern "C" {
#endif

void    spline_linear_basis_forward_kernel_Float (THCState *state,      THCudaTensor *basis, THCudaLongTensor *weight_index,       THCudaTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);
void    spline_linear_basis_forward_kernel_Double(THCState *state, THCudaDoubleTensor *basis, THCudaLongTensor *weight_index, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);
void spline_quadratic_basis_forward_kernel_Float (THCState *state,       THCudaTensor *basis, THCudaLongTensor *weight_index,       THCudaTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);
void spline_quadratic_basis_forward_kernel_Double(THCState *state, THCudaDoubleTensor *basis, THCudaLongTensor *weight_index, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);
void     spline_cubic_basis_forward_kernel_Float (THCState *state,       THCudaTensor *basis, THCudaLongTensor *weight_index,       THCudaTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);
void     spline_cubic_basis_forward_kernel_Double(THCState *state, THCudaDoubleTensor *basis, THCudaLongTensor *weight_index, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernel_size, THCudaByteTensor *is_open_spline, int K);

#ifdef __cplusplus
}
#endif
