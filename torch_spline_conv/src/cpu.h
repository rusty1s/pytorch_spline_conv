void     spline_linear_basis_forward_Float( THFloatTensor *basis, THLongTensor *weight_index,  THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void    spline_linear_basis_forward_Double(THDoubleTensor *basis, THLongTensor *weight_index, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void  spline_quadratic_basis_forward_Float( THFloatTensor *basis, THLongTensor *weight_index,  THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void spline_quadratic_basis_forward_Double(THDoubleTensor *basis, THLongTensor *weight_index, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void      spline_cubic_basis_forward_Float( THFloatTensor *basis, THLongTensor *weight_index,  THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void     spline_cubic_basis_forward_Double(THDoubleTensor *basis, THLongTensor *weight_index, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);

void     spline_linear_basis_backward_Float( THFloatTensor *grad_pseudo, THLongTensor *grad_basis,  THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline);
void    spline_linear_basis_backward_Double(THDoubleTensor *grad_pseudo, THLongTensor *grad_basis, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline);
void  spline_quadratic_basis_backward_Float( THFloatTensor *grad_pseudo, THLongTensor *grad_basis,  THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline);
void spline_quadratic_basis_backward_Double(THDoubleTensor *grad_pseudo, THLongTensor *grad_basis, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline);
void      spline_cubic_basis_backward_Float( THFloatTensor *grad_pseudo, THLongTensor *grad_basis,  THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline);
void     spline_cubic_basis_backward_Double(THDoubleTensor *grad_pseudo, THLongTensor *grad_basis, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline);

void  spline_weighting_forward_Float( THFloatTensor *output, THFloatTensor *input,  THFloatTensor *weight,  THFloatTensor *basis,  THLongTensor *weight_index);
void spline_weighting_forward_Double(THDoubleTensor *output, THDoubleTensor *input, THDoubleTensor *weight, THDoubleTensor *basis, THLongTensor *weight_index);

void  spline_weighting_backward_input_Float( THFloatTensor *grad_input,  THFloatTensor *grad_output,  THFloatTensor *weight,  THFloatTensor *basis, THLongTensor *weight_index);
void spline_weighting_backward_input_Double(THDoubleTensor *grad_input, THDoubleTensor *grad_output, THDoubleTensor *weight, THDoubleTensor *basis, THLongTensor *weight_index);

void  spline_weighting_backward_basis_Float( THFloatTensor *grad_basis,  THFloatTensor *grad_output,  THFloatTensor *input,  THFloatTensor *weight, THLongTensor *weight_index);
void spline_weighting_backward_basis_Double(THDoubleTensor *grad_basis, THDoubleTensor *grad_output, THDoubleTensor *input, THDoubleTensor *weight, THLongTensor *weight_index);

void  spline_weighting_backward_weight_Float( THFloatTensor *grad_weight,  THFloatTensor *grad_output,  THFloatTensor *input,  THFloatTensor *basis, THLongTensor *weight_index);
void spline_weighting_backward_weight_Double(THDoubleTensor *grad_weight, THDoubleTensor *grad_output, THDoubleTensor *input, THDoubleTensor *basis, THLongTensor *weight_index);

