void spline_basis_linear_Float(THFloatTensor *basis, THLongTensor *weight_index, THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void spline_basis_linear_Double(THDoubleTensor *basis, THLongTensor *weight_index, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);

void spline_basis_quadratic_Float(THFloatTensor *basis, THLongTensor *weight_index, THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void spline_basis_quadratic_Double(THDoubleTensor *basis, THLongTensor *weight_index, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);

void spline_basis_cubic_Float(THFloatTensor *basis, THLongTensor *weight_index, THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void spline_basis_cubic_Double(THDoubleTensor *basis, THLongTensor *weight_index, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);

void spline_edgewise_forward_Float(THFloatTensor *output, THFloatTensor *input, THFloatTensor *weight, THFloatTensor *basis, THLongTensor *weight_index);
void spline_edgewise_backward_Double(THDoubleTensor *output, THDoubleTensor *input, THDoubleTensor *weight, THDoubleTensor *basis, THLongTensor *weight_index);

void spline_edgewise_backward_Float(THFloatTensor *grad_input, THFloatTensor *grad_weight, THFloatTensor *grad_output, THFloatTensor *input, THFloatTensor *weight, THFloatTensor *basis, THLongTensor *weight_index);
void spline_edgewise_backward_Double(THDoubleTensor *grad_input, THDoubleTensor *grad_weight, THDoubleTensor *grad_output, THDoubleTensor *input, THDoubleTensor *weight, THDoubleTensor *basis, THLongTensor *weight_index);
