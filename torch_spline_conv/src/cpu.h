void spline_basis_linear_Float(THFloatTensor *basis, THLongTensor *weight_index, THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void spline_basis_linear_Double(THDoubleTensor *basis, THLongTensor *weight_index, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);

void spline_basis_quadratic_Float(THFloatTensor *basis, THLongTensor *weight_index, THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void spline_basis_quadratic_Double(THDoubleTensor *basis, THLongTensor *weight_index, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);

void spline_basis_cubic_Float(THFloatTensor *basis, THLongTensor *weight_index, THFloatTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
void spline_basis_cubic_Double(THDoubleTensor *basis, THLongTensor *weight_index, THDoubleTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K);
