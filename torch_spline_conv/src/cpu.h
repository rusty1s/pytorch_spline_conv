void spline_linear_Float (THFloatTensor  *amount, THLongTensor *index, THFloatTensor  *input, THLongTensor *kernel, THByteTensor *open);
void spline_linear_Double(THDoubleTensor *amount, THLongTensor *index, THDoubleTensor *input, THLongTensor *kernel, THByteTensor *open);

void spline_quadratic_Float (THFloatTensor  *amount, THLongTensor *index, THFloatTensor  *input, THLongTensor *kernel, THByteTensor *open);
void spline_quadratic_Double(THDoubleTensor *amount, THLongTensor *index, THDoubleTensor *input, THLongTensor *kernel, THByteTensor *open);

void spline_cubic_Float (THFloatTensor  *amount, THLongTensor *index, THFloatTensor  *input, THLongTensor *kernel, THByteTensor *open);
void spline_cubic_Double(THDoubleTensor *amount, THLongTensor *index, THDoubleTensor *input, THLongTensor *kernel, THByteTensor *open);
