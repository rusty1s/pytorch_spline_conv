#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

void spline_(linear)(THFloatTensor *amount, THLongTensor *index, THFloatTensor *input, THLongTensor *kernel, THByteTensor *open) {
  // s = (m+1)^d
  // amount: E x s
  // index: E x s
  // input: E x d
  // kernel: d
  // open: d
  //
  int64_t i, d;
  int64_t E = THLongTensor_size(index, 0);
  int64_t K = THLongTensor_size(index, 1);
  int64_t D = THLongTensor_size(kernel, 0);
  for (i = 0; i < E * K; i++) {
    for (d = 0; d < D; d++) {
    }
  }
}

void spline_(quadratic)(THFloatTensor *amount, THLongTensor *index, THFloatTensor *input, THLongTensor *kernel, THByteTensor *open) {
  int64_t i;
  for (i = 0; i < THLongTensor_size(input, dim); i++) {
  }
}

void spline_(cubic)(THFloatTensor *amount, THLongTensor *index, THFloatTensor *input, THLongTensor *kernel, THByteTensor *open) {
  int64_t i;
  for (i = 0; i < THLongTensor_size(input, dim); i++) {
  }
}

#endif
