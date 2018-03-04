#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

#define SPLINE_BASIS(M, basis, weight_index, pseudo, kernel_size, is_open_spline, K, CODE) { \
  int64_t *kernel_size_data = kernel_size->storage->data + kernel_size->storageOffset; \
  uint8_t *is_open_spline_data = is_open_spline->storage->data + is_open_spline->storageOffset; \
  int64_t D = THTensor_(size)(pseudo, 1); \
  int64_t S = THLongTensor_size(weight_index, 1); \
  int64_t s, d, k, k_mod, i, offset; real value, b; \
\
  TH_TENSOR_DIM_APPLY3(real, basis, int64_t, weight_index, real, pseudo, 1, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM, \
    for (s = 0; s < S; s++) { \
      b = 1; i = 0; k = s; offset = K; \
      for (d = 0; d < D; d++) { \
        offset /= kernel_size_data[d]; \
        k_mod = k % (M + 1); \
        k /= (M + 1); \
        value = *(pseudo_data + d * pseudo_stride) * (kernel_size_data[d] - M * is_open_spline_data[d]); \
        i += ((((int64_t) value) + k_mod) % kernel_size_data[d]) * offset; \
        value -= floor(value); \
        CODE \
        b *= value; \
      } \
      basis_data[s * basis_stride] = b; \
      weight_index_data[s * weight_index_stride] = i; \
    }) \
}

void spline_(basis_linear)(THTensor *basis, THLongTensor *weight_index, THTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K) {
  SPLINE_BASIS(1, basis, weight_index, pseudo, kernel_size, is_open_spline, K,
    value = (1 - k_mod) * value + k_mod * (1 - value);
  )
}

void spline_(basis_quadratic)(THTensor *basis, THLongTensor *weight_index, THTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K) {
  SPLINE_BASIS(2, basis, weight_index, pseudo, kernel_size, is_open_spline, K,
    if (k_mod == 0) value = 0.5 * (1 - value) * (1 - value);
    else if (k_mod == 1) value = -value * value + value + 0.5;
    else value = 0.5 * value * value;
  )
}

void spline_(basis_cubic)(THTensor *basis, THLongTensor *weight_index, THTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K) {
  SPLINE_BASIS(3, basis, weight_index, pseudo, kernel_size, is_open_spline, K,
    if (k_mod == 0) value = (1 - value) * (1 - value) * (1 - value) / 6.0;
    else if (k_mod == 1) value = (3 * value * value * value - 6 * value * value + 4) / 6.0;
    else if (k_mod == 2) value = (-3 * value * value * value + 3 * value * value + 3 * value + 1) / 6.0;
    else value = value * value * value / 6.0;
  )
}

#endif
