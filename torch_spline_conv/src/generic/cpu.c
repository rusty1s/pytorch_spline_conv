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
    value = (1 - k_mod) * (1 - value) + k_mod * value;
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

void spline_(weighting_forward)(THTensor *output, THTensor *input, THTensor *weight, THTensor *basis, THLongTensor *weight_index) {
  real *weight_data = weight->storage->data + weight->storageOffset;
  int64_t M_out = THTensor_(size)(output, 1);
  int64_t M_in = THTensor_(size)(input, 1);
  int64_t S = THLongTensor_size(weight_index, 1);
  int64_t m_out, m_in, s, i; real b, value;

  TH_TENSOR_DIM_APPLY4(real, output, real, input, real, basis, int64_t, weight_index, 1,
    for (m_out = 0; m_out < M_out; m_out++) {
      value = 0;
      for (s = 0; s < S; s++) {
        b = *(basis_data + s * basis_stride);
        i = *(weight_index_data + s * weight_index_stride);
        for (m_in = 0; m_in < M_in; m_in++) {
          value += b * *(weight_data + i * M_in * M_out + m_in * M_out + m_out) * *(input_data + m_in * input_stride);
        }
      }
      output_data[m_out * output_stride] = value;
    }
  )
}

void spline_(weighting_backward)(THTensor *grad_input, THTensor *grad_weight, THTensor *grad_output, THTensor *input, THTensor *weight, THTensor *basis, THLongTensor *weight_index) {
  real *weight_data = weight->storage->data + weight->storageOffset;
  real *grad_weight_data = grad_weight->storage->data + grad_weight->storageOffset;
  int64_t M_out = THTensor_(size)(grad_output, 1);
  int64_t M_in = THTensor_(size)(input, 1);
  int64_t S = THLongTensor_size(weight_index, 1);
  int64_t m_out, m_in, s, i, w_idx; real g, b;

  TH_TENSOR_DIM_APPLY5(real, grad_input, real, grad_output, real, input, real, basis, int64_t, weight_index, 1,
    for (m_out = 0; m_out < M_out; m_out++) {
      g = *(grad_output_data + m_out * grad_output_stride);
      for (s = 0; s < S; s++) {
        b = *(basis_data + s * basis_stride);
        i = *(weight_index_data + s * weight_index_stride);
        for (m_in = 0; m_in < M_in; m_in++) {
          w_idx = i * M_in * M_out + m_in * M_out + m_out;
          grad_input_data[m_in] += b * g * *(weight_data + w_idx);
          grad_weight_data[w_idx] += b * g * *(input_data + m_in * input_stride);
        }
      }
    }
  )
}

#endif
