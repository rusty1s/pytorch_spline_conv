#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

void spline_(basis_linear)(THTensor *basis, THLongTensor *weight_index, THTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K) {
  int64_t *kernel_size_data = kernel_size->storage->data + kernel_size->storageOffset;
  uint8_t *is_open_spline_data = is_open_spline->storage->data + is_open_spline->storageOffset;
  int64_t D = THTensor_(size)(pseudo, 1);
  int64_t S = THLongTensor_size(weight_index, 1);
  int64_t s, d;

  TH_TENSOR_DIM_APPLY3(real, basis, int64_t, weight_index, real, pseudo, 1, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (s = 0; s < S; s++) {
      real b = 1; int64_t i = 0;
      int64_t k = s;
      int64_t bla = K;
      for (d = 0; d < D; d++) {
        bla /= kernel_size_data[d];
        int64_t mod = k % 2;
        k >>= 1;

        real value = *(pseudo_data + d * pseudo_stride) * (kernel_size_data[d] - is_open_spline_data[d]);
        int64_t bot = ((int64_t) value) % kernel_size_data[d];
        int64_t top = ((int64_t) (value + 1)) % kernel_size_data[d];
        value -= floor(value);

        b *= (1 - mod) * value + mod * (1 - value);
        i += ((1 - mod) * top + mod * bot) * bla;
      }
      basis_data[s * basis_stride] = b;
      weight_index_data[s * weight_index_stride] = i;
    })
}

#endif
