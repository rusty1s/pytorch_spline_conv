#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

void spline_(basis_linear)(THTensor *basis, THLongTensor *weight_index, THTensor *pseudo, THTensor *kernel_size, THByteTensor *is_open_spline, int K) {
  int64_t *kernel_size_data = kernel_size->storage->data + kernel_size->storageOffset;
  uint8_t *is_open_spline_data = is_open_spline->storage->data + is_open_spline->storageOffset;

  int64_t k, s, S, d, D;
  real value;
  D = THTensor_(size)(pseudo, 1);
  S = THLongTensor_size(weight_index, 1);
  TH_TENSOR_DIM_APPLY3(real, basis, int64_t, weight_index, real, pseudo, 1, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
    for (s = 0; s < S; s++) {
  /*     /1* k = K; *1/ */
      real b = 1; int64_t i = 0;

      for (d = 0; d < D; d++) {
  /*       /1* k /= kernel_size[d]; *1/ */




        value = *(pseudo_data + d * pseudo_stride) * (kernel_size_data[d] - is_open_spline_data[d]);
        int64_t bot = ((int64_t) value) % kernel_size_data[d];
        int64_t top = ((int64_t) (value + 1)) % kernel_size_data[d];
        value -= floor(value);

        int mod = s % 2;
        b *= (1 - mod) * value + mod * (1 - value);
        i += (1 - mod) * top + mod * bot;


  /*       /1* int bot = int64_t(value); *1/ */
  /*       /1* int top = (bot + 1) % kernel_size[d]; *1/ */
  /*       /1* bot %= kernel_size[d]; *1/ */
      }
      basis_data[s * basis_stride] = b;
      weight_index_data[s * weight_index_stride] = i;
    })
}


/* void spline_(linear)(THFloatTensor *amount, THLongTensor *index, THFloatTensor *input, THLongTensor *kernel, THByteTensor *open) { */
/*   // s = (m+1)^d */
/*   // amount: E x s */
/*   // index: E x s */
/*   // input: E x d */
/*   // kernel: d */
/*   // open: d */
/*   // */
/*   int64_t i, d; */
/*   int64_t E = THLongTensor_size(index, 0); */
/*   int64_t K = THLongTensor_size(index, 1); */
/*   int64_t D = THLongTensor_size(kernel, 0); */
/*   for (i = 0; i < E * K; i++) { */
/*     for (d = 0; d < D; d++) { */
/*     } */
/*   } */
/* } */

/* void spline_(quadratic)(THFloatTensor *amount, THLongTensor *index, THFloatTensor *input, THLongTensor *kernel, THByteTensor *open) { */
/*   int64_t i; */
/*   for (i = 0; i < THLongTensor_size(input, dim); i++) { */
/*   } */
/* } */

/* void spline_(cubic)(THFloatTensor *amount, THLongTensor *index, THFloatTensor *input, THLongTensor *kernel, THByteTensor *open) { */
/*   int64_t i; */
/*   for (i = 0; i < THLongTensor_size(input, dim); i++) { */
/*   } */
/* } */

#endif
