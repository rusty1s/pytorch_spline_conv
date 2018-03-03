#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

void spline_(basis_linear)(THTensor *basis, THLongTensor *weight_index, THTensor *pseudo, THTensor *kernel_size, THByteTensor *is_open_spline, int K) {
  int64_t k, s, S, d, D;
  real value;
  D = THTensor_(size)(pseudo, 1);
  S = THLongTEnsor_size(weight_index, 1);
  TH_TENSOR_DIM_APPLY3(real, basis, int64_t, weight_index, real, pseudo, 1, TH_TENSOR_DIM_APPLY3_SIZE_EX_EXCEPT_DIM,
    for (s = 0; s < S; s++) {
      /* k = K; */
      /* b = 1; i = 0; */

      for (d = 0; d < D; d++) {
        /* k /= kernel_size[d]; */




        /* value = *(pseudo_data + d * pseudo_stride) * (kernel_size[d] - is_open_spline[d]); */

        /* int bot = int64_t(value); */
        /* int top = (bot + 1) % kernel_size[d]; */
        /* bot %= kernel_size[d]; */
      }
      basis_data[s * basis_stride] = 1;
      weight_index[s * weight_index_stride] = 2;
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
