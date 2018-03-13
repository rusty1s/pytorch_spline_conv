#include <TH/TH.h>

#include "THTensorDimApply.h"

#define spline_(NAME) TH_CONCAT_4(spline_, NAME, _, Real)

#define SPLINE_BASIS_FORWARD(M, basis, weight_index, pseudo, kernel_size, is_open_spline, K, CODE) { \
  int64_t *kernel_size_data = kernel_size->storage->data + kernel_size->storageOffset; \
  uint8_t *is_open_spline_data = is_open_spline->storage->data + is_open_spline->storageOffset; \
  int64_t S = THLongTensor_size(weight_index, 1); \
  int64_t D = THTensor_(size)(pseudo, 1); \
  int64_t s, d, k, k_mod, i, offset; real b, value; \
\
  TH_TENSOR_DIM_APPLY3(real, basis, int64_t, weight_index, real, pseudo, 1, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM, \
    for (s = 0; s < S; s++) { \
      b = 1; i = 0; k = s; offset = K; \
      for (d = 0; d < D; d++) { \
        offset /= kernel_size_data[d]; \
        k_mod = k % (M + 1); \
        k /= M + 1; \
        value = *(pseudo_data + d * pseudo_stride) * (kernel_size_data[d] - M * is_open_spline_data[d]); \
        i += (((int64_t) value + k_mod) % kernel_size_data[d]) * offset; \
        value -= floor(value); \
        CODE \
        b *= value; \
      } \
      basis_data[s * basis_stride] = b; \
      weight_index_data[s * weight_index_stride] = i; \
    }) \
}

#define SPLINE_BASIS_BACKWARD(M, grad_pseudo, grad_basis, pseudo, kernel_size, is_open_spline, EVAL_CODE, GRAD_CODE) { \
  int64_t *kernel_size_data = kernel_size->storage->data + kernel_size->storageOffset; \
  uint8_t *is_open_spline_data = is_open_spline->storage->data + is_open_spline->storageOffset; \
  int64_t D = THTensor_(size)(pseudo, 1); \
  int64_t S = THTensor_(size)(grad_basis, 1); \
  int64_t d, s, d_it, quotient, k_mod; real g_out, g, value;\
\
  TH_TENSOR_DIM_APPLY3(real, grad_pseudo, real, grad_basis, real, pseudo, 1, TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM, \
    for (d = 0; d < D; d++) { \
      g_out = 0; \
      quotient = pow(M + 1, d); \
      for (s = 0; s < S; s++) { \
        k_mod = (s / quotient) % (M + 1); \
        value = *(pseudo_data + d * pseudo_stride) * (kernel_size_data[d] - M * is_open_spline_data[d]); \
        value -= floor(value); \
        GRAD_CODE \
        g = value; \
\
        for (d_it = 0; d_it < D; d_it++) { \
          if (d_it != d) { \
            k_mod = (s / (int64_t) pow(M + 1, d_it)) % (M + 1); \
            value = *(pseudo_data + d_it * pseudo_stride) * (kernel_size_data[d_it] - M * is_open_spline_data[d_it]); \
            value -= floor(value); \
            EVAL_CODE \
            g *= value; \
          } \
        } \
        g_out += g * *(grad_basis_data + s * grad_basis_stride); \
      } \
      grad_pseudo_data[d * grad_pseudo_stride] = g_out * (kernel_size_data[d] - M * is_open_spline_data[d]); \
    } \
  ) \
}

#define SPLINE_WEIGHTING(TENSOR1, TENSOR2, TENSOR3, weight_index, M_IN, M_OUT, M_S, CODE) { \
  int64_t M_in = M_IN; int64_t M_out = M_OUT; int64_t S = M_S; \
  int64_t m_in, m_out, s, w_idx; real value; \
  TH_TENSOR_DIM_APPLY4(real, TENSOR1, real, TENSOR2, real, TENSOR3, int64_t, weight_index, 1, CODE) \
}

#include "generic/cpu.c"
#include "THGenerateFloatType.h"
#include "generic/cpu.c"
#include "THGenerateDoubleType.h"
