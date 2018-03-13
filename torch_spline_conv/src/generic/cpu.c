#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/cpu.c"
#else

void spline_(linear_basis_forward)(THTensor *basis, THLongTensor *weight_index, THTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K) {
  SPLINE_BASIS_FORWARD(1, basis, weight_index, pseudo, kernel_size, is_open_spline, K,
    value = 1 - value - k_mod + 2 * value * k_mod;
  )
}

void spline_(quadratic_basis_forward)(THTensor *basis, THLongTensor *weight_index, THTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K) {
  SPLINE_BASIS_FORWARD(2, basis, weight_index, pseudo, kernel_size, is_open_spline, K,
    if (k_mod == 0) value = 0.5 * value * value - value + 0.5;
    else if (k_mod == 1) value = -value * value + value + 0.5;
    else value = 0.5 * value * value;
  )
}

void spline_(cubic_basis_forward)(THTensor *basis, THLongTensor *weight_index, THTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline, int K) {
  SPLINE_BASIS_FORWARD(3, basis, weight_index, pseudo, kernel_size, is_open_spline, K,
    if (k_mod == 0) { value = (1 - value); value = value * value * value / 6.0; }
    else if (k_mod == 1) value = (3 * value * value * value - 6 * value * value + 4) / 6;
    else if (k_mod == 2) value = (-3 * value * value * value + 3 * value * value + 3 * value + 1) / 6;
    else value = value * value * value / 6;
  )
}

void spline_(linear_basis_backward)(THTensor *grad_pseudo, THTensor *grad_basis, THTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline) {
  SPLINE_BASIS_BACKWARD(1, grad_pseudo, grad_basis, pseudo, kernel_size, is_open_spline,
    value = 1 - value - k_mod + 2 * value * k_mod;
    ,
    value = -1 + k_mod + k_mod;
  )
}

void spline_(quadratic_basis_backward)(THTensor *grad_pseudo, THTensor *grad_basis, THTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline) {
  SPLINE_BASIS_BACKWARD(2, grad_pseudo, grad_basis, pseudo, kernel_size, is_open_spline,
    if (k_mod == 0) value = 0.5 * value * value - value + 0.5;
    else if (k_mod == 1) value = -value * value + value + 0.5;
    else value = 0.5 * value * value;
    ,
    if (k_mod == 0) value = value - 1;
    else if (k_mod == 1) value = -2 * value + 1;
    else value = value;
  )
}

void spline_(cubic_basis_backward)(THTensor *grad_pseudo, THTensor *grad_basis, THTensor *pseudo, THLongTensor *kernel_size, THByteTensor *is_open_spline) {
  SPLINE_BASIS_BACKWARD(3, grad_pseudo, grad_basis, pseudo, kernel_size, is_open_spline,
    if (k_mod == 0) { value = (1 - value); value = value * value * value / 6.0; }
    else if (k_mod == 1) value = (3 * value * value * value - 6 * value * value + 4) / 6;
    else if (k_mod == 2) value = (-3 * value * value * value + 3 * value * value + 3 * value + 1) / 6;
    else value = value * value * value / 6;
    ,
    if (k_mod == 0) value = (-value * value + 2 * value - 1) / 2;
    else if (k_mod == 1) value = (3 * value * value - 4 * value) / 2;
    else if (k_mod == 2) value = (-3 * value * value + 2 * value + 1) / 2;
    else value = value * value / 2;
  )
}

void spline_(weighting_forward)(THTensor *output, THTensor *input, THTensor *weight, THTensor *basis, THLongTensor *weight_index) {
  real *weight_data = weight->storage->data + weight->storageOffset; real b;
  SPLINE_WEIGHTING(output, input, basis, weight_index, THTensor_(size)(weight, 1), THTensor_(size)(weight, 2), THLongTensor_size(weight_index, 1),
    for (m_out = 0; m_out < M_out; m_out++) {
      value = 0;
      for (s = 0; s < S; s++) {
        b = *(basis_data + s * basis_stride);
        w_idx = *(weight_index_data + s * weight_index_stride);
        for (m_in = 0; m_in < M_in; m_in++) {
          value += b * *(weight_data + w_idx * M_in * M_out + m_in * M_out + m_out) * *(input_data + m_in * input_stride);
        }
      }
      output_data[m_out * output_stride] = value;
    }
  )
}

void spline_(weighting_backward_input)(THTensor *grad_input, THTensor *grad_output, THTensor *weight, THTensor *basis, THLongTensor *weight_index) {
  real *weight_data = weight->storage->data + weight->storageOffset; real b;
  SPLINE_WEIGHTING(grad_input, grad_output, basis, weight_index, THTensor_(size)(weight, 1), THTensor_(size)(weight, 2), THLongTensor_size(weight_index, 1),
    for (m_in = 0; m_in < M_in; m_in++) {
      value = 0;
      for (s = 0; s < S; s++) {
        b = *(basis_data + s * basis_stride);
        w_idx = *(weight_index_data + s * weight_index_stride);
        for (m_out = 0; m_out < M_out; m_out++) {
          value += b * *(grad_output_data + m_out * grad_output_stride) * *(weight_data + w_idx * M_in * M_out + m_in * M_out + m_out);
        }
      }
      grad_input_data[m_in * grad_input_stride] = value;
    }
  )
}

void spline_(weighting_backward_basis)(THTensor *grad_basis, THTensor *grad_output, THTensor *input, THTensor *weight, THLongTensor *weight_index) {
  real *weight_data = weight->storage->data + weight->storageOffset;
  SPLINE_WEIGHTING(grad_basis, grad_output, input, weight_index, THTensor_(size)(weight, 1), THTensor_(size)(weight, 2), THLongTensor_size(weight_index, 1),
    for (m_out = 0; m_out < M_out; m_out++) {
      for (s = 0; s < S; s++) {
        w_idx = *(weight_index_data + s * weight_index_stride); value = 0;
        for (m_in = 0; m_in < M_in; m_in++) {
          value += *(input_data + m_in * input_stride) * *(weight_data + w_idx * M_in * M_out + m_in * M_out + m_out);
        }
        grad_basis_data[s * grad_basis_stride] += value * *(grad_output_data + m_out * grad_output_stride);
      }
    }
  )
}

void spline_(weighting_backward_weight)(THTensor *grad_weight, THTensor *grad_output, THTensor *input, THTensor *basis, THLongTensor *weight_index) {
  real *grad_weight_data = grad_weight->storage->data + grad_weight->storageOffset; real b;
  SPLINE_WEIGHTING(grad_output, input, basis, weight_index, THTensor_(size)(input, 1), THTensor_(size)(grad_output, 1), THLongTensor_size(weight_index, 1),
    for (m_out = 0; m_out < M_out; m_out++) {
      value = *(grad_output_data + m_out * grad_output_stride);
      for (s = 0; s < S; s++) {
        b = *(basis_data + s * basis_stride);
        w_idx = *(weight_index_data + s * weight_index_stride);
        for (m_in = 0; m_in < M_in; m_in++) {
          grad_weight_data[w_idx * M_in * M_out + m_in * M_out + m_out] += b * value * *(input_data + m_in * input_stride);
        }
      }
    }
  )
}

#endif
