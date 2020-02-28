#include "basis_cuda.h"

#include "utils.cuh"

std::tuple<torch::Tensor, torch::Tensor>
spline_basis_fw_cpu(torch::Tensor pseudo, torch::Tensor kernel_size,
                    torch::Tensor is_open_spline, int64_t degree) {
  return std::make_tuple(pseudo, kernel_size);
}

torch::Tensor spline_basis_bw_cpu(torch::Tensor grad_basis,
                                  torch::Tensor pseudo,
                                  torch::Tensor kernel_size,
                                  torch::Tensor is_open_spline,
                                  int64_t degree) {
  return grad_basis;
}
