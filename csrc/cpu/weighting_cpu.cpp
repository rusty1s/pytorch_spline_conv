#include "weighting_cpu.h"

#include "utils.h"

torch::Tensor spline_weighting_fw_cpu(torch::Tensor x, torch::Tensor weight,
                                      torch::Tensor basis,
                                      torch::Tensor weight_index) {
  return x;
}

torch::Tensor spline_weighting_bw_x_cpu(torch::Tensor grad_out,
                                        torch::Tensor weight,
                                        torch::Tensor basis,
                                        torch::Tensor weight_index) {
  return grad_out;
}

torch::Tensor spline_weighting_bw_weight_cpu(torch::Tensor grad_out,
                                             torch::Tensor x,
                                             torch::Tensor basis,
                                             torch::Tensor weight_index,
                                             int64_t kernel_size) {
  return grad_out;
}

torch::Tensor spline_weighting_bw_basis_cpu(torch::Tensor grad_out,
                                            torch::Tensor x,
                                            torch::Tensor weight,
                                            torch::Tensor weight_index) {
  return grad_out;
}
