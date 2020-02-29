#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor>
spline_basis_fw_cuda(torch::Tensor pseudo, torch::Tensor kernel_size,
                     torch::Tensor is_open_spline, int64_t degree);

torch::Tensor spline_basis_bw_cuda(torch::Tensor grad_basis,
                                   torch::Tensor pseudo,
                                   torch::Tensor kernel_size,
                                   torch::Tensor is_open_spline,
                                   int64_t degree);
