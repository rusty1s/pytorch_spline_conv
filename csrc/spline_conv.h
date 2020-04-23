#pragma once

#include <torch/extension.h>

int64_t cuda_version();

std::tuple<torch::Tensor, torch::Tensor>
spline_basis(torch::Tensor pseudo, torch::Tensor kernel_size,
             torch::Tensor is_open_spline, int64_t degree);

torch::Tensor spline_weighting(torch::Tensor x, torch::Tensor weight,
                               torch::Tensor basis, torch::Tensor weight_index);
