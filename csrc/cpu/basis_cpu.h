#pragma once

#include <torch/extension.h>

#if defined(__linux__) && defined(__x86_64__)
__asm__(".symver pow,pow@GLIBC_2.2.5");
#endif

std::tuple<torch::Tensor, torch::Tensor>
spline_basis_fw_cpu(torch::Tensor pseudo, torch::Tensor kernel_size,
                    torch::Tensor is_open_spline, int64_t degree);

torch::Tensor spline_basis_bw_cpu(torch::Tensor grad_basis,
                                  torch::Tensor pseudo,
                                  torch::Tensor kernel_size,
                                  torch::Tensor is_open_spline, int64_t degree);
