#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")

std::tuple<at::Tensor, at::Tensor> linear_fw_cuda(at::Tensor pseudo,
                                                  at::Tensor kernel_size,
                                                  at::Tensor is_open_spline);

std::tuple<at::Tensor, at::Tensor> quadratic_fw_cuda(at::Tensor pseudo,
                                                     at::Tensor kernel_size,
                                                     at::Tensor is_open_spline);

std::tuple<at::Tensor, at::Tensor> cubic_fw_cuda(at::Tensor pseudo,
                                                 at::Tensor kernel_size,
                                                 at::Tensor is_open_spline);

at::Tensor linear_bw_cuda(at::Tensor grad_basis, at::Tensor pseudo,
                          at::Tensor kernel_size, at::Tensor is_open_spline);

at::Tensor quadratic_bw_cuda(at::Tensor grad_basis, at::Tensor pseudo,
                             at::Tensor kernel_size, at::Tensor is_open_spline);

at::Tensor cubic_bw_cuda(at::Tensor grad_basis, at::Tensor pseudo,
                         at::Tensor kernel_size, at::Tensor is_open_spline);

std::tuple<at::Tensor, at::Tensor> linear_fw(at::Tensor pseudo,
                                             at::Tensor kernel_size,
                                             at::Tensor is_open_spline) {
  CHECK_CUDA(pseudo);
  CHECK_CUDA(kernel_size);
  CHECK_CUDA(is_open_spline);
  return linear_fw_cuda(pseudo, kernel_size, is_open_spline);
}

std::tuple<at::Tensor, at::Tensor> quadratic_fw(at::Tensor pseudo,
                                                at::Tensor kernel_size,
                                                at::Tensor is_open_spline) {
  CHECK_CUDA(pseudo);
  CHECK_CUDA(kernel_size);
  CHECK_CUDA(is_open_spline);
  return quadratic_fw_cuda(pseudo, kernel_size, is_open_spline);
}

std::tuple<at::Tensor, at::Tensor>
cubic_fw(at::Tensor pseudo, at::Tensor kernel_size, at::Tensor is_open_spline) {
  CHECK_CUDA(pseudo);
  CHECK_CUDA(kernel_size);
  CHECK_CUDA(is_open_spline);
  return cubic_fw_cuda(pseudo, kernel_size, is_open_spline);
}

at::Tensor linear_bw(at::Tensor grad_basis, at::Tensor pseudo,
                     at::Tensor kernel_size, at::Tensor is_open_spline) {
  CHECK_CUDA(grad_basis);
  CHECK_CUDA(pseudo);
  CHECK_CUDA(kernel_size);
  CHECK_CUDA(is_open_spline);
  return linear_bw_cuda(grad_basis, pseudo, kernel_size, is_open_spline);
}

at::Tensor quadratic_bw(at::Tensor grad_basis, at::Tensor pseudo,
                        at::Tensor kernel_size, at::Tensor is_open_spline) {
  CHECK_CUDA(grad_basis);
  CHECK_CUDA(pseudo);
  CHECK_CUDA(kernel_size);
  CHECK_CUDA(is_open_spline);
  return quadratic_bw_cuda(grad_basis, pseudo, kernel_size, is_open_spline);
}

at::Tensor cubic_bw(at::Tensor grad_basis, at::Tensor pseudo,
                    at::Tensor kernel_size, at::Tensor is_open_spline) {
  CHECK_CUDA(grad_basis);
  CHECK_CUDA(pseudo);
  CHECK_CUDA(kernel_size);
  CHECK_CUDA(is_open_spline);
  return cubic_bw_cuda(grad_basis, pseudo, kernel_size, is_open_spline);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_fw", &linear_fw, "Linear Basis Forward (CUDA)");
  m.def("quadratic_fw", &quadratic_fw, "Quadratic Basis Forward (CUDA)");
  m.def("cubic_fw", &cubic_fw, "Cubic Basis Forward (CUDA)");
  m.def("linear_bw", &linear_bw, "Linear Basis Backward (CUDA)");
  m.def("quadratic_bw", &quadratic_bw, "Quadratic Basis Backward (CUDA)");
  m.def("cubic_bw", &cubic_bw, "Cubic Basis Backward (CUDA)");
}
