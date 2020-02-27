#include <torch/extension.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")

at::Tensor weighting_fw_cuda(at::Tensor x, at::Tensor weight, at::Tensor basis,
                             at::Tensor weight_index);

at::Tensor weighting_bw_x_cuda(at::Tensor grad_out, at::Tensor weight,
                               at::Tensor basis, at::Tensor weight_index);

at::Tensor weighting_bw_w_cuda(at::Tensor grad_out, at::Tensor x,
                               at::Tensor basis, at::Tensor weight_index,
                               int64_t K);

at::Tensor weighting_bw_b_cuda(at::Tensor grad_out, at::Tensor x,
                               at::Tensor weight, at::Tensor weight_index);

at::Tensor weighting_fw(at::Tensor x, at::Tensor weight, at::Tensor basis,
                        at::Tensor weight_index) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight);
  CHECK_CUDA(basis);
  CHECK_CUDA(weight_index);
  return weighting_fw_cuda(x, weight, basis, weight_index);
}

at::Tensor weighting_bw_x(at::Tensor grad_out, at::Tensor weight,
                          at::Tensor basis, at::Tensor weight_index) {
  CHECK_CUDA(grad_out);
  CHECK_CUDA(weight);
  CHECK_CUDA(basis);
  CHECK_CUDA(weight_index);
  return weighting_bw_x_cuda(grad_out, weight, basis, weight_index);
}

at::Tensor weighting_bw_w(at::Tensor grad_out, at::Tensor x, at::Tensor basis,
                          at::Tensor weight_index, int64_t K) {
  CHECK_CUDA(grad_out);
  CHECK_CUDA(x);
  CHECK_CUDA(basis);
  CHECK_CUDA(weight_index);
  return weighting_bw_w_cuda(grad_out, x, basis, weight_index, K);
}

at::Tensor weighting_bw_b(at::Tensor grad_out, at::Tensor x, at::Tensor weight,
                          at::Tensor weight_index) {
  CHECK_CUDA(grad_out);
  CHECK_CUDA(x);
  CHECK_CUDA(weight);
  CHECK_CUDA(weight_index);
  return weighting_bw_b_cuda(grad_out, x, weight, weight_index);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weighting_fw", &weighting_fw, "Weighting Forward (CUDA)");
  m.def("weighting_bw_x", &weighting_bw_x, "Weighting Backward X (CUDA)");
  m.def("weighting_bw_w", &weighting_bw_w, "Weighting Backward Weight (CUDA)");
  m.def("weighting_bw_b", &weighting_bw_b, "Weighting Backward Basis (CUDA)");
}
#define BLOCKS(N) (N + THREADS - 1) / THREADS
