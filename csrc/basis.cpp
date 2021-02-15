#include <Python.h>
#include <torch/script.h>

#include "cpu/basis_cpu.h"

#ifdef WITH_CUDA
#include "cuda/basis_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__basis_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__basis_cpu(void) { return NULL; }
#endif
#endif

std::tuple<torch::Tensor, torch::Tensor>
spline_basis_fw(torch::Tensor pseudo, torch::Tensor kernel_size,
                torch::Tensor is_open_spline, int64_t degree) {
  if (pseudo.device().is_cuda()) {
#ifdef WITH_CUDA
    return spline_basis_fw_cuda(pseudo, kernel_size, is_open_spline, degree);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spline_basis_fw_cpu(pseudo, kernel_size, is_open_spline, degree);
  }
}

torch::Tensor spline_basis_bw(torch::Tensor grad_basis, torch::Tensor pseudo,
                              torch::Tensor kernel_size,
                              torch::Tensor is_open_spline, int64_t degree) {
  if (grad_basis.device().is_cuda()) {
#ifdef WITH_CUDA
    return spline_basis_bw_cuda(grad_basis, pseudo, kernel_size, is_open_spline,
                                degree);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spline_basis_bw_cpu(grad_basis, pseudo, kernel_size, is_open_spline,
                               degree);
  }
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SplineBasis : public torch::autograd::Function<SplineBasis> {
public:
  static variable_list forward(AutogradContext *ctx, Variable pseudo,
                               Variable kernel_size, Variable is_open_spline,
                               int64_t degree) {
    ctx->saved_data["degree"] = degree;
    auto result = spline_basis_fw(pseudo, kernel_size, is_open_spline, degree);
    auto basis = std::get<0>(result), weight_index = std::get<1>(result);
    ctx->save_for_backward({pseudo, kernel_size, is_open_spline});
    ctx->mark_non_differentiable({weight_index});
    return {basis, weight_index};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_basis = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto pseudo = saved[0], kernel_size = saved[1], is_open_spline = saved[2];
    auto degree = ctx->saved_data["degree"].toInt();
    auto grad_pseudo = spline_basis_bw(grad_basis, pseudo, kernel_size,
                                       is_open_spline, degree);
    return {grad_pseudo, Variable(), Variable(), Variable()};
  }
};

std::tuple<torch::Tensor, torch::Tensor>
spline_basis(torch::Tensor pseudo, torch::Tensor kernel_size,
             torch::Tensor is_open_spline, int64_t degree) {
  pseudo = pseudo.contiguous();
  auto result = SplineBasis::apply(pseudo, kernel_size, is_open_spline, degree);
  return std::make_tuple(result[0], result[1]);
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv::spline_basis", &spline_basis);
