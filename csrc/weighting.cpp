#include <Python.h>
#include <torch/script.h>

#include "cpu/weighting_cpu.h"

#ifdef WITH_CUDA
#include "cuda/weighting_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__weighting_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__weighting_cpu(void) { return NULL; }
#endif
#endif

torch::Tensor spline_weighting_fw(torch::Tensor x, torch::Tensor weight,
                                  torch::Tensor basis,
                                  torch::Tensor weight_index) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return spline_weighting_fw_cuda(x, weight, basis, weight_index);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spline_weighting_fw_cpu(x, weight, basis, weight_index);
  }
}

torch::Tensor spline_weighting_bw_x(torch::Tensor grad_out,
                                    torch::Tensor weight, torch::Tensor basis,
                                    torch::Tensor weight_index) {
  if (grad_out.device().is_cuda()) {
#ifdef WITH_CUDA
    return spline_weighting_bw_x_cuda(grad_out, weight, basis, weight_index);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spline_weighting_bw_x_cpu(grad_out, weight, basis, weight_index);
  }
}

torch::Tensor spline_weighting_bw_weight(torch::Tensor grad_out,
                                         torch::Tensor x, torch::Tensor basis,
                                         torch::Tensor weight_index,
                                         int64_t kernel_size) {
  if (grad_out.device().is_cuda()) {
#ifdef WITH_CUDA
    return spline_weighting_bw_weight_cuda(grad_out, x, basis, weight_index,
                                           kernel_size);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spline_weighting_bw_weight_cpu(grad_out, x, basis, weight_index,
                                          kernel_size);
  }
}

torch::Tensor spline_weighting_bw_basis(torch::Tensor grad_out, torch::Tensor x,
                                        torch::Tensor weight,
                                        torch::Tensor weight_index) {
  if (grad_out.device().is_cuda()) {
#ifdef WITH_CUDA
    return spline_weighting_bw_basis_cuda(grad_out, x, weight, weight_index);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spline_weighting_bw_basis_cpu(grad_out, x, weight, weight_index);
  }
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SplineWeighting : public torch::autograd::Function<SplineWeighting> {
public:
  static variable_list forward(AutogradContext *ctx, Variable x,
                               Variable weight, Variable basis,
                               Variable weight_index) {
    auto out = spline_weighting_fw(x, weight, basis, weight_index);
    ctx->save_for_backward({x, weight, basis, weight_index});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto x = saved[0], weight = saved[1], basis = saved[2],
         weight_index = saved[3];

    auto grad_x = Variable();
    if (torch::autograd::any_variable_requires_grad({x})) {
      grad_x = spline_weighting_bw_x(grad_out, weight, basis, weight_index);
    }

    auto grad_weight = Variable();
    if (torch::autograd::any_variable_requires_grad({weight})) {
      grad_weight = spline_weighting_bw_weight(grad_out, x, basis, weight_index,
                                               weight.size(0));
    }

    auto grad_basis = Variable();
    if (torch::autograd::any_variable_requires_grad({basis})) {
      grad_basis = spline_weighting_bw_basis(grad_out, x, weight, weight_index);
    }

    return {grad_x, grad_weight, grad_basis, Variable()};
  }
};

torch::Tensor spline_weighting(torch::Tensor x, torch::Tensor weight,
                               torch::Tensor basis,
                               torch::Tensor weight_index) {
  x = x.contiguous();
  weight = weight.contiguous();
  return SplineWeighting::apply(x, weight, basis, weight_index)[0];
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv::spline_weighting", &spline_weighting);
