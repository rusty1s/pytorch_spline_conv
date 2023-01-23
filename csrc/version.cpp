#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#ifdef USE_ROCM
#include <hip/hip_version.h>
#else
#include <cuda.h>
#endif
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__version_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__version_cpu(void) { return NULL; }
#endif
#endif

int64_t cuda_version() {
#ifdef WITH_CUDA
#ifdef USE_ROCM
  return HIP_VERSION;
#else
  return CUDA_VERSION;
#endif
#else
  return -1;
#endif
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv::cuda_version", [] { return cuda_version(); });
