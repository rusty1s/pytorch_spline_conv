#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

#define AT_DISPATCH_DEGREE_TYPES(degree, ...)                                  \
  [&] {                                                                        \
    switch (degree) {                                                          \
    case 1: {                                                                  \
      const int64_t DEGREE = 1;                                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case 2: {                                                                  \
      const int64_t DEGREE = 2;                                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case 3: {                                                                  \
      const int64_t DEGREE = 3;                                                \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      AT_ERROR("Basis degree not implemented");                                \
    }                                                                          \
  }()
