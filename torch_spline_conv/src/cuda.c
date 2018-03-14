#include <THC/THC.h>

#include "kernel.h"

#define spline_(NAME) TH_CONCAT_4(spline_, NAME, _cuda_, Real)
#define spline_kernel_(NAME) TH_CONCAT_4(spline_, NAME, _kernel_, Real)

extern THCState *state;

#include "generic/cuda.c"
#include "THCGenerateFloatType.h"
#include "generic/cuda.c"
#include "THCGenerateDoubleType.h"
