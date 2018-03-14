#include <THC.h>

#include "kernel.h"

#define spline_(NAME) TH_CONCAT_4(spline_, NAME, _kernel_, Real)

#include "generic/kernel.cu"
#include "THCGenerateFloatType.h"
#include "generic/kernel.cu"
#include "THCGenerateDoubleType.h"
