#include <THC.h>

#include "kernel.h"

#include "common.cuh"
#include "THCBasisForward.cuh"

#define spline_(NAME) TH_CONCAT_4(spline_, NAME, _kernel_, Real)
#define thc_(NAME) TH_CONCAT_4(thc_, NAME, _, Real)

#include "generic/common.cu"
#include "THCGenerateAllTypes.h"

#include "generic/kernel.cu"
#include "THCGenerateFloatType.h"
#include "generic/kernel.cu"
#include "THCGenerateDoubleType.h"
