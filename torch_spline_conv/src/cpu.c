#include <TH/TH.h>

#include "THTensorDimApply.h"

#define spline_(NAME) TH_CONCAT_4(spline_, NAME, _, Real)

#include "generic/cpu.c"
#include "THGenerateFloatType.h"
#include "generic/cpu.c"
#include "THGenerateDoubleType.h"
