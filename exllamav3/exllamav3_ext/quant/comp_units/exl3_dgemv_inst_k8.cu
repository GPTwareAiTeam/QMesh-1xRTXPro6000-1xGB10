#include <cuda_fp16.h>
#include "../../util.h"
#include "../../util.cuh"
#include "../exl3_dgemv_kernel.cuh"
#include "exl3_dgemv_instances.cuh"

EXL3_DGEMV_SEL_DEF(8)
