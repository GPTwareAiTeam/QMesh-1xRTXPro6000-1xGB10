#include "exl3_moe_inst_common.cuh"

// Paired trellis layout (quant/trellis_pair.cuh), mul1, CUDA only: the fused MoE kernel over a paired-layout pack
#if !defined(USE_ROCM)
fp_exl3_moe_kernel exl3_moe_kernel_k4_n128_cb2_pair() { return exl3_moe_kernel<4, 128, 2, MOE_TILESIZE_M, false, false, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k4_n256_cb2_pair() { return exl3_moe_kernel<4, 256, 2, MOE_TILESIZE_M, false, false, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k4_n128_cb2_m32_pair() { return exl3_moe_kernel<4, 128, 2, 32, false, false, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k4_n128_cb2_m64_pair() { return exl3_moe_kernel<4, 128, 2, 64, false, false, true>; }
#endif
