// Per-expert mixed-K decode kernels (mul1 only: half-bit rates exist only under mul1). Launchers for the
#include <algorithm>
#include "exl3_moe_coop_instances.cuh"
#include "../exl3_moe_coop_mixed_kernel.cuh"

MoeCoopKernel exl3_moe_coop_kernel_a_mixed(int Hi, bool wide)
{
    using namespace exl3_moe_coop_mixed_ns;
    const int smem = mixed_smem_a(Hi);
    if (wide) return { (void*) exl3_moe_coop_a_mixed_kernel<2, true>, smem };
    return { (void*) exl3_moe_coop_a_mixed_kernel<2, false>, smem };
}

MoeCoopKernel exl3_moe_coop_kernel_b_mixed(bool wide)
{
    using namespace exl3_moe_coop_mixed_ns;
    const int smem = mixed_smem_b();
    if (wide) return { (void*) exl3_moe_coop_b_mixed_kernel<2, true>, smem };
    return { (void*) exl3_moe_coop_b_mixed_kernel<2, false>, smem };
}

// Fused persistent kernel: the A and B bodies share one grid, so the shared-memory size is the larger of the two
MoeCoopKernel exl3_moe_coop_kernel_fused_mixed(int Hi, bool wide_a, bool wide_b)
{
    using namespace exl3_moe_coop_mixed_ns;
    // One grid runs both stages: its dynamic shared memory is the larger of the A and B layouts
    const int smem = std::max(mixed_smem_a(Hi), mixed_smem_b());
    if (wide_a)
    {
        if (wide_b) return { (void*) exl3_moe_coop_fused_mixed_kernel<true, true>, smem };
        return { (void*) exl3_moe_coop_fused_mixed_kernel<true, false>, smem };
    }
    if (wide_b) return { (void*) exl3_moe_coop_fused_mixed_kernel<false, true>, smem };
    return { (void*) exl3_moe_coop_fused_mixed_kernel<false, false>, smem };
}
