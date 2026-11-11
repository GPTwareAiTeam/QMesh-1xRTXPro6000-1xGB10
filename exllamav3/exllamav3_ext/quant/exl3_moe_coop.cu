// Fused decode-shaped MoE path (bsz 1..MAX_BSZN): launcher and tensor front end. Kernels in
// exl3_moe_coop_kernel.cuh, instantiated per (K, codebook) in comp_units/exl3_moe_coop_inst_k*.cu

#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <cmath>

#include "../util.h"
#include "../util.cuh"
#include "exl3_devctx.cuh"
#include "exl3_moe_coop.cuh"
#include "comp_units/exl3_moe_coop_instances.cuh"
#include "bits_k.cuh"
#define EXL3_MOE_COOP_DEFINE_ROT
#include "exl3_moe_coop_kernel.cuh"

// Output rows of a launch without routed slots (top-k 0): the empty sum, plus the shared-expert term
// when there is one. One warp per (row, 128-column chunk)
__global__ __launch_bounds__(32)
void exl3_moe_coop_empty_kernel(const MoeCoopParams p)
{
    exl3_moe_coop_ns::write_empty_row_chunk(p, blockIdx.x, blockIdx.y, threadIdx.x);
}

static MoeCoopKernel moe_coop_kernel_a(float K_, int cb, int Hi, bool wide, bool paired)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        TORCH_CHECK(cb == 2, "exl3_moe_coop: half-integer bitrates require the mul1 codebook");
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_a_h1(Hi, wide);
            case 2: return exl3_moe_coop_kernel_a_h2(Hi, wide);
            default: return exl3_moe_coop_kernel_a_h3(Hi, wide);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_a_k1(cb, Hi, wide);
        case 2: return exl3_moe_coop_kernel_a_k2(cb, Hi, wide, paired);
        case 3: return exl3_moe_coop_kernel_a_k3(cb, Hi, wide, paired);
        case 4: return exl3_moe_coop_kernel_a_k4(cb, Hi, wide, paired);
        case 5: return exl3_moe_coop_kernel_a_k5(cb, Hi, wide);
        case 6: return exl3_moe_coop_kernel_a_k6(cb, Hi, wide);
        case 7: return exl3_moe_coop_kernel_a_k7(cb, Hi, wide);
        default: return exl3_moe_coop_kernel_a_k8(cb, Hi, wide);
    }
}

static MoeCoopKernel moe_coop_kernel_b(float K_, int cb, bool wide, bool paired)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        TORCH_CHECK(cb == 2, "exl3_moe_coop: half-integer bitrates require the mul1 codebook");
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_b_h1(wide);
            case 2: return exl3_moe_coop_kernel_b_h2(wide);
            default: return exl3_moe_coop_kernel_b_h3(wide);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_b_k1(cb, wide);
        case 2: return exl3_moe_coop_kernel_b_k2(cb, wide, paired);
        case 3: return exl3_moe_coop_kernel_b_k3(cb, wide, paired);
        case 4: return exl3_moe_coop_kernel_b_k4(cb, wide, paired);
        case 5: return exl3_moe_coop_kernel_b_k5(cb, wide);
        case 6: return exl3_moe_coop_kernel_b_k6(cb, wide);
        case 7: return exl3_moe_coop_kernel_b_k7(cb, wide);
        default: return exl3_moe_coop_kernel_b_k8(cb, wide);
    }
}

// Batch-invariant launch shapes (notes/BatchFix.md 6.7-6.9): EXL3_MOE_COOP_INVARIANT=1 pins one tile geometry for
// both stages and k-split 1 at every bsz, so a row's k summation order does not depend on how many rows share the
// launch. The batched server sets it (qmesh/serve_pair.py, BATCH > 1). The default keeps the per-shape picks below,
// which a single-user server sees as one fixed shape per decode width anyway, and which measured 3.8 % more drafted
// tokens per round than the pinned shapes at B=1 (2026-11-07, d1 A/B/B/A 2.687 vs 2.589).
static bool moe_coop_invariant()
{
    static int mode = -2;
    if (mode == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_INVARIANT");
        mode = env ? atoi(env) : 0;
    }
    return mode != 0;
}

// Split-k factor for one stage: blocks per column chunk, partial outputs summed by the last-arriving block. The grid
// is slots x column tiles, which fills the device for a routed launch (bsz x top-k slots) but not for a single-slot
// launch such as the shared expert (one expert's tiles on a 100+ SM device): split k until there is a block per SM,
// keeping at least 64 k-slices (1024 columns) per split so the partials and their reduction stay cheap. 1 under
// EXL3_MOE_COOP_INVARIANT. EXL3_MOE_COOP_KSPLIT=n forces n for every stage (testing; clamped to the scratch)
static int moe_coop_pick_ksplit(int base, int kslices, int max_split, int device)
{
    static int forced = -2;
    if (forced == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_KSPLIT");
        forced = env ? atoi(env) : 0;
    }
    if (forced > 0) return std::min(forced, max_split);
    if (moe_coop_invariant()) return 1;
    const int want = CEIL_DIVIDE(DevCtx::instance().get_num_sms(device), std::max(base, 1));
    const int kcap = std::max(kslices / 64, 1);
    return std::max(1, std::min(std::min(want, kcap), max_split));
}

// Tile geometry: EXL3_MOE_COOP_WIDE = 0 / 1 forces one geometry for every stage (testing); under
// EXL3_MOE_COOP_INVARIANT one choice for both stages (narrow on Blackwell, wide before it); otherwise the wide
// (128-column, 4-way k-split) tile for long k and the narrow one for short k
static int moe_coop_wide_mode()
{
    static int mode = -2;
    if (mode == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_WIDE");
        mode = env ? atoi(env) : -1;
    }
    return mode;
}

static bool moe_coop_pick_wide(int kslices, int slots, int device)
{
    const int mode = moe_coop_wide_mode();
    if (mode >= 0) return mode != 0;
    if (DevCtx::instance().get_cc(device) < CC_BLACKWELL) return true;
    if (moe_coop_invariant()) return false;
    return kslices >= 256 || (kslices >= 128 && slots >= 32);
}

// Dynamic shared memory above the 48 KB default needs an opt-in per kernel (once per process)
static void moe_coop_smem_optin(void* kernel, int smem)
{
    if (smem <= 48 * 1024) return;
    static std::map<void*, int> done;
    auto it = done.find(kernel);
    if (it != done.end() && it->second >= smem) return;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    done[kernel] = smem;
}

// Kernel launch, optionally with programmatic stream serialization (PDL, CC >= 9.0, EXL3_MOE_PDL builds): the
// grid may start while its predecessor drains; the kernels order every dependent access behind griddepcontrol.wait
static void moe_coop_launch_kernel(void* kernel, int grid, int smem, void** args, cudaStream_t stream, bool pdl)
{
#if EXL3_MOE_PDL
    if (pdl)
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(grid);
        cfg.blockDim = dim3(MOE_COOP_THREADS);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        cuda_check(cudaLaunchKernelExC(&cfg, kernel, args));
        return;
    }
#endif
    cuda_check(cudaLaunchKernel(kernel, dim3(grid), dim3(MOE_COOP_THREADS), args, smem, stream));
}

void exl3_moe_coop_launch(const MoeCoopParams& p_in, float K_gu, float K_d, int cb, int device, cudaStream_t stream)
{
    MoeCoopParams p = p_in;
    { static int dbg = std::getenv("EXL3_MOE_COOP_DBG") ? atoi(std::getenv("EXL3_MOE_COOP_DBG")) : 0; p.dbg = dbg; }
    const int slots = p.bsz * p.topk;
    if (!slots)
    {
        if (p.bsz && p.H_out)
        {
            exl3_moe_coop_empty_kernel<<<dim3(p.bsz, CEIL_DIVIDE(p.H_out, 128)), 32, 0, stream>>>(p);
            cuda_check(cudaPeekAtLastError());
        }
        return;
    }
    const int nproj = p.gated ? 2 : 1;
    p.a_global = p.bsz > 1;

    // Measured per stage (qmesh/notes/CoopPerf.md): gate/up (A) at bsz 1 is fastest on the narrow tile, down (B) at
    // bsz > 1 on the wide tile. Other cases keep the k-length rule; EXL3_MOE_COOP_WIDE overrides all, and
    // EXL3_MOE_COOP_INVARIANT takes one geometry for both stages at every bsz (moe_coop_pick_wide)
    const bool forced = moe_coop_wide_mode() >= 0 || moe_coop_invariant();
    const bool wide_a = forced ? moe_coop_pick_wide(p.Hi / 16, slots, device) : (p.bsz > 1 && moe_coop_pick_wide(p.Hi / 16, slots, device));
    const bool wide_b = forced ? moe_coop_pick_wide(p.I / 16, slots, device) : (p.bsz > 1 || moe_coop_pick_wide(p.I / 16, slots, device));
    // Mixed-K (any per-expert rate table set, or a folded shared expert): the *_mixed kernels dispatch each block's
    // tile on its expert's rate (table entry, the uniform rate below for a null table / 0 entry, or the folded
    // shared expert's own), so the uniform K / half flags only reach them as uni_k2_*
    const bool mixed = p.k2_g || p.k2_u || p.k2_d || p.sh_fold;
    TORCH_CHECK(!mixed || cb == 2, "exl3_moe_coop: mixed per-expert bitrates require the mul1 codebook");
    p.uni_k2_gu = (uint8_t) lroundf(2.0f * K_gu);
    p.uni_k2_d = (uint8_t) lroundf(2.0f * K_d);
    // Paired layout (p.paired, from the pack's layout tag): mul1 only (the paired instances exist for the mul1 kernels)
    TORCH_CHECK(!p.paired || cb == 2, "exl3_moe_coop: paired trellis layout requires the mul1 codebook");
    MoeCoopKernel ka = mixed ? exl3_moe_coop_kernel_a_mixed(p.Hi, wide_a) : moe_coop_kernel_a(K_gu, cb, p.Hi, wide_a, p.paired);
    MoeCoopKernel kb = mixed ? exl3_moe_coop_kernel_b_mixed(wide_b) : moe_coop_kernel_b(K_d, cb, wide_b, p.paired);
    moe_coop_smem_optin(ka.kernel, ka.smem);
    moe_coop_smem_optin(kb.kernel, kb.smem);

    // Paired tiles load two k-slices at once: every warp's k range must start and end on a slice pair. At the shipped
    // routed shapes (Hi / 16 = 320, I / 16 = 144 over 2 or 8 warps) it does; a shape where it doesn't is refused
    // (lowering split-k instead would change the fp32 summation order, and the kernel would trap)
    auto pair_aligned = [] (int kslices, int ksplit, bool wide)
    {
        const int wkk = wide ? exl3_moe_coop_ns::tile_wk<true>() : exl3_moe_coop_ns::tile_wk<false>();
        for (int ks = 0; ks < ksplit; ++ks)
        {
            const int kb = kslices * ks / ksplit, ke = kslices * (ks + 1) / ksplit;
            const int chunk = CEIL_DIVIDE(ke - kb, wkk);
            for (int w = 0; w < wkk; ++w)
            {
                const int s0 = kb + w * chunk, n = std::max(0, std::min(chunk, ke - s0));
                if (n > 0 && ((s0 | n) & 1)) return false;
            }
        }
        return true;
    };
    const int base_a = slots * nproj * (p.I / (wide_a ? 128 : MOE_COOP_COLS));
    const int base_b = slots * (p.Ho / (wide_b ? 128 : MOE_COOP_COLS));
    const int max_split = std::max(1, p.slots_max / slots);
    p.ksplit_a = moe_coop_pick_ksplit(base_a, p.Hi / 16, max_split, device);
    p.ksplit_b = moe_coop_pick_ksplit(base_b, p.I / 16, max_split, device);
    TORCH_CHECK(!p.paired || (pair_aligned(p.Hi / 16, p.ksplit_a, wide_a) && pair_aligned(p.I / 16, p.ksplit_b, wide_b)),
                "exl3_moe_coop: paired trellis layout needs slice-pair aligned k ranges at this shape (k-split ",
                p.ksplit_a, " / ", p.ksplit_b, ")");
    const int grid_a = base_a * p.ksplit_a;
    const int grid_b = base_b * p.ksplit_b;

    void* args[] = { (void*) &p };
    if (p.a_global)
    {
        const int rot_items = slots * (p.Hi / 128) * nproj;
        const int rot_grid = CEIL_DIVIDE(rot_items, MOE_COOP_THREADS / 32);
        exl3_moe_coop_ns::exl3_moe_coop_rot_kernel<<<rot_grid, MOE_COOP_THREADS, 0, stream>>>(p);
    }
    static const bool single = std::getenv("EXL3_MOE_COOP_SINGLE") && atoi(std::getenv("EXL3_MOE_COOP_SINGLE")) != 0;
    if (single && mixed)
    {
        // One persistent launch for A then B (queue of grid_a + grid_b items). The fused kernel does not zero its
        // counters, so they are cleared here in stream order before the launch: queue head, ready flags, A/B counters
        MoeCoopKernel kf = exl3_moe_coop_kernel_fused_mixed(p.Hi, wide_a, wide_b);
        moe_coop_smem_optin(kf.kernel, kf.smem);
        int per_sm = 0;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kf.kernel, MOE_COOP_THREADS, kf.smem));
        int n_sm = 0;
        cuda_check(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, device));
        p.fused = 1;
        p.nA = grid_a;
        p.nB = grid_b;
        p.resident = std::max(1, std::min(per_sm * n_sm, grid_a + grid_b));
        cuda_check(cudaMemsetAsync(p.queue_head, 0, sizeof(int), stream));
        cuda_check(cudaMemsetAsync(p.ctr_a, 0, sizeof(int) * (p.ctr_a_len + p.ctr_b_len), stream));
        cuda_check(cudaMemsetAsync(p.ready_a, 0, sizeof(int) * p.ctr_a_len, stream));
        cuda_check(cudaLaunchKernel(kf.kernel, dim3(p.resident), dim3(MOE_COOP_THREADS), args, kf.smem, stream));
        cuda_check(cudaPeekAtLastError());
        return;
    }
    // B may overlap A's drain (PDL): it reads only routing and weights before its griddepcontrol.wait. A is launched
    // normally (its routing reads must follow the router). EXL3_MOE_COOP_PDL=0 disables at run time
    static const bool pdl_env = !std::getenv("EXL3_MOE_COOP_PDL") || atoi(std::getenv("EXL3_MOE_COOP_PDL")) != 0;
    const bool pdl_b = pdl_env && DevCtx::instance().get_cc(device) >= CC_HOPPER;
    moe_coop_launch_kernel(ka.kernel, grid_a, ka.smem, args, stream, false);
    moe_coop_launch_kernel(kb.kernel, grid_b, kb.smem, args, stream, pdl_b);
    cuda_check(cudaPeekAtLastError());
}

MoeCoopParams exl3_moe_coop_prepare
(
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_gate_w,
    float& K_gu, float& K_d, int& cb
)
{
    TORCH_CHECK_DTYPE(ctr, kInt);
    TORCH_CHECK(ctr.is_contiguous(), "exl3_moe_coop: ctr must be contiguous");
    for (const at::Tensor* t : { &g_trellis, &g_suh, &g_svh, &u_trellis, &u_suh, &u_svh, &d_trellis, &d_suh, &d_svh })
        TORCH_CHECK(t->scalar_type() == at::kLong && t->is_contiguous(), "exl3_moe_coop: pointer tables must be contiguous int64");
    TORCH_CHECK(!(mcg && mul1), "exl3_moe_coop: specified both mcg and mul1");
    TORCH_CHECK(Kg >= 1 && Kg <= 8 && Ku >= 1 && Ku <= 8 && Kd >= 1 && Kd <= 8, "exl3_moe_coop: K out of range");
    TORCH_CHECK(!gated || Kg == Ku, "exl3_moe_coop: gate and up must share a bit width");
    TORCH_CHECK(act >= 0 && act <= 3, "exl3_moe_coop: unknown activation");

    MoeCoopParams p = {};
    p.Hi = Hi;
    p.I = (int) gu_u.size(-1);
    p.Ho = (int) d_out.size(-1);
    p.H_out = (int) out.size(-1);
    TORCH_CHECK(p.Hi % 128 == 0 && p.I % 128 == 0 && p.Ho % 128 == 0 && p.H_out <= p.Ho, "exl3_moe_coop: Hi/I/Ho shape");
    TORCH_CHECK(p.Hi > 0 && p.I > 0 && p.Ho > 0, "exl3_moe_coop: empty Hi/I/Ho dimension");

    TORCH_CHECK_DTYPE(had_g, kHalf);
    TORCH_CHECK_DTYPE(had_u, kHalf);
    TORCH_CHECK_DTYPE(act_out, kHalf);
    TORCH_CHECK_DTYPE(d_out, kFloat);
    TORCH_CHECK_DTYPE(out, kFloat);
    p.gu_f32 = gu_u.scalar_type() == at::kFloat;
    if (!p.gu_f32) TORCH_CHECK_DTYPE(gu_u, kHalf);
    TORCH_CHECK(gu_g.scalar_type() == gu_u.scalar_type(), "exl3_moe_coop: gate/up scratch dtype mismatch");
    // Scratch capacity in slots: the smallest of the per-slot buffers
    const int slots_max = (int) std::min({ had_g.numel() / p.Hi, had_u.numel() / p.Hi, gu_g.numel() / p.I,
                                           gu_u.numel() / p.I, act_out.numel() / p.I, d_out.numel() / p.Ho });
    auto check_scratch = [&] (const at::Tensor& t, int width)
    {
        TORCH_CHECK(t.is_contiguous() && t.size(-1) == width, "exl3_moe_coop: scratch shape");
    };
    check_scratch(had_g, p.Hi);
    check_scratch(had_u, p.Hi);
    check_scratch(gu_g, p.I);
    check_scratch(gu_u, p.I);
    check_scratch(act_out, p.I);
    check_scratch(d_out, p.Ho);
    TORCH_CHECK(out.dim() >= 2 && out.stride(-1) == 1, "exl3_moe_coop: out shape");
    TORCH_CHECK(slots_max >= 1 && slots_max <= 512, "exl3_moe_coop: scratch must hold 1..512 slots");

    // Counters: the gate/up stage's block covers every slot the scratch can hold, the down
    // stage's every token row the output can hold
    const int rows_max = (int) out.size(-2);
    p.slots_max = slots_max;
    p.ctr_a_len = slots_max * (p.I / 128);
    p.ctr_b_len = rows_max * (p.Ho / 128);
    TORCH_CHECK(ctr.numel() >= exl3_moe_coop_ctr_len(slots_max, rows_max, p.I, p.Ho), "exl3_moe_coop: counter scratch too small");
    p.ctr_a = (int*) ctr.data_ptr();
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.runs = p.ctr_b + p.ctr_b_len;
    p.ready_a = p.runs + (2 + (slots_max + 1) + slots_max);
    p.queue_head = p.ready_a + p.ctr_a_len;
    p.exit_ctr = p.queue_head + 1;
    p.rows_max = rows_max;

    p.g_trellis = (const int64_t*) g_trellis.data_ptr();
    p.g_suh = (const int64_t*) g_suh.data_ptr();
    p.g_svh = (const int64_t*) g_svh.data_ptr();
    p.u_trellis = (const int64_t*) u_trellis.data_ptr();
    p.u_suh = (const int64_t*) u_suh.data_ptr();
    p.u_svh = (const int64_t*) u_svh.data_ptr();
    p.d_trellis = (const int64_t*) d_trellis.data_ptr();
    p.d_suh = (const int64_t*) d_suh.data_ptr();
    p.d_svh = (const int64_t*) d_svh.data_ptr();
    p.g_bias = g_bias ? (const int64_t*) g_bias->data_ptr() : nullptr;
    p.u_bias = u_bias ? (const int64_t*) u_bias->data_ptr() : nullptr;
    p.d_bias = d_bias ? (const int64_t*) d_bias->data_ptr() : nullptr;
    p.n_local = (int) u_trellis.size(0);
    p.plain_from = p.n_local;
    p.act = act;
    p.act_limit = act_limit;
    p.gated = gated;

    p.had_g = (half*) had_g.data_ptr();
    p.had_u = (half*) had_u.data_ptr();
    p.gu_g = gu_g.data_ptr();
    p.gu_u = gu_u.data_ptr();
    p.act_out = (half*) act_out.data_ptr();
    p.d_out = (float*) d_out.data_ptr();
    p.out = (float*) out.data_ptr();
    p.out_stride = (int) out.stride(-2);

    if (sh_gate_w)
    {
        TORCH_CHECK_DTYPE(sh_gate_w.value(), kHalf);
        TORCH_CHECK(sh_gate_w->is_contiguous(), "exl3_moe_coop: shared gate weight must be contiguous");
        p.sh_gate_w = (const half*) sh_gate_w->data_ptr();
        p.sh_gate_n = (int) sh_gate_w->numel();
    }

    K_gu = gated ? Kg : Ku;
    K_d = Kd;
    cb = mcg ? 1 : (mul1 ? 2 : 0);
    return p;
}

void exl3_moe_coop_run
(
    MoeCoopParams p, float K_gu, float K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = x.device().index();

    // Per-call checks kept to what can actually vary between calls
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "exl3_moe_coop: x must be (bsz, H) with unit column stride");
    p.x = (const half*) x.data_ptr();
    p.x_stride = (int) x.stride(0);
    p.bsz = (int) x.size(0);
    p.H = (int) x.size(1);
    p.sel_k = (int) sel.size(-1);
    p.topk = p.sel_k + (p.sh_fold ? 1 : 0);
    p.sel = (const int64_t*) sel.data_ptr();
    p.rw = (const half*) rw.data_ptr();
    const int slots = p.bsz * p.topk;
    TORCH_CHECK(p.H % 4 == 0 && p.H <= p.Hi, "exl3_moe_coop: H/Hi shape");
    TORCH_CHECK(slots <= p.slots_max && p.bsz <= p.rows_max, "exl3_moe_coop: batch exceeds the scratch");
    TORCH_CHECK(p.min_expert < 0 || p.max_expert - p.min_expert <= p.n_local, "exl3_moe_coop: expert range exceeds tables");
    TORCH_CHECK(!(sh_out && p.sh_fold), "exl3_moe_coop: a folded shared expert takes no separate shared output");
    if (sh_out)
    {
        TORCH_CHECK(p.H_out == p.H, "exl3_moe_coop: shared expert output width must match the routed output");
        p.sh_out = (const float*) sh_out->data_ptr();
    }
    else if (!p.sh_fold)
        p.sh_gate_w = nullptr;
    if (p.sh_out || p.sh_fold)
    {
        TORCH_CHECK(p.H_out == p.H, "exl3_moe_coop: shared expert output width must match the routed output");
        if (p.sh_gate_w) TORCH_CHECK(p.sh_gate_n == p.H && p.H % 2 == 0, "exl3_moe_coop: shared gate width");
    }
    exl3_moe_coop_launch(p, K_gu, K_d, cb, device, stream);
}

// Rates the mixed-K kernels instantiate (exl3_moe_coop_mixed_kernel.cuh, qmesh/notes/mixedk_gen.py CASES), as k2
static bool moe_coop_mixed_rate(int k2)
{
    switch (k2) { case 4: case 5: case 6: case 7: case 8: case 10: case 12: case 16: return true; default: return false; }
}

bool exl3_moe_coop_set_fold(MoeCoopParams& p, const at::Tensor& ptrs, int64_t g_bias, int64_t u_bias, int64_t d_bias,
                            float Ksh_gu, float Ksh_d, float K_gu, float K_d, int cb)
{
    TORCH_CHECK(ptrs.scalar_type() == at::kLong && ptrs.device().is_cpu() && ptrs.is_contiguous() && ptrs.numel() == 9,
                "exl3_moe_coop_set_fold: ptrs must be a contiguous int64 CPU tensor of 9 pointers");
    // The fold runs through the mixed-K kernels (mul1), at every rate involved
    if (cb != 2) return false;
    const int k2s[4] = { (int) lroundf(2.0f * Ksh_gu), (int) lroundf(2.0f * Ksh_d), (int) lroundf(2.0f * K_gu), (int) lroundf(2.0f * K_d) };
    for (int k2 : k2s) if (!moe_coop_mixed_rate(k2)) return false;
    // Bias presence is per launch (the kernels test the routed table pointers)
    if ((g_bias != 0) != (p.g_bias != nullptr) || (u_bias != 0) != (p.u_bias != nullptr) || (d_bias != 0) != (p.d_bias != nullptr))
        return false;
    const int64_t* q = ptrs.data_ptr<int64_t>();
    p.sh_g_trellis = q[0]; p.sh_g_suh = q[1]; p.sh_g_svh = q[2];
    p.sh_u_trellis = q[3]; p.sh_u_suh = q[4]; p.sh_u_svh = q[5];
    p.sh_d_trellis = q[6]; p.sh_d_suh = q[7]; p.sh_d_svh = q[8];
    p.sh_g_bias = g_bias; p.sh_u_bias = u_bias; p.sh_d_bias = d_bias;
    p.sh_k2_gu = (uint8_t) k2s[0];
    p.sh_k2_d = (uint8_t) k2s[1];
    p.sh_fold = 1;
    return true;
}

void exl3_moe_coop
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w,
    const c10::optional<at::Tensor>& k2_g,     // per local expert half-bit rates (uint8, nullable): mixed-K layer
    const c10::optional<at::Tensor>& k2_u,
    const c10::optional<at::Tensor>& k2_d,
    const c10::optional<at::Tensor>& sh_fold_ptrs,
    float sh_fold_K_gu,
    float sh_fold_K_d,
    bool paired,
    int plain_from
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(rw, kHalf);
    TORCH_CHECK(sel.is_contiguous() && rw.is_contiguous(), "exl3_moe_coop: sel/rw must be contiguous");
    TORCH_CHECK(sel.sizes() == rw.sizes() && sel.dim() == 2 && sel.size(0) == x.size(0), "exl3_moe_coop: sel/rw must be (bsz, topk)");
    if (sh_out)
    {
        TORCH_CHECK_DTYPE(sh_out.value(), kFloat);
        TORCH_CHECK(sh_out->is_contiguous() && sh_out->size(-1) == x.size(1) && sh_out->numel() >= x.size(0) * x.size(1),
                    "exl3_moe_coop: shared expert output shape");
    }
    float K_gu, K_d; int cb;
    // Mixed layers: per-expert rates come from the tables (k2_*), and the scalar K is ignored by the kernels. Pass the
    // gate rate for up too so the uniform gate/up guard in prepare holds; the per-expert pairing is checked below.
    const bool mixed = k2_g.has_value() || k2_u.has_value() || k2_d.has_value();
    const float Ku_eff = mixed ? Kg : Ku;
    MoeCoopParams p = exl3_moe_coop_prepare(Hi, g_trellis, g_suh, g_svh, u_trellis, u_suh, u_svh, d_trellis, d_suh, d_svh,
                                            g_bias, u_bias, d_bias, Kg, Ku_eff, Kd, mcg, mul1, act, act_limit, gated,
                                            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out, sh_gate_w, K_gu, K_d, cb);
    p.min_expert = min_expert;
    p.max_expert = max_expert;
    p.paired = paired ? 1 : 0;
    if (plain_from >= 0) p.plain_from = std::min(plain_from, p.n_local);
    if (mixed)
    {
        TORCH_CHECK(k2_g && k2_u && k2_d, "exl3_moe_coop: mixed-K needs all three rate tables");
        for (const at::Tensor* t : { &k2_g.value(), &k2_u.value(), &k2_d.value() })
        {
            TORCH_CHECK(t->scalar_type() == at::kByte && t->is_contiguous() && t->is_cuda(),
                        "exl3_moe_coop: rate tables must be contiguous uint8 CUDA tensors");
        }
        // gate and up of one expert must share a width. Checked by the table builder (host, once at load), NOT here:
        // a device comparison in the launch path would synchronize and break CUDA-graph capture.
        TORCH_CHECK(k2_g->numel() >= 1 && k2_u->numel() == k2_g->numel(), "exl3_moe_coop: gate/up rate table length");
        p.k2_g = k2_g->data_ptr<uint8_t>();
        p.k2_u = k2_u->data_ptr<uint8_t>();
        p.k2_d = k2_d->data_ptr<uint8_t>();
    }
    if (sh_fold_ptrs)
        TORCH_CHECK(exl3_moe_coop_set_fold(p, sh_fold_ptrs.value(), 0, 0, 0, sh_fold_K_gu, sh_fold_K_d, K_gu, K_d, cb),
                    "exl3_moe_coop: shared expert cannot fold into this launch");
    exl3_moe_coop_run(p, K_gu, K_d, cb, x, sel, rw, sh_out);
}
