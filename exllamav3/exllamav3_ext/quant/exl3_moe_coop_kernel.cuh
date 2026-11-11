#pragma once

// Fused decode-shaped MoE kernels (bsz 1..MAX_BSZN), two ordinary launches per layer:
//
//   rot (bsz > 1): per (slot, chunk, projection) warp: x row (zero-padded to Hi) * suh -> Hadamard
//                 into the had_g / had_u scratch. At bsz 1 kernel A does this itself into shared
//                 memory and the launch is skipped.
//   A (gate/up):  one block per (expert run, 32-column group, projection). Slots are grouped by
//                 expert inside every block (identical redundant computation from sel/rw): a run
//                 is up to ROWS slots that picked the same expert, and the block runs the GEMV
//                 tile with those slots as the rows of one m16 MMA, so an expert's weights are read
//                 once per run rather than once per slot. Per row, the last of the 4 x nproj
//                 arrivals covering a 128-column chunk (completion counter) finishes the chunk:
//                 output rotation * svh, bias, activation, down-input rotation * down suh, stored
//                 as the down input.
//   B (down):     one block per (expert run, group), same multi-row tile on the rotated
//                 activation; the last arrival of a (token, chunk) (counter over active slots x 4
//                 groups) rotates every slot's result * svh, adds the down bias, sums the slots in
//                 a fixed order, merges the shared expert and stores the trimmed row.
//
// The grid is sized for the no-duplicates case (slots x groups x projections); blocks past the
// actual run count exit at once.
//
// Fused launch (EXL3_MOE_COOP_SINGLE=1, mixed-K): one persistent grid drains a queue of the A items [0, nA)
// followed by the B items [nA, nA + nB). B waits on ready_a per (slot, chunk); see exl3_moe_coop_fused_mixed_kernel.

#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include "../compat.cuh"
#include "exl3_gemv_kernel.cuh"
#include "hadamard_inner.cuh"
#include "exl3_moe_coop.cuh"
#include "trellis_pair.cuh"

namespace exl3_moe_coop_ns {

constexpr int WK = 8;                           // k-split (warps per block)
constexpr int WNT = MOE_COOP_WNT;               // adjacent n-tiles per warp
constexpr int PF = 4;                           // prefetch ring depth
constexpr int FOLD = 4;                         // fp16 -> fp32 fold cadence
constexpr int THREADS = WK * 32;
constexpr int COLS = WNT * 16;                  // columns per warp
constexpr int ROWS = 8;                         // slots of one expert per block (MMA rows)
template <bool WIDE> constexpr int tile_wn() { return WIDE ? 4 : 1; }          // warps along n
template <bool WIDE> constexpr int tile_wk() { return WK / tile_wn<WIDE>(); }  // warps along k
template <bool WIDE> constexpr int tile_cols() { return tile_wn<WIDE>() * COLS; }
template <bool WIDE> constexpr int tile_gpc() { return tile_cols<WIDE>() >= 128 ? 1 : 128 / tile_cols<WIDE>(); }  // block groups per 128-chunk
template <bool WIDE> constexpr int tile_cpb() { return tile_cols<WIDE>() >= 128 ? tile_cols<WIDE>() / 128 : 1; }  // 128-chunks per block
constexpr int MAX_SLOTS = 512;                  // supports 48 verify rows with up to 10 slots each
constexpr int STAGE_WORDS = WNT * 8 * 8;        // words per warp per k-slice at 8 bpw
constexpr int STAGE_STRIDE = STAGE_WORDS;      // words per warp of the staging area
constexpr float R_SCALE = 0.088388347648f;      // 1/sqrt(128)

static_assert(THREADS == MOE_COOP_THREADS, "block size mismatch");

// Register-dequant widths (shuffle extraction, no staging). 3 bpw stays on the staged generic
// path on Ampere where its register form loses to the block-pipelined kernel)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
template <int bits>
constexpr bool tile_reg() { return bits == 2 || bits == 4; }
#else
template <int bits>
constexpr bool tile_reg() { return bits == 2 || bits == 3 || bits == 4; }
#endif
// Shared-memory layout is sized on the host, which doesn't know the device arch: reserve the
// staging area for every width some arch stages (unused on the others)
// HALF: half-integer rate bits + 0.5 (mul1 only), always through the staged generic path
template <int bits, bool HALF = false>
constexpr bool tile_may_stage() { return HALF || !(bits == 2 || bits == 4); }

// Dynamic shared memory layout (bytes)
template <int bits, bool HALF = false>
__host__ __device__ constexpr int smem_stage_bytes() { return tile_may_stage<bits, HALF>() ? WK * STAGE_STRIDE * 4 : 0; }
constexpr int smem_red_bytes() { return WK * ROWS * COLS * 4; }   // same for both geometries: wk x ROWS x cols
constexpr int smem_part_bytes() { return WK * 128 * 4; }
template <int bits, bool HALF = false>
__host__ __device__ constexpr int smem_a_bytes(int Hi) { return Hi * 2 + smem_red_bytes() + smem_stage_bytes<bits, HALF>(); }
template <int bits, bool HALF = false>
__host__ __device__ constexpr int smem_b_bytes() { return smem_red_bytes() + smem_stage_bytes<bits, HALF>() + smem_part_bytes(); }

// ---------------------------------------------------------------------------------------------
// Elementwise helpers (fp32 forms of activation_kernels.cuh, which can't be included twice)

__device__ __forceinline__ float act_silu(float x)
{
    float e = __expf(-x);
    return x * __fdividef(1.0f, 1.0f + e);
}

__device__ __forceinline__ float act_gelu(float x)
{
    const float c = 0.797884560803f;  // sqrt(2/Pi)
    float t = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh_opt(t));
}

__device__ __forceinline__ float act_relu2(float x)
{
    x = fmaxf(0.0f, x);
    return x * x;
}

// gpt-oss clamped swiglu: gate clamped from above only, up symmetrically, alpha = 1.702 in the
// sigmoid and +1 on the up path
__device__ __forceinline__ float act_oai_swiglu(float g, float u, float limit)
{
    if (limit != 0.0f)
    {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    float glu = g * (1.0f / (1.0f + __expf(-1.702f * g)));
    return (u + 1.0f) * glu;
}

// Gated activation with the act_limit semantics of act_mul_kernel_*: gate lane activated, up lane
// clamped symmetrically and the activated gate from above when a limit is set. Gateless experts
// pass g = u, giving relu(u) * u = relu2(u) exactly
__device__ __forceinline__ float act_gate(int act, bool gated, float g, float u, float limit)
{
    if (act == MOE_COOP_ACT_SILU_OAI) return act_oai_swiglu(g, u, limit);
    float x;
    if (!gated) x = fmaxf(0.0f, u);
    else if (act == MOE_COOP_ACT_SILU) x = act_silu(g);
    else if (act == MOE_COOP_ACT_GELU) x = act_gelu(g);
    else x = act_relu2(g);
    if (limit != 0.0f)
    {
        u = fminf(fmaxf(u, -limit), limit);
        x = fminf(x, limit);
    }
    return x * u;
}

// 128-element Hadamard across the warp on 4 values per lane, same butterfly as had_*_r_128_inner
#ifndef EXL3_MOE_PROBE
#define EXL3_MOE_PROBE 0
#endif

__device__ __forceinline__ void had128(float& v0, float& v1, float& v2, float& v3, int lane)
{
#if EXL3_MOE_PROBE == 4
    return;
#endif
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    v0 = h0 * R_SCALE;
    v1 = h1 * R_SCALE;
    v2 = h2 * R_SCALE;
    v3 = h3 * R_SCALE;
}

__device__ __forceinline__ void unpack_h4(uint2 u, float& v0, float& v1, float& v2, float& v3)
{
    half2 a = *reinterpret_cast<half2*>(&u.x);
    half2 b = *reinterpret_cast<half2*>(&u.y);
    v0 = __low2float(a);
    v1 = __high2float(a);
    v2 = __low2float(b);
    v3 = __high2float(b);
}

__device__ __forceinline__ void load_h4(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    unpack_h4(*((const uint2*) p), v0, v1, v2, v3);
}

// L2-coherent loads for data written by other blocks of the same launch
__device__ __forceinline__ void load_h4_cg(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    unpack_h4(__ldcg((const uint2*) p), v0, v1, v2, v3);
}

__device__ __forceinline__ void load_f4_cg(const float* p, float& v0, float& v1, float& v2, float& v3)
{
    float4 f = __ldcg((const float4*) p);
    v0 = f.x; v1 = f.y; v2 = f.z; v3 = f.w;
}

__device__ __forceinline__ void store_h4(half* p, float v0, float v1, float v2, float v3)
{
    half2 a = __floats2half2_rn(v0, v1);
    half2 b = __floats2half2_rn(v2, v3);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *((uint2*) p) = u;
}

__device__ __forceinline__ void scale_h4(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    float s0, s1, s2, s3;
    load_h4(p, s0, s1, s2, s3);
    v0 *= s0; v1 *= s1; v2 *= s2; v3 *= s3;
}

__device__ __forceinline__ void add_h4(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    float s0, s1, s2, s3;
    load_h4(p, s0, s1, s2, s3);
    v0 += s0; v1 += s1; v2 += s2; v3 += s3;
}

struct SlotInfo
{
    int row;
    int local;
    float w;
    bool active;
};

__device__ __forceinline__ SlotInfo slot_info(const MoeCoopParams& p, int s)
{
    SlotInfo si;
    si.row = s / p.topk;
    const int k = s - si.row * p.topk;
    if (p.sh_fold && k == p.sel_k)
    {
        // Folded shared expert: one past the routed tables, weight 1 (its gate is applied by the down reduction)
        si.local = p.n_local;
        si.w = 1.0f;
        si.active = true;
        return si;
    }
    const int idx = si.row * p.sel_k + k;
    int64_t e = p.sel[idx];
    si.w = __half2float(p.rw[idx]);
    bool ok = si.w != 0.0f;
    if (p.min_expert >= 0)
    {
        ok = ok && e >= p.min_expert && e < p.max_expert;
        e -= p.min_expert;
    }
    si.local = ok ? (int) e : 0;
    si.active = ok;
    return si;
}

// Per-expert table entry of local expert `local`, the folded shared expert's own pointer for local == n_local
__device__ __forceinline__ int64_t tab_ptr(const MoeCoopParams& p, const int64_t* tab, int64_t sh, int local)
{
    return (p.sh_fold && local == p.n_local) ? sh : tab[local];
}
#define TAB_H(name, local) ((const half*) tab_ptr(p, p.name, p.sh_##name, (local)))

// Rate (k2) of local expert `local` for one projection: the folded shared expert's, the table entry, or the launch's
// uniform rate when the table is null or the entry is 0 (mixed-K kernels only)
__device__ __forceinline__ int expert_k2(const MoeCoopParams& p, const uint8_t* tab, int uni, int sh, int local)
{
    if (p.sh_fold && local == p.n_local) return sh;
    const int k2 = tab ? tab[local] : 0;
    return k2 ? k2 : uni;
}

// ---------------------------------------------------------------------------------------------
// Slot grouping (bsz > 1): active slots sorted by (local expert, slot) and cut into runs of at
// most ROWS slots of one expert. Built once per call by block 0 of the rotation kernel into the
// run table (p.runs: [n_runs, -, run_start[0..], order[0..]]), read by every block of A and B.
// At bsz 1 top-k picks are distinct, so runs are the slots themselves and no table is built

__device__ __forceinline__ void build_runs
(
    const MoeCoopParams& p, int16_t* sh_order, int16_t* sh_local, int* sh_scan, int* sh_res
)
{
    const int slots = p.bsz * p.topk;
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    int* run_start = p.runs + 2;
    int* order = p.runs + 2 + p.slots_max + 1;

    // Routing of every slot: local expert or -1 when inactive
    for (int s = t; s < slots; s += THREADS)
    {
        const SlotInfo si = slot_info(p, s);
        sh_local[s] = si.active ? (int16_t) si.local : (int16_t) -1;
    }
    if (t == 0) { sh_res[0] = 0; sh_res[1] = 0; }
    __syncthreads();

    // Rank of every active slot by (expert, slot)
    for (int s = t; s < slots; s += THREADS)
    {
        const int my_e = sh_local[s];
        if (my_e < 0) continue;
        int rank = 0;
        for (int u = 0; u < slots; ++u)
        {
            const int eu = sh_local[u];
            rank += (eu >= 0 && (eu < my_e || (eu == my_e && u < s))) ? 1 : 0;
        }
        sh_order[rank] = (int16_t) s;
        atomicMax(sh_res, rank + 1);
    }
    __syncthreads();
    const int n_active = sh_res[0];

    // Run starts: a new expert, or ROWS slots into the same expert (first index of an expert =
    // number of active slots with a smaller expert)
    // Prefix scans in block-sized tiles; each tile carries the previous tile's run count.
    for (int tile = 0; tile < n_active; tile += THREADS)
    {
        const int s = tile + t;
        int is_start = 0;
        if (s < n_active)
        {
            const int e = sh_local[sh_order[s]];
            int first = 0;
            for (int u = 0; u < slots; ++u)
                first += (sh_local[u] >= 0 && sh_local[u] < e) ? 1 : 0;
            is_start = (s == 0 || sh_local[sh_order[s - 1]] != e || (s - first) % ROWS == 0);
            order[s] = sh_order[s];
        }
        int v = is_start;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            int n = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += n;
        }
        if (lane == 31) sh_scan[warp] = v;
        __syncthreads();
        int base = 0, total = 0;
        for (int w = 0; w < WK; ++w)
        {
            if (w < warp) base += sh_scan[w];
            total += sh_scan[w];
        }
        const int prior = sh_res[1];
        if (is_start) run_start[prior + base + v - 1] = s;
        __syncthreads();
        if (t == 0) sh_res[1] = prior + total;
        __syncthreads();
    }
    if (t == 0)
    {
        run_start[sh_res[1]] = n_active;
        p.runs[0] = sh_res[1];
    }
}

// The block's run: its slot indices into rows[] (shared) and the row count; false past the end
__device__ __forceinline__ bool read_run(const MoeCoopParams& p, int run_idx, int* rows, int& nrows, int* sh_res)
{
    if (!p.a_global)
    {
        // bsz 1: run = slot (an inactive slot returns false; kernel A checks for empty rows)
        if (run_idx >= p.bsz * p.topk) return false;
        const SlotInfo si = slot_info(p, run_idx);
        if (!si.active) return false;
        if (threadIdx.x == 0) rows[0] = run_idx;
        nrows = 1;
        __syncthreads();
        return true;
    }
    const int n_runs = p.runs[0];
    if (run_idx >= n_runs) return false;
    const int start = p.runs[2 + run_idx];
    const int end = p.runs[2 + run_idx + 1];
    nrows = end - start;
    if (threadIdx.x < nrows) rows[threadIdx.x] = p.runs[2 + p.slots_max + 1 + start + threadIdx.x];
    __syncthreads();
    return true;
}

// ---------------------------------------------------------------------------------------------
// GEMV tile: one 32-column group of an (nrows, k) x (k, n) product, the rows being the run's slots
// (row r reads A at A2 + rows[r] * a_stride2 half2 and writes C at rows[r] * c_stride), the whole
// block splitting k across its 16 warps. Body follows exl3_gemv_kernel (MMODE 1, CFG 0) for
// 2/3/4 bpw; other widths stage each k-slice's tile words in shared memory and decode with
// dq_dispatch

// Candidate switches (each one bit-identical to the baseline arithmetic; see OPTIMIZATION_NOTES.md):
//   EXL3_MOE_HOIST     issue the trellis prefetch ring before the block prologue (bsz-1 input rotation, fused /
//                      PDL waits), so the first DRAM round trip overlaps the prologue instead of following it
//   EXL3_MOE_HALF_REG  2.5 / 3.5 (and 1.5) bpw decode from registers by lane shuffles, as exl3_gemv_kernel does,
//                      instead of staging every k-slice through shared memory
//   EXL3_MOE_DEEP_PF   prefetch ring depth 8 (was 4) for widths with <= 2 warp loads per k-slice; keeps the fold
//                      cadence (FOLD = 4), so the fp16 -> fp32 rounding points are unchanged
//   EXL3_MOE_PDL       programmatic dependent launch: kernel B (and A) start while the previous kernel drains and
//                      prefetch weights before griddepcontrol.wait; counters are reset after the wait
#ifndef EXL3_MOE_HOIST
#define EXL3_MOE_HOIST 1
#endif
#ifndef EXL3_MOE_HALF_REG
#define EXL3_MOE_HALF_REG 0
#endif
#ifndef EXL3_MOE_DEEP_PF
#define EXL3_MOE_DEEP_PF 0
#endif
#ifndef EXL3_MOE_PDL
#define EXL3_MOE_PDL 1
#endif
// k-loop form (qmesh/notes/CoopKernel2.md; each bit-identical to the previous loop on every measured K / row count,
// default on, = 0 restores the previous form for A/B):
//   EXL3_MOE_STEADY    full-ring iterations of the k loop run without bounds checks and with compile-time fold points;
//                      the partial-warp loads of 3 bpw / half-integer register tiles clamp the lane instead of
//                      predicating it (lanes past the tile re-read its last word and are never shuffled from)
//   EXL3_MOE_LATE      decode straight from the ring slot and refill it after the slice's MMAs (no register copy of
//                      the slot; PF - 1 slices of load lead instead of PF)
//   EXL3_MOE_TMMA      transposed tile product: the decoded weight registers (f0[0], f1[0], f0[1], f1[1]) are the
//                      m16n8k16 A fragment of W^T (16 columns x 16 k) and the activation words the B fragment (16 k x 8
//                      rows), so one MMA per 16x16 tile instead of two whose A rows 8-15 are zero. Same products, same
//                      fp16 accumulator chain and fold points. CUDA sm_80+ only (sm_75 splits the MMA in k halves, ROCm
//                      has its own fragment layout); elsewhere the two-MMA form stays
#ifndef EXL3_MOE_STEADY
#define EXL3_MOE_STEADY 1
#endif
#ifndef EXL3_MOE_LATE
#define EXL3_MOE_LATE 1
#endif
#ifndef EXL3_MOE_TMMA
#define EXL3_MOE_TMMA 1
#endif
#if EXL3_MOE_TMMA && !defined(USE_ROCM) && !EXL3_SM75
#define EXL3_MOE_TMMA_ON 1
#else
#define EXL3_MOE_TMMA_ON 0
#endif
// Load ordering (qmesh/notes/CoopKernel2.md §10). In the shipped SASS the top-stall sample on the first activation
// LD.E of a ring block is the weight ring's wait, not the A load's latency: ptxas reuses the ring registers of the
// consumed slots as decode temporaries and issues all PF x LOADS refills of a block in one late cluster on one
// scoreboard, so the next block's first slice waits for every refill (lead 193-216 SASS instructions of a 416 block
// at K3, 108-136 of 377 at K4, instead of PF - 1 slices).
//   EXL3_MOE_ORD = 1   ring refills as ld.volatile.global and A fragment words as ld.volatile, in program order:
//                      volatile accesses keep their order through nvvm and ptxas, so the refill of slot d issues
//                      before slice d + 1's A load, i.e. right after slot d's decode
//   EXL3_MOE_ORD = 2   refills ordered among themselves only (A loads plain)
//   EXL3_MOE_ORD = 3   as 1 with the refills as asm volatile ld.global.cs (keeps the evict-first hint)
// Same words loaded, same arithmetic: same outputs
#ifndef EXL3_MOE_ORD
#define EXL3_MOE_ORD 0
#endif
// Paired weight layout (qmesh/notes/CoopKernel2.md §11, quant/trellis_pair.cuh): tiles instantiated with PAIRED read
// a routed expert's trellis permuted so that, per (k-slice pair, warp column chunk of WNT tiles), lane L's words of
// both slices are contiguous: one 16 B (K3 / K4) or 8 B (K2) load per lane fetches two ring slots. Same words as the
// plain layout, same decode / MMA order: same outputs. Chosen per launch (MoeCoopParams.paired, from the pack's
// layout tag) for the routed experts of integer 2-4 bpw; the folded shared expert and every other rate stay plain
template <int bits, bool HALF>
constexpr bool pair_ok() { return trellis_pair::pairable(bits, HALF); }

__device__ __forceinline__ uint32_t ld_ord_g(const uint32_t* p)
{
    uint32_t v;
#if EXL3_MOE_ORD == 3
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
#else
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
#endif
    return v;
}
__device__ __forceinline__ uint32_t ld_ord(const uint32_t* p)
{
    uint32_t v;
    asm volatile("ld.volatile.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

#if EXL3_MOE_TMMA_ON
// Transposed tile product (EXL3_MOE_TMMA): C^T (16 columns x 8 rows) += W^T (16 x 16 k) * X^T (16 k x 8 rows). The A
// fragment of W^T is the decoded B fragments of W in the order (f0[0], f1[0], f0[1], f1[1]); the B fragment of X^T is
// the activation words (row lane/4, k pairs 2(lane%4) and +8). c[0] = (column lane/4, rows 2(lane%4), +1),
// c[1] = (column 8 + lane/4, same rows)
__device__ __forceinline__ void mma_t_h(const FragB& f0, const FragB& f1, uint2 aw, FragC_h& c)
{
    const uint32_t* w0 = reinterpret_cast<const uint32_t*>(&f0);
    const uint32_t* w1 = reinterpret_cast<const uint32_t*>(&f1);
    uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(w0[0]), "r"(w1[0]), "r"(w0[1]), "r"(w1[1]),
           "r"(aw.x), "r"(aw.y)
    );
}
#endif

// griddepcontrol (sm_90+): no-ops when the launch carries no programmatic-serialization attribute
__device__ __forceinline__ void pdl_wait()
{
#if EXL3_MOE_PDL && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger()
{
#if EXL3_MOE_PDL && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

struct NoPre { __device__ __forceinline__ void operator()() const {} };

template <int bits, bool HALF>
constexpr bool tile_reg_k() { return HALF ? (EXL3_MOE_HALF_REG != 0) : tile_reg<bits>(); }

template <int bits, bool HALF>
constexpr int tile_loads()
{
    constexpr bool REG = tile_reg_k<bits, HALF>();
    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;
    return REG ? (HALF ? (bits == 1 ? WNT / 2 : WNT) : (bits == 2 ? WNT / 2 : WNT)) : CEIL_DIVIDE(WNT * TWORDS, 32);
}

template <int bits, bool HALF>
constexpr int tile_pf() { return (EXL3_MOE_DEEP_PF && tile_loads<bits, HALF>() <= 2) ? 8 : PF; }

template <int bits, int cb, bool WIDE, bool HALF = false, bool PAIRED = false, typename Pre = NoPre>
__device__ __forceinline__ void gemv_tile
(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    size_t a_stride2,
    const int* __restrict__ rows,
    int nrows,
    void* __restrict__ C,
    size_t c_stride,
    bool c_f32,
    int k_begin,                        // k-slice range of this block (split-k across blocks)
    int k_end,
    int ntiles,
    int group,                          // block column group of tile_cols<WIDE>() columns
    float* __restrict__ sh_red,         // [WKK][ROWS][TCOLS]
    uint32_t* __restrict__ sh_stage,    // [WK][STAGE_STRIDE], staged widths / cp.async ring only
    Pre pre = Pre()                     // block prologue, run after the prefetch ring is issued (block-uniform)
)
{
    constexpr bool REG = tile_reg_k<bits, HALF>();
    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;              // uint32 per 16x16 tile
    constexpr int GWORDS = WNT * TWORDS;                                        // uint32 per warp per k-slice
    constexpr bool HALF_TWO = HALF && bits == 1;                                // 1.5 bpw: two tiles per load
    constexpr int LOADS = tile_loads<bits, HALF>();
    constexpr int LSTRIDE = (REG && HALF) ? (HALF_TWO ? 2 * TWORDS : TWORDS) : (REG && bits == 3) ? 24 : 32;
    constexpr int PFD = tile_pf<bits, HALF>();
    static_assert(PFD % FOLD == 0, "fold cadence must divide the ring depth");
    constexpr int WN = tile_wn<WIDE>();
    constexpr int WKK = tile_wk<WIDE>();
    constexpr int TCOLS = tile_cols<WIDE>();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int wn = warp % WN;                   // adjacent warps read adjacent column pieces
    const int wk = warp / WN;

    const int kslices = k_end - k_begin;
    const int chunk = CEIL_DIVIDE(kslices, WKK);
    const int ks0 = k_begin + wk * chunk;
    const int myn = max(0, min(chunk, k_end - ks0));
    // Partial-warp register tiles (3 bpw, half-integer): STEADY loads every lane, lanes past the tile re-reading the
    // tile's last word (same sector, never shuffled from) instead of a predicated load
    constexpr bool CLAMP = EXL3_MOE_STEADY && REG && (bits == 3 || HALF);
    const int lane_ld = CLAMP ? min(lane, (HALF ? LSTRIDE : 24) - 1) : lane;
#if EXL3_MOE_PROBE == 5
    // Probe 5: block-contiguous layout (a load-time repack): the warp's k-slices of its column group back to back, same
    // bytes per slice, so each warp streams one contiguous range (k_end slices per group; split-k 1)
    const size_t slice_stride = GWORDS;
    const uint32_t* bp = B32 + ((size_t) (group * WN + wn) * k_end + ks0) * GWORDS + lane;
#else
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + (group * WN + wn) * GWORDS + lane_ld;
#endif
    [[maybe_unused]] uint32_t* stage = sh_stage + warp * STAGE_STRIDE;

    // Probe builds only (qmesh/coop_fast, -DEXL3_MOE_PROBE=n; 0 in every shipped build): 1 no weight loads (lane-derived
    // words, no DRAM), 2 no trellis decode (words reinterpreted as fp16), 3 no MMA (fragments folded by HADD2),
    // 4 no Hadamard (input / output rotations skipped), 5 block-contiguous weight addressing (the layout a load-time
    // repack would give). Results are garbage; timing only
    auto ldw = [] (const uint32_t* q) -> uint32_t
    {
#if EXL3_MOE_ORD
        return ld_ord_g(q);
#else
        return __ldcs(q);
#endif
    };
    auto ld_b = [&] (int i, int l) -> uint32_t
    {
#if EXL3_MOE_PROBE == 1
        return (uint32_t) (lane * 0x9E3779B1u) ^ (uint32_t) (i * 0x85EBCA77u + l);
#endif
        if constexpr (CLAMP)
            return ldw(bp + (size_t) i * slice_stride + l * LSTRIDE);
        else if constexpr (REG && HALF)
            return lane < LSTRIDE ? ldw(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else if constexpr (REG && bits == 3)
            return lane < 24 ? ldw(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else if constexpr (!REG)
            return (l * 32 + lane < GWORDS) ? ldw(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return ldw(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    // Prefetch ring first: the loads depend only on the weight pointer, not on the prologue. (A cp.async shared-memory
    // ring of 2-6 slices in place of this register ring measured 0-12 % slower at every K / row count on sm_120;
    // qmesh/notes/CoopKernel.md)
    uint32_t pf[PFD][LOADS];
    // Paired layout (pair_ok): the slices (2j, 2j + 1) of the expert are one region of 2 x slice_stride words, cut into
    // one block of 2 GWORDS words per warp column chunk c (chunk-major); inside a block lane L holds its LOADS plain
    // words of the even slice, then of the odd one (pair_layout.py). One 16 B (LOADS 2) or 8 B (LOADS 1) load per lane
    // fills ring slots d and d + 1 (lanes past a 3 bpw tile clamp to lane 23 as above)
    static_assert(!PAIRED || (pair_ok<bits, HALF>() && REG && EXL3_MOE_STEADY && EXL3_MOE_LATE && EXL3_MOE_PROBE == 0 && PFD % 2 == 0
                              && (LOADS == 1 || LOADS == 2)),
                  "paired tiles need the register-ring STEADY + LATE loop (2-4 bpw register tiles, even ring depth)");
    [[maybe_unused]] const uint32_t* bpp = B32 + (size_t) ks0 * slice_stride + (size_t) (group * WN + wn) * 2 * GWORDS
                                         + (size_t) lane_ld * 2 * LOADS;
    auto ld_pair = [&] (int i, int d)
    {
        if constexpr (PAIRED)
        {
            const uint32_t* q = bpp + (size_t) i * slice_stride;
            if constexpr (LOADS == 2)
            {
                const uint4 v = __ldcs(reinterpret_cast<const uint4*>(q));
                pf[d][0] = v.x; pf[d][1] = v.y; pf[d + 1][0] = v.z; pf[d + 1][1] = v.w;
            }
            else
            {
                const uint2 v = __ldcs(reinterpret_cast<const uint2*>(q));
                pf[d][0] = v.x; pf[d + 1][0] = v.y;
            }
        }
    };
    // Slice pairs never straddle a warp's k range at the launches the host allows (exl3_moe_coop.cu checks the shapes);
    // a range that would is a launch bug, not a fallback
    if constexpr (PAIRED)
        if (myn > 0 && ((ks0 | myn) & 1)) __trap();
    auto prefetch = [&] ()
    {
        #pragma unroll
        for (int d = 0; d < PFD; ++d)
        {
            if constexpr (PAIRED)
            {
                if ((d & 1) == 0 && d < myn) ld_pair(d, d);
            }
            else if (d < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);
            }
        }
    };
#if EXL3_MOE_HOIST
    prefetch();
    pre();
#else
    pre();
    prefetch();
#endif

    // A fragment row of this lane: rows[lane / 4] of the run (rows[] may be written by the prologue's caller only)
    const int r0 = lane >> 2;
    const bool r0_ok = r0 < nrows;
    const size_t a_row0 = r0_ok ? (size_t) rows[r0] * a_stride2 : 0;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    [[maybe_unused]] uint32_t x_pack = 0;
    if constexpr (HALF)
    {
        // Same lane constants as exl3_gemv_kernel's half-integer path / dq8_half: word indices (< TWORDS) and
        // funnel shifts of both window groups, six 5-bit fields
        constexpr int bits2 = 2 * bits + 1;
        constexpr int gspan = 18 + 3 * bits;
        const int t_offset = lane << 3;
        const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;
        const int e3 = e7 - 2 * bits2;
        const uint32_t hi7 = ((e7 - 1) / 32) % TWORDS, lo7 = ((e7 - gspan) / 32) % TWORDS, s7 = ((e7 - 1) / 32 + 1) * 32 - e7;
        const uint32_t hi3 = ((e3 - 1) / 32) % TWORDS, lo3 = ((e3 - gspan) / 32) % TWORDS, s3 = ((e3 - 1) / 32 + 1) * 32 - e3;
        x_pack = lo7 | (hi7 << 5) | (s7 << 10) | (lo3 << 15) | (hi3 << 20) | (s3 << 25);
    }
    else if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    else if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }
    #define XPK(i) ((x_pack >> (5 * (i))) & 31u)   // 0: lo7, 1: hi7, 2: s7, 3: lo3, 4: hi3, 5: s3

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    // A fragment words of slice i: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
    auto ld_a = [&] (int i) -> uint2
    {
        const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
        const uint32_t* ap = reinterpret_cast<const uint32_t*>(A2) + a_row0 + a_col;
#if EXL3_MOE_ORD == 1 || EXL3_MOE_ORD == 3
        return r0_ok ? make_uint2(ld_ord(ap), ld_ord(ap + 4)) : make_uint2(0u, 0u);
#else
        return r0_ok ? make_uint2(ap[0], ap[4]) : make_uint2(0u, 0u);
#endif
    };

    // One k-slice: decode of the slice's weight words bw, MMAs against its A fragment aw, fp32 fold on the fixed
    // cadence (every FOLD slices of the warp's range, and at its end), identical for every loop form. fold: 1 = fold
    // now, 0 = don't, -1 = decide at run time from i
    auto step = [&] (int i, const uint32_t (&bw)[LOADS], uint2 aw, int fold)
    {
            if constexpr (!REG)
            {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (l * 32 + lane < GWORDS)
                        stage[l * 32 + lane] = bw[l];
                __syncwarp();
            }

            [[maybe_unused]] FragB a01, a23;      // two-MMA form (and probe 3) only; TMMA takes aw directly
            reinterpret_cast<uint32_t&>(a01[0]) = aw.x;
            reinterpret_cast<uint32_t&>(a23[0]) = aw.y;
            a01[1] = hzero;
            a23[1] = hzero;
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
#if EXL3_MOE_PROBE == 2
                {
                    const uint32_t w = bw[(t * LOADS) / WNT];
                    reinterpret_cast<uint32_t&>(f0[0]) = w;
                    reinterpret_cast<uint32_t&>(f0[1]) = w ^ 0x00010001u;
                    reinterpret_cast<uint32_t&>(f1[0]) = w ^ 0x00020002u;
                    reinterpret_cast<uint32_t&>(f1[1]) = w ^ 0x00030003u;
                }
#else
                if constexpr (!REG)
                {
                    dq_dispatch<bits, cb, HALF>(stage + t * TWORDS, lane << 3, f0, f1);
                }
                else if constexpr (HALF)
                {
                    const uint32_t w = HALF_TWO ? bw[t >> 1] : bw[t];
                    const int base = HALF_TWO ? (t & 1) * TWORDS : 0;
                    uint32_t a7 = __shfl_sync(0xffffffffu, w, base + XPK(0));
                    uint32_t b7 = __shfl_sync(0xffffffffu, w, base + XPK(1));
                    uint32_t a3 = __shfl_sync(0xffffffffu, w, base + XPK(3));
                    uint32_t b3 = __shfl_sync(0xffffffffu, w, base + XPK(4));
                    exl3_gemv_ns::dq8_regs_half<bits, cb>(a7, b7, XPK(2), a3, b3, XPK(5), f0, f1);
                }
                else if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }
#endif

#if EXL3_MOE_PROBE == 3
                ch[t][0][0] = __hadd2(ch[t][0][0], __hmul2(a01[0], __hadd2(f0[0], f0[1])));
                ch[t][1][0] = __hadd2(ch[t][1][0], __hmul2(a23[0], __hadd2(f1[0], f1[1])));
#elif EXL3_MOE_TMMA_ON
                {
                    // Transposed accumulator: ch[t][f][0] = (column t * 16 + f * 8 + lane / 4, rows 2 (lane % 4), +1)
                    FragC_h c;
                    c[0] = ch[t][0][0];
                    c[1] = ch[t][1][0];
                    mma_t_h(f0, f1, aw, c);
                    ch[t][0][0] = c[0];
                    ch[t][1][0] = c[1];
                }
#else
                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
#endif
            }

            if (fold > 0 || (fold < 0 && ((i + 1) % FOLD == 0 || i + 1 == myn)))
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
    };

    // Ring slot d of block ib: consume, refill with slice i + PFD when it exists, run the slice. (An activation-fragment
    // register ring 1-4 slices ahead measured 0-3 % at best and spilled at depth 4; notes/CoopKernel2.md)
    auto slot = [&] (int i, int d, bool steady, int fold)
    {
        const uint2 aw = ld_a(i);
        if constexpr (EXL3_MOE_LATE)
        {
            // Decode straight from the ring slot, refill it after the slice (no register copy of the slot). Paired: one
            // load refills both slots of a pair after the pair's second slice (myn even, so slice i + PFD exists
            // whenever i - 1 + PFD does)
            step(i, pf[d], aw, fold);
            if constexpr (PAIRED)
            {
                if ((d & 1) && (steady || i - 1 + PFD < myn)) ld_pair(i - 1 + PFD, d - 1);
            }
            else if (steady || i + PFD < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PFD, l);
            }
            return;
        }
        uint32_t bw[LOADS];
        #pragma unroll
        for (int l = 0; l < LOADS; ++l)
            bw[l] = pf[d][l];
        if (steady || i + PFD < myn)
        {
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(i + PFD, l);
        }
        step(i, bw, aw, fold);
    };

    int ib = 0;
    if constexpr (EXL3_MOE_STEADY)
    {
        // Full blocks whose refills all exist: no bounds checks, fold points known per ring slot (ib % PFD == 0 and
        // PFD % FOLD == 0, and i + 1 < myn holds throughout)
        for (; ib + 2 * PFD <= myn; ib += PFD)
        {
            #pragma unroll
            for (int d = 0; d < PFD; ++d)
                slot(ib + d, d, true, (d + 1) % FOLD == 0 ? 1 : 0);
        }
    }
    for (; ib < myn; ib += PFD)
    {
        #pragma unroll
        for (int d = 0; d < PFD; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;
            slot(i, d, false, -1);
        }
    }
    #undef XPK

    // Cross-warp reduction over the k splits. Lane l holds row l/4, cols tile*16 + frag*8 + 2*(l%4) (+1); under TMMA
    // col tile*16 + frag*8 + l/4, rows 2*(l%4) (+1)
#if EXL3_MOE_TMMA_ON && EXL3_MOE_PROBE != 3
    {
        const int rr = 2 * (lane & 3);
        float* red = sh_red + wk * ROWS * TCOLS + wn * COLS + (lane >> 2);
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
            {
                const int col = t * 16 + f * 8;
                if (rr + 0 < nrows) red[(rr + 0) * TCOLS + col] = acc0[t][f].x;
                if (rr + 1 < nrows) red[(rr + 1) * TCOLS + col] = acc0[t][f].y;
            }
    }
#else
    if (r0_ok)
    {
        const int c0 = 2 * (lane & 3);
        float* red = sh_red + (wk * ROWS + r0) * TCOLS + wn * COLS;
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
            {
                const int col = t * 16 + f * 8 + c0;
                red[col + 0] = acc0[t][f].x;
                red[col + 1] = acc0[t][f].y;
            }
    }
#endif
    __syncthreads();

    for (int o = threadIdx.x; o < TCOLS * nrows; o += THREADS)     // up to 128 x 8 outputs per block
    {
        const int r = o / TCOLS;
        const int c = o % TCOLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WKK; ++j)
            sum += sh_red[(j * ROWS + r) * TCOLS + c];
        const size_t idx = (size_t) rows[r] * c_stride + group * TCOLS + c;
        if (c_f32) ((float*) C)[idx] = sum;
        else       ((half*) C)[idx] = __float2half_rn(sum);
    }
    __syncthreads();
}

// Completion counter: every thread fences its own stores, then one thread arrives. Returns true
// (block-uniform) for the last arriving block, which may then read the other blocks' results
// through L2 (load_*_cg)
__device__ __forceinline__ bool arrive_last(int* counter, int expected, int* sh_flag)
{
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        *sh_flag = (atomicAdd(counter, 1) == expected - 1);
    __syncthreads();
    bool last = *sh_flag;
    if (last) __threadfence();
    return last;
}

// Number of active slots of a token
__device__ __forceinline__ int token_active_slots(const MoeCoopParams& p, int row)
{
    int n = 0;
    for (int k = 0; k < p.topk; ++k)
        n += slot_info(p, row * p.topk + k).active ? 1 : 0;
    return n;
}

// A token with no active slot on this rank (expert-parallel sharding: every pick lives elsewhere)
// gets no arrival in the down stage, so nobody would write its output row. Writes one 128-chunk of
// that row: the shared-expert term when there is one, else zeros. Warp-level
__device__ __forceinline__ void write_empty_row_chunk(const MoeCoopParams& p, int row, int chunk, int lane)
{
    const int col = chunk * 128 + lane * 4;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    if (p.sh_out)
    {
        float gv = 1.0f;
        if (p.sh_gate_w)
        {
            const half* xr = p.x + (size_t) row * p.x_stride;
            float dot = 0.0f;
            for (int i = lane * 2; i < p.H; i += 64)
            {
                half2 xv = *((const half2*) (xr + i));
                half2 wv = *((const half2*) (p.sh_gate_w + i));
                dot += __low2float(xv) * __low2float(wv) + __high2float(xv) * __high2float(wv);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, o);
            gv = 1.0f / (1.0f + __expf(-dot));
        }
        const float* sh = p.sh_out + (size_t) row * p.H + col;
        if (col + 0 < p.H) o0 = gv * sh[0];
        if (col + 1 < p.H) o1 = gv * sh[1];
        if (col + 2 < p.H) o2 = gv * sh[2];
        if (col + 3 < p.H) o3 = gv * sh[3];
    }
    float* dst = p.out + (size_t) row * p.out_stride + col;
    if (col + 3 < p.H_out)
        *((float4*) dst) = make_float4(o0, o1, o2, o3);
    else
    {
        if (col + 0 < p.H_out) dst[0] = o0;
        if (col + 1 < p.H_out) dst[1] = o1;
        if (col + 2 < p.H_out) dst[2] = o2;
    }
}

// Input rotation of one 128-chunk of a slot's x row: zero-padded from H to Hi, * suh, Hadamard
__device__ __forceinline__ void rotate_chunk(const MoeCoopParams& p, int row, const half* suh, int c, half* dst, int lane)
{
    const int col = c * 128 + lane * 4;
    const half* xr = p.x + (size_t) row * p.x_stride;
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    if (col + 3 < p.H)
        load_h4(xr + col, v0, v1, v2, v3);
    else
    {
        if (col + 0 < p.H) v0 = __half2float(xr[col + 0]);
        if (col + 1 < p.H) v1 = __half2float(xr[col + 1]);
        if (col + 2 < p.H) v2 = __half2float(xr[col + 2]);
    }
    scale_h4(suh + col, v0, v1, v2, v3);
    had128(v0, v1, v2, v3, lane);
    store_h4(dst + col, v0, v1, v2, v3);
}

// ---------------------------------------------------------------------------------------------
// Rotation pre-kernel (bsz > 1): warp per (slot, chunk, projection) into had_g / had_u

#ifdef EXL3_MOE_COOP_DEFINE_ROT
__global__ __launch_bounds__(THREADS, MOE_COOP_MINB)
void exl3_moe_coop_rot_kernel(const MoeCoopParams p)
{
    __shared__ int16_t sh_order[MAX_SLOTS];
    __shared__ int16_t sh_local[MAX_SLOTS];
    __shared__ int sh_scan[WK];
    __shared__ int sh_res[4];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nproj = p.gated ? 2 : 1;
    const int chunks = p.Hi / 128;
    const int slots = p.bsz * p.topk;
    if (blockIdx.x == 0) build_runs(p, sh_order, sh_local, sh_scan, sh_res);
    const int item = blockIdx.x * (THREADS / 32) + warp;
    if (item >= slots * chunks * nproj) return;
    const int s = item / (chunks * nproj);
    const int rem = item % (chunks * nproj);
    const int c = rem / nproj;
    const bool is_gate = p.gated && (rem % nproj) == 0;
    const SlotInfo si = slot_info(p, s);
    if (!si.active)
    {
        // Empty token row (no active slot on this rank): its first slot's warps write the output
        // chunks c, c + chunks, ... (Hi chunks vs Ho chunks may differ)
        if (rem % nproj == 0 && s % p.topk == 0 && token_active_slots(p, si.row) == 0)
            for (int oc = c; oc < p.Ho / 128; oc += chunks)
                write_empty_row_chunk(p, si.row, oc, lane);
        return;
    }
    const half* suh = is_gate ? TAB_H(g_suh, si.local) : TAB_H(u_suh, si.local);
    half* dst = (is_gate ? p.had_g : p.had_u) + (size_t) s * p.Hi;
    rotate_chunk(p, si.row, suh, c, dst, lane);
}
#else
__global__ void exl3_moe_coop_rot_kernel(const MoeCoopParams p);
#endif

// Fused launch (EXL3_MOE_COOP_SINGLE, mixed-K): A publishes each finished (slot, chunk) of act_out to ready_a,
// and B waits for every (slot, chunk) of its run before reading act_out. Inert for the two-launch kernels (p.fused == 0)
__device__ __forceinline__ void publish_a_ready(const MoeCoopParams& p, int s, int chunk, int warp)
{
    if (warp == 0) __threadfence();           // warp 0 stored act_out: each storing lane fences its own writes
    __syncthreads();
    if (threadIdx.x == 0) atomicExch(p.ready_a + (size_t) s * (p.I / 128) + chunk, 1);
    __syncthreads();
}

__device__ __forceinline__ unsigned long long globaltimer_ns()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Bounded wait (a broken publish protocol traps instead of hanging) until every (slot, chunk) of the run's rows is published
__device__ __forceinline__ void wait_a_ready(const MoeCoopParams& p, const int* rows, int nrows)
{
    constexpr unsigned long long TIMEOUT_NS = 20000000000ull;
    const int nch = p.I / 128;
    const int total = nrows * nch;
    for (int i = threadIdx.x; i < total; i += THREADS)
    {
        int* flag = p.ready_a + (size_t) rows[i / nch] * nch + i % nch;
        const unsigned long long t0 = globaltimer_ns();
        while (atomicAdd(flag, 0) == 0)
        {
            __nanosleep(64);
            if (globaltimer_ns() - t0 > TIMEOUT_NS) __trap();
        }
    }
    __syncthreads();
    __threadfence();
}
// ---------------------------------------------------------------------------------------------
// Kernel A: gate/up GEMV per (expert run, group, projection) with the chunk epilogue (rotation,
// bias, activation, down-input rotation) by the last arrival per (slot, chunk)

template <int bits, int cb, bool WIDE, bool HALF = false, bool PAIRED = false>
__device__ __forceinline__
void moe_a_body(const MoeCoopParams& p, int item, uint32_t* smem_dyn)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    constexpr int GPC = tile_gpc<WIDE>();
    constexpr int CPB = tile_cpb<WIDE>();
    half* sh_A = (half*) smem_dyn;
    float* sh_red = (float*) (smem_dyn + p.Hi / 2);
    uint32_t* sh_stage = smem_dyn + p.Hi / 2 + WK * ROWS * COLS;
    __shared__ int sh_flag;
    __shared__ int sh_res[4];
    __shared__ int sh_rows[ROWS];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nproj = p.gated ? 2 : 1;
    const int ng = p.I / TCOLS;


    const bool is_gate = p.gated && (item % nproj) == 0;
    const int rem = item / nproj;
    const int ks = rem % p.ksplit_a;                 // split-k index: partial output rows
    const int rem2 = rem / p.ksplit_a;
    const int run_idx = rem2 / ng;
    const int group = rem2 % ng;
    int nrows = 0;
    if (!read_run(p, run_idx, sh_rows, nrows, sh_res))
    {
        // bsz 1: an inactive slot's blocks (up projection, split 0) write the output chunks of an
        // empty token row, one warp per chunk, strided by the number of gate/up groups
        if (!p.a_global && run_idx < p.bsz * p.topk && !is_gate && ks == 0 && run_idx % p.topk == 0)
        {
            const int row = run_idx / p.topk;
            if (token_active_slots(p, row) == 0)
                for (int oc = group + warp * ng; oc < p.Ho / 128; oc += ng * WK)
                    write_empty_row_chunk(p, row, oc, lane);
        }
        return;
    }
    const int local = slot_info(p, sh_rows[0]).local;
    const int kslices_all = p.Hi / 16;
    const int k_begin = kslices_all * ks / p.ksplit_a;
    const int k_end = kslices_all * (ks + 1) / p.ksplit_a;

    const half* suh = is_gate ? TAB_H(g_suh, local) : TAB_H(u_suh, local);
    const half2* A2;
    size_t a_stride2;
    if (p.a_global)
    {
        A2 = (const half2*) (is_gate ? p.had_g : p.had_u);
        a_stride2 = (size_t) p.Hi / 2;
    }
    else
    {
        // bsz 1: every run is a single slot; its input is rotated into shared memory by the prologue below. A zero
        // row stride makes the tile read it whatever the slot index (which still selects the C row)
        A2 = (const half2*) sh_A;
        a_stride2 = 0;
    }
    // Block prologue, run by gemv_tile once the weight prefetch ring is in flight
    auto prologue = [&] ()
    {
        pdl_wait();
        if (!p.a_global)
        {
            const SlotInfo si = slot_info(p, sh_rows[0]);
            for (int c = warp; c < p.Hi / 128; c += WK)
                rotate_chunk(p, si.row, suh, c, sh_A, lane);
            __syncthreads();
        }
    };

    {
        const uint32_t* B32 = (const uint32_t*) (is_gate ? tab_ptr(p, p.g_trellis, p.sh_g_trellis, local)
                                                         : tab_ptr(p, p.u_trellis, p.sh_u_trellis, local));
        // Partial output rows of split ks live at slot + ks * slots (ksplit * slots <= slots_max)
        const size_t part = (size_t) ks * (p.bsz * p.topk) * p.I * (p.gu_f32 ? 4 : 2);
        void* C = (void*) (((char*) (is_gate ? p.gu_g : p.gu_u)) + part);
        gemv_tile<bits, cb, WIDE, HALF, PAIRED>(B32, A2, a_stride2, sh_rows, nrows, C, p.I, p.gu_f32, k_begin, k_end, p.I / 16, group, sh_red, sh_stage, prologue);
        pdl_trigger();
    }

    for (int r = 0; r < nrows; ++r)
    for (int j = 0; j < CPB; ++j)
    {
        const int s = sh_rows[r];
        const int chunk = (group * CPB + j) / GPC;
        if (!arrive_last(p.ctr_a + (size_t) s * (p.I / 128) + chunk, GPC * nproj * p.ksplit_a, &sh_flag)) continue;
        if (warp == 0 && !(p.dbg & 1))
        {
            const int col = chunk * 128 + lane * 4;
            const size_t off = (size_t) s * p.I + col;
            const size_t pstride = (size_t) (p.bsz * p.topk) * p.I;

            // Sum of the split-k partials (fixed order)
            auto load_sum = [&] (const void* buf, float& v0, float& v1, float& v2, float& v3)
            {
                v0 = v1 = v2 = v3 = 0.0f;
                for (int q = 0; q < p.ksplit_a; ++q)
                {
                    float t0, t1, t2, t3;
                    if (p.gu_f32) load_f4_cg(((const float*) buf) + q * pstride + off, t0, t1, t2, t3);
                    else          load_h4_cg(((const half*) buf) + q * pstride + off, t0, t1, t2, t3);
                    v0 += t0; v1 += t1; v2 += t2; v3 += t3;
                }
            };
            float u0, u1, u2, u3;
            load_sum(p.gu_u, u0, u1, u2, u3);
            had128(u0, u1, u2, u3, lane);
            scale_h4(TAB_H(u_svh, local) + col, u0, u1, u2, u3);
            if (p.u_bias) add_h4(TAB_H(u_bias, local) + col, u0, u1, u2, u3);

            float g0 = u0, g1 = u1, g2 = u2, g3 = u3;
            if (p.gated)
            {
                load_sum(p.gu_g, g0, g1, g2, g3);
                had128(g0, g1, g2, g3, lane);
                scale_h4(TAB_H(g_svh, local) + col, g0, g1, g2, g3);
                if (p.g_bias) add_h4(TAB_H(g_bias, local) + col, g0, g1, g2, g3);
            }

            float a0 = act_gate(p.act, p.gated, g0, u0, p.act_limit);
            float a1 = act_gate(p.act, p.gated, g1, u1, p.act_limit);
            float a2 = act_gate(p.act, p.gated, g2, u2, p.act_limit);
            float a3 = act_gate(p.act, p.gated, g3, u3, p.act_limit);

            scale_h4(TAB_H(d_suh, local) + col, a0, a1, a2, a3);
            had128(a0, a1, a2, a3, lane);
            store_h4(p.act_out + off, a0, a1, a2, a3);
        }
        if (p.fused) publish_a_ready(p, s, chunk, warp);
        __syncthreads();
    }
}

// Two-launch kernel A: resets the down stage's counters for this call (its previous use finished in stream order)
template <int bits, int cb, bool WIDE, bool HALF = false, bool PAIRED = false>
__global__ __launch_bounds__(THREADS, MOE_COOP_MINB)
void exl3_moe_coop_a_kernel(const MoeCoopParams p)
{
    extern __shared__ uint32_t smem_dyn[];
#if EXL3_MOE_PDL
    moe_a_body<bits, cb, WIDE, HALF, PAIRED>(p, blockIdx.x, smem_dyn);
    // PDL: this grid may start before its predecessor finishes; touch the shared counters only after it
    pdl_wait();
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * THREADS)
        p.ctr_b[i] = 0;
#else
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * THREADS)
        p.ctr_b[i] = 0;
    moe_a_body<bits, cb, WIDE, HALF, PAIRED>(p, blockIdx.x, smem_dyn);
#endif
}

// ---------------------------------------------------------------------------------------------
// Kernel B: down GEMV per (expert run, group); the last arrival of a (token, chunk) reduces the
// token's slots (rotation * svh, down bias, routing weight) in a fixed order, merges the shared
// expert and stores the trimmed row

template <int bits, int cb, bool WIDE, bool HALF = false, bool PAIRED = false>
__device__ __forceinline__
void moe_b_body(const MoeCoopParams& p, int item, uint32_t* smem_dyn)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    constexpr int GPC = tile_gpc<WIDE>();
    constexpr int CPB = tile_cpb<WIDE>();
    float* sh_red = (float*) smem_dyn;
    uint32_t* sh_stage = smem_dyn + WK * ROWS * COLS;
    float* sh_part = (float*) (smem_dyn + WK * ROWS * COLS + smem_stage_bytes<bits, HALF>() / 4);
    __shared__ int sh_flag;
    __shared__ int sh_n_active;
    __shared__ float sh_gate;
    __shared__ int sh_res[4];
    __shared__ int sh_rows[ROWS];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ng = p.Ho / TCOLS;


    const int ks = item % p.ksplit_b;
    const int rem = item / p.ksplit_b;
    const int run_idx = rem / ng;
    const int group = rem % ng;
    int nrows = 0;
    if (!read_run(p, run_idx, sh_rows, nrows, sh_res)) return;
    auto prologue = [&] ()
    {
        pdl_wait();
        if (p.fused) wait_a_ready(p, sh_rows, nrows);
    };
    const int local = slot_info(p, sh_rows[0]).local;
    const int kslices_all = p.I / 16;
    const int k_begin = kslices_all * ks / p.ksplit_b;
    const int k_end = kslices_all * (ks + 1) / p.ksplit_b;

    {
        const uint32_t* B32 = (const uint32_t*) tab_ptr(p, p.d_trellis, p.sh_d_trellis, local);
        float* C = p.d_out + (size_t) ks * (p.bsz * p.topk) * p.Ho;
        gemv_tile<bits, cb, WIDE, HALF, PAIRED>(B32, (const half2*) p.act_out, (size_t) p.I / 2, sh_rows, nrows, C, p.Ho, true,
                                  k_begin, k_end, p.Ho / 16, group, sh_red, sh_stage, prologue);
    }

    for (int r = 0; r < nrows; ++r)
    for (int j = 0; j < CPB; ++j)
    {
        const int s = sh_rows[r];
        const int row = s / p.topk;
        const int chunk = (group * CPB + j) / GPC;

        // Active slots of this token: the number of arrivals to expect for (row, chunk)
        if (threadIdx.x == 0)
        {
            int n = 0;
            for (int k = 0; k < p.topk; ++k)
                n += slot_info(p, row * p.topk + k).active ? 1 : 0;
            sh_n_active = n;
        }
        __syncthreads();
        if (!arrive_last(p.ctr_b + (size_t) row * (p.Ho / 128) + chunk, sh_n_active * GPC * p.ksplit_b, &sh_flag)) continue;
        if (p.dbg & 2) continue;

        const int col = chunk * 128 + lane * 4;

        // Shared-expert gate: sigmoid(x_row . w), fp32 across the block
        if (p.sh_gate_w)
        {
            const half* xr = p.x + (size_t) row * p.x_stride;
            float dot = 0.0f;
            for (int i = threadIdx.x * 2; i < p.H; i += THREADS * 2)
            {
                half2 xv = *((const half2*) (xr + i));
                half2 wv = *((const half2*) (p.sh_gate_w + i));
                dot += __low2float(xv) * __low2float(wv) + __high2float(xv) * __high2float(wv);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) sh_part[warp] = dot;
            __syncthreads();
            if (threadIdx.x == 0)
            {
                float tt = 0.0f;
                for (int w = 0; w < WK; ++w) tt += sh_part[w];
                sh_gate = 1.0f / (1.0f + __expf(-tt));
            }
            __syncthreads();
        }

        // Each warp takes slots warp, warp + WK, ...; per-warp partials are summed in warp order. The folded shared
        // expert's slot is left out here and merged after the routed sum, through its gate, like a separate sh_out
        auto slot_out = [&] (int sk, int local, float& v0, float& v1, float& v2, float& v3)
        {
            v0 = 0.0f; v1 = 0.0f; v2 = 0.0f; v3 = 0.0f;
            for (int q = 0; q < p.ksplit_b; ++q)     // split-k partials, fixed order
            {
                float t0, t1, t2, t3;
                load_f4_cg(p.d_out + ((size_t) q * (p.bsz * p.topk) + sk) * p.Ho + col, t0, t1, t2, t3);
                v0 += t0; v1 += t1; v2 += t2; v3 += t3;
            }
            had128(v0, v1, v2, v3, lane);
            scale_h4(TAB_H(d_svh, local) + col, v0, v1, v2, v3);
            if (p.d_bias) add_h4(TAB_H(d_bias, local) + col, v0, v1, v2, v3);
        };
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        for (int k = warp; k < p.sel_k; k += WK)
        {
            const int sk = row * p.topk + k;
            const SlotInfo sj = slot_info(p, sk);
            if (!sj.active) continue;
            float v0, v1, v2, v3;
            slot_out(sk, sj.local, v0, v1, v2, v3);
            o0 += sj.w * v0;
            o1 += sj.w * v1;
            o2 += sj.w * v2;
            o3 += sj.w * v3;
        }
        float* part = sh_part + warp * 128 + lane * 4;
        part[0] = o0; part[1] = o1; part[2] = o2; part[3] = o3;
        __syncthreads();
        if (warp == 0)
        {
            o0 = 0.0f; o1 = 0.0f; o2 = 0.0f; o3 = 0.0f;
            #pragma unroll
            for (int w = 0; w < WK; ++w)
            {
                const float* q = sh_part + w * 128 + lane * 4;
                o0 += q[0]; o1 += q[1]; o2 += q[2]; o3 += q[3];
            }

            if (p.sh_out || p.sh_fold)
            {
                const float gv = p.sh_gate_w ? sh_gate : 1.0f;
                float s0, s1, s2, s3;
                if (p.sh_fold)
                    slot_out(row * p.topk + p.sel_k, p.n_local, s0, s1, s2, s3);
                else
                {
                    const float* sh = p.sh_out + (size_t) row * p.H + col;
                    s0 = col + 0 < p.H ? sh[0] : 0.0f;
                    s1 = col + 1 < p.H ? sh[1] : 0.0f;
                    s2 = col + 2 < p.H ? sh[2] : 0.0f;
                    s3 = col + 3 < p.H ? sh[3] : 0.0f;
                }
                if (col + 0 < p.H) o0 += gv * s0;
                if (col + 1 < p.H) o1 += gv * s1;
                if (col + 2 < p.H) o2 += gv * s2;
                if (col + 3 < p.H) o3 += gv * s3;
            }

            float* dst = p.out + (size_t) row * p.out_stride + col;
            if (col + 3 < p.H_out)
                *((float4*) dst) = make_float4(o0, o1, o2, o3);
            else
            {
                if (col + 0 < p.H_out) dst[0] = o0;
                if (col + 1 < p.H_out) dst[1] = o1;
                if (col + 2 < p.H_out) dst[2] = o2;
            }
        }
        __syncthreads();
    }
}

// Two-launch kernel B: resets the gate/up stage's counters for the next call
template <int bits, int cb, bool WIDE, bool HALF = false, bool PAIRED = false>
__global__ __launch_bounds__(THREADS, MOE_COOP_MINB)
void exl3_moe_coop_b_kernel(const MoeCoopParams p)
{
    extern __shared__ uint32_t smem_dyn[];
#if EXL3_MOE_PDL
    moe_b_body<bits, cb, WIDE, HALF, PAIRED>(p, blockIdx.x, smem_dyn);
    // PDL: this grid may start before its predecessor finishes; touch the shared counters only after it
    pdl_wait();
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_a_len; i += gridDim.x * THREADS)
        p.ctr_a[i] = 0;
#else
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_a_len; i += gridDim.x * THREADS)
        p.ctr_a[i] = 0;
    moe_b_body<bits, cb, WIDE, HALF, PAIRED>(p, blockIdx.x, smem_dyn);
#endif
}

}  // namespace exl3_moe_coop_ns
