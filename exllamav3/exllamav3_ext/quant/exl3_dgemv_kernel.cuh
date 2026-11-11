#pragma once

// Dense decode GEMV for EXL3 trellis weights, 1..48 rows, any integer bitrate dq_dispatch decodes.
//
// Decode-width dense projections are latency-bound, not bandwidth-bound: a 5120x512 6 bpw matrix is
// 2 MB (~1.1 us at DRAM rate) but costs ~20 us in the cooperative block-pipelined exl3_gemm, whose
// time goes to the cooperative launch, two grid-wide barriers (input Hadamard, output Hadamard) and a
// long per-warp serial k loop. This kernel removes all three:
//
// - ordinary launch, no grid barrier. Split-K across blocks: block (column block, k split) covers 128
//   output columns (one output Hadamard group, one n-tile per warp) and a contiguous run of 128-wide
//   k groups (one input Hadamard group each), so every block computes exactly the slice of the input
//   transform it consumes: the transform is done once per (block, group), nothing global
// - every warp streams its weights with cp.async (L2 evict-first) into a warp-private two-stage ring
//   and decodes them from shared memory with the standard dq_dispatch, so any K works and all of a
//   group's loads are in flight at once
// - fp32 MMA accumulation (m16n8k16, up to 3 row tiles = 48 rows sharing each decoded tile)
// - per-split fp32 partials in a fixed workspace; the last-arriving split of a column block (counter,
//   self-resetting) sums the partials in split order and runs the output Hadamard/svh epilogue with
//   the same arithmetic as exl3_gemm's. Deterministic: fixed reduction order, no float atomics
//
// The single-matrix kernel takes EXL3_GEMM_ARGS (A_had unused, `locks` carries the workspace) and the
// multi-matrix one EXL3_MGEMM_ARGS (slot j on gridDim.z), so exl3_gemm_gr / exl3_mgemm_gr record and
// patch their graph parameters unchanged.

#include <cuda_fp16.h>
#include "../ptx.cuh"
#include "../util.cuh"
#include "exl3_dq.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"

#define EXL3_DGEMV_MAX_M 48
#define EXL3_DGEMV_WARPS 8
#define EXL3_DGEMV_THREADS (EXL3_DGEMV_WARPS * 32)
#define EXL3_DGEMV_CNT_INTS 16384       // completion counters at the head of the workspace
#ifndef EXL3_DGEMV_TARGET_BLOCKS
#define EXL3_DGEMV_TARGET_BLOCKS 256    // blocks per matrix the split rule aims for (P7: 188 SMs)
#endif
#define EXL3_DGEMV_MAX_SLOTS 8          // matrices per multi-matrix launch
#define EXL3_DGEMV_SLOT_FLOATS (2 * 1024 * 1024)   // partials per matrix (8 MB): 48 rows x 5120 x 8 splits
#define EXL3_DGEMV_A_STRIDE 136         // halves per staged input row: 128 + 8 keeps fragment reads conflict-free

namespace exl3_dgemv_ns {

template <int bits> constexpr int tile_words() { return bits * 8; }              // uint32 per 16x16 tile
template <int bits> constexpr int group_words() { return 8 * tile_words<bits>(); }  // one warp, one k group

// k splits of one matrix: a pure function of its shape and row count, so a matrix computes the same
// bits alone or inside a multi-matrix launch (and the summation order is fixed). About
// EXL3_DGEMV_TARGET_BLOCKS blocks per matrix, never more splits than 128-wide k groups, partials of
// at most EXL3_DGEMV_SLOT_FLOATS, every split owning >= 1 group
__host__ __device__ __forceinline__ int splits_for(int size_m, int size_k, int size_n)
{
    const int kgroups = size_k / 128;
    const int ncb = size_n / 128;
    int s = (EXL3_DGEMV_TARGET_BLOCKS + ncb - 1) / ncb;
    const int cap = EXL3_DGEMV_SLOT_FLOATS / (size_m * size_n);
    s = s > cap ? cap : s;
    s = s > kgroups ? kgroups : s;
    s = s < 1 ? 1 : s;
    const int kw = (kgroups + s - 1) / s;
    return (kgroups + kw - 1) / kw;
}

extern __shared__ int4 dgemv_smem[];

template <int bits, int cb, int MT, bool c_fp32>
__device__ __forceinline__ void body
(
    const half* __restrict__ A,
    const uint32_t* __restrict__ B32,
    void* __restrict__ C,
    const half* __restrict__ suh,
    const half* __restrict__ svh,
    float* __restrict__ ws,             // this matrix's partials, [splits][size_m][size_n]
    int* __restrict__ cnt,              // this matrix's counters, one per column block
    const int size_m,
    const int size_k,
    const int size_n,
    const int splits,
    const int cbi,
    const int split
)
{
    constexpr int TW = tile_words<bits>();
    constexpr int GW = group_words<bits>();
    constexpr int ROWS = 16 * MT;
    constexpr int WARPS = EXL3_DGEMV_WARPS;
    constexpr int AS = EXL3_DGEMV_A_STRIDE;
    constexpr float R_SCALE = 0.088388347648f;  // 1/sqrt(128)

    uint32_t* sh_w = (uint32_t*) dgemv_smem;                 // [2][WARPS][GW]
    half* sh_a = (half*) (sh_w + 2 * WARPS * GW);            // [2][ROWS][AS]
    int* sh_flag = (int*) (sh_a + 2 * ROWS * AS);

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int kgroups = size_k / 128;
    const int kw = CEIL_DIVIDE(kgroups, splits);
    const int g0 = split * kw;
    const int ng = MIN(kw, kgroups - g0);                   // <= 0 only for a degenerate split: zero partial
    const int ntiles = size_n / 16;
    const int tile = cbi * WARPS + warp;

    // Weights of k group gi (8 k slices of this warp's n-tile) -> ring slot gi & 1. Always commits, so
    // the wait count below is uniform
    auto issue_w = [&] (int gi)
    {
        if (gi < ng)
        {
            uint32_t* dst = sh_w + ((gi & 1) * WARPS + warp) * GW;
            const uint32_t* src = B32 + ((size_t) (g0 + gi) * 8 * ntiles + tile) * TW;
            constexpr int V = GW / 4;          // 16-byte chunks
            constexpr int TV = TW / 4;
            #pragma unroll
            for (int i = lane; i < V; i += 32)
            {
                const int s = i / TV;
                const int j = i - s * TV;
                cp_async_stream(dst + s * TW + j * 4, src + (size_t) s * ntiles * TW + j * 4);
            }
        }
        cp_async_fence();
    };

    // Input scales + Hadamard of k group gi for every live row -> input ring slot gi & 1. MMA rows past
    // size_m read whatever the slot holds: an output row depends only on its own input row, and those
    // rows are never stored
    auto had_a = [&] (int gi)
    {
        half* dst = sh_a + (gi & 1) * ROWS * AS;
        const int kofs = (g0 + gi) * 128;
        for (int r = warp; r < size_m; r += WARPS)
            had_hf_r_128_inner<true, false>(A + (size_t) r * size_k + kofs, dst + r * AS, suh + kofs, R_SCALE);
    };

    FragC acc[MT][2];
    #pragma unroll
    for (int m = 0; m < MT; ++m) { acc[m][0] = {}; acc[m][1] = {}; }

    if (ng > 0)
    {
        issue_w(0);
        issue_w(1);
        had_a(0);
        __syncthreads();

        const int r0 = lane >> 2;
        const int q2 = 2 * (lane & 3);

        for (int gi = 0; gi < ng; ++gi)
        {
            if (gi + 1 < ng) had_a(gi + 1);
            cp_async_wait<1>();
            __syncwarp();

            const uint32_t* wb = sh_w + ((gi & 1) * WARPS + warp) * GW;
            const half* ab = sh_a + (gi & 1) * ROWS * AS;

            #pragma unroll
            for (int s = 0; s < 8; ++s)
            {
                FragB f0, f1;
                dq_dispatch<bits, cb>(wb + s * TW, lane << 3, f0, f1);
                #pragma unroll
                for (int m = 0; m < MT; ++m)
                {
                    const half* ar = ab + (m * 16 + r0) * AS + s * 16 + q2;
                    FragA fa;
                    fa[0] = *(const half2*) (ar);
                    fa[1] = *(const half2*) (ar + 8 * AS);
                    fa[2] = *(const half2*) (ar + 8);
                    fa[3] = *(const half2*) (ar + 8 * AS + 8);
                    ptx_mma_m16n8k16(fa, f0, acc[m][0]);
                    ptx_mma_m16n8k16(fa, f1, acc[m][1]);
                }
            }

            __syncthreads();      // ring slots gi & 1 free, input of gi + 1 visible
            issue_w(gi + 2);
        }
        cp_async_wait<0>();
    }
    __syncthreads();      // rings free: the single-split staging below overlays them

    // Partial of this block. Lane holds rows r0 (c0, c1) and r0 + 8 (c2, c3), cols f * 8 + q2 (+1).
    // Split-K: to this split's workspace slab (row stride size_n). One split: straight to shared
    // memory (row stride 128), the block finishes its column block alone
    float* sh_c = (float*) dgemv_smem;
    {
        float* dst = splits > 1 ? ws + (size_t) split * size_m * size_n + tile * 16 : sh_c + warp * 16;
        const int stride = splits > 1 ? size_n : 128;
        const int r0 = lane >> 2;
        const int q2 = 2 * (lane & 3);
        #pragma unroll
        for (int m = 0; m < MT; ++m)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
            {
                const int col = f * 8 + q2;
                const int ra = m * 16 + r0;
                const int rb = ra + 8;
                if (ra < size_m) *((float2*) (dst + (size_t) ra * stride + col)) = make_float2(acc[m][f][0], acc[m][f][1]);
                if (rb < size_m) *((float2*) (dst + (size_t) rb * stride + col)) = make_float2(acc[m][f][2], acc[m][f][3]);
            }
    }

    // Last arriving split of this column block finishes it
    if (splits > 1) __threadfence();
    __syncthreads();
    if (splits > 1)
    {
        if (threadIdx.x == 0)
        {
            int old = atomicAdd(cnt + cbi, 1);
            int last = old == splits - 1;
            if (last) cnt[cbi] = 0;
            *sh_flag = last;
        }
        __syncthreads();
        if (!*sh_flag) return;
        __threadfence();
    }

    const int c0 = cbi * 128;
    for (int r = warp; r < size_m; r += WARPS)
    {
        float4 v;
        if (splits == 1)
            v = ((const float4*) (sh_c + r * 128))[lane];
        else
        {
            v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int s = 0; s < splits; ++s)
            {
                float4 p = __ldcg(((const float4*) (ws + ((size_t) s * size_m + r) * size_n + c0)) + lane);
                v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
            }
        }

        // Same epilogue as exl3_gemm: C holds the raw sum (rounded to the output type), then the output
        // Hadamard with post scale runs over it in place. Each lane reads back only what it wrote
        if constexpr (c_fp32)
        {
            float* cp = ((float*) C) + (size_t) r * size_n + c0;
            ((float4*) cp)[lane] = v;
            had_ff_r_128_inner<false, true>(cp, cp, svh + c0, R_SCALE);
        }
        else
        {
            half* cp = ((half*) C) + (size_t) r * size_n + c0;
            ((half4*) cp)[lane] = half4(__float2half_rn(v.x), __float2half_rn(v.y),
                                        __float2half_rn(v.z), __float2half_rn(v.w));
            had_hf_r_128_inner<false, true>(cp, cp, svh + c0, R_SCALE);
        }
    }
}

}  // namespace exl3_dgemv_ns

// One matrix. Grid: (size_n / 128) * splits_for(size_m, size_k, size_n) blocks, column block fastest;
// `locks` is the workspace (EXL3_DGEMV_CNT_INTS counters, then the partials)
template <int bits, int cb, int MT, bool c_fp32>
__global__ __launch_bounds__(EXL3_DGEMV_THREADS)
void exl3_dgemv_kernel(EXL3_GEMM_ARGS)
{
    const int ncb = size_n / 128;
    const int splits = exl3_dgemv_ns::splits_for(size_m, size_k, size_n);
    const int cbi = blockIdx.x % ncb;
    const int split = blockIdx.x / ncb;
    if (split >= splits) return;
    exl3_dgemv_ns::body<bits, cb, MT, c_fp32>
    (
        A, (const uint32_t*) B, C, suh, svh,
        (float*) (locks + EXL3_DGEMV_CNT_INTS), locks,
        size_m, size_k, size_n, splits, cbi, split
    );
}

// Several matrices, slot j = blockIdx.z (blockIdx.y stays 0: the Hadamard helpers index scales by it).
// Slot j uses matrix B_indices[j] (or j), input A[j] (or the broadcast A), output C_list[mat] at width
// size_n_list[mat] (or C[j] at size_n), and splits_for(size_m, size_k, its width) k splits, exactly as
// a single-matrix launch would. Grid x: the largest ncb * splits over the slots (host computed); blocks
// past a slot's own count exit. Slot j owns counters [j * CNT_INTS / MAX_SLOTS, +CNT_INTS / MAX_SLOTS)
// and partials [j * SLOT_FLOATS, +SLOT_FLOATS). Unweighted, unfiltered, single-token, unsliced only
// (checked by the host)
template <int bits, int cb, int MT, bool c_fp32>
__global__ __launch_bounds__(EXL3_DGEMV_THREADS)
void exl3_dmgemv_kernel(EXL3_MGEMM_ARGS)
{
    const int j = blockIdx.z;
    const int mat = B_indices ? (int) B_indices[j] : j;
    if (mat < 0) return;
    const int n_j = size_n_list ? size_n_list[mat] : size_n;
    const int ncb = n_j / 128;
    const int splits = exl3_dgemv_ns::splits_for(size_m, size_k, n_j);
    const int cbi = blockIdx.x % ncb;
    const int split = blockIdx.x / ncb;
    if (split >= splits) return;

    const half* A_j = bszm_in == 1 ? A : A + (size_t) j * size_m * size_k;
    void* C_j;
    if (C_list) C_j = C_list[mat];
    else if constexpr (c_fp32) C_j = (void*) (((float*) C) + (size_t) j * size_m * size_n);
    else                       C_j = (void*) (((half*) C) + (size_t) j * size_m * size_n);

    float* ws = (float*) (locks + EXL3_DGEMV_CNT_INTS) + (size_t) j * EXL3_DGEMV_SLOT_FLOATS;
    int* cnt = locks + j * (EXL3_DGEMV_CNT_INTS / EXL3_DGEMV_MAX_SLOTS);

    exl3_dgemv_ns::body<bits, cb, MT, c_fp32>
    (
        A_j, (const uint32_t*) B_list[mat], C_j, suh_list[mat], svh_list[mat],
        ws, cnt, size_m, size_k, n_j, splits, cbi, split
    );
}
