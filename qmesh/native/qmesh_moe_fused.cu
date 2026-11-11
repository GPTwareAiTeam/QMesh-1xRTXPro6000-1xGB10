/* qmesh_moe_fused: the cold-expert MoE forward as one launch per projection.
 *
 * The shipped path is four kernels per bitrate group:
 *
 *     had_in(x)  -> a13 (rows x H fp16)          exl3_moe_had_in
 *     gemm13(a13)                                exl3_moe_gemm   (w13 = 2*I columns)
 *     glu(inter) -> a2 (rows x I fp16)           exl3_moe_glu_had_in
 *     gemm2(a2, accumulate into the token rows)  exl3_moe_gemm   (H columns)
 *
 * Every arrow is a full write and read of an activation tensor whose only consumer is the next
 * kernel in the list, and the first one is written *twice* (the w13 trellis is one tensor with two
 * shard groups, and the vendor kernel indexes its A slice as `A + grp*m*k`, so had_in stores a copy
 * per group even though the groups differ only in the suh scale they are transformed with).
 *
 * These kernels fold the Hadamard transforms into the GEMMs that consume them:
 *
 *     stage 1: a13 is never built. Each block transforms the 128-wide k tile it is about to read
 *              (one warp per A row, `had128_warp_in`, straight into the shared A tile), runs the
 *              w13 GEMM, and writes the fp16 *output* transform to `inter` -- still needed there,
 *              because the down projection's k extent (I) is 18 blocks wide and no single block of
 *              the first stage owns a whole row of it.
 *     stage 2: a2 is never built. The A tile of each k step is `had128_warp_glu_in(inter_gate,
 *              inter_up)` over that k tile, then the down GEMM, then the routing-weighted combine.
 *
 * Numerics: bit for bit the same sums. The A values handed to the mma are the same fp16 values the
 * separate transform kernels store (the same `had128_warp_*` helper, the same rounding to fp16
 * before the tensor core sees it), the k accumulation order is ascending 16-element steps either
 * way, and the accumulator is fp32 (cuda-exl3's default; CUDA_EXL3_FP16_ACC is refused here because
 * a fused kernel has no second pass to fall back on). What differs is only the shared-memory tile
 * shape, which cannot change a sum.
 *
 * Tile: BM rows x BN=128 columns x BK=128, one warp row (WARPS_M = 1) and eight warps across n, so
 * each 16x16 trellis tile is decoded exactly once per block -- the vendor's own shape for
 * block_m=16. BK=128 rather than the vendor's 64 is what makes the fused transform possible: the
 * Hadamard is defined on 128 contiguous elements, which is exactly one A row of one k tile, and a
 * warp holds one such row as 4 halfs per lane, in lane order. So the A tile is *not* XOR-swizzled
 * (it cannot be: the transform writes it); its row stride is padded by one int4 instead (17 int4 =
 * 272 B), which puts the eight rows of an ldmatrix.x4 on eight distinct 4-bank groups.
 */
#include "qmesh_moe_cpp.h"

#include <cstdint>

#include <cuda_fp16.h>

/* The vendored device-side primitives: fragments, mma/cp.async wrappers, ldsm4, the trellis
 * decoders and the 128-point Hadamard. Read-only, shared with the vendor's own kernels. */
#include "../../vendor/cuda-exl3/src/cuda_exl3/csrc/exl3_dq.cuh"
#include "../../vendor/cuda-exl3/src/cuda_exl3/csrc/exl3_had.cuh"

namespace qmf {

constexpr int BN = 128;      /* == HAD_N: a block owns whole Hadamard blocks */
constexpr int BK = 128;      /* == HAD_N: one warp transforms one A row segment */
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;
constexpr int WARP_N = 16;   /* 16x16 trellis tile -> one warp, decoded once */
constexpr int WARPS_N = BN / WARP_N;
constexpr int NBLK = WARP_N / 8;   /* n8 fragments per warp */
constexpr int A_COLS = BK / 8;     /* int4 per A row */
constexpr int A_STRIDE = A_COLS + 1;
constexpr int KSTEPS = BK / 16;
constexpr int NTILES = BN / 16;

template <int BITS, int BM>
struct Cfg {
    static constexpr int MBLK = BM / 16;
    static constexpr int TILE_I4 = 2 * BITS;          /* trellis bytes per 16x16 tile, in int4 */
    static constexpr int TILE_U32 = 8 * BITS;
    static constexpr int SH_A_I4 = BM * A_STRIDE;
    static constexpr int SH_B_I4 = KSTEPS * NTILES * TILE_I4;
    static constexpr int SH_STAGE_I4 = SH_A_I4 + SH_B_I4;
    static constexpr int C_STRIDE = BN + 8;
    static constexpr int C_I4 = (BM * C_STRIDE * 2 + 15) / 16;
    static constexpr int SMEM_I4 = SH_STAGE_I4 * 2 > C_I4 ? SH_STAGE_I4 * 2 : C_I4;
};

/* fp32 accumulator, as cuda-exl3's Acc<false>: the element order of an mma D fragment is regs
 * (0,1),(2,3) for rows lane>>2 and (lane>>2)+8. */
struct AccF {
    using T = FragC;
    __device__ __forceinline__ static void zero(T& c) {
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = 0.0f;
    }
    __device__ __forceinline__ static void mma(const FragA& a, const FragB& b, T& c) {
        mma_m16n8k16(a, b, c);
    }
    __device__ __forceinline__ static half geth(const T& c, int i) { return __float2half(c[i]); }
};

/* Trellis staging: KSTEPS x NTILES packed 16x16 tiles, same layout and indexing as the vendor's
 * load_stage, so a block reads exactly the bytes it always read. */
template <int BITS, int BM>
__device__ __forceinline__ void load_b(int4* sh, const uint16_t* __restrict__ Bq, int kt, int n0,
                                       int n_tiles_full, int t) {
    using C = Cfg<BITS, BM>;
    int4* sh_b = sh + A_STRIDE * BM;
    const int k0 = kt * BK;
#pragma unroll
    for (int i = t; i < C::SH_B_I4; i += NTHREADS) {
        const int chunk = i % C::TILE_I4;
        const int tile = i / C::TILE_I4;
        const int ks = tile / NTILES;
        const int nt = tile % NTILES;
        const int4* src = ((const int4*)Bq) +
                          ((size_t)(k0 / 16 + ks) * n_tiles_full + n0 / 16 + nt) * C::TILE_I4 +
                          chunk;
        cp_async16(sh_b + tile * C::TILE_I4 + chunk, src);
    }
}

/* MMA over one staged k tile. Ascending ks (and so ascending k) is what keeps the accumulation
 * order identical to the unfused path. */
template <int BITS, int CB, int BM>
__device__ __forceinline__ void compute_tile(const int4* sh, FragC (&frag_c)[Cfg<BITS, BM>::MBLK][NBLK],
                                             int warp, int lane) {
    using C = Cfg<BITS, BM>;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    const int4* sh_a = sh;
    const uint32_t* sh_b = (const uint32_t*)(sh + A_STRIDE * BM);
#pragma unroll
    for (int ks = 0; ks < KSTEPS; ++ks) {
        FragA frag_a[C::MBLK];
#pragma unroll
        for (int mb = 0; mb < C::MBLK; ++mb) {
            const int r = (lane & 7) + 8 * ((lane >> 3) & 1);
            const int R = warp_m * BM + mb * 16 + r;
            const int c = ks * 2 + (lane >> 4);
            ldsm4(frag_a[mb], sh_a + R * A_STRIDE + c);
        }
        FragB frag_b[NBLK];
#pragma unroll
        for (int nb = 0; nb < NBLK; nb += 2) {
            const int nt = (warp_n * WARP_N + nb * 8) / 16;
            const uint32_t* tile = sh_b + (ks * NTILES + nt) * C::TILE_U32;
            dq_dispatch<BITS, CB, false>(tile, lane << 3, frag_b[nb], frag_b[nb + 1]);
        }
#pragma unroll
        for (int mb = 0; mb < C::MBLK; ++mb)
#pragma unroll
            for (int nb = 0; nb < NBLK; ++nb) AccF::mma(frag_a[mb], frag_b[nb], frag_c[mb][nb]);
    }
}

/* Shared C tile -> the Hadamard output transform's input. Same writer the vendor epilogue uses. */
template <int BITS, int BM>
__device__ __forceinline__ void store_c(half* sh_c, const FragC (&frag_c)[Cfg<BITS, BM>::MBLK][NBLK],
                                        int warp, int lane) {
    using C = Cfg<BITS, BM>;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
#pragma unroll
    for (int mb = 0; mb < C::MBLK; ++mb) {
#pragma unroll
        for (int nb = 0; nb < NBLK; ++nb) {
            const int col = warp_n * WARP_N + nb * 8 + 2 * (lane & 3);
            half* p0 = sh_c + (warp_m * BM + mb * 16 + (lane >> 2)) * C::C_STRIDE + col;
            half* p1 = p0 + 8 * C::C_STRIDE;
            p0[0] = AccF::geth(frag_c[mb][nb], 0);
            p0[1] = AccF::geth(frag_c[mb][nb], 1);
            p1[0] = AccF::geth(frag_c[mb][nb], 2);
            p1[1] = AccF::geth(frag_c[mb][nb], 3);
        }
    }
}

__device__ __forceinline__ void zero_row(half* dst, int lane) {
    half4 z;
    z.x = __float2half2_rn(0.0f);
    z.y = __float2half2_rn(0.0f);
    ((half4*)dst)[lane] = z;
}

/* ---- stage 1: had_in + w13 GEMM + out-had ----------------------------------
 *
 * One block: block_m rows of the alignment (all the same expert e, starting at routed row m0) by
 * 128 output columns (n0). The w13 trellis holds both shard groups (gate, up), so the block column
 * picks the group's suh scale -- the same two A copies the unfused path built, without building
 * them. XT is the activation's storage type: half for the per-group path (which converts the
 * request once), bf16 for the layer path, which reads the request slot's bf16 directly --
 * ActVec<bf16>::load is bf16 -> float -> half RN, the value k_bf16_to_half stores, so the A tile is
 * the same bits either way.
 */
template <int BITS, int CB, int BM, typename XT>
__device__ __forceinline__ void stage_w13_body(
    int4* smem, const XT* __restrict__ x, const int* __restrict__ sorted_ids, int nlv,
    const half* __restrict__ suh, half* __restrict__ inter, const uint16_t* __restrict__ bq,
    const half* __restrict__ svh, int e, int m0, int n0, int k, int n_total, int shard_w,
    int n_shards, int64_t b_expert_stride, int m_valid, int top_k)
{
    using C = Cfg<BITS, BM>;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;

    const int grp = n0 / shard_w;
    const uint16_t* Bq = bq + (size_t)e * b_expert_stride;
    const half* S = svh + (size_t)e * n_total + n0;
    const half* As = suh + ((size_t)e * n_shards + grp) * k;
    const int n_tiles_full = n_total / 16;
    const int kt_total = k / BK;

    /* The A tile: one warp per row, one Hadamard per row, straight into shared. Padding rows and
     * rows of a dead block carry zeros -- the same values cp.async's zero-fill used to supply. */
    auto load_a = [&](int stage, int kt) {
        half* sh_a = (half*)(smem + stage * C::SH_STAGE_I4);
        const int k0 = kt * BK;
        for (int r = warp; r < BM; r += NWARPS) {
            const int grow = m0 + r;
            const int idx = grow < nlv ? sorted_ids[grow] : -1;
            half* dst = sh_a + r * (A_STRIDE * 8);
            if (idx >= 0 && idx < m_valid)
                had128_warp_in<XT>(x + (size_t)(idx / top_k) * k + k0, dst, As + k0, lane);
            else
                zero_row(dst, lane);
        }
    };

    FragC frag_c[C::MBLK][NBLK];
#pragma unroll
    for (int i = 0; i < C::MBLK; ++i)
#pragma unroll
        for (int j = 0; j < NBLK; ++j) AccF::zero(frag_c[i][j]);

    if (kt_total > 0) {
        load_a(0, 0);
        load_b<BITS, BM>(smem, Bq, 0, n0, n_tiles_full, t);
        cp_async_fence();
    }

    for (int kt = 0; kt < kt_total; ++kt) {
        const int slot = kt & 1;
        cp_async_wait<0>();   /* the B tiles of this slot have landed */
        __syncthreads();      /* ... and every warp is done with the slot's previous use */
        if (kt + 1 < kt_total) {
            load_b<BITS, BM>(smem + (slot ^ 1) * C::SH_STAGE_I4, Bq, kt + 1, n0, n_tiles_full, t);
            cp_async_fence();
            load_a(slot ^ 1, kt + 1);
        }
        compute_tile<BITS, CB, BM>(smem + slot * C::SH_STAGE_I4, frag_c, warp, lane);
    }

    /* Out transform, and with it the fp16 rounding the down projection's A tile must see. */
    __syncthreads();
    half* sh_c = (half*)smem;
    store_c<BITS, BM>(sh_c, frag_c, warp, lane);
    __syncthreads();
    for (int r = warp; r < BM; r += NWARPS) {
        const int grow = m0 + r;
        if (grow >= nlv) continue;
        const int idx = sorted_ids[grow];
        if (idx >= m_valid) continue;   /* padding row: stage 2 zeroes its own A tile */
        had128_warp_out<half>(sh_c + r * C::C_STRIDE, inter + (size_t)grow * n_total + n0, S,
                              lane);
    }
}

/* Routing weight of pick `idx`, as k_weights builds it: the per-group path hands the stage the
 * converted fp32 vector; the layer path reads the request's bf16 and rounds it the same way
 * (bf16 -> float -> half -> float). A pick whose id is -1 is never in sorted_ids, so k_weights'
 * zero branch has no counterpart here. */
__device__ __forceinline__ float route_w(const float* __restrict__ w, int idx) { return w[idx]; }
__device__ __forceinline__ float route_w(const __nv_bfloat16* __restrict__ w, int idx) {
    return __half2float(__float2half(__bfloat162float(w[idx])));
}

/* ---- stage 2: glu + had_in + down GEMM + combine ---------------------------
 *
 * One block: block_m routed rows by 128 of the H output columns. The rows are already in routed
 * order (stage 1's output), so the A tile is a straight read of `inter`'s two halves. SLOT=false is
 * the per-group kernels' fused combine, which accumulates each routed row into its token's row of
 * the fp32 answer; SLOT=true (the layer path) stores each routed row to its pick's own fp32 slot,
 * out[idx][n_total], for k_layer_reduce to sum in pick order.
 */
template <int BITS, int CB, int BM, typename WT, bool SLOT = false>
__device__ __forceinline__ void stage_down_body(
    int4* smem, const half* __restrict__ inter, const int* __restrict__ sorted_ids, int nlv,
    const half* __restrict__ suh, float* __restrict__ out, const uint16_t* __restrict__ bq,
    const half* __restrict__ svh, const WT* __restrict__ w, int e, int m0, int n0, int k,
    int n_total, int64_t b_expert_stride, int m_valid, int top_k)
{
    using C = Cfg<BITS, BM>;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;

    const uint16_t* Bq = bq + (size_t)e * b_expert_stride;
    const half* S = svh + (size_t)e * n_total;
    const half* As = suh + (size_t)e * k;
    const int n_tiles_full = n_total / 16;
    const int kt_total = k / BK;

    auto load_a = [&](int stage, int kt) {
        half* sh_a = (half*)(smem + stage * C::SH_STAGE_I4);
        const int k0 = kt * BK;
        for (int r = warp; r < BM; r += NWARPS) {
            const int grow = m0 + r;
            const int idx = grow < nlv ? sorted_ids[grow] : -1;
            half* dst = sh_a + r * (A_STRIDE * 8);
            if (idx >= 0 && idx < m_valid) {
                const half* gate = inter + (size_t)grow * 2 * k + k0;
                had128_warp_glu_in<half>(gate, gate + k, dst, As + k0, lane);
            } else {
                zero_row(dst, lane);
            }
        }
    };

    FragC frag_c[C::MBLK][NBLK];
#pragma unroll
    for (int i = 0; i < C::MBLK; ++i)
#pragma unroll
        for (int j = 0; j < NBLK; ++j) AccF::zero(frag_c[i][j]);

    if (kt_total > 0) {
        load_a(0, 0);
        load_b<BITS, BM>(smem, Bq, 0, n0, n_tiles_full, t);
        cp_async_fence();
    }

    for (int kt = 0; kt < kt_total; ++kt) {
        const int slot = kt & 1;
        cp_async_wait<0>();
        __syncthreads();
        if (kt + 1 < kt_total) {
            load_b<BITS, BM>(smem + (slot ^ 1) * C::SH_STAGE_I4, Bq, kt + 1, n0, n_tiles_full, t);
            cp_async_fence();
            load_a(slot ^ 1, kt + 1);
        }
        compute_tile<BITS, CB, BM>(smem + slot * C::SH_STAGE_I4, frag_c, warp, lane);
    }

    __syncthreads();
    half* sh_c = (half*)smem;
    store_c<BITS, BM>(sh_c, frag_c, warp, lane);
    __syncthreads();
    for (int r = warp; r < BM; r += NWARPS) {
        const int grow = m0 + r;
        if (grow >= nlv) continue;
        const int idx = sorted_ids[grow];
        if (idx >= m_valid) continue;   /* padding row belongs to no token */
        if constexpr (SLOT)
            had128_warp_out_slot(sh_c + r * C::C_STRIDE, out + (size_t)idx * n_total + n0, S + n0,
                                 lane, route_w(w, idx));
        else
            had128_warp_out_acc<float>(sh_c + r * C::C_STRIDE,
                                       out + (size_t)(idx / top_k) * n_total + n0, S + n0, lane,
                                       route_w(w, idx));
    }
}

/* Per-group kernels: grid (n_total/BN, n_blocks), one block row per expert block of one group's
 * alignment. */
template <int BITS, int CB, int BM>
__global__ __launch_bounds__(NTHREADS) void k_stage_w13(
    const half* __restrict__ x, const int* __restrict__ sorted_ids,
    const half* __restrict__ suh, half* __restrict__ inter, const uint16_t* __restrict__ bq,
    const half* __restrict__ svh, const int* __restrict__ eids, const int* __restrict__ n_rows,
    int k, int n_total, int shard_w, int n_shards, int64_t b_expert_stride, int m_valid, int top_k)
{
    extern __shared__ __align__(16) int4 smem[];
    const int nlv = n_rows ? *n_rows : 0x7fffffff;
    const int m0 = blockIdx.y * BM;
    if (m0 >= nlv) return;   /* the alignment pads eids for the worst case; retire the surplus */
    const int e = eids[blockIdx.y];
    if (e < 0) return;
    stage_w13_body<BITS, CB, BM, half>(smem, x, sorted_ids, nlv, suh, inter, bq, svh, e, m0,
                                       blockIdx.x * BN, k, n_total, shard_w, n_shards,
                                       b_expert_stride, m_valid, top_k);
}

template <int BITS, int CB, int BM>
__global__ __launch_bounds__(NTHREADS) void k_stage_down(
    const half* __restrict__ inter, const int* __restrict__ sorted_ids,
    const half* __restrict__ suh, float* __restrict__ out, const uint16_t* __restrict__ bq,
    const half* __restrict__ svh, const int* __restrict__ eids, const int* __restrict__ n_rows,
    const float* __restrict__ w, int k, int n_total, int64_t b_expert_stride, int m_valid,
    int top_k)
{
    extern __shared__ __align__(16) int4 smem[];
    const int nlv = n_rows ? *n_rows : 0x7fffffff;
    const int m0 = blockIdx.y * BM;
    if (m0 >= nlv) return;
    const int e = eids[blockIdx.y];
    if (e < 0) return;
    stage_down_body<BITS, CB, BM, float>(smem, inter, sorted_ids, nlv, suh, out, bq, svh, w, e, m0,
                                         blockIdx.x * BN, k, n_total, b_expert_stride, m_valid,
                                         top_k);
}

/* ---- the layer path: every bitrate group of a layer in one launch per projection ----------------
 *
 * The per-group path pays, per piece: bf16->half and weight conversion kernels, a histogram and a
 * one-thread alignment per group, an 8-byte-per-group readback with a host sync, then per group a
 * cursor/n_rows clear, a fill, a scatter and the two stage launches. In the shipped decode trace
 * (f9d57851, rows 4/6, two groups) that is ~95 us of a ~580 us call before and between the stage
 * kernels, and each group's stage launch is a short grid with its own tail wave.
 *
 * Here one plan kernel builds the whole layer's alignment on device (all groups, one block list)
 * and checks the ids; then one w13 launch and one down launch run every group's blocks, and one
 * reduce launch combines. The grid's y extent is the host's bound on the block count (no
 * readback); a block past the plan's count retires on its first load. Per block the math is the
 * per-group kernels' own body.
 *
 * The combine is deterministic. The down stage writes each routed row's routing-weighted output to
 * that pick's own fp32 slot (slot[pick][H], one writer per element, no atomics), and k_layer_reduce
 * adds a token's top_k slots in pick order. The answer depends only on the picks, never on the
 * order blocks retire in: the folded-atomic epilogue made every run of the layer path (and of the
 * per-group path) sum a token's experts in a different order, which moves greedy argmax on
 * near-tie tokens run to run (measured 2026-11-07: DSPARK=0, 8 prompts x 128, 0-3/8 prompts
 * identical between two runs of the same setting, first divergences at top-2 margins 0-0.03).
 * A pick whose id is -1 or belongs to no group of this layer has no slot writer; the plan marks it
 * dead (live[pick] = 0) and the reduce skips it, so no slot is read before it is written. */
constexpr int PLAN_THREADS = 1024;
constexpr int PLAN_EXPERTS = 384;                          /* expert stride of a (group, expert) cell */
constexpr int PLAN_SLOTS = QMC_MAX_GROUPS * PLAN_EXPERTS;

/* One block plans. Outputs: sorted[nblk*BM] (pick index per routed row, m_valid for a padding
 * row), blk[nblk] = (group, expert) per block, *nblk_out, live[m_valid] (1 for a pick that one
 * group of this layer serves, else 0: k_layer_reduce reads a slot only where live is 1), and *bad,
 * set when an id is outside [-1, n_cold), remaps outside its group, is served by more than one
 * group (two writers of one slot), or the plan would exceed max_blk (then nblk = 0, every pick is
 * dead and the reduce answers zeros). clear_bad clears the flag first (the request's first piece). */
template <int BM>
__global__ __launch_bounds__(PLAN_THREADS) void k_layer_plan(
    const qmc_layer_desc L, const int32_t* __restrict__ ids, int m_valid, int max_blk,
    int32_t* __restrict__ sorted, int2* __restrict__ blk, int32_t* __restrict__ nblk_out,
    int32_t* __restrict__ bad, int clear_bad, int32_t* __restrict__ live)
{
    __shared__ int cnt[PLAN_SLOTS];
    __shared__ int off[PLAN_SLOTS];
    __shared__ int warp_sum[PLAN_THREADS / 32];
    __shared__ int s_bad;
    const int t = threadIdx.x;
    for (int i = t; i < PLAN_SLOTS; i += PLAN_THREADS) cnt[i] = 0;
    if (t == 0) s_bad = 0;
    __syncthreads();

    /* histogram: a cold id belongs to exactly one group (the others' remaps hold -1 for it) */
    for (int i = t; i < m_valid; i += PLAN_THREADS) {
        const int v = ids[i];
        if (v < -1 || v >= L.n_cold) {
            s_bad = 1;
            continue;
        }
        if (v < 0) continue;
        int served = 0;
        for (int g = 0; g < L.n_groups; g++) {
            const int e = L.g[g].remap[v];
            if (e >= L.g[g].n_experts) s_bad = 1;
            else if (e >= 0) {
                atomicAdd(&cnt[g * PLAN_EXPERTS + e], 1);
                served++;
            }
        }
        if (served > 1) s_bad = 1;
    }
    __syncthreads();

    /* exclusive scan of blocks per cell in (group, expert) order: each group's own alignment
     * order, groups concatenated */
    constexpr int PER = (PLAN_SLOTS + PLAN_THREADS - 1) / PLAN_THREADS;
    int loc[PER];
    int sum = 0;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const int i = t * PER + j;
        const int b = i < PLAN_SLOTS ? (cnt[i] + BM - 1) / BM : 0;
        loc[j] = sum;
        sum += b;
    }
    const int lane = t & 31, warp = t >> 5;
    int incl = sum;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, d);
        if (lane >= d) incl += y;
    }
    if (lane == 31) warp_sum[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int w = lane < PLAN_THREADS / 32 ? warp_sum[lane] : 0;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, d);
            if (lane >= d) w += y;
        }
        if (lane < PLAN_THREADS / 32) warp_sum[lane] = w; /* inclusive per warp */
    }
    __syncthreads();
    const int base = (warp ? warp_sum[warp - 1] : 0) + incl - sum;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const int i = t * PER + j;
        if (i < PLAN_SLOTS) off[i] = base + loc[j];
    }
    const int total_blk = warp_sum[PLAN_THREADS / 32 - 1];
    __syncthreads();

    /* the host's bound is a proof (picks + one block per expert cannot exceed it); a plan past it
     * is a broken invariant and is refused rather than written past the buffers */
    const bool over = total_blk > max_blk;
    if (!over) {
        for (int i = t; i < PLAN_SLOTS; i += PLAN_THREADS) {
            const int nb = (cnt[i] + BM - 1) / BM;
            for (int b = 0; b < nb; b++)
                blk[off[i] + b] = make_int2(i / PLAN_EXPERTS, i % PLAN_EXPERTS);
        }
        for (int i = t; i < total_blk * BM; i += PLAN_THREADS) sorted[i] = m_valid;
    }
    __syncthreads();   /* cnt is read above and reset below; the padding fill precedes the scatter */
    for (int i = t; i < PLAN_SLOTS; i += PLAN_THREADS) cnt[i] = 0; /* reused as the cursor */
    __syncthreads();
    if (!over) {
        for (int i = t; i < m_valid; i += PLAN_THREADS) {
            const int v = ids[i];
            if (v < 0 || v >= L.n_cold) continue;
            for (int g = 0; g < L.n_groups; g++) {
                const int e = L.g[g].remap[v];
                if (e < 0 || e >= L.g[g].n_experts) continue;
                const int c = g * PLAN_EXPERTS + e;
                sorted[off[c] * BM + atomicAdd(&cnt[c], 1)] = i;
            }
        }
    }
    /* live mirrors the scatter above exactly: a pick is live iff one of its group remaps took it
     * into sorted (and the plan was written). s_bad is final here (set only before the first
     * barrier), so a double-served pick, whose slot would have two writers, is never read. */
    for (int i = t; i < m_valid; i += PLAN_THREADS) {
        const int v = ids[i];
        int served = 0;
        if (!over && v >= 0 && v < L.n_cold)
            for (int g = 0; g < L.n_groups; g++) {
                const int e = L.g[g].remap[v];
                if (e >= 0 && e < L.g[g].n_experts) served++;
            }
        live[i] = (served == 1 && !s_bad) ? 1 : 0;
    }
    if (t == 0) {
        *nblk_out = over ? 0 : total_blk;
        if (clear_bad || s_bad || over) *bad = (s_bad || over) ? 1 : 0;
    }
}

/* One block per (128 output columns, plan block). The group's bitrate and codebook select the
 * instantiation at run time; every branch is the per-group kernel's body. */
#define QMF_LAYER_SWITCH(BITS_, CALL)                                                          \
    switch ((BITS_) * 4 + G.cb) {                                                              \
    case 2 * 4 + 0: CALL(2, 0); break;                                                         \
    case 2 * 4 + 1: CALL(2, 1); break;                                                         \
    case 2 * 4 + 2: CALL(2, 2); break;                                                         \
    case 3 * 4 + 0: CALL(3, 0); break;                                                         \
    case 3 * 4 + 1: CALL(3, 1); break;                                                         \
    case 3 * 4 + 2: CALL(3, 2); break;                                                         \
    case 4 * 4 + 0: CALL(4, 0); break;                                                         \
    case 4 * 4 + 1: CALL(4, 1); break;                                                         \
    case 4 * 4 + 2: CALL(4, 2); break;                                                         \
    default: break;                                                                            \
    }

/* Persistent: a grid sized to the GPU's resident capacity walks the plan's (block, column) items,
 * column fastest, so every group's blocks share one wave schedule and nothing is launched for a
 * block the plan did not make (the host never learns the count). The barrier at the end of an item
 * is the one the per-group kernels get for free from retiring: the next item's first tile write
 * lands in the shared memory the previous epilogue is still reading. */
template <int BM>
__global__ __launch_bounds__(NTHREADS) void k_layer_w13(
    const qmc_layer_desc L, const __nv_bfloat16* __restrict__ x, const int* __restrict__ sorted,
    const int2* __restrict__ blk, const int* __restrict__ nblk, half* __restrict__ inter, int k,
    int m_valid, int top_k)
{
    extern __shared__ __align__(16) int4 smem[];
    const int nb = *nblk;
    const int n_total = 2 * L.inter;
    const int cols = n_total / BN;
    const int nlv = nb * BM;
    for (int item = blockIdx.x; item < nb * cols; item += gridDim.x) {
        const int by = item / cols, bx = item % cols;
        const int2 be = blk[by];
        const qmc_layer_group& G = L.g[be.x];
#define QMF_W13(B_, C_)                                                                        \
    stage_w13_body<B_, C_, BM, __nv_bfloat16>(                                                 \
        smem, x, sorted, nlv, (const half*)G.suh13, inter, (const uint16_t*)G.t13,             \
        (const half*)G.svh13, be.y, by * BM, bx * BN, k, n_total, L.inter, 2, G.bs13, m_valid, \
        top_k)
        QMF_LAYER_SWITCH(G.bits13, QMF_W13)
#undef QMF_W13
        __syncthreads();
    }
}

/* `slots` is [m_valid, n_total] fp32: each live pick's routing-weighted down output, one writer per
 * element (a pick is one routed row of exactly one block), so no atomics and nothing to zero. */
template <int BM>
__global__ __launch_bounds__(NTHREADS) void k_layer_down(
    const qmc_layer_desc L, const half* __restrict__ inter, const int* __restrict__ sorted,
    const int2* __restrict__ blk, const int* __restrict__ nblk, const __nv_bfloat16* __restrict__ w,
    float* __restrict__ slots, int n_total, int m_valid, int top_k)
{
    extern __shared__ __align__(16) int4 smem[];
    const int nb = *nblk;
    const int cols = n_total / BN;
    const int nlv = nb * BM;
    for (int item = blockIdx.x; item < nb * cols; item += gridDim.x) {
        const int by = item / cols, bx = item % cols;
        const int2 be = blk[by];
        const qmc_layer_group& G = L.g[be.x];
#define QMF_DOWN(B_, C_)                                                                       \
    stage_down_body<B_, C_, BM, __nv_bfloat16, true>(                                          \
        smem, inter, sorted, nlv, (const half*)G.suh2, slots, (const uint16_t*)G.t2,           \
        (const half*)G.svh2, w, be.y, by * BM, bx * BN, L.inter, n_total, G.bs2, m_valid,      \
        top_k)
        QMF_LAYER_SWITCH(G.bits2, QMF_DOWN)
#undef QMF_DOWN
        __syncthreads();
    }
}
#undef QMF_LAYER_SWITCH

/* acc[token][h] = sum over k = 0..top_k-1, in that order, of slots[token*top_k + k][h] for the live
 * picks; a token with no live pick gets 0. One thread per 4 columns (float4), every element of acc
 * written exactly once, so the accumulator needs no clearing and the sum has one fixed order. */
__global__ void k_layer_reduce(const float* __restrict__ slots, const int32_t* __restrict__ live,
                               float* __restrict__ acc, int rows, int top_k, int h4)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * h4) return;
    const int token = i / h4, c = i % h4;
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < top_k; k++) {
        const int p = token * top_k + k;
        if (!live[p]) continue;
        const float4 v = ((const float4*)slots)[(size_t)p * h4 + c];
        s.x += v.x;
        s.y += v.y;
        s.z += v.z;
        s.w += v.w;
    }
    ((float4*)acc)[i] = s;
}

/* Resident blocks of a kernel at `smem` bytes, times the SM count: the persistent grid. A failed
 * query is 0, which the caller refuses. */
template <class K>
int resident_grid(K fn, int smem) {
    int dev = 0, sms = 0, per = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, NTHREADS, smem) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return sms * (per > 0 ? per : 1);
}

/* A tile shape wider than 48 KiB needs the dynamic-shared attribute raised first; block_m=16 (the
 * only instantiated shape here) fits under the static limit, so this is a no-op it does not have to
 * know about. The call is kept so a future tile cannot silently fail to launch. */
template <class K>
cudaError_t raise_smem(K fn, int bytes) {
    if (bytes <= 48 * 1024) return cudaSuccess;
    return cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

/* Shared memory of one layer-path block: the widest tile any instantiated bitrate stages. */
constexpr int LAYER_SMEM = Cfg<4, 16>::SMEM_I4 * 16;

int launch_layer(const qmc_layer_desc& L, const void* x_bf16, const int32_t* ids,
                 const void* w_bf16, int m_valid, int top_k, int hidden, int max_blk,
                 int32_t* sorted, void* blk, int32_t* nblk, int32_t* bad, int clear_bad,
                 int32_t* live, float* slots, void* inter, float* acc, int rows, cudaStream_t s) {
    static int grid_w13 = 0, grid_down = 0;
    if (!grid_w13) grid_w13 = resident_grid(k_layer_w13<16>, LAYER_SMEM);
    if (!grid_down) grid_down = resident_grid(k_layer_down<16>, LAYER_SMEM);
    if (grid_w13 <= 0 || grid_down <= 0) return 2;
    if (raise_smem(k_layer_w13<16>, LAYER_SMEM) != cudaSuccess ||
        raise_smem(k_layer_down<16>, LAYER_SMEM) != cudaSuccess)
        return 2;
    if (hidden % 4 != 0 || m_valid != rows * top_k) return 2;
    k_layer_plan<16><<<1, PLAN_THREADS, 0, s>>>(L, ids, m_valid, max_blk, sorted, (int2*)blk,
                                               nblk, bad, clear_bad, live);
    k_layer_w13<16><<<grid_w13, NTHREADS, LAYER_SMEM, s>>>(
        L, (const __nv_bfloat16*)x_bf16, sorted, (const int2*)blk, nblk, (half*)inter, hidden,
        m_valid, top_k);
    k_layer_down<16><<<grid_down, NTHREADS, LAYER_SMEM, s>>>(
        L, (const half*)inter, sorted, (const int2*)blk, nblk, (const __nv_bfloat16*)w_bf16, slots,
        hidden, m_valid, top_k);
    const int h4 = hidden / 4;
    const int n = rows * h4;
    k_layer_reduce<<<(n + 255) / 256, 256, 0, s>>>(slots, live, acc, rows, top_k, h4);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

#define QMF_ARGS_W13                                                                          \
    x, sorted_ids, suh, inter, bq, svh, eids, n_rows, k, n_total, shard_w, n_shards,           \
        b_expert_stride, m_valid, top_k
#define QMF_ARGS_DOWN                                                                         \
    inter, sorted_ids, suh, out, bq, svh, eids, n_rows, w, k, n_total, b_expert_stride,         \
        m_valid, top_k

template <int BITS, int CB, int BM>
int launch_w13(const half* x, const int* sorted_ids, const half* suh, half* inter,
               const uint16_t* bq, const half* svh, const int* eids, const int* n_rows, int k,
               int n_total, int shard_w, int n_shards, int64_t b_expert_stride, int m_valid,
               int top_k, int m_blocks, cudaStream_t s) {
    using C = Cfg<BITS, BM>;
    const int smem = C::SMEM_I4 * 16;
    const cudaError_t attrs = raise_smem(k_stage_w13<BITS, CB, BM>, smem);
    if (attrs != cudaSuccess) return 2;
    dim3 grid(n_total / BN, m_blocks);
    k_stage_w13<BITS, CB, BM><<<grid, NTHREADS, smem, s>>>(QMF_ARGS_W13);
    /* A refused launch leaves `inter` stale and `acc` untouched; the caller must not report that as a
     * computed layer, so it is an error here, not a silent zero. */
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <int BITS, int CB, int BM>
int launch_down(const half* inter, const int* sorted_ids, const half* suh, float* out,
                const uint16_t* bq, const half* svh, const int* eids, const int* n_rows,
                const float* w, int k, int n_total, int64_t b_expert_stride, int m_valid,
                int top_k, int m_blocks, cudaStream_t s) {
    using C = Cfg<BITS, BM>;
    const int smem = C::SMEM_I4 * 16;
    const cudaError_t attrs = raise_smem(k_stage_down<BITS, CB, BM>, smem);
    if (attrs != cudaSuccess) return 2;
    dim3 grid(n_total / BN, m_blocks);
    k_stage_down<BITS, CB, BM><<<grid, NTHREADS, smem, s>>>(QMF_ARGS_DOWN);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

#undef QMF_ARGS_W13
#undef QMF_ARGS_DOWN

/* Every bitrate the pack can carry and every codebook id is instantiated: one branch per (bits,
 * cb), one tile shape per block_m. 3 bit needs no kernel of its own: `dq_dispatch<3, ...>` decodes
 * the 3-bit tile out of the same 24 words it is packed into (`TILE_U32 = 8*BITS` is the dq8 ring
 * exactly), the staged B tile is 6 int4 per row-tile instead of 4 or 8, and `Cfg` derives every
 * other size from BITS. A combination that is not here returns 1 and the caller stays on the
 * vendor path. */
#define QMF_TRY(B_, C_, BM_)                                                                   \
    if (bits == B_ && cb == C_ && block_m == BM_) return launch_w13<B_, C_, BM_>(              \
        x, sorted_ids, su, inth, bq, sv, ei, nr, k, n_total, shard_w, n_shards,                 \
        b_expert_stride, m_valid, top_k, m_blocks, s)
#define QMF_TRY_DOWN(B_, C_, BM_)                                                              \
    if (bits == B_ && cb == C_ && block_m == BM_) return launch_down<B_, C_, BM_>(             \
        ih, sorted_ids, su, oth, bq, sv, ei, nr, w, k, n_total, b_expert_stride,                \
        m_valid, top_k, m_blocks, s)

}  // namespace qmf

extern "C" int qmc_fused_w13(const void* x_h, const void* sorted, const void* suh, void* inter,
                             const void* trellis, const void* svh, const void* eids,
                             const void* n_rows, int k, int n_total, int shard_w, int n_shards,
                             int bits, int cb, int block_m, int64_t b_expert_stride, int m_valid,
                             int top_k, int m_blocks, cudaStream_t s) {
    const half* x = (const half*)x_h;
    const int* sorted_ids = (const int*)sorted;
    const half* su = (const half*)suh;
    const uint16_t* bq = (const uint16_t*)trellis;
    const half* sv = (const half*)svh;
    const int* ei = (const int*)eids;
    const int* nr = (const int*)n_rows;
    half* inth = (half*)inter;
    using namespace qmf;
    QMF_TRY(2, 0, 16);
    QMF_TRY(2, 1, 16);
    QMF_TRY(2, 2, 16);
    QMF_TRY(3, 0, 16);
    QMF_TRY(3, 1, 16);
    QMF_TRY(3, 2, 16);
    QMF_TRY(4, 0, 16);
    QMF_TRY(4, 1, 16);
    QMF_TRY(4, 2, 16);
    return 1;
}

extern "C" int qmc_fused_down(const void* inter, const void* sorted, const void* suh, void* out,
                              const void* trellis, const void* svh, const void* eids,
                              const void* n_rows, const void* w_f32, int k, int n_total, int bits,
                              int cb, int block_m, int64_t b_expert_stride, int m_valid,
                              int top_k, int m_blocks, cudaStream_t s) {
    const half* ih = (const half*)inter;
    const int* sorted_ids = (const int*)sorted;
    const half* su = (const half*)suh;
    const uint16_t* bq = (const uint16_t*)trellis;
    const half* sv = (const half*)svh;
    const int* ei = (const int*)eids;
    const int* nr = (const int*)n_rows;
    const float* w = (const float*)w_f32;
    float* oth = (float*)out;
    using namespace qmf;
    QMF_TRY_DOWN(2, 0, 16);
    QMF_TRY_DOWN(2, 1, 16);
    QMF_TRY_DOWN(2, 2, 16);
    QMF_TRY_DOWN(3, 0, 16);
    QMF_TRY_DOWN(3, 1, 16);
    QMF_TRY_DOWN(3, 2, 16);
    QMF_TRY_DOWN(4, 0, 16);
    QMF_TRY_DOWN(4, 1, 16);
    QMF_TRY_DOWN(4, 2, 16);
    return 1;
}

extern "C" int qmc_fused_layer(const qmc_layer_desc* L, const void* x_bf16, const int32_t* ids,
                               const void* w_bf16, int m_valid, int top_k, int hidden, int max_blk,
                               int32_t* sorted, void* blk, int32_t* nblk, int32_t* bad,
                               int clear_bad, int32_t* live, float* slots, void* inter, float* acc,
                               int rows, cudaStream_t s) {
    if (!L || !live || !slots || L->n_groups <= 0 || L->n_groups > QMC_MAX_GROUPS ||
        hidden % qmf::BN || L->inter % qmf::BN || (rows * hidden) % 4)
        return 1;
    for (int g = 0; g < L->n_groups; g++) {
        const qmc_layer_group& G = L->g[g];
        if (G.bits13 < 2 || G.bits13 > 4 || G.bits2 < 2 || G.bits2 > 4 || G.cb < 0 || G.cb > 2 ||
            G.n_experts > qmf::PLAN_EXPERTS)
            return 1;
    }
    return qmf::launch_layer(*L, x_bf16, ids, w_bf16, m_valid, top_k, hidden, max_blk, sorted,
                             blk, nblk, bad, clear_bad, live, slots, inter, acc, rows, s);
}
