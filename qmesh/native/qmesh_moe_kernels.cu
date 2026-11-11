/* qmesh_moe_kernels: the device steps the Python path used to do with torch ops.
 *
 * align_block() in qmesh/exl3_moe_cuda.py is vLLM's moe_align_block_size in torch: nonzero, sort,
 * bincount, cumsum, repeat_interleave and a scatter. Two of those read a value back to the host
 * (`(loc >= 0).sum()` and `pos.max()`), so the Python path stalls the stream at least twice per
 * group per call before a single MoE kernel runs. Here the histogram is one kernel and the
 * scatter another, and the only host round trip is the per-group count vector, which is needed
 * anyway to choose block_m.
 *
 * The scatter's within-expert order is `atomicAdd` order, not the stable argsort order: the same
 * picks land in the same expert's block either way and each block's rows are independent, so the
 * only difference is the order of the fp32 atomics in the down-projection epilogue (<= 1 ulp per
 * row, measured against the Python path in the ticket).
 */
#include "qmesh_moe_cpp.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

/* `src` is the request's bf16 x where the seam put it: the pinned slab's device alias on the served
 * path (the NIC wrote it; no H2D staging copy), or a device buffer in the benches. Loaded with
 * ld.global.cv so no line cached from the slot's previous request can be read. */
__global__ void k_bf16_to_half(const uint16_t *__restrict__ src, __half *__restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const unsigned short v = __ldcv(reinterpret_cast<const unsigned short *>(src) + i);
    dst[i] = __float2half(__bfloat162float(__ushort_as_bfloat16(v)));
  }
}

/* Routing weights, bit for bit as _forward_piece built them: bf16 -> float -> half (the `.half()`
 * on the client's bf16 weight) -> float (CudaExl3Moe.forward's `w.float()`), and zero for a pick
 * whose id is -1. bf16 -> float alone would be *more* accurate than the Python path and would
 * show up as an error in the comparison, so the extra rounding is deliberate. */
__global__ void k_weights(const int32_t *__restrict__ ids, const uint16_t *__restrict__ w,
                          float *__restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (ids[i] < 0) {
    dst[i] = 0.f;
    return;
  }
  const float v = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(w + i));
  dst[i] = __half2float(__float2half(v));
}

/* One pass over the request's picks. `bad` is set for an id outside [-1, n_cold): the client only
 * ever sends a cold-local index or -1, and a value outside that range would index the remap table
 * out of bounds - it is a client bug, refused, never a kernel to run on garbage. */
__global__ void k_hist(const int32_t *__restrict__ ids, int npick, int n_cold,
                       const int32_t *__restrict__ remap, int n_experts,
                       int32_t *__restrict__ counts, int32_t *__restrict__ bad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npick) return;
  const int32_t v = ids[i];
  if (v < -1 || v >= n_cold) {
    atomicExch(bad, 1);
    return;
  }
  if (v < 0) return;
  const int32_t e = remap[v];
  if (e >= n_experts) atomicExch(bad, 1);
  else if (e >= 0) atomicAdd(counts + e, 1);
}

__global__ void k_fill_i32(int32_t *__restrict__ dst, int32_t v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = v;
}

/* Padded row position for a pick: blk_off[e]*block_m + rank_within_expert[e]. Everything the
 * kernels need beyond that - which expert owns a block, the live-row count - is written here or
 * prefilled by the caller. An id outside [0, n_cold) is skipped: it reads remap[v], and the caller
 * only learns that the request was bad from the same flag k_hist sets, i.e. after this kernel has
 * already run. */
__global__ void k_scatter(const int32_t *__restrict__ ids, int npick, int n_cold,
                          const int32_t *__restrict__ remap, int32_t *__restrict__ cursor,
                          const int32_t *__restrict__ blk_off, int block_m,
                          int32_t *__restrict__ sorted_ids, int32_t *__restrict__ n_rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npick) return;
  const int32_t v = ids[i];
  if (v < 0 || v >= n_cold) return;
  const int32_t e = remap[v];
  if (e < 0) return;
  const int32_t r = atomicAdd(cursor + e, 1);
  const int32_t pos = blk_off[e] * block_m + r;
  sorted_ids[pos] = i;
  atomicMax(n_rows, pos + 1);
}

/* align_block()'s host loop, on device: one thread walks the expert counts in id order, so the
 * block offsets, the block -> expert map and the padded row count are identical to the Python
 * path's. n_experts <= 384 and this runs once per group per call, next to a 74 ms gemm.
 *
 * `nblk_pad` is the caller's bound on the block count (min(n_experts, m_valid)); blocks past the
 * live ones are marked -1, the value align_block uses for "belongs to no expert". The consumer
 * retires them on its first check, so a caller that cannot afford the readback still gets a
 * completely defined eids array - stale entries from an earlier block_m would select a real
 * expert's weights and compute a block of rows nobody reads. */
__global__ void k_align(const int32_t *__restrict__ counts, int n_experts, int block_m,
                        int nblk_pad, int32_t *__restrict__ blk_off, int32_t *__restrict__ eids,
                        int32_t *__restrict__ nblk, int32_t *__restrict__ live) {
  if (threadIdx.x || blockIdx.x) return;
  int off = 0, nb = 0, lv = 0;
  for (int e = 0; e < n_experts; e++) {
    const int cnt = counts[e];
    const int blocks = cnt > 0 ? (cnt + block_m - 1) / block_m : 0;
    blk_off[e] = off;
    for (int i = 0; i < blocks; i++) eids[nb + i] = e;
    nb += blocks;
    off += blocks;
    lv += cnt;
  }
  for (int i = nb; i < nblk_pad; i++) eids[i] = -1;
  *nblk = nb;
  *live = lv;
}

constexpr int kThreads = 256;

inline int grid_for(int n) { return (n + kThreads - 1) / kThreads; }

/* Response rounding, for a seam slot that carries halves instead of floats. Round-to-nearest-even,
 * the conversion torch's .bfloat16()/.half() does, so the number a client sees is the one the
 * Python path's own cast would have produced. `dst` is the pinned response slot itself (its device
 * alias): the rounding is the copy, so there is no device staging buffer and no D2H. */
__global__ void k_f32_to_half(const float *src, void *dst, int n, int mode) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = src[i];
  if (mode == 1)
    ((__nv_bfloat16 *)dst)[i] = __float2bfloat16_rn(v);
  else
    ((__half *)dst)[i] = __float2half_rn(v);
}

/* The deterministic combine's fp32 fold: the Python path's `total = out.float() if total is None
 * else total + out.float()`, with out the half the vendor's exl3_moe_combine returned. One thread
 * per output element, so the sum is the same one on every run and on every block_m - which is the
 * whole point of the mode (see run_piece). `accumulate` false assigns instead of adding, so a slab
 * whose only contributing group is group 0 holds exactly the combined values, -0.0 included. */
__global__ void k_h2f_add(const __half *src, float *dst, int n, int accumulate) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = __half2float(src[i]);
  dst[i] = accumulate ? dst[i] + v : v;
}

}  // namespace

void qmc_k_bf16_to_half(const uint16_t *src, void *dst_half, int n, cudaStream_t s) {
  if (n <= 0) return;
  k_bf16_to_half<<<grid_for(n), kThreads, 0, s>>>(src, (__half *)dst_half, n);
}

void qmc_k_weights(const int32_t *ids, const uint16_t *w_bf16, float *dst, int n, cudaStream_t s) {
  if (n <= 0) return;
  k_weights<<<grid_for(n), kThreads, 0, s>>>(ids, w_bf16, dst, n);
}

void qmc_k_hist(const int32_t *ids, int npick, int n_cold, const int32_t *remap, int n_experts,
                int32_t *counts, int32_t *bad, cudaStream_t s) {
  if (npick <= 0) return;
  k_hist<<<grid_for(npick), kThreads, 0, s>>>(ids, npick, n_cold, remap, n_experts, counts, bad);
}

void qmc_k_fill_i32(int32_t *dst, int32_t v, int n, cudaStream_t s) {
  if (n <= 0) return;
  k_fill_i32<<<grid_for(n), kThreads, 0, s>>>(dst, v, n);
}

void qmc_k_scatter(const int32_t *ids, int npick, int n_cold, const int32_t *remap,
                   int32_t *cursor, const int32_t *blk_off, int block_m, int32_t *sorted_ids,
                   int32_t *n_rows, cudaStream_t s) {
  if (npick <= 0) return;
  k_scatter<<<grid_for(npick), kThreads, 0, s>>>(ids, npick, n_cold, remap, cursor, blk_off,
                                                block_m, sorted_ids, n_rows);
}

void qmc_k_align(const int32_t *counts, int n_experts, int block_m, int nblk_pad, int32_t *blk_off,
                 int32_t *eids, int32_t *nblk, int32_t *live, cudaStream_t s) {
  if (n_experts <= 0) {
    /* No group has an expert, so no block is live and the padded tail covers all of eids. */
    qmc_k_fill_i32(eids, -1, nblk_pad, s);
    qmc_k_fill_i32(nblk, 0, 1, s);
    qmc_k_fill_i32(live, 0, 1, s);
    return;
  }
  k_align<<<1, 1, 0, s>>>(counts, n_experts, block_m, nblk_pad, blk_off, eids, nblk, live);
}

void qmc_k_f32_to_half(const float *src, void *dst, int n, int mode, cudaStream_t s) {
  if (n <= 0) return;
  k_f32_to_half<<<grid_for(n), kThreads, 0, s>>>(src, dst, n, mode);
}

/* The fused stage kernels live in their own file so they can be compile-checked alone, but
 * build_moe.sh compiles exactly one .cu per architecture: include it here rather than growing the
 * build script (which the seam server owns). */
#include "qmesh_moe_fused.cu"

void qmc_k_h2f_add(const void *src_half, float *dst, int n, int accumulate, cudaStream_t s) {
  if (n <= 0) return;
  k_h2f_add<<<grid_for(n), kThreads, 0, s>>>((const __half *)src_half, dst, n, accumulate);
}
