/* qmesh_moe_cpp: the GX cold-expert forward, moved out of Python.
 *
 * Why this exists
 * ----------------
 * worker_seam_server._forward_piece built the request's tensors in Python on every call: a numpy
 * view over the pinned request slot, a 21-41 MiB `.copy()` out of it, `.view(torch.bfloat16)
 * .to(device).half()`, an int64 astype of the ids, a bf16->float of the weights, then
 * `stream.synchronize()` and `part.float().cpu().numpy()` into the pinned response slot. None of
 * that touches math: it is 60 MiB of host staging traffic and a dozen allocator round trips per
 * call, and it runs on the GIL. This module does the same forward from the C proxy thread:
 *
 *   cudaMemcpyAsync pinned request slot -> persistent device buffers
 *   convert bf16 -> half / float on device
 *   per bitrate group: histogram pick counts -> (host: block_m, block offsets) -> scatter
 *   cuda-exl3 had_in / gemm13 / glu / gemm2(accumulate into one fp32 accumulator)
 *   cudaMemcpyAsync accumulator -> pinned response slot
 *
 * The response is fp32 because the shipped seam slot is (off_resp's stride, rows*HIDDEN*4, in
 * qmesh_expert_seam.cu and the client's Seam.collect in remote_seam_host.py).
 * SEAM_MOE_CPP_OUT=bf16|fp16 rounds the answer on device and copies half the bytes instead; the
 * slot and the client's collect have to change together for that to be a wire format, so it stays
 * a measurement switch until that lands.
 *
 * The kernels are the vendored cuda-exl3 ones, called through the torch dispatcher
 * (`cuda_exl3_C::exl3_moe_*`), so this file links libtorch but never imports Python. Everything
 * Python still does is load-time: it restacks each layer's experts per bitrate (qmesh/exl3_moe_cuda)
 * and hands the device pointers to qmc_add_layer once. No numpy view, no per-call allocation, no
 * GIL on the request path.
 *
 * Installation
 * ------------
 * qmc_compute has exactly the qes_server_fn signature, so it is installed with the shipped
 * qes_server_set_fn -- the seam ABI, its refusal codes and its protocol are untouched:
 *
 *   void *ctx; qmc_open(dev, HIDDEN, TOPK, ws_rows, &ctx);
 *   for layer in 0..NL-1: qmc_add_layer(ctx, layer, n_groups, groups);
 *   qes_server_set_fn(s, (qes_server_fn)qmc_compute, ctx);
 *
 * Return codes are the ones qes.py already understands: 0 answered, 4 refused for memory, 5
 * refused because the compute raised with the CUDA context still alive. Nothing here exits the
 * process except the one case worker_seam_server refuses to survive either, a CUDA context that
 * can no longer run a kernel (then it raises its hand with return code 5 and a SEAM_MOE_CPP_FAIL
 * line; the caller's fail-stop already exits 70 on a dead context via _cuda_ok).
 */
#ifndef QMESH_MOE_CPP_H
#define QMESH_MOE_CPP_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QMC_MAX_GROUPS 8

/* One bitrate group of one layer, as qmesh/exl3_moe_cuda.CudaExl3Moe already built it. Every
 * pointer is a device pointer that stays alive for the server's lifetime (the stacked tensors are
 * the layer's resident copy of the cold pack). */
typedef struct {
  int32_t n_experts;   /* experts in this group */
  int32_t n_cold;      /* length of remap: this layer's cold-expert count */
  int32_t words13;     /* trellis13.size(3): 16*bits (+1 word when the tile is fractional) */
  int32_t words2;      /* trellis2.size(3) */
  int32_t inter;       /* I, from trellis2.size(1)*16 */
  int32_t cb;          /* codebook id */
  const void *trellis13, *suh13, *svh13;
  const void *trellis2, *suh2, *svh2;
  const int32_t *remap; /* n_cold -> [0, n_experts) or -1 for "another group owns this pick" */
} qmc_group;

/* Context lifecycle. `ws_rows` is the piece ceiling (_pick_step's ws_rows: a request is served in
 * pieces of at most that many rows); 0 = one piece. */
int qmc_open(int device, int hidden, int topk, int ws_rows, void **out);
int qmc_add_layer(void *ctx, int layer, int n_groups, const qmc_group *groups);
int qmc_compute(void *ctx, int layer, int rows, const void *x_bf16, const int32_t *ids,
                const void *w_bf16, float *out_f32);
/* Timing of the last served call: wall ns inside qmc_compute, and the device span of the same
 * call in ns (first launch -> last launch, both events on the compute stream). */
int qmc_last_call_ns(void *ctx, int64_t *wall_ns, int64_t *dev_ns);
int qmc_stats(void *ctx, int64_t *calls, int64_t *refused);
/* Response format of the D2H: 0 = fp32 (the shipped slot), 1 = bf16, 2 = fp16. The slot's stride
 * follows it, so whoever reads the response has to agree with this value - off_resp in
 * qmesh_expert_seam.cu and Seam.collect in remote_seam_host.py are the other half of that
 * agreement. set_out_mode returns the old mode, or -1 for an unknown mode. */
int qmc_out_mode(void *ctx);
int qmc_set_out_mode(void *ctx, int mode);

/* Installing on the seam proxy, without a Python frame in the path. qmc_entry() returns the address
 * of qmc_compute for qes_server_set_fn(s, fn, ctx) - the same call qes.py's Server.set_fn makes,
 * with the C symbol instead of a ctypes callback. Layers this context does not hold (38/39: the
 * mixed-K pack has no cuda-exl3 stacks) are then handed to `fn`, a qes_server_fn whose `fnctx` is
 * whatever the caller passes: install() gives it the server's own trampoline, so those two layers
 * reach Server.compute exactly as they did before this module existed, and the other 38 never
 * leave C. Passing fn = NULL restores the refusal for an unregistered layer. */
void *qmc_entry(void);
int qmc_set_fallback(void *ctx, void *fn, void *fnctx);
/* Alignment knobs. block_m = 0 keeps the automatic tier (pick_block_m on the previous call's live
 * count); a positive value freezes the tier, which a captured graph needs. set_block_m returns the
 * old value or -1. dump_align copies the last piece's alignment of `layer`/`group` back to the
 * caller (qmc_k_align's output, for comparing against align_block); it needs SEAM_MOE_CPP_DUMP=1,
 * which is what allocates the host mirror, and returns -1 with qmc_last_error() set otherwise. */
int qmc_set_block_m(void *ctx, int block_m);
int qmc_dump_align(void *ctx, int layer, int group, int32_t *sorted, int32_t *eids, int *nblk,
                   int *total, int *block_m);
const char *qmc_last_error(void);
void qmc_close(void *ctx);

/* -- kernels (qmesh_moe_kernels.cu; plain CUDA so the compute file can stay torch-only) --------- */
void qmc_k_bf16_to_half(const uint16_t *src, void *dst_half, int n, cudaStream_t s);
void qmc_k_weights(const int32_t *ids, const uint16_t *w_bf16, float *dst, int n, cudaStream_t s);
void qmc_k_hist(const int32_t *ids, int npick, int n_cold, const int32_t *remap, int n_experts,
                int32_t *counts, int32_t *bad, cudaStream_t s);
void qmc_k_fill_i32(int32_t *dst, int32_t v, int n, cudaStream_t s);
void qmc_k_scatter(const int32_t *ids, int npick, int n_cold, const int32_t *remap,
                   int32_t *cursor, const int32_t *blk_off, int block_m, int32_t *sorted_ids,
                   int32_t *n_rows, cudaStream_t s);
/* counts -> block offsets, per-block expert id, block count and padded row count, on device: the
 * alignment align_block() computes host-side, without the readback that made a call uncapturable.
 * `nblk_pad` is the caller's bound on the block count; eids[nblk..nblk_pad) is filled with -1 (a
 * block that belongs to no expert, which the vendored kernel retires on its first check), so a
 * caller that must not sync can size the sorted buffer from the bound instead of the readback. */
void qmc_k_align(const int32_t *counts, int n_experts, int block_m, int nblk_pad, int32_t *blk_off,
                 int32_t *eids, int32_t *nblk, int32_t *live, cudaStream_t s);
/* mode 1 = bf16, 2 = fp16, round-to-nearest-even. */
void qmc_k_f32_to_half(const float *src, void *dst, int n, int mode, cudaStream_t s);
/* The deterministic combine's fold: dst[i] = dst[i] + (float)src_half[i], or dst[i] = (float)
 * src_half[i] when accumulate is 0. Same element order every call and every block_m. */
void qmc_k_h2f_add(const void *src_half, float *dst, int n, int accumulate, cudaStream_t s);

/* -- fused stages (qmesh_moe_fused.cu) ---------------------------------------------------------- *
 * One launch per projection instead of the vendor's two per projection: the input Hadamard (and, for
 * the down projection, the SwiGLU) is folded into the GEMM's A tile, and the output Hadamard and the
 * routing-weighted combine into its epilogue. Pointers are device pointers, the layout is exactly
 * what exl3_moe_had_in / exl3_moe_glu_had_in / exl3_moe_gemm take, and the k order is unchanged, so
 * the answer is the same bits. `block_m` must be 16, `bits` 2, 3 or 4 and `cb` 0..2; any other
 * combination returns 1 and the caller must use the vendor ops. Returns 0 when the launches were
 * issued. */
int qmc_fused_w13(const void *x_h, const void *sorted_ids, const void *suh13, void *inter,
                  const void *trellis13, const void *svh13, const void *eids, const void *n_rows,
                  int k, int n_total, int shard_w, int n_shards, int bits, int cb, int block_m,
                  int64_t b_expert_stride, int m_valid, int top_k, int m_blocks, cudaStream_t s);
int qmc_fused_down(const void *inter, const void *sorted_ids, const void *suh2, void *acc,
                   const void *trellis2, const void *svh2, const void *eids, const void *n_rows,
                   const void *w_f32, int k, int n_total, int bits, int cb, int block_m,
                   int64_t b_expert_stride, int m_valid, int top_k, int m_blocks, cudaStream_t s);

/* -- the layer path (SEAM_MOE_CPP_LAYER, qmesh_moe_fused.cu) ------------------------------------ *
 * Every bitrate group of one layer in three launches and no host round trip: a plan kernel builds
 * the whole layer's block_m=16 alignment on device (all groups concatenated in group, expert order),
 * zeroes the fp32 accumulator and flags a bad id; then one persistent w13 launch and one persistent
 * down launch walk every group's blocks, picking each block's (bits, cb) instantiation at run time.
 * The request's bf16 activation and routing weights are read in place (the same half/fp32 values the
 * per-group path's conversion kernels produce). Per routed row the math is the per-group fused
 * stage's own body. The descriptor is passed by value (kernel parameter space), built once per layer
 * at registration. */
typedef struct {
  const void *t13, *suh13, *svh13, *t2, *suh2, *svh2;
  const int32_t *remap;
  int64_t bs13, bs2; /* trellis elements per expert, w13 and down */
  int32_t n_experts, bits13, bits2, cb; /* bits = trellis words / 16, per projection */
} qmc_layer_group;

typedef struct {
  qmc_layer_group g[QMC_MAX_GROUPS];
  int32_t n_groups, n_cold, inter, pad_;
} qmc_layer_desc;

/* sorted: >= max_blk*16 int32; blk: >= max_blk int2; nblk, bad: one int32 each; live: m_valid int32;
 * slots: m_valid*hidden floats (each live pick's routing-weighted down output); inter: max_blk*16
 * rows of 2*inter halves; acc: rows*hidden floats, every element written by the final fixed-order
 * reduce (acc[t] = sum over k of slots[t*top_k + k] in pick order), so the answer does not depend on
 * block retirement order. m_valid must equal rows*top_k. max_blk must bound the plan's block count
 * (picks + blocks of partial fill); a plan past it sets *bad and answers zeros. Returns 0 when
 * issued, 1/2 for a refused launch. The plan supports n_experts <= 384 per group. */
int qmc_fused_layer(const qmc_layer_desc *L, const void *x_bf16, const int32_t *ids,
                    const void *w_bf16, int m_valid, int top_k, int hidden, int max_blk,
                    int32_t *sorted, void *blk, int32_t *nblk, int32_t *bad, int clear_bad,
                    int32_t *live, float *slots, void *inter, float *acc, int rows,
                    cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif /* QMESH_MOE_CPP_H */
