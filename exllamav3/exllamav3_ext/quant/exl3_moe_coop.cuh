#pragma once

#include <ATen/Tensor.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Fused decode-shaped MoE path (bsz <= MAX_BSZN): two launches per layer run every (token, expert)
// slot through gate/up, activation and down, replacing the per-projection mgemm CUDA graphs of
// BC_BlockSparseMLP. See exl3_moe_coop_kernel.cuh / exl3_moe_coop.cu.

#define MOE_COOP_ACT_SILU 0
#define MOE_COOP_ACT_GELU 1
#define MOE_COOP_ACT_RELU2 2
#define MOE_COOP_ACT_SILU_OAI 3

#define MOE_COOP_THREADS 256
#ifndef MOE_COOP_MINB
#define MOE_COOP_MINB 3
#endif
#define MOE_COOP_WNT 2                  // adjacent 16-column tiles per warp (32-column groups per block)
#define MOE_COOP_COLS (MOE_COOP_WNT * 16)

struct MoeCoopParams
{
    // Inputs
    const half* x;              // (bsz, H), row stride x_stride
    int x_stride;
    const int64_t* sel;         // (bsz, sel_k) global expert indices
    const half* rw;             // (bsz, sel_k) routing weights
    int bsz;
    int topk;                   // slots per row: the sel_k routed picks, plus the folded shared expert (sh_fold)
    int sel_k;                  // routed picks per row (row stride of sel / rw)
    int H;                      // x width (the experts' input width before padding)
    int Hi;                     // gate/up in_features, padded (multiple of 128)
    int I;                      // intermediate width, padded (multiple of 128)
    int Ho;                     // down out_features, padded (multiple of 128)
    int H_out;                  // output width (<= Ho)
    int min_expert;             // -1: no expert-range filtering; else sel in [min, max) is local
    int max_expert;

    // Per-expert pointer tables (int64 device arrays indexed by local expert)
    const int64_t* g_trellis;   const int64_t* g_suh;   const int64_t* g_svh;
    const int64_t* u_trellis;   const int64_t* u_suh;   const int64_t* u_svh;
    const int64_t* d_trellis;   const int64_t* d_suh;   const int64_t* d_svh;
    const int64_t* g_bias;      // nullable
    const int64_t* u_bias;
    const int64_t* d_bias;
    int act;                    // MOE_COOP_ACT_*
    float act_limit;
    bool gated;                 // false: no gate projection, activation is relu(u) * u

    // Scratch (leading dim >= bsz * topk)
    half* had_g;                // (slots, Hi) rotated gate input (bsz > 1, written by the rot kernel)
    half* had_u;                // (slots, Hi) rotated up input
    bool a_global;              // true: the GEMV reads had_g / had_u; false (bsz 1): rotated in-block
    void* gu_g;                 // (slots, I) gate GEMV output, half or float (gu_f32)
    void* gu_u;                 // (slots, I)
    bool gu_f32;
    half* act_out;              // (slots, I) activation, rotated for the down projection
    float* d_out;               // (slots, Ho) down GEMV output before the output rotation
    int* ctr_a;                 // (slots_max, I / 128) completion counters of the gate/up stage
    int* ctr_b;                 // (MAX_BSZN, Ho / 128) completion counters of the down stage
    int ctr_a_len;
    int ctr_b_len;
    int* ready_a;               // (slots_max, I / 128) fused launch: 1 once the chunk's act_out is stored
    int* queue_head;            // fused launch: next work item (atomic), reset by the last CTA to exit
    int* exit_ctr;              // fused launch: CTAs that finished, the last one resets the scratch
    int nA;                     // fused launch: gate/up items (queue [0, nA)), then down items [nA, nA + nB)
    int nB;
    int resident;               // fused launch: persistent grid size (CTAs resident across the device)
    int fused;                  // 1: the stage bodies publish ready_a / wait on it (fused launch only)
    int ksplit_a;               // split-k blocks per chunk per stage (launcher; partial rows at slot + q * slots)
    int ksplit_b;
    int dbg;                    // diagnostics (EXL3_MOE_COOP_DBG): 1 skip A epilogue, 2 skip B reduction
    int* runs;                  // run table (bsz > 1): [n_runs, -, run_start[slots_max + 1], order[slots_max]]
    int slots_max;
    int rows_max;               // token rows the output scratch holds
    int n_local;                // entries in the pointer tables
    int paired;                 // 1: the routed experts' integer 2-4 bpw trellises are in the paired layout (pack layout
                                // tag, quant/trellis_pair.cuh); the folded shared expert is always plain
    int plain_from;             // with paired: local experts >= plain_from are plain (ColdSteal slots, the folded shared
                                // expert); n_local unless the tables carry steal slots
    int sh_gate_n;              // shared gate weight length (checked against H per call)
    // Mixed-K (nullable; null: every expert of the projection uses the launch's uniform rate).
    // Per local expert, half-bit rate k2 = 2 * bits + half (see bits_k.cuh); an entry of 0 also means the uniform
    // rate (uni_k2_*), so a uniform layer can carry a table and change single entries (stolen experts) in place.
    // Read by the *_mixed kernels only.
    const uint8_t* k2_g;
    const uint8_t* k2_u;
    const uint8_t* k2_d;
    uint8_t uni_k2_gu;          // launch's uniform gate/up and down rates as k2 (set by the launcher)
    uint8_t uni_k2_d;

    // Folded shared expert (sh_fold = 1): slot sel_k of every row runs the shared expert inside the routed launch,
    // weight 1 (times the sigmoid gate when sh_gate_w is set), through these pointers and rates. It shares Hi / I /
    // Ho, activation, codebook and bias presence with the routed experts (checked by exl3_moe_coop_set_fold). Its
    // local index is n_local (one past the routed tables)
    int sh_fold;
    int64_t sh_g_trellis, sh_g_suh, sh_g_svh;
    int64_t sh_u_trellis, sh_u_suh, sh_u_svh;
    int64_t sh_d_trellis, sh_d_suh, sh_d_svh;
    int64_t sh_g_bias, sh_u_bias, sh_d_bias;
    uint8_t sh_k2_gu;
    uint8_t sh_k2_d;

    // Output
    float* out;                 // (bsz, H_out), row stride out_stride
    int out_stride;

    // Shared expert as a separate launch's output (nullable): out += gate * sh_out, gate = sigmoid(x . sh_gate_w) or
    // 1. With sh_fold, sh_out is null and the gate multiplies the folded slot instead
    const float* sh_out;        // (bsz, H), row stride H
    const half* sh_gate_w;      // (H,) or null
};

// Launch with plain device pointers (called from BC_BlockSparseMLP). K: gate/up and down bit
// widths (gate and up always share one, the converter allocates them as one group); cb: 0 default,
// 1 mcg, 2 mul1 codebook, uniform across the three projections
void exl3_moe_coop_launch(const MoeCoopParams& p, float K_gu, float K_d, int cb, int device, cudaStream_t stream);

// Fold a shared expert into p's routed launch (slot sel_k of every row). ptrs: int64 CPU tensor of the 9 device
// pointers gate trellis/suh/svh, up trellis/suh/svh, down trellis/suh/svh; biases as device pointers or 0.
// Ksh_gu / Ksh_d: the shared expert's rates; K_gu / K_d / cb: the routed launch's. Returns false, leaving p
// unchanged, when the fold is not possible: differing rates outside mul1 or outside the mixed-K kernels' rate set,
// or bias presence that differs from the routed experts'
bool exl3_moe_coop_set_fold(MoeCoopParams& p, const at::Tensor& ptrs, int64_t g_bias, int64_t u_bias, int64_t d_bias,
                            float Ksh_gu, float Ksh_d, float K_gu, float K_d, int cb);

// Static part of the parameter block, validated once from the module's tensors (BC construction);
// K_gu / K_d / cb come back through the out-params. Gate tables/scratch are ignored when
// gated == false but must still be valid tensors (pass the up ones)
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
);

// Stage routed selections and weights with the shared expert appended to each row.
void exl3_moe_fold_stage(const at::Tensor& sel, const at::Tensor& rw,
                         at::Tensor& fold_sel, at::Tensor& fold_rw, int rows, int topk, int expert);

// Per-call part: input rows, routing, optional shared-expert output (bsz rows, width H), launch
void exl3_moe_coop_run
(
    MoeCoopParams p, float K_gu, float K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
);

// Tensor front end (prepare + run; the test entry point)
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
    const c10::optional<at::Tensor>& k2_g,     // per local expert half-bit rates (uint8): mixed-K layer, nullable
    const c10::optional<at::Tensor>& k2_u,
    const c10::optional<at::Tensor>& k2_d,
    const c10::optional<at::Tensor>& sh_fold_ptrs,  // int64 CPU (9,): fold this shared expert into the launch, nullable
    float sh_fold_K_gu,
    float sh_fold_K_d,
    bool paired = false,       // routed-expert trellises in the paired layout (quant/trellis_pair.cuh)
    int plain_from = -1        // with paired: local experts >= plain_from are plain (-1: all routed experts paired)
);

// Length of the int32 scratch for a module: completion counters, run table, and the fused-launch
// ready flags, queue head and exit counter (slots_max = MAX_BSZN * top_k)
inline int64_t exl3_moe_coop_ctr_len(int64_t slots_max, int64_t bsz_max, int64_t I, int64_t Ho)
{
    return slots_max * (I / 128) + bsz_max * (Ho / 128) + 2 + (slots_max + 1) + slots_max
        + slots_max * (I / 128) + 2;
}
