#pragma once

#include <ATen/Tensor.h>

class Graph;

void dsa_topk_prepare(const at::Tensor& scores_static, int k_pad);

void dsa_topk_gr
(
    const at::Tensor& scores,
    at::Tensor& indices,
    int k,
    Graph* graph,
    const c10::optional<at::Tensor>& t_ptr = {},
    int t_seq = 0
);

void dsa_topk
(
    const at::Tensor& scores,
    at::Tensor indices,
    int64_t k,
    const c10::optional<at::Tensor>& t_ptr,
    int64_t t_seq
);

// Candidate-block entry compaction: ascending entry ids of the kept blocks (see dsa_topk.cu)
void dsa_compact_ent
(
    const at::Tensor& keep,              // (nb,) bool, device
    at::Tensor out,                      // (cap,) int32, device, -1 past the count
    int64_t bs,
    int64_t ec
);

// Multi-row compaction (MTP verify slab): row s keeps its own blocks and its own entry bound
// min((pos_base + s) / cr, ec_max); same ascending order and -1 padding as dsa_compact_ent
void dsa_compact_ent_rows
(
    const at::Tensor& keep,              // (rows, nb) bool, device, contiguous
    at::Tensor out,                      // (rows, cap) int32, device, -1 past each row's count
    int64_t bs,
    int64_t ec_max,
    int64_t pos_base,                    // pos0 + 1 + r0
    int64_t cr                           // compress_rate
);

// Same, plus the pool row of each kept entry: out_rows[t] = pages[out[t] / epp] * epp + out[t] % epp
void dsa_compact_ent_rows_map
(
    const at::Tensor& keep,              // (rows, nb) bool, device, contiguous
    at::Tensor out,                      // (rows, cap) int32, device, -1 past each row's count
    at::Tensor out_rows,                 // (rows, cap) int32, device, -1 padded, pool rows
    const at::Tensor& pages,             // (np,) int32, device, block table
    int64_t epp,                         // pool entries per page
    int64_t bs,
    int64_t ec_max,
    int64_t pos_base,
    int64_t cr
);

void dsa_topk_tile
(
    const at::Tensor& scores,
    at::Tensor ws_idx,
    at::Tensor ws_scr,
    at::Tensor ws_cnt,
    int64_t slot,
    int64_t k,
    int64_t idx_offset
);
void dsa_topk_merge_tiles
(
    const at::Tensor& ws_idx,
    const at::Tensor& ws_scr,
    const at::Tensor& ws_cnt,
    at::Tensor out_idx,
    const c10::optional<at::Tensor>& out_scr,
    const c10::optional<at::Tensor>& out_cnt,
    int64_t k
);
void dsa_seq_state_gr
(
    const at::Tensor& cache_seqlens,
    at::Tensor& arr,
    int bsz,
    int q_len,
    Graph* graph
);
