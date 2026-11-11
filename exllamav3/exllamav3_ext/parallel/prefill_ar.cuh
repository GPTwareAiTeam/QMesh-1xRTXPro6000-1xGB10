#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <torch/extension.h>

// Prefill all-reduce: peer-push sum of a rank's activations, compressed to int8 with one bf16 scale per 32
// contiguous elements (mode 0) or pushed exactly as fp32 (mode 1). Both modes compute the same expression on
// both ranks, so the result is bit-identical across ranks and reproducible against a fp32 reference for mode 1.

// Arena layout for `cap` elements and `fmt`: payload, scales (int8 only), arrival flags, consumption acks.
// Returns {payload_bytes, scale_bytes, flag_bytes, ack_bytes, total_bytes}.
std::vector<int64_t> p2p_prefill_ar_layout(int64_t cap, int64_t fmt);

// Resident launch width: grid used by the launcher is clamped to this occupancy times the device SM count.
int64_t p2p_prefill_ar_occupancy(int64_t fmt);

// Zeroed device arena of `bytes`, exported/imported with the existing p2p_dev_arena_handle/open helpers.
int64_t p2p_prefill_ar_alloc(int64_t bytes, int64_t device);

// out <- x_rank0 + x_rank1 over `n` elements of fp32 x, with the fixed operand order of `rank`. `mine` is this
// rank's arena, `peer` the peer's; `cap` (arena capacity in elements) and `fmt` must match on both ranks, and
// `seq` must increase by one per call on both ranks.
void p2p_prefill_ar_launch(torch::Tensor x, torch::Tensor out, int64_t peer, int64_t mine, int64_t n, int64_t cap,
                           int64_t fmt, int64_t grid, int64_t seq, int64_t rank);
