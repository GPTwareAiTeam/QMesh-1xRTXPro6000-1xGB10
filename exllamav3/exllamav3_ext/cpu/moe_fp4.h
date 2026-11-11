#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <vector>

// CPU MoE experts in MXFP4 (the DeepSeek-V4.1 checkpoint's own routed-expert format: E2M1 nibbles, one E8M0 scale per
// 32 inputs per output row). See moe_fp4.cpp for the layout and the math.
//
// A layer is registered once from the checkpoint tensors (copied and repacked into the kernel's layout, owned by the
// layer), then run per call with the routing results. Experts are indexed by the caller's local id (0 .. n-1).

// weight: [out, in/2] uint8 (low nibble = even input), scale: [out, in/32] uint8 (E8M0, 2^(s-127)).
// Projections per expert: gate (w1) [I, H], up (w3) [I, H], down (w2) [H, I]. Returns a handle.
int64_t exl3_moe_fp4_make_layer
(
    const std::vector<at::Tensor>& gate_w, const std::vector<at::Tensor>& gate_s,
    const std::vector<at::Tensor>& up_w, const std::vector<at::Tensor>& up_s,
    const std::vector<at::Tensor>& down_w, const std::vector<at::Tensor>& down_s,
    double act_limit
);

void exl3_moe_fp4_free_layer(int64_t handle);

// Shotgun variants (moe_fp4_variants.inc, notes/CPUKernelShotgun.md): same contract as exl3_moe_fp4_forward, cfg =
// [alloc, phases, spin, streams, pf, pfhint, amx]. Benchmark only; the worker uses exl3_moe_fp4_forward_raw.
void exl3_moe_fp4_forward_variant
(
    int64_t handle, const at::Tensor& x, const at::Tensor& selected, const at::Tensor& weights, at::Tensor& out,
    int64_t num_threads, const std::vector<int64_t>& cfg
);

// x [m, H] fp16, selected [m, topk] int64 (local expert ids, < 0 = skip), weights [m, topk] fp16, out [m, H] fp32
// (overwritten with the routing-weighted sum of SiLU-gated experts, act clamped at act_limit as the GPU coop kernel).
void exl3_moe_fp4_forward
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
);

// Same computation, raw pointers and int32 ids (the CPU MoE worker's slot layout, cpu/moe_handoff.cu)
void exl3_moe_fp4_forward_raw
(
    int64_t handle, const at::Half* x, const int32_t* selected, const at::Half* weights,
    float* out, int rows, int topk, int threads
);

// Bytes of one expert in the kernel layout (all three projections)
int64_t exl3_moe_fp4_expert_bytes(int64_t handle);
