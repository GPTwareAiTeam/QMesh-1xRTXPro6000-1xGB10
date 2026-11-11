#pragma once

#include <ATen/Tensor.h>

// paired: the trellis is a routed expert of a paired-layout pack (quant/trellis_pair.cuh); 2 / 3 / 4 bpw read paired,
// other rates plain. Dense linears always pass false

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K,
    bool mcg,
    bool mul1,
    bool paired = false
);

void reconstruct_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K,
    bool mcg,
    bool mul1,
    int64_t n_offset,
    bool paired = false
);

void reconstruct_had_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    at::Tensor suh,
    at::Tensor svh,
    float K,
    bool mcg,
    bool mul1,
    int64_t n_offset,
    bool paired = false
);

void reconstruct_had_batch
(
    at::Tensor unpacked,
    at::Tensor packed_ptrs,
    at::Tensor suh_ptrs,
    at::Tensor svh_ptrs,
    float K,
    bool mcg,
    bool mul1,
    bool paired = false
);

void reconstruct_batch
(
    at::Tensor unpacked,
    at::Tensor packed_ptrs,
    float K,
    bool mcg,
    bool mul1,
    bool paired = false
);
