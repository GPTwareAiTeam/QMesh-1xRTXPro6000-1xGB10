#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Dense decode GEMV (exl3_dgemv_kernel.cuh): ordinary-launch split-K GEMV for EXL3 trellis weights at
// 1..48 rows, mul1 codebook, K 4/5/6/8. EXL3_DGEMV=1 (default) = measured envelope, 2 = every
// hard-eligible call, 0 = off. The try_launch functions take the caller's EXL3_GEMM_ARGS / EXL3_MGEMM_ARGS
// arrays (same parameter positions, so graph recording in exl3_gemm_gr / exl3_mgemm_gr is
// unchanged), replace the `locks` slot with the GEMV workspace, and return the launched kernel or
// nullptr (nothing launched: the caller's own path runs).

bool exl3_dgemv_enabled();

// True when kernel instances exist for (K, cb, c_fp32) at size_m rows. Tests use this to hard-fail
// on a missing instantiation instead of silently measuring the fallback
bool exl3_dgemv_has(int K, int cb, int size_m, bool c_fp32);

// Mode 0 off / 1 measured envelope / 2 every hard-eligible call; -1 = from EXL3_DGEMV (default 1)
void exl3_dgemv_set_enabled(int enabled);
// k splits a matrix of this shape runs with
int exl3_dgemv_splits_for(int size_m, int size_k, int size_n);

void* exl3_dgemv_try_launch
(
    void** kernel_args,         // EXL3_GEMM_ARGS order; [6] (locks) is replaced
    int size_m, int size_k, int size_n,
    int K, bool half_k, int cb, bool c_fp32, bool has_su_sv,
    int device, cudaStream_t stream
);

void* exl3_dmgemv_try_launch
(
    void** kernel_args,         // EXL3_MGEMM_ARGS order; [6] (locks) is replaced
    int size_m, int size_k, int size_n,
    int K, bool half_k, int cb, bool c_fp32,
    int bszm_in, int bszm_out, bool weighted, int min_index, int num_tokens, bool sliced,
    int device, cudaStream_t stream
);
