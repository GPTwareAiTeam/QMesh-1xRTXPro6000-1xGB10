#pragma once

// Dense decode GEMV instances (exl3_dgemv_kernel.cuh), one translation unit per bitrate, mul1
// codebook only (every dense tensor the runtime ships is mul1). Bitrates 4, 5, 6 and 8: the widths
// the shipped and d5 packs use for dense matrices and the head. MT = row tiles of 16 (1..3 = up to
// 48 rows); multi selects the multi-matrix (mgemm-signature) kernel. nullptr = no instance.

void* exl3_dgemv_sel_k4(int MT, bool c_fp32, bool multi);
void* exl3_dgemv_sel_k5(int MT, bool c_fp32, bool multi);
void* exl3_dgemv_sel_k6(int MT, bool c_fp32, bool multi);
void* exl3_dgemv_sel_k8(int MT, bool c_fp32, bool multi);

#define EXL3_DGEMV_SEL_MT_(K_, MT_) \
    case MT_: \
        if (multi) return c_fp32 ? (void*) exl3_dmgemv_kernel<K_, 2, MT_, true> : (void*) exl3_dmgemv_kernel<K_, 2, MT_, false>; \
        return c_fp32 ? (void*) exl3_dgemv_kernel<K_, 2, MT_, true> : (void*) exl3_dgemv_kernel<K_, 2, MT_, false>;

#define EXL3_DGEMV_SEL_DEF(K_) \
    void* exl3_dgemv_sel_k##K_(int MT, bool c_fp32, bool multi) \
    { \
        switch (MT) \
        { \
            EXL3_DGEMV_SEL_MT_(K_, 1) \
            EXL3_DGEMV_SEL_MT_(K_, 2) \
            EXL3_DGEMV_SEL_MT_(K_, 3) \
        } \
        return nullptr; \
    }
