#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "dsv4_pool_quant.cuh"
#include "cache/q_cache_kernels.cuh"
#include "graph.cuh"

/*
One block per (emitted entry w, job). Warps quantize four 32-value groups each of the entry's
D_c nope columns straight from the fp16 staging row into the packed pool row (words + group
scales); the remaining threads copy the rope columns. Entry count and destination follow the
compress kernel: nw = (pos0 + seq) / m - pos0 / m entries at rows ec0 + w, mapped through the
job's block-table row (epp entries per page).
*/

template <int bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void dsv4_pool_quant_scatter_kernel
(
    const half* __restrict__ stage,      // (B, nw_max, hd), job stride stage_stride
    uint32_t* __restrict__ pool_q,       // (rows, G * bits)
    half* __restrict__ pool_s,           // (rows, G)
    half* __restrict__ pool_r,           // (rows, D_r)
    const int* __restrict__ pool_bt,     // (B, bt_stride)
    const int bt_stride,
    const int epp,
    const int* __restrict__ pos_ptr,
    const int pos_base,
    const int m,
    const int seq,
    const int hd,
    const int D_c,
    const int stage_stride
)
{
    __shared__ uint32_t sh_pack[MAX_WARPS][32];
    const int job = blockIdx.y;
    const int w = blockIdx.x;
    const int pos0 = pos_ptr ? pos_ptr[job] : pos_base;
    const int ec0 = pos0 / m;
    const int nw = (pos0 + seq) / m - ec0;
    if (w >= nw) return;

    const half* row = stage + (size_t) job * stage_stride + (size_t) w * hd;
    const int e = ec0 + w;
    const size_t drow = (size_t) pool_bt[job * bt_stride + e / epp] * epp + e % epp;
    const int G = D_c / 32;
    const int D_r = hd - D_c;

    const int warp = threadIdx.x >> 5;
    const int g0 = warp * 4;
    if (g0 < G)
    {
        int active = min(4, G - g0);
        quant_block_x4<bits>
        (
            row + g0 * 32,
            pool_q + drow * (size_t) (G * bits) + g0 * bits,
            pool_s + drow * (size_t) G + g0,
            sh_pack[warp],
            active,
            0.0f
        );
    }
    for (int c = threadIdx.x; c < D_r; c += blockDim.x)
        pool_r[drow * (size_t) D_r + c] = row[D_c + c];
}

#define __(i) dsv4_pool_quant_scatter_kernel<i>
constexpr auto dsv4_pool_quant_scatter_kernel_instances = std::array
{
    __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __

void dsv4_pool_quant_scatter_gr
(
    const at::Tensor& stage,
    at::Tensor& pool_q,
    at::Tensor& pool_s,
    at::Tensor& pool_r,
    const at::Tensor& pool_bt,
    int position,
    const c10::optional<at::Tensor>& position_tensor,
    int m,
    int seq,
    int pool_epp,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(stage.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(stage, kHalf);
    TORCH_CHECK_DTYPE(pool_q, kInt);
    TORCH_CHECK_DTYPE(pool_s, kHalf);
    TORCH_CHECK_DTYPE(pool_r, kHalf);
    TORCH_CHECK_DTYPE(pool_bt, kInt);
    TORCH_CHECK(stage.is_contiguous() && pool_q.is_contiguous() && pool_s.is_contiguous() &&
                pool_r.is_contiguous() && pool_bt.is_contiguous(), "dsv4_pool_quant_scatter: non-contiguous");

    int hd = (int) stage.size(-1);
    int D_r = (int) pool_r.size(-1);
    int D_c = hd - D_r;
    int G = (int) pool_s.size(-1);
    TORCH_CHECK(D_c == 32 * G && D_c > 0, "dsv4_pool_quant_scatter: D_c must be a positive multiple of 32");
    int bits = (int) pool_q.size(-1) / G;
    TORCH_CHECK(2 <= bits && bits <= 8 && (int) pool_q.size(-1) == G * bits, "dsv4_pool_quant_scatter: bad packed width");
    TORCH_CHECK(pool_epp > 0 && m > 0, "dsv4_pool_quant_scatter: bad epp / m");

    int batch = stage.dim() == 3 ? (int) stage.size(0) : 1;
    int nw_max = (int) stage.size(-2);
    int stage_stride = batch > 1 ? (int) (nw_max * hd) : 0;
    TORCH_CHECK(pool_bt.dim() == 2 && (int) pool_bt.size(0) >= batch, "dsv4_pool_quant_scatter: block table rows < jobs");
    int bt_stride = batch > 1 ? (int) pool_bt.size(1) : 0;

    const int* pos_ptr = nullptr;
    if (position_tensor)
    {
        TORCH_CHECK_DTYPE(position_tensor.value(), kInt);
        pos_ptr = (const int*) position_tensor.value().data_ptr();
    }
    if (!batch || !seq) return;
    TORCH_CHECK(pool_bt.size(1) > 0, "dsv4_pool_quant_scatter: rows to scatter but no pages in block table");
    int nw = (position + seq) / m - position / m;
    int grid_w = pos_ptr ? seq / m + 1 : nw;
    TORCH_CHECK(grid_w <= nw_max, "dsv4_pool_quant_scatter: staging too small for this step");
    if (grid_w <= 0) return;

    int threads = CEIL_DIVIDE(G, 4) * 32;
    auto dsv4_pool_quant_scatter_fn = dsv4_pool_quant_scatter_kernel_instances[bits - 2];
    dsv4_pool_quant_scatter_fn<<<dim3(grid_w, batch), threads, 0, stream>>>
    (
        (const half*) stage.data_ptr(),
        (uint32_t*) pool_q.data_ptr(),
        (half*) pool_s.data_ptr(),
        (half*) pool_r.data_ptr(),
        (const int*) pool_bt.data_ptr(),
        bt_stride, pool_epp,
        pos_ptr, position, m, seq, hd, D_c, stage_stride
    );
    cuda_check(cudaPeekAtLastError());
}

void dsv4_pool_quant_scatter
(
    const at::Tensor& stage,
    at::Tensor pool_q,
    at::Tensor pool_s,
    at::Tensor pool_r,
    const at::Tensor& pool_bt,
    int position,
    const c10::optional<at::Tensor>& position_tensor,
    int m,
    int seq,
    int pool_epp
)
{
    dsv4_pool_quant_scatter_gr(stage, pool_q, pool_s, pool_r, pool_bt, position, position_tensor,
                               m, seq, pool_epp, nullptr);
}

__global__ void dsv4_pool_scatter_kernel(
    const int64_t* rows, const half* keys, const half* stage,
    half* pool_idx, half* pool_c, half* pool_r, int di, int dc, int dr)
{
    const int i = blockIdx.x;
    const size_t row = rows[i];
    for (int c = threadIdx.x; c < di; c += blockDim.x)
        pool_idx[row * di + c] = keys[(size_t) i * di + c];
    for (int c = threadIdx.x; c < dc; c += blockDim.x)
        pool_c[row * dc + c] = stage[(size_t) i * (dc + dr) + c];
    for (int c = threadIdx.x; c < dr; c += blockDim.x)
        pool_r[row * dr + c] = stage[(size_t) i * (dc + dr) + dc + c];
}

void dsv4_pool_scatter(
    const at::Tensor& rows, const at::Tensor& keys, const at::Tensor& stage,
    at::Tensor pool_idx, at::Tensor pool_c, at::Tensor pool_r)
{
    const at::cuda::OptionalCUDAGuard device_guard(stage.device());
    TORCH_CHECK_DTYPE(rows, kLong);
    TORCH_CHECK_DTYPE(keys, kHalf);
    TORCH_CHECK_DTYPE(stage, kHalf);
    TORCH_CHECK_DTYPE(pool_idx, kHalf);
    TORCH_CHECK_DTYPE(pool_c, kHalf);
    TORCH_CHECK_DTYPE(pool_r, kHalf);
    TORCH_CHECK(rows.dim() == 1 && keys.dim() == 2 && stage.dim() == 2 &&
                pool_idx.dim() == 2 && pool_c.dim() == 2 && pool_r.dim() == 2,
                "dsv4_pool_scatter: invalid dimensions");
    TORCH_CHECK(rows.is_contiguous() && keys.is_contiguous() && stage.is_contiguous() &&
                pool_idx.is_contiguous() && pool_c.is_contiguous() && pool_r.is_contiguous(),
                "dsv4_pool_scatter: non-contiguous tensor");
    TORCH_CHECK(rows.device() == stage.device() && keys.device() == stage.device() &&
                pool_idx.device() == stage.device() && pool_c.device() == stage.device() &&
                pool_r.device() == stage.device(), "dsv4_pool_scatter: device mismatch");
    TORCH_CHECK(keys.size(0) == rows.size(0) && stage.size(0) == rows.size(0) &&
                keys.size(1) == pool_idx.size(1) &&
                stage.size(1) == pool_c.size(1) + pool_r.size(1) &&
                pool_idx.size(0) == pool_c.size(0) && pool_c.size(0) == pool_r.size(0),
                "dsv4_pool_scatter: incompatible shapes");
    if (!rows.numel()) return;
    dsv4_pool_scatter_kernel<<<rows.numel(), 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        (const int64_t*) rows.data_ptr(), (const half*) keys.data_ptr(), (const half*) stage.data_ptr(),
        (half*) pool_idx.data_ptr(), (half*) pool_c.data_ptr(), (half*) pool_r.data_ptr(),
        (int) keys.size(1), (int) pool_c.size(1), (int) pool_r.size(1));
    cuda_check(cudaPeekAtLastError());
}
