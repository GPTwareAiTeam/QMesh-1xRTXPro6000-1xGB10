#include <cuda_fp16.h>
#include "exl3_dgemv.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "exl3_dgemv_kernel.cuh"
#include "exl3_devctx.cuh"
#include "comp_units/exl3_dgemv_instances.cuh"
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <utility>

/*
Dense decode GEMV dispatch (kernel: exl3_dgemv_kernel.cuh).

EXL3_DGEMV=1 (default) routes eligible exl3_gemm / exl3_mgemm calls here ahead of every other path; 0
leaves the vendor dispatch untouched. Eligible: mul1 codebook, integer K in {4, 5, 6, 8}, 1..48 rows,
k and n multiples of 128, input/output scales present; for exl3_mgemm additionally unweighted,
unfiltered, single-token and unsliced (the dense fans: x-side projection fan, q_b + indexer q_b fan,
wo_a's grouped slices, shared w1+w3).

Grid: one block per (128-column block, k split[, matrix]). The split count is a pure function of the
matrix's (rows, k, n) (exl3_dgemv_ns::splits_for): ~EXL3_DGEMV_TARGET_BLOCKS blocks per matrix, so a
matrix produces the same bits alone or inside a fan, and the partials are summed in split order (no
float atomics): deterministic for a given shape.

Workspace: fixed per (device, stream), allocated on first use and never reallocated (the pointer is
baked into captured graphs). Counters at its head self-reset; partials follow, one fixed slab per
multi-matrix slot.
*/

#define DGEMV_WS_BYTES (EXL3_DGEMV_CNT_INTS * 4 + EXL3_DGEMV_MAX_SLOTS * EXL3_DGEMV_SLOT_FLOATS * 4)

// 0 off, 1 measured envelope (default), 2 every hard-eligible call (testing). exl3_dgemv_set_enabled
// overrides the env (-1 restores it)
static int dgemv_mode_override = -1;

static int dgemv_mode()
{
    if (dgemv_mode_override >= 0) return dgemv_mode_override;
    static const int env = [] { const char* e = std::getenv("EXL3_DGEMV"); return e ? atoi(e) : 1; }();
    return env;
}

bool exl3_dgemv_enabled() { return dgemv_mode() != 0; }

void exl3_dgemv_set_enabled(int mode) { dgemv_mode_override = mode; }

static void* select_kernel(int K, int MT, bool c_fp32, bool multi)
{
    switch (K)
    {
        case 4: return exl3_dgemv_sel_k4(MT, c_fp32, multi);
        case 5: return exl3_dgemv_sel_k5(MT, c_fp32, multi);
        case 6: return exl3_dgemv_sel_k6(MT, c_fp32, multi);
        case 8: return exl3_dgemv_sel_k8(MT, c_fp32, multi);
    }
    return nullptr;
}

static int smem_bytes(int K, int MT)
{
    return 2 * EXL3_DGEMV_WARPS * 64 * K * 4 + 2 * 16 * MT * EXL3_DGEMV_A_STRIDE * 2 + 16;
}

bool exl3_dgemv_has(int K, int cb, int size_m, bool c_fp32)
{
    if (cb != 2 || size_m < 1 || size_m > EXL3_DGEMV_MAX_M) return false;
    int MT = CEIL_DIVIDE(size_m, 16);
    return select_kernel(K, MT, c_fp32, false) && select_kernel(K, MT, c_fp32, true);
}

// One workspace per (device, stream): two streams may run dense GEMMs concurrently (the P7 compute stream
// and the head stream), and the self-resetting counters and partials must not be shared between them.
// The pointer is baked into captured graphs, so a buffer lives for the process. First use may be inside
// a capture (graphs capture on their own stream): the allocation then runs in relaxed capture mode and
// the counters are zeroed on a private non-blocking stream, neither of which is captured
static int* get_ws(int device, cudaStream_t stream)
{
    static std::mutex mtx;
    static std::map<std::pair<int, cudaStream_t>, int*> ws;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_pair(device, stream);
    auto it = ws.find(key);
    if (it != ws.end()) return it->second;

    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    cuda_check(cudaThreadExchangeStreamCaptureMode(&mode));
    int* p;
    cuda_check(cudaMalloc(&p, DGEMV_WS_BYTES));
    cudaStream_t init;
    cuda_check(cudaStreamCreateWithFlags(&init, cudaStreamNonBlocking));
    cuda_check(cudaMemsetAsync(p, 0, EXL3_DGEMV_CNT_INTS * sizeof(int), init));
    cuda_check(cudaStreamSynchronize(init));
    cuda_check(cudaStreamDestroy(init));
    cuda_check(cudaThreadExchangeStreamCaptureMode(&mode));
    ws[key] = p;
    return p;
}

int exl3_dgemv_splits_for(int size_m, int size_k, int size_n)
{
    return exl3_dgemv_ns::splits_for(size_m, size_k, size_n);
}

static bool launch
(
    void* kernel, int K, int MT, int grid_x, int grid_z, void** kernel_args, int device, cudaStream_t stream
)
{
    int smem = smem_bytes(K, MT);
    static std::mutex mtx;
    static std::set<std::pair<int, void*>> attr_set;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto key = std::make_pair(device, kernel);
        if (!attr_set.count(key))
        {
            if (smem > 48 * 1024)
                cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            attr_set.insert(key);
        }
    }
    cuda_check(cudaLaunchKernel(kernel, dim3(grid_x, 1, grid_z), dim3(EXL3_DGEMV_THREADS), kernel_args, smem, stream));
    return true;
}

// Hard constraints, then the envelope where the kernel measured faster than the vendor path on P7
// (notes/DenseGemv.md, graph-replay medians, real tensors, K 4/5/6/8):
//  - the head (k * n = 662M) never: DRAM-bound, the vendor kernel is at the bandwidth floor
//    (8 bpw 441 vs 502 us at 4 rows) and this kernel's grid has no k split left at that width
//  - 1 row: k * n <= 8M (wq_a, wkv, wo_a, compressor, indexer 1.1-1.8x; shared w1/w2 tie; wo_b, wq_b lose)
//  - 2..16 rows (one MMA row tile): every shape up to k * n = 48M wins (wq_b 32768-wide 1.0-1.1x,
//    wo_b 1.2-1.3x, the rest 1.5-3.1x)
//  - 17..32 rows: k * n <= 12M (wq_a, wkv, wo_a, compressor, indexer, shared w1/w3/w2 1.2-1.7x;
//    wo_b 8192 x 5120 and wq_b tie or lose)
//  - 33..48 rows: k * n <= 6.6M (wkv/compressor 1.3x, indexer 1.1x, wo_a 1.04x, wq_a 1.04-1.3x;
//    anything wider loses)
// EXL3_DGEMV=2 drops the envelope (hard constraints only) for testing
static bool eligible(int size_m, int size_k, int size_n, int K, bool half_k, int cb)
{
    if (half_k || cb != 2) return false;
    if (size_m < 1 || size_m > EXL3_DGEMV_MAX_M) return false;
    if (size_k <= 0 || size_k % 128 || size_n <= 0 || size_n % 128) return false;
    if (!(K == 4 || K == 5 || K == 6 || K == 8)) return false;
    if (dgemv_mode() == 2) return true;
    int64_t kn = (int64_t) size_k * size_n;
    if (size_m == 1) return kn <= 8 * 1024 * 1024;
    if (size_m <= 16) return kn <= 48 * 1024 * 1024;
    if (size_m <= 32) return kn <= 12 * 1024 * 1024;
    return kn <= 6600 * 1024;
}

void* exl3_dgemv_try_launch
(
    void** kernel_args,
    int size_m, int size_k, int size_n,
    int K, bool half_k, int cb, bool c_fp32, bool has_su_sv,
    int device, cudaStream_t stream
)
{
    if (!has_su_sv || !eligible(size_m, size_k, size_n, K, half_k, cb)) return nullptr;
    int MT = CEIL_DIVIDE(size_m, 16);
    void* kernel = select_kernel(K, MT, c_fp32, false);
    TORCH_CHECK(kernel, "exl3_dgemv: no instance for K=", K, " rows=", size_m);
    int splits = exl3_dgemv_ns::splits_for(size_m, size_k, size_n);
    TORCH_CHECK(size_n / 128 <= EXL3_DGEMV_CNT_INTS, "exl3_dgemv: too many column blocks");

    int* ws = get_ws(device, stream);
    void* args[10];
    for (int i = 0; i < 10; ++i) args[i] = kernel_args[i];
    args[6] = (void*) &ws;
    launch(kernel, K, MT, size_n / 128 * splits, 1, args, device, stream);
    return kernel;
}

void* exl3_dmgemv_try_launch
(
    void** kernel_args,
    int size_m, int size_k, int size_n,
    int K, bool half_k, int cb, bool c_fp32,
    int bszm_in, int bszm_out, bool weighted, int min_index, int num_tokens, bool sliced,
    int device, cudaStream_t stream
)
{
    if (weighted || min_index >= 0 || num_tokens != 1 || sliced) return nullptr;
    int slots = MAX(bszm_in, bszm_out);
    int ncb = size_n / 128;
    if (slots < 1 || slots > EXL3_DGEMV_MAX_SLOTS || ncb > EXL3_DGEMV_CNT_INTS / EXL3_DGEMV_MAX_SLOTS) return nullptr;
    if (!eligible(size_m, size_k, size_n, K, half_k, cb)) return nullptr;
    // Fan envelope (measured, graph replay): the launch's blocks share the device, so above one MMA row
    // tile it is the launch's weight bound (slots x k x widest n) that decides. 17..32 rows: <= 14M
    // (q_a + wkv, 2 x 5120 x 1280 = 13.1M, wins 1.7x; shared w1+w3 23.6M and wo_a x8 33.6M lose);
    // 33..48 rows: never (every measured fan ties or loses)
    if (dgemv_mode() != 2 && size_m > 16)
    {
        int64_t total = (int64_t) slots * size_k * size_n;
        if (size_m > 32 || total > 14 * 1024 * 1024) return nullptr;
    }
    int MT = CEIL_DIVIDE(size_m, 16);
    void* kernel = select_kernel(K, MT, c_fp32, true);
    TORCH_CHECK(kernel, "exl3_dmgemv: no instance for K=", K, " rows=", size_m);
    // Per-slot widths live on the device; ncb * splits_for(n) <= min(TARGET + ncb - 1, ncb * kgroups) for
    // every n <= size_n, so this grid covers every slot (blocks past a slot's own count exit at once)
    int grid_x = MIN(EXL3_DGEMV_TARGET_BLOCKS - 1 + ncb, ncb * (size_k / 128));

    int* ws = get_ws(device, stream);
    void* args[22];
    for (int i = 0; i < 22; ++i) args[i] = kernel_args[i];
    args[6] = (void*) &ws;
    launch(kernel, K, MT, grid_x, slots, args, device, stream);
    return kernel;
}
