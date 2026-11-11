// Prefill all-reduce over CUDA IPC device arenas.
//
// The idea of an 8-bit peer-push prefill all-reduce is diffbot's (`fp8ar`, huggingface.co/diffbot); this kernel is
// our own implementation of it, with int8 and exact fp32 modes.
//
// Own design: a tile pipeline per CTA. Every CTA owns tiles t = block, block + grid, ... and, for each tile,
// first pushes its own bytes into the peer's payload region (compressed to int8 with a bf16 scale per 32
// elements, or exact fp32), publishes a system-scope release flag, and only then consumes the tile it pushed
// one step earlier: it waits for the peer's flag, reads the peer's bytes out of its own payload region, and
// writes the fp32 sum to `out`. Pipelining the push of tile t against the consume of tile t-1 keeps a CTA's
// PCIe writes in flight while it does local reads and combines, and costs one kernel launch per collective
// instead of one launch (and one rendezvous) per call.
//
// Consistency: a tile in the peer's arena is only overwritten after the peer acknowledged consuming it, so one
// slot per tile is safe. Both ranks evaluate `rank0_value + rank1_value` in the same operand order, and the
// int8 payload is decoded with the bf16 scale transmitted next to it, so the result is bit-identical on both
// ranks. Mode 1 (fp32) carries no quantization at all.
//
// A tile is 65536 elements, every thread owns 16 contiguous elements, and a quantization block is the 32
// elements owned by an even/odd lane pair: the shared scale is one shfl_xor(1) away and a warp's int8 stores
// are 512 contiguous bytes.
#include <cstring>
#include <cstdio>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "p2p_oneshot.cuh"
#include "prefill_ar.cuh"
#include "../ptx.cuh"

#define PAR_TIMEOUT_NS (60ull * 1000000000ull)
#define PAR_TILE 262144
#define PAR_EPT 16
#define PAR_THREADS 256

static __host__ __device__ constexpr int64_t par_round16(int64_t v) { return (v + 15) & ~(int64_t) 15; }

static __host__ __device__ constexpr int64_t par_tiles(int64_t cap) { return (cap + PAR_TILE - 1) / PAR_TILE; }

std::vector<int64_t> p2p_prefill_ar_layout(int64_t cap, int64_t fmt)
{
    TORCH_CHECK(cap > 0 && cap % 512 == 0, "prefill_ar: cap must be a positive multiple of 512");
    TORCH_CHECK(fmt == 0 || fmt == 1, "prefill_ar: fmt must be 0 (int8) or 1 (fp32)");
    int64_t payload = cap * (fmt == 0 ? 1 : 4);
    int64_t scales = fmt == 0 ? par_round16(cap / 32 * 2) : 0;
    int64_t flags = par_round16(par_tiles(cap) * 8);
    int64_t acks = 2 * par_round16(par_tiles(cap) * 8);
    return {payload, scales, flags, acks, payload + scales + flags + acks};
}

// Wait until flag >= want, trapping (with a message) rather than hanging on a lost peer.
__device__ __forceinline__ void par_wait(const uint64_t* p, uint64_t want, int rank, int64_t tile, uint64_t seq,
                                         const char* what)
{
    uint64_t deadline = globaltimer_ns() + PAR_TIMEOUT_NS;
    while (ldg_acquire_sys_u64(p) < want)
    {
        #if __CUDA_ARCH__ >= 700
        __nanosleep(64);                                  // back off: a polling storm steals fabric bandwidth
        #endif
        if (globaltimer_ns() > deadline)
        {
            printf("prefill_ar: %s timeout rank %d tile %lld seq %llu\n", what, rank, (long long) tile,
                   (unsigned long long) seq);
            __trap();
        }
    }
}

// int8 codec: bf16 scale shared by the lane pair, and the quantized bytes plus that scale written to the peer.
__device__ __forceinline__ void par_quant16(const float* xs, uint4& v, __nv_bfloat16& sb)
{
    // lane pairs own whole 32-element blocks, so a pair mask keeps the shuffle legal on a partial last tile
    const unsigned mask = 3u << ((threadIdx.x & 31u) & ~1u);
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < PAR_EPT; j++) amax = fmaxf(amax, fabsf(xs[j]));
    amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 1));
    float s = amax / 127.0f;
    sb = __float2bfloat16_rn(s);
    float r = (__bfloat162float(sb) > 0.0f) ? 1.0f / __bfloat162float(sb) : 0.0f;
    int32_t w[4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
    {
        int q0 = max(-127, min(127, __float2int_rn(xs[j * 4 + 0] * r)));
        int q1 = max(-127, min(127, __float2int_rn(xs[j * 4 + 1] * r)));
        int q2 = max(-127, min(127, __float2int_rn(xs[j * 4 + 2] * r)));
        int q3 = max(-127, min(127, __float2int_rn(xs[j * 4 + 3] * r)));
        w[j] = (q0 & 0xff) | ((q1 & 0xff) << 8) | ((q2 & 0xff) << 16) | ((q3 & 0xff) << 24);
    }
    v = make_uint4((uint32_t) w[0], (uint32_t) w[1], (uint32_t) w[2], (uint32_t) w[3]);
}

// out <- rank0_value + rank1_value for this thread's 16 elements of tile `tile`.
template <int FMT>
__device__ __forceinline__ void par_consume16(const float* x, float* out,
                                              const uint8_t* __restrict__ my_pay, const uint8_t* __restrict__ sc,
                                              int64_t tile, int64_t n, int rank)
{
    const int64_t e0 = tile * PAR_TILE;
    const int64_t end = min(e0 + PAR_TILE, n);
    for (int64_t e = e0 + threadIdx.x * PAR_EPT; e < end; e += PAR_THREADS * PAR_EPT)
    {
        const float* xs = x + e;
        float* os = out + e;
        const int64_t rel = e - e0;
        if (FMT == 0)
        {
            const uint4 v = __ldcg((const uint4*) (my_pay + e0 + rel));
            const uint32_t w[4] = {v.x, v.y, v.z, v.w};
            const unsigned short sraw = __ldcg((const unsigned short*) (sc + ((e0 + rel) / 32) * 2));
            const float psb = __bfloat162float(__ushort_as_bfloat16(sraw));
            uint4 mine_code;
            __nv_bfloat16 sb;
            par_quant16(xs, mine_code, sb);
            const uint32_t mw[4] = {mine_code.x, mine_code.y, mine_code.z, mine_code.w};
            const float msb = __bfloat162float(sb);
            #pragma unroll
            for (int j = 0; j < PAR_EPT; j++)
            {
                float mine = (float) (int8_t) ((mw[j / 4] >> (8 * (j % 4))) & 0xff) * msb;
                float other = (float) (int8_t) ((w[j / 4] >> (8 * (j % 4))) & 0xff) * psb;
                os[j] = rank == 0 ? mine + other : other + mine;
            }
        }
        else
        {
            const uint4* pd = (const uint4*) (my_pay + e0 * 4 + rel * 4);
            const float* pf = (const float*) pd;
            #pragma unroll
            for (int j = 0; j < PAR_EPT; j++)
            {
                float other = __ldcg(pf + j);
                float mine = xs[j];
                os[j] = rank == 0 ? mine + other : other + mine;
            }
        }
    }
}

template <int FMT>
__global__ void __launch_bounds__(PAR_THREADS) prefill_ar_kernel
(
    const float* x,          // may alias `out`: the reduction runs in place
    float* out,
    int64_t n,
    int64_t cap,
    uint64_t seq,
    int rank,
    uint8_t* __restrict__ mine,
    uint8_t* __restrict__ peer
)
{
    __shared__ __align__(16) __nv_bfloat16 s_scale[PAR_TILE / 32];
    const int64_t payb = cap * (FMT == 0 ? 1 : 4);              // payload bytes, matching the host layout
    const int64_t off_s = FMT == 0 ? payb : 0;
    const int64_t off_f = FMT == 0 ? off_s + par_round16(cap / 32 * 2) : payb;
    const int64_t off_a = off_f + par_round16(par_tiles(cap) * 8);
    uint8_t* my_pay = mine;
    uint8_t* pe_pay = peer;
    uint8_t* my_sc = mine + off_s;
    uint8_t* pe_sc = peer + off_s;
    uint64_t* my_flag = (uint64_t*) (mine + off_f);
    uint64_t* pe_flag = (uint64_t*) (peer + off_f);
    uint64_t* my_ack = (uint64_t*) (mine + off_a);
    uint64_t* pe_ack = (uint64_t*) (peer + off_a);
    uint64_t* last_used = (uint64_t*) (mine + off_a + par_round16(par_tiles(cap) * 8));

    const int64_t ntiles = (n + PAR_TILE - 1) / PAR_TILE;
    int64_t prev = -1;

    for (int64_t t = blockIdx.x; t < ntiles; t += gridDim.x)
    {
        // Tiles absent from previous calls retain their own last-use sequence.
        if (threadIdx.x == 0) par_wait(my_ack + t, last_used[t], rank, t, seq, "ack");
        __syncthreads();

        const int64_t e0 = t * PAR_TILE;
        const int64_t e1 = (e0 + PAR_TILE < n) ? e0 + PAR_TILE : n;

        for (int64_t e = e0 + threadIdx.x * PAR_EPT; e < e1; e += PAR_THREADS * PAR_EPT)
        {
            const int64_t rel = e - e0;
            if (FMT == 0)
            {
                uint4 v;
                __nv_bfloat16 sb;
                par_quant16(x + e, v, sb);
                *((uint4*) (pe_pay + e0 + rel)) = v;
                if ((threadIdx.x & 1) == 0) s_scale[rel / 32] = sb;
            }
            else
            {
                const uint4* src = (const uint4*) (x + e);
                uint4* dst = (uint4*) (pe_pay + e0 * 4 + rel * 4);
                #pragma unroll
                for (int j = 0; j < 4; j++) dst[j] = src[j];
            }
        }

        if (FMT == 0)
        {
            __syncthreads();
            const int64_t nb = (e1 - e0) / 32;                 // 32-element blocks in this tile
            uint8_t* dst = pe_sc + (e0 / 32) * 2;
            const uint8_t* src = (const uint8_t*) s_scale;
            if ((nb & 7) == 0)                                 // 16 bytes = 8 blocks
                for (int64_t i = threadIdx.x; i < nb / 8; i += PAR_THREADS)
                    ((uint4*) dst)[i] = ((const uint4*) src)[i];
            else
                for (int64_t i = threadIdx.x; i < nb * 2; i += PAR_THREADS) dst[i] = src[i];
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) { stg_release_sys_u64(pe_flag + t, seq); last_used[t] = seq; }

        // (b) consume the tile pushed one step earlier, so the push above is still in flight
        if (prev >= 0)
        {
            if (threadIdx.x == 0) par_wait(my_flag + prev, seq, rank, prev, seq, "data");
            __syncthreads();
            const int64_t p1e = (prev * PAR_TILE + PAR_TILE < n) ? prev * PAR_TILE + PAR_TILE : n;
            par_consume16<FMT>(x, out, my_pay, my_sc, prev, p1e, rank);
            __syncthreads();
            if (threadIdx.x == 0) stg_release_sys_u64(pe_ack + prev, seq);
        }
        prev = t;
    }

    if (prev >= 0)                                             // last tile of this CTA
    {
        if (threadIdx.x == 0) par_wait(my_flag + prev, seq, rank, prev, seq, "data");
        __syncthreads();
        const int64_t p1e = (prev * PAR_TILE + PAR_TILE < n) ? prev * PAR_TILE + PAR_TILE : n;
        par_consume16<FMT>(x, out, my_pay, my_sc, prev, p1e, rank);
        __syncthreads();
        if (threadIdx.x == 0) stg_release_sys_u64(pe_ack + prev, seq);
    }
}

int64_t p2p_prefill_ar_alloc(int64_t bytes, int64_t device)
{
    TORCH_CHECK(bytes > 0, "prefill_ar: arena size must be positive");
    c10::cuda::CUDAGuard guard((c10::DeviceIndex) device);
    void* p = nullptr;
    TORCH_CHECK(cudaMalloc(&p, (size_t) bytes) == cudaSuccess, "prefill_ar: cudaMalloc failed");
    TORCH_CHECK(cudaMemset(p, 0, (size_t) bytes) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
                "prefill_ar: arena zeroing failed");
    return (int64_t) p;
}

int64_t p2p_prefill_ar_occupancy(int64_t fmt)
{
    int occ = 0;
    cudaError_t e = fmt == 0
        ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, prefill_ar_kernel<0>, PAR_THREADS, 0)
        : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, prefill_ar_kernel<1>, PAR_THREADS, 0);
    if (e != cudaSuccess)
    {
        cudaGetLastError();
        return 0;
    }
    return occ;
}

void p2p_prefill_ar_launch(torch::Tensor x, torch::Tensor out, int64_t peer, int64_t mine, int64_t n, int64_t cap,
                           int64_t fmt, int64_t grid, int64_t seq, int64_t rank)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat,
                "prefill_ar: x must be contiguous fp32 CUDA");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dtype() == torch::kFloat && out.numel() >= n,
                "prefill_ar: out must be a contiguous fp32 CUDA buffer of at least n elements");
    TORCH_CHECK(n > 0 && n % 512 == 0 && n <= cap, "prefill_ar: n must be a multiple of 512 and fit the arena");
    TORCH_CHECK(peer != 0 && mine != 0, "prefill_ar: arena not set up");
    TORCH_CHECK(grid >= 1 && grid <= 1024, "prefill_ar: grid out of range");
    TORCH_CHECK(seq >= 1, "prefill_ar: sequence must start at 1");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (fmt == 0)
        prefill_ar_kernel<0><<<(unsigned) grid, PAR_THREADS, 0, stream>>>(
            (const float*) x.data_ptr(), (float*) out.data_ptr(), n, cap, (uint64_t) seq, (int) rank,
            (uint8_t*) mine, (uint8_t*) peer);
    else
        prefill_ar_kernel<1><<<(unsigned) grid, PAR_THREADS, 0, stream>>>(
            (const float*) x.data_ptr(), (float*) out.data_ptr(), n, cap, (uint64_t) seq, (int) rank,
            (uint8_t*) mine, (uint8_t*) peer);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "prefill_ar launch failed");
}
