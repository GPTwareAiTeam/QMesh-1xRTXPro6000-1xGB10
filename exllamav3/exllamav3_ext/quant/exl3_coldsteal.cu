// ColdSteal device select and slot proxy: see exl3_coldsteal.cuh for the protocol.

#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "../util.h"
#include "../util.cuh"
#include "exl3_coldsteal.cuh"

#if defined(USE_ROCM)

void coldsteal_select(at::Tensor&, const c10::optional<at::Tensor>&, const at::Tensor&, at::Tensor&, at::Tensor&,
                      int64_t, int64_t, int64_t)
{
    TORCH_CHECK(false, "coldsteal_select: ColdSteal is not available on ROCm");
}
void coldsteal_cpu_split(at::Tensor&, at::Tensor&, double, int64_t) { TORCH_CHECK(false, "ColdSteal is not available on ROCm"); }
void coldsteal_policy(at::Tensor&, at::Tensor&, const c10::optional<at::Tensor>&, const at::Tensor&, at::Tensor&,
                      at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, const c10::optional<at::Tensor>&,
                      int64_t, int64_t, int64_t, int64_t, at::Tensor&, int64_t, int64_t,
                      const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&, int64_t, int64_t,
                      const c10::optional<at::Tensor>&, int64_t)
{
    TORCH_CHECK(false, "ColdSteal is not available on ROCm");
}
void coldsteal_policy_end(const at::Tensor&, int64_t, int64_t, int64_t) {}
int64_t coldsteal_host_alloc(int64_t, bool) { TORCH_CHECK(false, "ColdSteal is not available on ROCm"); }
void coldsteal_host_free(int64_t) {}
at::Tensor coldsteal_host_read(int64_t, int64_t) { return at::empty({0}, at::TensorOptions().dtype(at::kLong)); }
int64_t coldsteal_proxy_create(int64_t, int64_t) { TORCH_CHECK(false, "ColdSteal is not available on ROCm"); }
void coldsteal_proxy_add_layer(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                               int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {}
void coldsteal_proxy_launch(int64_t) {}
std::tuple<int64_t, int64_t, int64_t, std::string> coldsteal_proxy_stats(int64_t) { return { 0, 0, 0, "" }; }
void coldsteal_proxy_set_active(int64_t, int64_t) {}
int64_t coldsteal_proxy_retired(int64_t) { return -1; }
void coldsteal_proxy_destroy(int64_t) {}

#else

namespace {

__device__ __forceinline__ void cs_store_sys(int64_t* p, int64_t v)
{
    asm volatile("st.release.sys.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ int cs_load_slot(const int32_t* p)
{
    int v;
    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// One block. Picks are classified once (one slot_of read per pick), first occurrences of each cold expert ranked
// within their group (resident / miss) by (count desc, id asc), so the result does not depend on thread order.
__global__ __launch_bounds__(256)
void cs_select_kernel
(
    int64_t* __restrict__ sel,
    int64_t* __restrict__ sel_cpu,
    const int32_t* __restrict__ slot_of,
    int64_t* __restrict__ dev_seq,
    int64_t* __restrict__ stats,
    int64_t* words,
    const int n,
    const int n_hot,
    const int E,
    const int s_max
)
{
    __shared__ int32_t ex[CS_MAX_PICKS];        // cold expert id, -1 hot / invalid
    __shared__ int32_t sl[CS_MAX_PICKS];        // slot, -1 not resident
    __shared__ int32_t cnt[CS_MAX_PICKS];       // picks of this expert in the call (first occurrence only, else 0)
    __shared__ int32_t rank[CS_MAX_PICKS];
    __shared__ int64_t seq;
    __shared__ int n_miss, n_hit, n_stolen_picks, n_cold;
    const int t = threadIdx.x;

    if (t == 0)
    {
        seq = dev_seq[0] + 1;
        dev_seq[0] = seq;
        // `seen` before any slot_of read: the proxy's eviction fence relies on it (exl3_coldsteal.cuh)
        cs_store_sys(words, seq);
        __threadfence_system();
        n_miss = 0; n_hit = 0; n_stolen_picks = 0; n_cold = 0;
    }
    __syncthreads();

    for (int i = t; i < n; i += blockDim.x)
    {
        const int64_t e = sel[i];
        const bool cold = e >= n_hot && e < E;
        ex[i] = cold ? (int32_t) e : -1;
        sl[i] = cold ? cs_load_slot(slot_of + e) : -1;
    }
    __syncthreads();
    // Two picks of one expert can see different slot_of values when the proxy writes mid-call; either answer is
    // correct per pick (each pick is computed exactly once, here or on the worker). Grouping uses the first pick's.
    for (int i = t; i < n; i += blockDim.x)
    {
        int c = 0;
        bool first = ex[i] >= 0;
        if (first)
            for (int j = 0; j < n; ++j)
                if (ex[j] == ex[i]) { if (j < i) { first = false; break; } ++c; }
        cnt[i] = first ? c : 0;
    }
    __syncthreads();
    for (int i = t; i < n; i += blockDim.x)
    {
        if (!cnt[i]) { rank[i] = -1; continue; }
        const bool res = sl[i] >= 0;
        int r = 0;
        for (int j = 0; j < n; ++j)
            if (cnt[j] && (sl[j] >= 0) == res && (cnt[j] > cnt[i] || (cnt[j] == cnt[i] && ex[j] < ex[i]))) ++r;
        rank[i] = r;
        atomicAdd(res ? &n_hit : &n_miss, 1);
    }
    __syncthreads();
    // Rewrite: stolen picks -> local n_hot + slot (and off the worker's list); other cold picks -> inactive here
    for (int i = t; i < n; i += blockDim.x)
    {
        if (ex[i] < 0) continue;
        int f = 0;                                       // first occurrence carries the group's rank
        while (ex[f] != ex[i]) ++f;
        const bool steal = sl[f] >= 0 && rank[f] < s_max;
        atomicAdd(&n_cold, 1);
        if (steal)
        {
            sel[i] = (int64_t) n_hot + sl[f];
            if (sel_cpu) sel_cpu[i] = -1;
            atomicAdd(&n_stolen_picks, 1);
        }
        else sel[i] = CS_INACTIVE;
    }
    // Record (seqlock: begin, body, end)
    int64_t* rec = words + CS_REC_OFF;
    if (t == 0) { rec[0] = seq; __threadfence_system(); }
    __syncthreads();
    for (int i = t; i < n; i += blockDim.x)
    {
        if (!cnt[i] || rank[i] >= CS_REC_N) continue;
        if (sl[i] >= 0) rec[11 + rank[i]] = ex[i];
        else rec[2 + rank[i]] = ex[i];
    }
    __syncthreads();
    if (t == 0)
    {
        rec[1] = min(n_miss, CS_REC_N);
        rec[10] = min(n_hit, CS_REC_N);
        __threadfence_system();
        cs_store_sys(rec + 19, seq);
        stats[0] += 1;
        stats[1] += n_cold;
        stats[2] += n_stolen_picks;
        stats[3] += min(n_hit, s_max);
    }
}

// CPU tier split (notes/CPUExpertTier.md): of this call's worker-bound picks (sel_gx >= 0), the experts ranked
// last by pick count (fewest rows share them; ties: higher cold id first) up to n_cpu = ceil(frac * distinct) go to
// the host CPU: sel_cpu gets the cold id, sel_gx -1. Every other position of sel_cpu is -1. One block, graph-safe.
__global__ __launch_bounds__(256)
void cs_cpu_split_kernel(int64_t* __restrict__ sel_gx, int64_t* __restrict__ sel_cpu, const int n, const float frac,
                         const int max_cpu)
{
    __shared__ int32_t ex[CS_MAX_PICKS];
    __shared__ int32_t cnt[CS_MAX_PICKS];
    __shared__ int n_distinct;
    const int t = threadIdx.x;
    if (t == 0) n_distinct = 0;
    for (int i = t; i < n; i += blockDim.x) ex[i] = (int32_t) sel_gx[i];
    __syncthreads();
    for (int i = t; i < n; i += blockDim.x)
    {
        int c = 0; bool first = ex[i] >= 0;
        if (first)
            for (int j = 0; j < n; ++j)
                if (ex[j] == ex[i]) { if (j < i) { first = false; break; } ++c; }
        cnt[i] = first ? c : 0;
        if (first) atomicAdd(&n_distinct, 1);
    }
    __syncthreads();
    const int k = min(max_cpu, (int) ceilf(frac * (float) n_distinct));
    for (int i = t; i < n; i += blockDim.x)
    {
        sel_cpu[i] = -1;
        if (ex[i] < 0) continue;
        int f = 0;
        while (ex[f] != ex[i]) ++f;
        // rank among distinct experts by (count asc, id desc): number of experts strictly "less shared"
        int r = 0;
        for (int j = 0; j < n; ++j)
            if (cnt[j] && (cnt[j] < cnt[f] || (cnt[j] == cnt[f] && ex[j] > ex[f]))) ++r;
        if (r < k) { sel_cpu[i] = ex[i]; sel_gx[i] = -1; }
    }
}

// Proxy stream: rates of one slot, then its residency (one kernel, so the rates are visible first)
__global__ void cs_admit_kernel(int32_t* slot_of, int e, int slot, uint8_t* k2, int k2_stride, int k2_idx, int k2v)
{
    k2[k2_idx] = (uint8_t) k2v;
    k2[k2_stride + k2_idx] = (uint8_t) k2v;
    k2[2 * k2_stride + k2_idx] = (uint8_t) k2v;
    __threadfence();
    asm volatile("st.relaxed.gpu.s32 [%0], %1;" :: "l"(slot_of + e), "r"(slot) : "memory");
}

__global__ void cs_evict_kernel(int32_t* slot_of, int e)
{
    asm volatile("st.relaxed.gpu.s32 [%0], %1;" :: "l"(slot_of + e), "r"(-1) : "memory");
}

// ---- three-unit split policy (notes/SplitPolicy.md) ----------------------------------------------------------------

__device__ __forceinline__ uint64_t cs_mix(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ uint64_t cs_globaltimer()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ int cs_row_bucket(int rows)
{
    if (rows <= 6) return rows - 1;
    if (rows <= 8) return 6;
    if (rows <= 12) return 7;
    if (rows <= 16) return 8;
    if (rows <= 24) return 9;
    if (rows <= 32) return 10;
    return 11;
}

// One block. Same classification, ranking, rewrite and proxy record as cs_select_kernel; per call the policy (table or
// seeded explore) picks s resident experts to steal (P7, from VRAM slots), z non-resident ones for the P7 to compute
// straight from the pinned pool (zero-copy: the call writes table entries n_hot + S .. + z - 1 to those pool rows),
// c of the rest for the CPU tier (sel_t), and the GX keeps the remainder. Logs one CsPolRec per call (t1 is written by
// cs_policy_end_kernel).
__global__ __launch_bounds__(256)
void cs_policy_kernel
(
    int64_t* __restrict__ sel,
    int64_t* __restrict__ sel_cpu,
    int64_t* __restrict__ sel_t,            // null: no CPU tier (c = 0)
    const int32_t* __restrict__ slot_of,
    int64_t* __restrict__ dev_seq,
    int64_t* __restrict__ stats,
    int64_t* words,
    const int8_t* __restrict__ table,       // null: explore
    const int n,
    const int rows,
    const int n_hot,
    const int E,
    const int layer,
    const int s_cap,
    const int c_cap,
    const uint64_t salt,
    CsPolRec* ring,
    const int64_t ring_mask,
    unsigned long long* ring_head,
    int64_t* layer_cur,
    // zero-copy (z_cap 0: off)
    const int z_cap,
    const int S,                            // steal slots: zero-copy entries start at n_hot + S
    int64_t* __restrict__ zc_tables,        // (9, n_cs) pointer tables of the launch
    uint8_t* __restrict__ zc_k2,            // (3, n_cs) rates of the launch
    const int n_cs,
    const uint8_t* zc_pool,                 // pinned pool (device-addressable host pointer), row r = expert n_hot + r
    const uint8_t* zc_pool_k2,              // pinned (E - n_hot) row rates
    const int64_t* __restrict__ zc_off,     // (9,) byte offsets inside a row, table-row order
    const int64_t zc_stride
)
{
    __shared__ int32_t ex[CS_MAX_PICKS];        // cold expert id, -1 hot / invalid
    __shared__ int32_t sl[CS_MAX_PICKS];        // slot, -1 not resident
    __shared__ int32_t cnt[CS_MAX_PICKS];       // picks of this cold expert (first occurrence only, else 0)
    __shared__ int32_t rank[CS_MAX_PICKS];      // resident / miss rank by (count desc, id asc)
    __shared__ uint8_t stolen[CS_MAX_PICKS], tocpu[CS_MAX_PICKS];
    __shared__ int8_t zslot[CS_MAX_PICKS];      // zero-copy entry index of this (first-occurrence) expert, -1 none
    __shared__ int64_t seq;
    __shared__ uint64_t t0;
    __shared__ int n_miss, n_hit, n_cold, n_hot_picks, n_hot_distinct, s_sel, c_sel, z_sel, n_stolen_picks,
                   n_cpu_picks, n_zc_picks;
    const int t = threadIdx.x;

    if (t == 0)
    {
        t0 = cs_globaltimer();
        seq = dev_seq[0] + 1;
        dev_seq[0] = seq;
        cs_store_sys(words, seq);
        __threadfence_system();
        n_miss = 0; n_hit = 0; n_cold = 0; n_hot_picks = 0; n_hot_distinct = 0; n_stolen_picks = 0; n_cpu_picks = 0;
        n_zc_picks = 0;
    }
    __syncthreads();

    for (int i = t; i < n; i += blockDim.x)
    {
        const int64_t e = sel[i];
        const bool cold = e >= n_hot && e < E;
        ex[i] = cold ? (int32_t) e : -1;
        sl[i] = cold ? cs_load_slot(slot_of + e) : -1;
        stolen[i] = 0; tocpu[i] = 0; zslot[i] = -1;
        if (sel_t) sel_t[i] = -1;
    }
    __syncthreads();
    for (int i = t; i < n; i += blockDim.x)
    {
        const int64_t e = sel[i];
        if (e >= 0 && e < n_hot)
        {
            atomicAdd(&n_hot_picks, 1);
            bool first = true;
            for (int j = 0; j < i; ++j) if (sel[j] == e) { first = false; break; }
            if (first) atomicAdd(&n_hot_distinct, 1);
        }
        int c = 0;
        bool first = ex[i] >= 0;
        if (first)
            for (int j = 0; j < n; ++j)
                if (ex[j] == ex[i]) { if (j < i) { first = false; break; } ++c; }
        cnt[i] = first ? c : 0;
        if (ex[i] >= 0) atomicAdd(&n_cold, 1);
    }
    __syncthreads();
    for (int i = t; i < n; i += blockDim.x)
    {
        if (!cnt[i]) { rank[i] = -1; continue; }
        const bool res = sl[i] >= 0;
        int r = 0;
        for (int j = 0; j < n; ++j)
            if (cnt[j] && (sl[j] >= 0) == res && (cnt[j] > cnt[i] || (cnt[j] == cnt[i] && ex[j] < ex[i]))) ++r;
        rank[i] = r;
        atomicAdd(res ? &n_hit : &n_miss, 1);
    }
    __syncthreads();
    if (t == 0)
    {
        const int nd = n_hit + n_miss, h = n_hit;
        int s, z, c;
        const int cc = sel_t ? c_cap : 0;
        const int zc = zc_tables ? z_cap : 0;
        if (table)
        {
            // P7 hot load level (distinct hot experts of this call) from the table's per-rows-bucket thresholds
            const int rbk = cs_row_bucket(rows);
            const int8_t* thr = table + CS_POL_TABLE_CELLS * CS_POL_V + rbk * 2;
            const int lv = n_hot_distinct <= (uint8_t) thr[0] ? 0 : (n_hot_distinct <= (uint8_t) thr[1] ? 1 : 2);
            const int8_t* v = table + ((((size_t) lv * CS_POL_ROW_B + rbk) * (CS_POL_N + 1) + min(nd, CS_POL_N)) *
                                       (CS_POL_H + 1) + min(h, CS_POL_H)) * CS_POL_V;
            s = min((int) v[0], min(h, s_cap));
            z = min((int) v[2], min(n_miss, zc));
            c = min((int) v[1], min(nd - s - z, cc));
        }
        else
        {
            const uint64_t r = cs_mix((uint64_t) seq ^ ((uint64_t) layer << 48) ^ salt);
            const int sm = min(h, s_cap);
            s = (int) (r % (uint64_t) (sm + 1));
            const int zm = min(n_miss, zc);
            z = (int) ((r >> 20) % (uint64_t) (zm + 1));
            const int cm = min(nd - s - z, cc);
            c = (int) ((r >> 40) % (uint64_t) (cm + 1));
        }
        s_sel = max(s, 0); z_sel = max(z, 0); c_sel = max(c, 0);
    }
    __syncthreads();
    // Steal: the s most-picked resident experts. Zero-copy: the z most-picked non-resident (one PCIe read serves every
    // row that picked it). CPU: of the rest, the c least-picked (ties: higher id first)
    for (int i = t; i < n; i += blockDim.x)
    {
        if (!cnt[i]) continue;
        if (sl[i] >= 0 && rank[i] < s_sel) stolen[i] = 1;
        if (sl[i] < 0 && rank[i] < z_sel)
        {
            zslot[i] = (int8_t) rank[i];
            // Table entry n_hot + S + rank -> this expert's pool row (graph-safe: device tables, written here)
            const int col = n_hot + S + rank[i];
            const int64_t row = (int64_t) (ex[i] - n_hot);
            const uint8_t* base = zc_pool + row * zc_stride;
            for (int q = 0; q < 9; ++q) zc_tables[(size_t) q * n_cs + col] = (int64_t) (base + zc_off[q]);
            const uint8_t k2v = zc_pool_k2[row];
            for (int q = 0; q < 3; ++q) zc_k2[(size_t) q * n_cs + col] = k2v;
        }
    }
    __syncthreads();
    for (int i = t; i < n; i += blockDim.x)
    {
        if (!cnt[i] || stolen[i] || zslot[i] >= 0) continue;
        int r = 0;
        for (int j = 0; j < n; ++j)
            if (cnt[j] && !stolen[j] && zslot[j] < 0 && (cnt[j] < cnt[i] || (cnt[j] == cnt[i] && ex[j] > ex[i]))) ++r;
        if (r < c_sel) tocpu[i] = 1;
    }
    __threadfence();
    __syncthreads();
    for (int i = t; i < n; i += blockDim.x)
    {
        if (ex[i] < 0) continue;
        int f = 0;
        while (ex[f] != ex[i]) ++f;
        if (stolen[f])
        {
            sel[i] = (int64_t) n_hot + sl[f];
            sel_cpu[i] = -1;
            atomicAdd(&n_stolen_picks, 1);
        }
        else if (zslot[f] >= 0)
        {
            sel[i] = (int64_t) n_hot + S + zslot[f];
            sel_cpu[i] = -1;
            atomicAdd(&n_zc_picks, 1);
        }
        else
        {
            sel[i] = CS_INACTIVE;
            if (tocpu[f])
            {
                sel_t[i] = sel_cpu[i];
                sel_cpu[i] = -1;
                atomicAdd(&n_cpu_picks, 1);
            }
        }
    }
    // Proxy record (seqlock), as cs_select_kernel
    int64_t* rec = words + CS_REC_OFF;
    if (t == 0) { rec[0] = seq; __threadfence_system(); }
    __syncthreads();
    for (int i = t; i < n; i += blockDim.x)
    {
        if (!cnt[i] || rank[i] >= CS_REC_N) continue;
        if (sl[i] >= 0) rec[11 + rank[i]] = ex[i];
        else rec[2 + rank[i]] = ex[i];
    }
    __syncthreads();
    if (t == 0)
    {
        rec[1] = min(n_miss, CS_REC_N);
        rec[10] = min(n_hit, CS_REC_N);
        __threadfence_system();
        cs_store_sys(rec + 19, seq);
        stats[0] += 1;
        stats[1] += n_cold;
        stats[2] += n_stolen_picks;
        stats[3] += s_sel;
        if (ring)
        {
            const unsigned long long idx = atomicAdd(ring_head, 1ull);
            layer_cur[0] = (int64_t) idx;
            CsPolRec* p = ring + (idx & (unsigned long long) ring_mask);
            p->t0 = t0;
            p->t1 = 0;
            const int16_t f[16] = { (int16_t) layer, (int16_t) rows, (int16_t) (n_hit + n_miss), (int16_t) n_hit,
                                    (int16_t) s_sel, (int16_t) c_sel, (int16_t) n_hot_picks, (int16_t) n_hot_distinct,
                                    (int16_t) n_cold, (int16_t) n_stolen_picks, (int16_t) n_cpu_picks,
                                    (int16_t) (table ? 0 : 1), (int16_t) z_sel, (int16_t) n_zc_picks, 0, 0 };
            for (int k = 0; k < 16; ++k) p->f[k] = f[k];
        }
    }
}

__global__ void cs_policy_end_kernel(const int64_t* layer_cur, CsPolRec* ring, const int64_t ring_mask, int64_t* pub)
{
    const int64_t idx = layer_cur[0];
    ring[idx & ring_mask].t1 = cs_globaltimer();
    __threadfence_system();
    cs_store_sys(pub, idx + 1);
}

}  // namespace

void coldsteal_policy
(
    at::Tensor& sel, at::Tensor& sel_cpu, const c10::optional<at::Tensor>& sel_t, const at::Tensor& slot_of,
    at::Tensor& dev_seq, at::Tensor& stats, int64_t words, int64_t n_hot, int64_t layer, int64_t s_cap, int64_t c_cap,
    const c10::optional<at::Tensor>& table, int64_t salt, int64_t ring, int64_t ring_mask, int64_t ring_head,
    at::Tensor& layer_cur, int64_t z_cap, int64_t S, const c10::optional<at::Tensor>& zc_tables,
    const c10::optional<at::Tensor>& zc_k2, int64_t zc_pool, int64_t zc_pool_k2, const c10::optional<at::Tensor>& zc_off,
    int64_t zc_stride
)
{
    const at::cuda::OptionalCUDAGuard device_guard(sel.device());
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(sel_cpu, kLong);
    TORCH_CHECK_DTYPE(slot_of, kInt);
    TORCH_CHECK_DTYPE(layer_cur, kLong);
    TORCH_CHECK(sel.is_contiguous() && sel_cpu.is_contiguous() && slot_of.is_contiguous(), "coldsteal_policy: contiguous");
    const int n = (int) sel.numel();
    TORCH_CHECK(n <= CS_MAX_PICKS && sel_cpu.numel() >= n, "coldsteal_policy: picks / sel_cpu size");
    int64_t* t_ptr = nullptr;
    if (sel_t)
    {
        TORCH_CHECK_DTYPE(sel_t.value(), kLong);
        TORCH_CHECK(sel_t->is_contiguous() && sel_t->numel() >= n, "coldsteal_policy: sel_t size");
        t_ptr = (int64_t*) sel_t->data_ptr();
    }
    const int8_t* tab = nullptr;
    if (table)
    {
        TORCH_CHECK(table->scalar_type() == at::kChar && table->numel() == CS_POL_TABLE_BYTES,
                    "coldsteal_policy: table int8[", CS_POL_TABLE_BYTES, "]");
        tab = (const int8_t*) table->data_ptr();
    }
    int64_t* zt = nullptr;
    uint8_t* zk = nullptr;
    const int64_t* zo = nullptr;
    int n_cs = 0;
    if (z_cap > 0)
    {
        TORCH_CHECK(zc_tables && zc_k2 && zc_off && zc_pool && zc_pool_k2, "coldsteal_policy: zero-copy state missing");
        TORCH_CHECK(zc_tables->scalar_type() == at::kLong && zc_tables->dim() == 2 && zc_tables->size(0) == 9 &&
                    zc_k2->scalar_type() == at::kByte && zc_k2->size(1) == zc_tables->size(1) &&
                    zc_off->scalar_type() == at::kLong && zc_off->numel() == 9, "coldsteal_policy: zero-copy tables");
        n_cs = (int) zc_tables->size(1);
        TORCH_CHECK(n_hot + S + z_cap <= n_cs, "coldsteal_policy: tables hold ", n_cs, " entries, need ",
                    n_hot + S + z_cap);
        zt = (int64_t*) zc_tables->data_ptr(); zk = (uint8_t*) zc_k2->data_ptr(); zo = (const int64_t*) zc_off->data_ptr();
    }
    if (!n) return;
    const int rows = (int) (sel.dim() == 2 ? sel.size(0) : 1);
    cs_policy_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>
    (
        (int64_t*) sel.data_ptr(), (int64_t*) sel_cpu.data_ptr(), t_ptr, (const int32_t*) slot_of.data_ptr(),
        (int64_t*) dev_seq.data_ptr(), (int64_t*) stats.data_ptr(), (int64_t*) words, tab, n, rows, (int) n_hot,
        (int) slot_of.numel(), (int) layer, (int) s_cap, (int) c_cap, (uint64_t) salt,
        reinterpret_cast<CsPolRec*>(ring), ring_mask, reinterpret_cast<unsigned long long*>(ring_head),
        (int64_t*) layer_cur.data_ptr(), (int) z_cap, (int) S, zt, zk, n_cs,
        reinterpret_cast<const uint8_t*>(zc_pool), reinterpret_cast<const uint8_t*>(zc_pool_k2), zo, zc_stride
    );
    cuda_check(cudaPeekAtLastError());
}

void coldsteal_policy_end(const at::Tensor& layer_cur, int64_t ring, int64_t ring_mask, int64_t pub)
{
    const at::cuda::OptionalCUDAGuard device_guard(layer_cur.device());
    cs_policy_end_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        (const int64_t*) layer_cur.data_ptr(), reinterpret_cast<CsPolRec*>(ring), ring_mask, (int64_t*) pub);
    cuda_check(cudaPeekAtLastError());
}

void coldsteal_select
(
    at::Tensor& sel,
    const c10::optional<at::Tensor>& sel_cpu,
    const at::Tensor& slot_of,
    at::Tensor& dev_seq,
    at::Tensor& stats,
    int64_t words,
    int64_t n_hot,
    int64_t s_max
)
{
    const at::cuda::OptionalCUDAGuard device_guard(sel.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(slot_of, kInt);
    TORCH_CHECK_DTYPE(dev_seq, kLong);
    TORCH_CHECK_DTYPE(stats, kLong);
    TORCH_CHECK(sel.is_contiguous() && slot_of.is_contiguous(), "coldsteal_select: sel / slot_of must be contiguous");
    const int n = (int) sel.numel();
    TORCH_CHECK(n <= CS_MAX_PICKS, "coldsteal_select: ", n, " picks exceed CS_MAX_PICKS");
    TORCH_CHECK(words != 0 && stats.numel() >= 4 && dev_seq.numel() >= 1, "coldsteal_select: state not allocated");
    int64_t* cpu = nullptr;
    if (sel_cpu)
    {
        TORCH_CHECK_DTYPE(sel_cpu.value(), kLong);
        TORCH_CHECK(sel_cpu->is_contiguous() && sel_cpu->numel() >= n, "coldsteal_select: sel_cpu too small");
        cpu = (int64_t*) sel_cpu->data_ptr();
    }
    if (!n) return;
    cs_select_kernel<<<1, 256, 0, stream>>>
    (
        (int64_t*) sel.data_ptr(), cpu, (const int32_t*) slot_of.data_ptr(), (int64_t*) dev_seq.data_ptr(),
        (int64_t*) stats.data_ptr(), (int64_t*) words, n, (int) n_hot, (int) slot_of.numel(), (int) s_max
    );
    cuda_check(cudaPeekAtLastError());
}

void coldsteal_cpu_split(at::Tensor& sel_gx, at::Tensor& sel_cpu, double frac, int64_t max_cpu)
{
    const at::cuda::OptionalCUDAGuard device_guard(sel_gx.device());
    TORCH_CHECK_DTYPE(sel_gx, kLong);
    TORCH_CHECK_DTYPE(sel_cpu, kLong);
    TORCH_CHECK(sel_gx.is_contiguous() && sel_cpu.is_contiguous() && sel_cpu.numel() >= sel_gx.numel(),
                "coldsteal_cpu_split: contiguous int64, sel_cpu >= sel_gx");
    const int n = (int) sel_gx.numel();
    TORCH_CHECK(n <= CS_MAX_PICKS, "coldsteal_cpu_split: ", n, " picks exceed CS_MAX_PICKS");
    if (!n) return;
    cs_cpu_split_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        (int64_t*) sel_gx.data_ptr(), (int64_t*) sel_cpu.data_ptr(), n, (float) frac, (int) max_cpu);
    cuda_check(cudaPeekAtLastError());
}

int64_t coldsteal_host_alloc(int64_t nbytes, bool mapped)
{
    TORCH_CHECK(nbytes > 0, "coldsteal_host_alloc: size must be positive");
    void* p = nullptr;
    cudaError_t err = cudaHostAlloc(&p, (size_t) nbytes, mapped ? (cudaHostAllocMapped | cudaHostAllocPortable)
                                                               : cudaHostAllocPortable);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "coldsteal_host_alloc: cudaHostAlloc(", nbytes, ") failed: ", cudaGetErrorString(err));
    }
    if (mapped) memset(p, 0, (size_t) nbytes);
    return (int64_t) p;
}

void coldsteal_host_free(int64_t ptr)
{
    if (ptr) cudaFreeHost((void*) ptr);
}

at::Tensor coldsteal_host_read(int64_t ptr, int64_t n)
{
    TORCH_CHECK(ptr != 0 && n >= 0, "coldsteal_host_read: bad block or count");
    return at::from_blob((void*) ptr, {n}, at::TensorOptions().dtype(at::kLong)).clone();
}

// ---- proxy ------------------------------------------------------------------------------------------------------

namespace {

struct CsLayer
{
    volatile int64_t* words;
    int32_t* slot_of;
    uint8_t* k2;
    int k2_stride;
    uint8_t* slots;
    const uint8_t* pool;
    const uint8_t* pool_k2;
    int E, n_hot, S;
    size_t stride, scale_bytes, tstride, tunit;

    int64_t last_rec = 0;
    std::vector<int> slot_expert;       // -1 empty
    std::vector<int64_t> slot_use;      // tick of last hit / admission
    std::vector<int64_t> slot_fence;    // reusable once seen >= fence; 0 = free, -1 = eviction write in flight
    std::vector<cudaEvent_t> slot_ev;   // eviction write landed
    std::vector<int> host_slot_of;      // expert -> slot (host mirror)
    cudaEvent_t admit_ev = nullptr;     // last admission landed (one admission in flight per layer)
    bool admit_busy = false;
};

struct CsProxy
{
    int device;
    int max_out;
    cudaStream_t stream = nullptr;
    std::vector<CsLayer> layers;
    std::thread thr;
    std::atomic<bool> stop { false };
    std::atomic<int64_t> copies { 0 }, evictions { 0 }, records { 0 };
    std::mutex m;
    std::string why;
    int64_t tick = 0;

    void fail(const std::string& w)
    {
        std::lock_guard<std::mutex> lk(m);
        if (why.empty())
        {
            why = w;
            fprintf(stderr, " !! ColdSteal proxy: %s (admission stopped; resident slots stay valid)\n", w.c_str());
            fflush(stderr);
        }
    }

    bool read_record(CsLayer& L, int64_t* rec)
    {
        const volatile int64_t* r = L.words + CS_REC_OFF;
        for (int attempt = 0; attempt < 4; ++attempt)
        {
            const int64_t end = r[19];
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            for (int i = 0; i < CS_REC_WORDS; ++i) rec[i] = r[i];
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            const int64_t begin = r[0];
            if (begin == end && end == rec[19] && rec[0] == end) return end != 0;
        }
        return false;
    }

    // Slots [active, S) are retired (growable KV borrows their memory, notes/GrowKV.md): never admitted into, and a
    // resident one is evicted through the usual fenced path. retired_ready reports when every retired slot of every
    // layer is empty with its eviction fence passed, i.e. no launch can read it any more.
    std::atomic<int> active { 1 << 30 };
    std::atomic<int64_t> retired_clean { -1 };

    // One pass over one layer; returns true when it issued work
    bool step(CsLayer& L, bool allow_admit)
    {
        bool did = false;
        const int64_t seen = L.words[0];
        const int act = std::min(L.S, active.load(std::memory_order_acquire));
        // Retire landed admissions and evictions
        if (L.admit_busy && cudaEventQuery(L.admit_ev) == cudaSuccess) L.admit_busy = false;
        for (int s = 0; s < L.S; ++s)
            if (L.slot_fence[s] == -1 && cudaEventQuery(L.slot_ev[s]) == cudaSuccess)
                L.slot_fence[s] = L.words[0] + 1;       // read after the write landed (exl3_coldsteal.cuh)
        // Retired slots: evict their experts (one write each, fenced like an LRU eviction). Not while an admission is
        // in flight, which may target one of them
        if (!L.admit_busy)
            for (int s = act; s < L.S; ++s)
                if (L.slot_expert[s] >= 0)
                {
                    const int old = L.slot_expert[s];
                    cs_evict_kernel<<<1, 1, 0, stream>>>(L.slot_of, old);
                    if (cudaEventRecord(L.slot_ev[s], stream) != cudaSuccess) { fail("evict event"); return did; }
                    L.host_slot_of[old] = -1;
                    L.slot_expert[s] = -1;
                    L.slot_fence[s] = -1;
                    evictions.fetch_add(1, std::memory_order_relaxed);
                    did = true;
                }

        int64_t rec[CS_REC_WORDS];
        if (!read_record(L, rec) || rec[0] == L.last_rec) return did;
        L.last_rec = rec[0];
        records.fetch_add(1, std::memory_order_relaxed);
        ++tick;
        const int nh = (int) std::min<int64_t>(rec[10], CS_REC_N), nm = (int) std::min<int64_t>(rec[1], CS_REC_N);
        std::vector<int> hit_slots;
        for (int i = 0; i < nh; ++i)
        {
            const int e = (int) rec[11 + i];
            if (e < L.n_hot || e >= L.E) continue;
            const int s = L.host_slot_of[e];
            if (s >= 0) { L.slot_use[s] = tick; hit_slots.push_back(s); }
        }
        if (L.admit_busy || !nm || !allow_admit || act <= 0) return did;
        // Most-picked miss that has a pool row and is not resident
        int e = -1;
        for (int i = 0; i < nm && e < 0; ++i)
        {
            const int c = (int) rec[2 + i];
            if (c >= L.n_hot && c < L.E && L.host_slot_of[c] < 0 && L.pool_k2[c - L.n_hot]) e = c;
        }
        if (e < 0) return did;
        // A free slot, else start evicting the least recently used one not hit by this record
        int free_s = -1;
        for (int s = 0; s < act && free_s < 0; ++s)
            if (L.slot_expert[s] < 0 && L.slot_fence[s] >= 0 && seen >= L.slot_fence[s]) free_s = s;
        if (free_s < 0)
        {
            bool evicting = false;
            for (int s = 0; s < act; ++s) if (L.slot_expert[s] < 0) evicting = true;   // already on its way
            if (!evicting)
            {
                int v = -1;
                for (int s = 0; s < act; ++s)
                {
                    if (std::find(hit_slots.begin(), hit_slots.end(), s) != hit_slots.end()) continue;
                    if (v < 0 || L.slot_use[s] < L.slot_use[v]) v = s;
                }
                if (v >= 0)
                {
                    const int old = L.slot_expert[v];
                    cs_evict_kernel<<<1, 1, 0, stream>>>(L.slot_of, old);
                    if (cudaEventRecord(L.slot_ev[v], stream) != cudaSuccess) { fail("evict event"); return did; }
                    L.host_slot_of[old] = -1;
                    L.slot_expert[v] = -1;
                    L.slot_fence[v] = -1;
                    evictions.fetch_add(1, std::memory_order_relaxed);
                    did = true;
                }
            }
            return did;
        }
        // Admission: bytes, then rates + residency
        const int r = e - L.n_hot;
        const int k2v = L.pool_k2[r];
        const size_t tb = (size_t) k2v * L.tunit;      // trellis bytes at k2 / 2 bits
        if (tb > L.tstride) { fail("pool row rate " + std::to_string(k2v) + " exceeds the slot's trellis region"); return did; }
        const uint8_t* src = L.pool + (size_t) r * L.stride;
        uint8_t* dst = L.slots + (size_t) free_s * L.stride;
        cudaError_t err = cudaMemcpyAsync(dst, src, L.scale_bytes + tb, cudaMemcpyHostToDevice, stream);
        for (int p = 1; p < 3 && err == cudaSuccess; ++p)
            err = cudaMemcpyAsync(dst + L.scale_bytes + p * L.tstride, src + L.scale_bytes + p * L.tstride, tb,
                                  cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) { fail(std::string("slot copy: ") + cudaGetErrorString(err)); return did; }
        cs_admit_kernel<<<1, 1, 0, stream>>>(L.slot_of, e, free_s, L.k2, L.k2_stride, L.n_hot + free_s, k2v);
        if (cudaEventRecord(L.admit_ev, stream) != cudaSuccess) { fail("admit event"); return did; }
        L.admit_busy = true;
        L.slot_expert[free_s] = e;
        L.slot_use[free_s] = tick;
        L.host_slot_of[e] = free_s;
        copies.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void run()
    {
        cudaSetDevice(device);
        int idle = 0;
        while (!stop.load(std::memory_order_relaxed))
        {
            bool any = false;
            bool ok;
            { std::lock_guard<std::mutex> lk(m); ok = why.empty(); }
            if (ok)
            {
                // Bound the copies queued ahead of the PCIe link: at most max_out admissions in flight
                int busy = 0;
                for (auto& L : layers) busy += L.admit_busy ? 1 : 0;
                for (auto& L : layers)
                {
                    const bool was = L.admit_busy;
                    any |= step(L, busy < max_out);
                    if (!was && L.admit_busy) ++busy;
                }
                // Growable KV: report the active count once every retired slot is drained in every layer. Clean = no
                // expert, its eviction write landed (slot_of no longer names the slot, so no select from here on
                // routes a pick to it) and no admission in flight. The caller synchronizes the device before
                // unmapping, so no launch issued earlier is still reading the slot either.
                const int act = active.load(std::memory_order_acquire);
                bool clean = true;
                for (auto& L : layers)
                {
                    if (L.admit_busy && act < L.S) { clean = false; break; }
                    for (int s = std::max(0, act); s < L.S && clean; ++s)
                        if (L.slot_expert[s] >= 0 || L.slot_fence[s] < 0) clean = false;
                    if (!clean) break;
                }
                retired_clean.store(clean ? act : -1, std::memory_order_release);
            }
            if (any) idle = 0;
            else if (++idle > 2000) { std::this_thread::sleep_for(std::chrono::microseconds(50)); idle = 0; }
        }
    }
};

}  // namespace

int64_t coldsteal_proxy_create(int64_t device, int64_t max_out)
{
    auto* p = new CsProxy();
    p->device = (int) device;
    p->max_out = (int) std::max<int64_t>(1, max_out);
    int prev = -1;
    cuda_check(cudaGetDevice(&prev));
    cuda_check(cudaSetDevice(p->device));
    // Non-blocking: a stream that takes part in the legacy default stream's implicit synchronization would wait for
    // the compute stream's kernels before every copy
    cuda_check(cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking));
    cuda_check(cudaSetDevice(prev));
    return (int64_t) p;
}

void coldsteal_proxy_add_layer
(
    int64_t handle, int64_t words, int64_t slot_of, int64_t k2, int64_t k2_stride, int64_t slots, int64_t pool,
    int64_t pool_k2, int64_t E, int64_t n_hot, int64_t S, int64_t stride, int64_t scale_bytes, int64_t tstride,
    int64_t tunit
)
{
    auto* p = (CsProxy*) handle;
    TORCH_CHECK(p && !p->thr.joinable(), "coldsteal_proxy_add_layer: proxy already running");
    CsLayer L;
    L.words = (volatile int64_t*) words;
    L.slot_of = (int32_t*) slot_of;
    L.k2 = (uint8_t*) k2;
    L.k2_stride = (int) k2_stride;
    L.slots = (uint8_t*) slots;
    L.pool = (const uint8_t*) pool;
    L.pool_k2 = (const uint8_t*) pool_k2;
    L.E = (int) E; L.n_hot = (int) n_hot; L.S = (int) S;
    L.stride = (size_t) stride; L.scale_bytes = (size_t) scale_bytes; L.tstride = (size_t) tstride;
    L.tunit = (size_t) tunit;
    TORCH_CHECK(tunit > 0 && 6 * tunit <= tstride && (size_t) scale_bytes + 3 * (size_t) tstride <= (size_t) stride,
                "coldsteal_proxy_add_layer: slot layout does not fit the stride");
    L.slot_expert.assign(S, -1);
    L.slot_use.assign(S, 0);
    L.slot_fence.assign(S, 0);
    L.host_slot_of.assign(E, -1);
    int prev = -1;
    cuda_check(cudaGetDevice(&prev));
    cuda_check(cudaSetDevice(p->device));
    L.slot_ev.resize(S);
    for (auto& ev : L.slot_ev) cuda_check(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    cuda_check(cudaEventCreateWithFlags(&L.admit_ev, cudaEventDisableTiming));
    cuda_check(cudaSetDevice(prev));
    p->layers.push_back(std::move(L));
}

void coldsteal_proxy_launch(int64_t handle)
{
    auto* p = (CsProxy*) handle;
    TORCH_CHECK(p && !p->thr.joinable(), "coldsteal_proxy_launch: already running");
    p->thr = std::thread([p] { p->run(); });
}

std::tuple<int64_t, int64_t, int64_t, std::string> coldsteal_proxy_stats(int64_t handle)
{
    auto* p = (CsProxy*) handle;
    std::lock_guard<std::mutex> lk(p->m);
    return { p->copies.load(), p->evictions.load(), p->records.load(), p->why };
}

// Growable KV (notes/GrowKV.md): slots [active, S) of every layer stop being used. Returns at once; poll
// coldsteal_proxy_retired until it reports `active`, then the retired slots' memory may be released.
void coldsteal_proxy_set_active(int64_t handle, int64_t active)
{
    auto* p = (CsProxy*) handle;
    p->active.store((int) std::max<int64_t>(0, active), std::memory_order_release);
}

// The active count for which every layer's retired slots are empty with their eviction fence passed (published by
// the proxy thread after each pass); -1 while a retire is still draining
int64_t coldsteal_proxy_retired(int64_t handle)
{
    auto* p = (CsProxy*) handle;
    return p->retired_clean.load(std::memory_order_acquire);
}

void coldsteal_proxy_destroy(int64_t handle)
{
    auto* p = (CsProxy*) handle;
    if (!p) return;
    p->stop.store(true);
    if (p->thr.joinable()) p->thr.join();
    cudaSetDevice(p->device);
    cudaStreamSynchronize(p->stream);
    for (auto& L : p->layers)
    {
        for (auto ev : L.slot_ev) cudaEventDestroy(ev);
        cudaEventDestroy(L.admit_ev);
    }
    cudaStreamDestroy(p->stream);
    delete p;
}

#endif  // !defined(USE_ROCM)
