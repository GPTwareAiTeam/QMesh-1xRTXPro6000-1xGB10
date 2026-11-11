// CPU MoE experts in MXFP4 (DeepSeek-V4.1's routed-expert format), AVX-512 VNNI.
//
// Weights: W[o, k] = e2m1(nib[o, k]) * 2^(s[o, k / 32] - 127). With the E2M1 values doubled (0 1 2 3 4 6 8 12, signed)
// every weight is an exact int in [-12, 12]: W = v[o, k] * 2^(s - 128). Activations are quantized to int8 per 32-input
// block (x = x8 * qa), so one block's contribution is
//     y[o] += 2^(s[o, kb] - 128) * qa[kb] * sum_{k in kb} v[o, k] * x8[k]
// The integer dot is VNNI: vpdpbusd(u8 weights, s8 activations) with the weights offset by +12 (a 16-entry vpshufb LUT
// turns a nibble straight into 12 + v), and the offset removed per block as 12 * sum(x8).
//
// Layout (repacked at registration, owned by the layer): per projection, output tiles of 16 rows; per tile, per 32-input
// block, 256 B of nibbles then (separate array) 16 E8M0 bytes. Within a block, 64 B load L (L = 0..3) holds in its low
// nibbles the expanded bytes of vpdpbusd step q = 2L and in its high nibbles step q = 2L + 1; expanded byte o * 4 + j of
// step q is v[o, 32 kb + 4 q + j]. A worker owns whole tiles, so each reads one sequential stream (k-major in the tile).
//
// One call (moe_fp4_variants.inc forward_variant): quantize the token rows, gate/up tile items over all experts,
// SiLU-gate + requantize per slot, down tile items (routing weight folded in), then the per-token sum. Phases run on a
// persistent pool with persistent scratch; items are taken from an atomic counter, and an item is a pair of tiles
// streamed interleaved (two DRAM streams per worker) unless 16+ rows share an expert (notes/CPUKernelShotgun.md).

#include "moe_fp4.h"

#include <ATen/ATen.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <memory>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(__x86_64__) && defined(__GNUC__)
#include <cpuid.h>
#endif

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define FP4_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
#define FP4_X86 1
#define FP4_RELAX() _mm_pause()
#else
#define FP4_TARGET
#define FP4_X86 0
#define FP4_RELAX() std::this_thread::yield()
#endif

namespace {

constexpr int TILE = 16;            // output rows per tile
constexpr int KB = 32;              // inputs per scale block
constexpr int BLK_BYTES = 256;      // packed nibbles per (tile, block)
constexpr int RB = 8;               // activation rows per register block

struct Proj
{
    uint8_t* w = nullptr;           // [tiles][kbn][256]
    uint8_t* s = nullptr;           // [tiles][kbn][16]
    int out = 0, in = 0, tiles = 0, kbn = 0;
};

struct Expert { Proj g, u, d; };

struct Layer
{
    std::vector<Expert> experts;
    int H = 0, I = 0;
    float act_limit = 0.f;
    void* arena = nullptr;
    size_t arena_bytes = 0;
    size_t expert_bytes = 0;
};

// ---- persistent pool ---------------------------------------------------------------------------------------------

struct Pool
{
    std::vector<std::thread> threads;
    std::atomic<uint64_t> gen { 0 };
    std::atomic<int> pending { 0 };
    void (*fn)(void*, int) = nullptr;
    void* ctx = nullptr;
    int n = 1;
    std::atomic<bool> stop { false };
    std::mutex m;
    std::condition_variable cv;
    std::atomic<int> sleepers { 0 };

    void ensure(int want)
    {
        if (want <= n && !threads.empty()) return;
        if (want <= 1) { n = 1; return; }
        shutdown();
        stop = false;
        n = want;
        // The start generation is read here, not in the thread: a run() issued before a new thread is scheduled must
        // still be seen by it as new work
        const uint64_t g0 = gen.load(std::memory_order_acquire);
        for (int i = 1; i < n; ++i) threads.emplace_back([this, i, g0] { loop(i, g0); });
    }

    void loop(int id, uint64_t seen)
    {
        while (true)
        {
            int spins = 0;
            uint64_t g;
            while ((g = gen.load(std::memory_order_acquire)) == seen)
            {
                if (stop.load(std::memory_order_relaxed)) return;
                if (++spins < spin_limit) { FP4_RELAX(); continue; }
                std::unique_lock<std::mutex> lk(m);
                sleepers.fetch_add(1);
                cv.wait_for(lk, std::chrono::milliseconds(2), [&] { return gen.load() != seen || stop.load(); });
                sleepers.fetch_sub(1);
                spins = 0;
            }
            seen = g;
            if (stop.load()) return;
            if (id < active) fn(ctx, id);
            pending.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    int active = 1;
    int spin_limit = 20000;     // pause-spins before a worker parks on the condvar (shotgun variants tune it)

    void run(void (*f)(void*, int), void* c, int workers)
    {
        workers = std::max(1, std::min(workers, n));
        if (workers == 1) { f(c, 0); return; }
        fn = f; ctx = c; active = workers;
        pending.store(n - 1, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_acq_rel);
        if (sleepers.load()) { std::lock_guard<std::mutex> lk(m); cv.notify_all(); }
        f(c, 0);
        while (pending.load(std::memory_order_acquire) != 0) FP4_RELAX();
    }

    void shutdown()
    {
        if (threads.empty()) return;
        stop = true;
        { std::lock_guard<std::mutex> lk(m); cv.notify_all(); }
        gen.fetch_add(1);
        for (auto& t : threads) t.join();
        threads.clear();
        n = 1;
    }

    ~Pool() { shutdown(); }
};

Pool g_pool;
std::mutex g_pool_mutex;

// ---- activation quantization -------------------------------------------------------------------------------------

// x [k] fp32 -> x8 [k] int8, qa [k/32] (block scale), corr [k/32] = 12 * sum(x8)
inline void quant_row(const float* x, int k, int8_t* x8, float* qa, int32_t* corr)
{
    for (int b = 0; b < k / KB; ++b)
    {
        const float* xb = x + b * KB;
        float mx = 0.f;
        for (int i = 0; i < KB; ++i) mx = std::max(mx, std::fabs(xb[i]));
        const float q = mx / 127.f;
        const float inv = mx > 0.f ? 127.f / mx : 0.f;
        int32_t s = 0;
        for (int i = 0; i < KB; ++i)
        {
            const int v = (int) std::nearbyint(xb[i] * inv);
            x8[b * KB + i] = (int8_t) v;
            s += v;
        }
        qa[b] = q;
        corr[b] = 12 * s;
    }
}

// ---- tile GEMV ---------------------------------------------------------------------------------------------------

struct RowPtrs
{
    const int8_t* x8[RB];
    const float* qa[RB];
    const int32_t* corr[RB];
    float* out[RB];                 // out row base (tile offset added inside)
    float wscale[RB];               // multiplier folded into the stored result
};

// Scalar reference of the same math (non-x86 / no VNNI)
void tile_gemv_scalar(const Proj& P, int tile, const RowPtrs& rp, int R)
{
    static const int lut[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
    const uint8_t* w = P.w + (size_t) tile * P.kbn * BLK_BYTES;
    const uint8_t* s = P.s + (size_t) tile * P.kbn * TILE;
    float acc[RB][TILE] = {};
    for (int kb = 0; kb < P.kbn; ++kb, w += BLK_BYTES, s += TILE)
        for (int r = 0; r < R; ++r)
        {
            int32_t ia[TILE] = {};
            for (int L = 0; L < 4; ++L)
                for (int b = 0; b < 64; ++b)
                {
                    const int o = b / 4, j = b % 4;
                    ia[o] += lut[w[L * 64 + b] & 15] * rp.x8[r][kb * KB + 8 * L + j];
                    ia[o] += lut[w[L * 64 + b] >> 4] * rp.x8[r][kb * KB + 8 * L + 4 + j];
                }
            for (int o = 0; o < TILE; ++o)
                acc[r][o] += std::ldexp((float) ia[o], (int) std::max<int>(s[o], 1) - 128) * rp.qa[r][kb];
        }
    for (int r = 0; r < R; ++r)
        for (int o = 0; o < TILE; ++o) rp.out[r][tile * TILE + o] = acc[r][o] * rp.wscale[r];
}

bool has_vnni()
{
#if FP4_X86
    static const bool v = __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw");
    return v;
#else
    return false;
#endif
}

// ---- one forward: moe_fp4_variants.inc (forward_variant; exl3_moe_fp4_forward_raw picks the shipped config) ------

struct Group { int expert; int first; int count; };   // slots [first, first + count) of `slots`

// No MADV_HUGEPAGE hint and no later collapse: with THP defrag=madvise every first touch of a hinted region compacts
// synchronously once free memory runs low (stalled a 40-layer load past 30 min), and a background MADV_COLLAPSE of the
// ~170 GB tier cost the P7 client 11 % per decode step for <= 5 % kernel speed. 4K pages.
void* huge_alloc(size_t bytes)
{
    const size_t align = 2u << 20;
    bytes = (bytes + align - 1) / align * align;
    void* p = std::aligned_alloc(align, bytes);
    TORCH_CHECK(p, "exl3_moe_fp4: cannot allocate ", bytes, " bytes");
    return p;
}

// Repack one projection: src weight [out, in/2] (low nibble = even input), scale [out, in/32]
void repack(Proj& P, const at::Tensor& wt, const at::Tensor& st, uint8_t* wdst, uint8_t* sdst)
{
    const uint8_t* w = wt.data_ptr<uint8_t>();
    const uint8_t* s = st.data_ptr<uint8_t>();
    const int out = P.out, in = P.in, half = in / 2;
    P.w = wdst; P.s = sdst;
    for (int t = 0; t < P.tiles; ++t)
        for (int kb = 0; kb < P.kbn; ++kb)
        {
            uint8_t* blk = wdst + ((size_t) t * P.kbn + kb) * BLK_BYTES;
            for (int L = 0; L < 4; ++L)
                for (int b = 0; b < 64; ++b)
                {
                    const int o = t * TILE + b / 4, j = b % 4;
                    const int k0 = kb * KB + 8 * L + j, k1 = k0 + 4;
                    const uint8_t* row = w + (size_t) o * half;
                    const int n0 = (k0 & 1) ? row[k0 >> 1] >> 4 : row[k0 >> 1] & 15;
                    const int n1 = (k1 & 1) ? row[k1 >> 1] >> 4 : row[k1 >> 1] & 15;
                    blk[L * 64 + b] = (uint8_t) (n0 | (n1 << 4));
                }
            for (int o = 0; o < TILE; ++o)
                sdst[((size_t) t * P.kbn + kb) * TILE + o] = s[(size_t) (t * TILE + o) * (in / KB) + kb];
        }
    (void) out;
}

#include "moe_fp4_variants.inc"

}  // namespace

int64_t exl3_moe_fp4_make_layer
(
    const std::vector<at::Tensor>& gate_w, const std::vector<at::Tensor>& gate_s,
    const std::vector<at::Tensor>& up_w, const std::vector<at::Tensor>& up_s,
    const std::vector<at::Tensor>& down_w, const std::vector<at::Tensor>& down_s,
    double act_limit
)
{
    const size_t n = gate_w.size();
    TORCH_CHECK(n > 0 && gate_s.size() == n && up_w.size() == n && up_s.size() == n && down_w.size() == n &&
                down_s.size() == n, "exl3_moe_fp4_make_layer: per-expert lists differ in length");
    const int I = (int) gate_w[0].size(0), H = (int) gate_w[0].size(1) * 2;
    TORCH_CHECK(I % TILE == 0 && H % TILE == 0 && I % KB == 0 && H % KB == 0, "exl3_moe_fp4: dims must be multiples of 32");
    auto chk = [&] (const at::Tensor& w, const at::Tensor& s, int out, int in)
    {
        TORCH_CHECK(w.device().is_cpu() && s.device().is_cpu() && w.is_contiguous() && s.is_contiguous() &&
                    w.scalar_type() == at::kByte && s.scalar_type() == at::kByte, "exl3_moe_fp4: CPU uint8 contiguous tensors");
        TORCH_CHECK(w.size(0) == out && w.size(1) == in / 2 && s.size(0) == out && s.size(1) == in / KB,
                    "exl3_moe_fp4: tensor shapes [", w.size(0), ",", w.size(1), "] / [", s.size(0), ",", s.size(1),
                    "] for a ", out, "x", in, " projection");
    };
    auto* L = new Layer();
    L->H = H; L->I = I; L->act_limit = (float) act_limit;
    const size_t gw = (size_t) I * H / 2, gs = (size_t) I * H / KB, dw = (size_t) H * I / 2, ds = (size_t) H * I / KB;
    L->expert_bytes = 2 * (gw + gs) + dw + ds;
    L->arena_bytes = L->expert_bytes * n;
    L->arena = huge_alloc(L->arena_bytes);
    L->experts.resize(n);
    uint8_t* base = (uint8_t*) L->arena;
    std::vector<std::thread> th;
    const int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::atomic<size_t> next { 0 };
    auto work = [&]
    {
        for (size_t e; (e = next.fetch_add(1)) < n; )
        {
            chk(gate_w[e], gate_s[e], I, H); chk(up_w[e], up_s[e], I, H); chk(down_w[e], down_s[e], H, I);
            Expert& E = L->experts[e];
            uint8_t* p = base + e * L->expert_bytes;
            for (Proj* P : { &E.g, &E.u }) { P->out = I; P->in = H; P->tiles = I / TILE; P->kbn = H / KB; }
            E.d.out = H; E.d.in = I; E.d.tiles = H / TILE; E.d.kbn = I / KB;
            repack(E.g, gate_w[e], gate_s[e], p, p + gw); p += gw + gs;
            repack(E.u, up_w[e], up_s[e], p, p + gw); p += gw + gs;
            repack(E.d, down_w[e], down_s[e], p, p + dw);
        }
    };
    for (int i = 0; i < nt; ++i) th.emplace_back(work);
    for (auto& t : th) t.join();
    return (int64_t) L;
}

void exl3_moe_fp4_free_layer(int64_t handle)
{
    auto* L = (Layer*) handle;
    if (!L) return;
    std::free(L->arena);
    delete L;
}

int64_t exl3_moe_fp4_expert_bytes(int64_t handle)
{
    return (int64_t) ((Layer*) handle)->expert_bytes;
}

void exl3_moe_fp4_forward_raw
(
    int64_t handle, const at::Half* x, const int32_t* selected, const at::Half* weights,
    float* out, int rows, int topk, int threads
)
{
    const Layer* L = (const Layer*) handle;
    TORCH_CHECK(L, "exl3_moe_fp4_forward_raw: null layer");
    // Shotgun winner (notes/CPUKernelShotgun.md, stage 2, 200 calls x 3 reps, layer 5, DRAM-cold experts): persistent
    // scratch + two interleaved tile streams per item, 12-20 % under v1 at every split-policy CPU shape (1-8 experts x
    // 1 row, rows 4-8 sharing one). One stream when 16+ rows share an expert (compute-bound there, 2-6 % better).
    // Most rows sharing one expert in this call
    std::vector<int32_t> ids;
    ids.reserve((size_t) rows * topk);
    for (int i = 0; i < rows * topk; ++i) if (selected[i] >= 0) ids.push_back(selected[i]);
    std::sort(ids.begin(), ids.end());
    int max_rows = 0;
    for (size_t i = 0, j; i < ids.size(); i = j)
    {
        for (j = i; j < ids.size() && ids[j] == ids[i]; ++j) {}
        max_rows = std::max(max_rows, (int) (j - i));
    }
    VarCfg v;
    v.alloc = 1; v.phases = 5; v.spin = 0; v.streams = max_rows >= 16 ? 1 : 2; v.pf = 8; v.pfhint = 0; v.amx = 0;
    forward_variant(L, v, x, selected, weights, out, rows, topk, threads);
}

void exl3_moe_fp4_forward
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
)
{
    const Layer* L = (const Layer*) handle;
    TORCH_CHECK(L, "exl3_moe_fp4_forward: null layer");
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2 &&
                x.size(1) == L->H, "exl3_moe_fp4_forward: x must be CPU fp16 [m, H]");
    TORCH_CHECK(selected.scalar_type() == at::kLong && weights.scalar_type() == at::kHalf &&
                selected.sizes() == weights.sizes() && selected.size(0) == x.size(0) && selected.is_contiguous() &&
                weights.is_contiguous(), "exl3_moe_fp4_forward: selected/weights [m, topk] int64/fp16");
    TORCH_CHECK(out.scalar_type() == at::kFloat && out.is_contiguous() && out.sizes() == x.sizes(),
                "exl3_moe_fp4_forward: out must be fp32 [m, H]");
    std::vector<int32_t> sel(selected.numel());
    const int64_t* in = selected.data_ptr<int64_t>();
    for (size_t i = 0; i < sel.size(); ++i)
    {
        TORCH_CHECK(in[i] < (int64_t) L->experts.size(), "exl3_moe_fp4_forward: expert id ", in[i], " >= ",
                    L->experts.size());
        sel[i] = in[i] < 0 ? -1 : (int32_t) in[i];
    }
    exl3_moe_fp4_forward_raw(handle, x.data_ptr<at::Half>(), sel.data(), weights.data_ptr<at::Half>(),
                             out.data_ptr<float>(), (int) x.size(0), (int) selected.size(1), (int) num_threads);
}

void exl3_moe_fp4_forward_variant
(
    int64_t handle, const at::Tensor& x, const at::Tensor& selected, const at::Tensor& weights, at::Tensor& out,
    int64_t num_threads, const std::vector<int64_t>& cfg
)
{
    const Layer* L = (const Layer*) handle;
    TORCH_CHECK(L && x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2 && x.size(1) == L->H &&
                selected.scalar_type() == at::kLong && weights.scalar_type() == at::kHalf &&
                out.scalar_type() == at::kFloat && out.is_contiguous() && out.sizes() == x.sizes(),
                "exl3_moe_fp4_forward_variant: x fp16 [m, H], selected int64 / weights fp16 [m, topk], out fp32 [m, H]");
    TORCH_CHECK(cfg.size() == 7, "exl3_moe_fp4_forward_variant: cfg = [alloc, phases, spin, streams, pf, pfhint, amx]");
    VarCfg v;
    v.alloc = (int) cfg[0]; v.phases = (int) cfg[1]; v.spin = (int) cfg[2]; v.streams = (int) cfg[3];
    v.pf = (int) cfg[4]; v.pfhint = (int) cfg[5]; v.amx = (int) cfg[6];
    TORCH_CHECK((v.phases == 5 || v.phases == 3 || v.phases == 2) && (v.streams == 1 || v.streams == 2),
                "exl3_moe_fp4_forward_variant: phases 5/3/2, streams 1/2");
    std::vector<int32_t> sel(selected.numel());
    const int64_t* in = selected.data_ptr<int64_t>();
    for (size_t i = 0; i < sel.size(); ++i) sel[i] = in[i] < 0 ? -1 : (int32_t) in[i];
    forward_variant(L, v, x.data_ptr<at::Half>(), sel.data(), weights.data_ptr<at::Half>(), out.data_ptr<float>(),
                    (int) x.size(0), (int) selected.size(1), (int) num_threads);
}
