// Shotgun variants of the MXFP4 forward (notes/CPUKernelShotgun.md). Included at the end of moe_fp4.cpp's anonymous
// namespace: shares the packed layout, Pool, tile_gemv_scalar and quant_row. One parameterized forward; each knob is one
// candidate idea, so a single harness sweeps them and their combinations. exl3_moe_fp4_forward_raw runs the stage-2
// winner config (alloc 1, phases 5, streams 2, or 1 when 16+ rows share an expert).
//
//   alloc    0 = per-call std::vector buffers (v1)        1 = persistent scratch, grown only (A1)
//   phases   5 = v1 (quant x | gate+up | act | down | reduce)
//            3 = quant x fused into gate+up (lazy per row, first owner quantizes), act fused into the up-tile epilogue
//                of the worker that finishes a slot's last gate/up tile (A2)
//            2 = A2 + down tiles write straight into out[] (one owner per (row, tile): no y buffer, no reduce) (A3)
//   spin     0 = v1 pool (spin 20k then park)    1 = spin-only for this call (A4)
//   streams  1 = v1 (one tile per item)          2 = an item is a pair of tiles (t, t + T/2), interleaved per 32-block (B1)
//   pf       prefetch distance in 256 B blocks (v1 8) (B2)
//   pfhint   0 = T0 (v1)  1 = T1 (L2) (B3)
//   amx      0 = VNNI always   R = AMX int8 tiles when a group has >= R rows (D1; AMX path: see tile_amx)

struct VarCfg { int alloc = 0, phases = 5, spin = 0, streams = 1, pf = 8, pfhint = 0, amx = 0; };

#if FP4_X86
// VNNI tile GEMV with tunable prefetch and 1 or 2 interleaved tiles (math: file header of moe_fp4.cpp).
template <int R, int NS>
FP4_TARGET
void tile_gemv_v(const Proj& P, const int* tiles, const RowPtrs& rp, int pf, int pfhint)
{
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m512i nib = _mm512_set1_epi8(0x0f);
    const uint8_t* w[NS];
    const uint8_t* s[NS];
    for (int q = 0; q < NS; ++q)
    {
        w[q] = P.w + (size_t) tiles[q] * P.kbn * BLK_BYTES;
        s[q] = P.s + (size_t) tiles[q] * P.kbn * TILE;
    }
    __m512 facc[NS][R];
    for (int q = 0; q < NS; ++q)
        for (int r = 0; r < R; ++r) facc[q][r] = _mm512_setzero_ps();
    const size_t pfo = (size_t) pf * BLK_BYTES;
    for (int kb = 0; kb < P.kbn; ++kb)
    {
        for (int q = 0; q < NS; ++q)
        {
            const char* p = (const char*) (w[q] + pfo);
            if (pfhint)
            {
                _mm_prefetch(p, _MM_HINT_T1); _mm_prefetch(p + 64, _MM_HINT_T1);
                _mm_prefetch(p + 128, _MM_HINT_T1); _mm_prefetch(p + 192, _MM_HINT_T1);
            }
            else
            {
                _mm_prefetch(p, _MM_HINT_T0); _mm_prefetch(p + 64, _MM_HINT_T0);
                _mm_prefetch(p + 128, _MM_HINT_T0); _mm_prefetch(p + 192, _MM_HINT_T0);
            }
        }
        __m512i iacc[NS][R];
        for (int q = 0; q < NS; ++q)
            for (int r = 0; r < R; ++r) iacc[q][r] = _mm512_setzero_si512();
#pragma GCC unroll 4
        for (int L = 0; L < 4; ++L)
        {
            const int o0 = kb * KB + 8 * L;
            __m512i a0v[R], a1v[R];
            for (int r = 0; r < R; ++r)
            {
                int32_t a0, a1;
                std::memcpy(&a0, rp.x8[r] + o0, 4);
                std::memcpy(&a1, rp.x8[r] + o0 + 4, 4);
                a0v[r] = _mm512_set1_epi32(a0); a1v[r] = _mm512_set1_epi32(a1);
            }
            for (int q = 0; q < NS; ++q)
            {
                const __m512i v = _mm512_loadu_si512((const void*) (w[q] + L * 64));
                const __m512i wl = _mm512_shuffle_epi8(lut, _mm512_and_si512(v, nib));
                const __m512i wh = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v, 4), nib));
                for (int r = 0; r < R; ++r)
                {
                    iacc[q][r] = _mm512_dpbusd_epi32(iacc[q][r], wl, a0v[r]);
                    iacc[q][r] = _mm512_dpbusd_epi32(iacc[q][r], wh, a1v[r]);
                }
            }
        }
        for (int q = 0; q < NS; ++q)
        {
            const __m512i e = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) s[q]));
            const __m512 sv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(_mm512_max_epi32(e, _mm512_set1_epi32(1)), _mm512_set1_epi32(1)), 23));
            for (int r = 0; r < R; ++r)
            {
                const __m512 f = _mm512_cvtepi32_ps(_mm512_sub_epi32(iacc[q][r], _mm512_set1_epi32(rp.corr[r][kb])));
                facc[q][r] = _mm512_fmadd_ps(f, _mm512_mul_ps(sv, _mm512_set1_ps(rp.qa[r][kb])), facc[q][r]);
            }
            w[q] += BLK_BYTES; s[q] += TILE;
        }
    }
    for (int q = 0; q < NS; ++q)
        for (int r = 0; r < R; ++r)
            _mm512_storeu_ps(rp.out[r] + tiles[q] * TILE, _mm512_mul_ps(facc[q][r], _mm512_set1_ps(rp.wscale[r])));
}

template <int NS>
void run_tiles_v(const Proj& P, const int* tiles, const RowPtrs& rp, int R, int pf, int pfhint)
{
    switch (R)
    {
        case 1: tile_gemv_v<1, NS>(P, tiles, rp, pf, pfhint); return;
        case 2: tile_gemv_v<2, NS>(P, tiles, rp, pf, pfhint); return;
        case 3: tile_gemv_v<3, NS>(P, tiles, rp, pf, pfhint); return;
        case 4: tile_gemv_v<4, NS>(P, tiles, rp, pf, pfhint); return;
        case 5: tile_gemv_v<5, NS>(P, tiles, rp, pf, pfhint); return;
        case 6: tile_gemv_v<6, NS>(P, tiles, rp, pf, pfhint); return;
        case 7: tile_gemv_v<7, NS>(P, tiles, rp, pf, pfhint); return;
        default: tile_gemv_v<8, NS>(P, tiles, rp, pf, pfhint); return;
    }
}

// ---- AMX int8 (D1) ----------------------------------------------------------------------------------------------
//
// One 16-row output tile x R rows (R <= 16). Per 64-input chunk (two scale blocks), the tile's nibbles are expanded
// into an int8 B tile in the VNNI tile layout (16 rows of K = 64 bytes as 16 x (4 x 16) quads: B[k/4][n*4 + k%4]), the
// activations are an A tile [R x 64] int8; tdpbssd accumulates int32 [R x 16]. Each 32-block keeps its own scale, so
// one chunk issues two tdpbssd into two accumulators (one per block, K = 32 each: A/B tiles with 32-byte rows) and the
// epilogue applies scale * qa per block. Weights are signed here (no +12 offset), so no corr term.

bool amx_ready()
{
    static int ok = -1;
    if (ok < 0)
    {
        ok = 0;
        unsigned a, b, cc, d;
        if (__get_cpuid_count(7, 0, &a, &b, &cc, &d) && (d & (1u << 25)) && (d & (1u << 24)))   // AMX-INT8, AMX-TILE
        {
            constexpr int ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18;
            ok = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0 ? 1 : 0;
        }
    }
    return ok == 1;
}

struct alignas(64) TileCfg
{
    uint8_t palette = 1, start = 0, rsv[14] = {};
    uint16_t colsb[16] = {};
    uint8_t rows[16] = {};
};

// Per-thread AMX permission is process-wide after arch_prctl; tile config is per thread: load it per call
__attribute__((target("amx-tile,amx-int8,avx512f,avx512bw,avx512vl,avx512vnni")))
void tile_amx(const Proj& P, int tile, const RowPtrs& rp, int R)
{
    alignas(64) int8_t A[16][32];           // R rows x 32 activations (one block)
    alignas(64) int8_t B[8][64];            // K = 32 as 8 quads x (16 cols x 4)
    alignas(64) int32_t C[16][16];
    TileCfg cfg;
    cfg.rows[0] = (uint8_t) R; cfg.colsb[0] = 64;       // tmm0: C [R x 16] int32
    cfg.rows[1] = (uint8_t) R; cfg.colsb[1] = 32;       // tmm1: A [R x 32] int8
    cfg.rows[2] = 8; cfg.colsb[2] = 64;                 // tmm2: B [8 quads x 64] int8
    _tile_loadconfig(&cfg);
    static const int8_t lut[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
    const __m512i lutv = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) lut));
    const __m512i nib = _mm512_set1_epi8(0x0f);
    const uint8_t* w = P.w + (size_t) tile * P.kbn * BLK_BYTES;
    const uint8_t* s = P.s + (size_t) tile * P.kbn * TILE;
    __m512 facc[16];
    for (int r = 0; r < R; ++r) facc[r] = _mm512_setzero_ps();
    for (int kb = 0; kb < P.kbn; ++kb, w += BLK_BYTES, s += TILE)
    {
        _mm_prefetch((const char*) (w + 8 * BLK_BYTES), _MM_HINT_T0);
        _mm_prefetch((const char*) (w + 8 * BLK_BYTES + 64), _MM_HINT_T0);
        _mm_prefetch((const char*) (w + 8 * BLK_BYTES + 128), _MM_HINT_T0);
        _mm_prefetch((const char*) (w + 8 * BLK_BYTES + 192), _MM_HINT_T0);
        // Packed load L: low nibbles = step 2L (inputs 8L + j), high = step 2L + 1 (inputs 8L + 4 + j), byte o*4 + j.
        // VNNI B layout quad q covers inputs 4q..4q+3: quad 2L = low nibbles of load L, quad 2L + 1 = high nibbles.
        for (int L = 0; L < 4; ++L)
        {
            const __m512i v = _mm512_loadu_si512((const void*) (w + L * 64));
            _mm512_store_si512((void*) B[2 * L], _mm512_shuffle_epi8(lutv, _mm512_and_si512(v, nib)));
            _mm512_store_si512((void*) B[2 * L + 1], _mm512_shuffle_epi8(lutv, _mm512_and_si512(_mm512_srli_epi16(v, 4), nib)));
        }
        for (int r = 0; r < R; ++r) std::memcpy(A[r], rp.x8[r] + kb * KB, 32);
        _tile_zero(0);
        _tile_loadd(1, A, 32);
        _tile_loadd(2, B, 64);
        _tile_dpbssd(0, 1, 2);
        _tile_stored(0, C, 64);
        const __m512i e = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) s));
        const __m512 sv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(_mm512_max_epi32(e, _mm512_set1_epi32(1)), _mm512_set1_epi32(1)), 23));
        for (int r = 0; r < R; ++r)
        {
            const __m512 f = _mm512_cvtepi32_ps(_mm512_load_si512((const void*) C[r]));
            facc[r] = _mm512_fmadd_ps(f, _mm512_mul_ps(sv, _mm512_set1_ps(rp.qa[r][kb])), facc[r]);
        }
    }
    _tile_release();
    for (int r = 0; r < R; ++r)
        _mm512_storeu_ps(rp.out[r] + tile * TILE, _mm512_mul_ps(facc[r], _mm512_set1_ps(rp.wscale[r])));
}
#endif

// Signed-weight AMX path has no +12 offset: the shared quant_row writes corr = 12 * sum, unused there.

struct VCtx
{
    const Layer* L = nullptr;
    VarCfg cfg;
    int m = 0, topk = 0, H = 0, I = 0, S = 0;
    const at::Half* x = nullptr;
    float* out = nullptr;
    std::vector<int> slot_row;
    std::vector<float> slot_w;
    std::vector<Group> groups;
    std::vector<std::vector<int>> row_slots;
    std::vector<int8_t> x8; std::vector<float> xqa; std::vector<int32_t> xcorr;
    std::vector<float> gu;
    std::vector<int8_t> a8; std::vector<float> aqa; std::vector<int32_t> acorr;
    std::vector<float> y;
    std::vector<int> item_group, item_tile;
    std::unique_ptr<std::atomic<int>[]> row_state;      // phases <= 3: 0 todo, 1 in progress, 2 quantized
    std::unique_ptr<std::atomic<int>[]> slot_left;      // phases <= 3: gate/up tile items left per slot group
    size_t cap_rows = 0, cap_groups = 0;
    std::atomic<int> next { 0 };
};

VCtx g_vctx;

inline void v_quant_row_x(VCtx& c, int r)
{
    alignas(64) float tmp[8192];
    for (int i = 0; i < c.H; ++i) tmp[i] = (float) c.x[(size_t) r * c.H + i];
    quant_row(tmp, c.H, &c.x8[(size_t) r * c.H], &c.xqa[(size_t) r * (c.H / KB)], &c.xcorr[(size_t) r * (c.H / KB)]);
}

// Lazily quantize row r (phases <= 3): first claimant quantizes, others wait for it
inline void v_need_row(VCtx& c, int r)
{
    auto& st = c.row_state[r];
    int z = 0;
    if (st.load(std::memory_order_acquire) == 2) return;
    if (st.compare_exchange_strong(z, 1, std::memory_order_acq_rel))
    {
        v_quant_row_x(c, r);
        st.store(2, std::memory_order_release);
        return;
    }
    while (st.load(std::memory_order_acquire) != 2) FP4_RELAX();
}

inline void v_act_slot(VCtx& c, int s)
{
    alignas(64) float a[8192];
    const float* g = &c.gu[(size_t) s * 2 * c.I];
    const float* u = g + c.I;
    const float lim = c.L->act_limit;
    for (int i = 0; i < c.I; ++i)
    {
        float xg = g[i] / (1.f + std::exp(-g[i]));
        float v = u[i];
        if (lim != 0.f) { v = std::min(std::max(v, -lim), lim); xg = std::min(xg, lim); }
        a[i] = xg * v;
    }
    quant_row(a, c.I, &c.a8[(size_t) s * c.I], &c.aqa[(size_t) s * (c.I / KB)], &c.acorr[(size_t) s * (c.I / KB)]);
}

void v_run_tiles(VCtx& c, const Proj& P, const int* tiles, int nt, const RowPtrs& rp, int R)
{
#if FP4_X86
    if (has_vnni())
    {
        if (c.cfg.amx && R >= c.cfg.amx && amx_ready())
        {
            for (int q = 0; q < nt; ++q) tile_amx(P, tiles[q], rp, R);
            return;
        }
        if (nt == 2) run_tiles_v<2>(P, tiles, rp, R, c.cfg.pf, c.cfg.pfhint);
        else run_tiles_v<1>(P, tiles, rp, R, c.cfg.pf, c.cfg.pfhint);
        return;
    }
#endif
    for (int q = 0; q < nt; ++q) tile_gemv_scalar(P, tiles[q], rp, R);
}

// Items: (group, unit) with unit = a tile or a tile pair (streams 2: t and t + half)
inline int v_units(int tiles, int streams) { return streams == 2 ? (tiles + 1) / 2 : tiles; }
inline int v_unit_tiles(int u, int tiles, int streams, int* out)
{
    if (streams == 2)
    {
        const int half = (tiles + 1) / 2;
        out[0] = u;
        if (u + half < tiles) { out[1] = u + half; return 2; }
        return 1;
    }
    out[0] = u;
    return 1;
}

void vphase_quant_x(void* vc, int)
{
    VCtx& c = *(VCtx*) vc;
    for (int r; (r = c.next.fetch_add(1)) < c.m; ) v_quant_row_x(c, r);
}

// gate+up items: for each group, units of the gate projection then the up projection
void vphase_gate_up(void* vc, int)
{
    VCtx& c = *(VCtx*) vc;
    const int n = (int) c.item_group.size();
    const bool fused = c.cfg.phases <= 3;
    for (int it; (it = c.next.fetch_add(1)) < n; )
    {
        const int gi = c.item_group[it];
        const Group& G = c.groups[gi];
        const Expert& E = c.L->experts[G.expert];
        const int units = v_units(E.g.tiles, c.cfg.streams);
        const int uidx = c.item_tile[it];
        const bool up = uidx >= units;
        const Proj& P = up ? E.u : E.g;
        int tl[2];
        const int nt = v_unit_tiles(up ? uidx - units : uidx, P.tiles, c.cfg.streams, tl);
        for (int r0 = 0; r0 < G.count; r0 += RB)
        {
            const int R = std::min(RB, G.count - r0);
            RowPtrs rp;
            for (int r = 0; r < R; ++r)
            {
                const int slot = G.first + r0 + r;
                const int row = c.slot_row[slot];
                if (fused) v_need_row(c, row);
                rp.x8[r] = &c.x8[(size_t) row * c.H];
                rp.qa[r] = &c.xqa[(size_t) row * (c.H / KB)];
                rp.corr[r] = &c.xcorr[(size_t) row * (c.H / KB)];
                rp.out[r] = &c.gu[((size_t) slot * 2 + (up ? 1 : 0)) * c.I];
                rp.wscale[r] = 1.f;
            }
            v_run_tiles(c, P, tl, nt, rp, R);
        }
        if (fused && c.slot_left[gi].fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            // Last gate/up unit of this group: its slots' gate and up outputs are complete
            std::atomic_thread_fence(std::memory_order_acquire);
            for (int k = 0; k < G.count; ++k) v_act_slot(c, G.first + k);
        }
    }
}

void vphase_act(void* vc, int)
{
    VCtx& c = *(VCtx*) vc;
    for (int s; (s = c.next.fetch_add(1)) < c.S; ) v_act_slot(c, s);
}

// down: phases 5/3 write y[slot] then reduce; phases 2 own (row, unit) and sum every slot of the row in registers
void vphase_down(void* vc, int)
{
    VCtx& c = *(VCtx*) vc;
    const int n = (int) c.item_group.size();
    for (int it; (it = c.next.fetch_add(1)) < n; )
    {
        const Group& G = c.groups[c.item_group[it]];
        const Proj& P = c.L->experts[G.expert].d;
        int tl[2];
        const int nt = v_unit_tiles(c.item_tile[it], P.tiles, c.cfg.streams, tl);
        for (int r0 = 0; r0 < G.count; r0 += RB)
        {
            const int R = std::min(RB, G.count - r0);
            RowPtrs rp;
            for (int r = 0; r < R; ++r)
            {
                const int slot = G.first + r0 + r;
                rp.x8[r] = &c.a8[(size_t) slot * c.I];
                rp.qa[r] = &c.aqa[(size_t) slot * (c.I / KB)];
                rp.corr[r] = &c.acorr[(size_t) slot * (c.I / KB)];
                rp.out[r] = &c.y[(size_t) slot * c.H];
                rp.wscale[r] = c.slot_w[slot];
            }
            v_run_tiles(c, P, tl, nt, rp, R);
        }
    }
}

// phases 2: item = (row, unit); for each slot of the row compute its down tile(s) into a stack buffer, sum, store
void vphase_down_direct(void* vc, int)
{
    VCtx& c = *(VCtx*) vc;
    const int dunits = v_units(c.H / TILE, c.cfg.streams);
    const int n = c.m * dunits;
    for (int it; (it = c.next.fetch_add(1)) < n; )
    {
        const int row = it / dunits, u = it % dunits;
        int tl[2];
        const int nt = v_unit_tiles(u, c.H / TILE, c.cfg.streams, tl);
        alignas(64) float acc[2][TILE] = {};
        alignas(64) float tmp[2 * TILE * 1];
        for (int s : c.row_slots[row])
        {
            int e = -1;
            // slot -> its group's expert (groups are contiguous slot ranges)
            for (const Group& G : c.groups) if (s >= G.first && s < G.first + G.count) { e = G.expert; break; }
            const Proj& D = c.L->experts[e].d;
            RowPtrs rp;
            rp.x8[0] = &c.a8[(size_t) s * c.I];
            rp.qa[0] = &c.aqa[(size_t) s * (c.I / KB)];
            rp.corr[0] = &c.acorr[(size_t) s * (c.I / KB)];
            // tile_gemv_v stores at out + tile * TILE: point out so that it lands in tmp
            for (int q = 0; q < nt; ++q)
            {
                int one = tl[q];
                rp.out[0] = tmp - (size_t) one * TILE;
                rp.wscale[0] = c.slot_w[s];
                v_run_tiles(c, D, &one, 1, rp, 1);
                for (int o = 0; o < TILE; ++o) acc[q][o] += tmp[o];
            }
        }
        for (int q = 0; q < nt; ++q)
            std::memcpy(c.out + (size_t) row * c.H + (size_t) tl[q] * TILE, acc[q], TILE * sizeof(float));
    }
}

void vphase_reduce(void* vc, int)
{
    VCtx& c = *(VCtx*) vc;
    constexpr int CH = 1024;
    const int chunks = (c.H + CH - 1) / CH;
    for (int it; (it = c.next.fetch_add(1)) < c.m * chunks; )
    {
        const int r = it / chunks, o0 = (it % chunks) * CH, o1 = std::min(c.H, o0 + CH);
        float* dst = c.out + (size_t) r * c.H;
        std::fill(dst + o0, dst + o1, 0.f);
        for (int s : c.row_slots[r])
        {
            const float* src = &c.y[(size_t) s * c.H];
            for (int o = o0; o < o1; ++o) dst[o] += src[o];
        }
    }
}

template <class T> inline void v_size(std::vector<T>& v, size_t n, bool persist)
{
    if (persist) { if (v.size() < n) v.resize(n); }
    else { std::vector<T>().swap(v); v.resize(n); }
}

void forward_variant(const Layer* L, const VarCfg& cfg, const at::Half* x, const int32_t* selected,
                     const at::Half* weights, float* out, int rows, int topk, int threads)
{
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    VCtx& c = g_vctx;
    c.L = L; c.cfg = cfg; c.m = rows; c.topk = topk; c.H = L->H; c.I = L->I; c.x = x; c.out = out;
    const int E = (int) L->experts.size();
    c.slot_row.clear(); c.slot_w.clear(); c.groups.clear();
    c.row_slots.assign(rows, {});
    // Group slots by expert (stable in row order), without an E-sized vector of vectors
    std::vector<int> order(rows * topk);
    int n_sel = 0;
    for (int i = 0; i < rows * topk; ++i) if (selected[i] >= 0) { TORCH_CHECK(selected[i] < E, "expert id"); order[n_sel++] = i; }
    std::stable_sort(order.begin(), order.begin() + n_sel, [&](int a, int b) { return selected[a] < selected[b]; });
    for (int k = 0; k < n_sel; ++k)
    {
        const int i = order[k], e = selected[i], r = i / topk;
        if (c.groups.empty() || c.groups.back().expert != e) c.groups.push_back({ e, (int) c.slot_row.size(), 0 });
        c.groups.back().count++;
        c.row_slots[r].push_back((int) c.slot_row.size());
        c.slot_row.push_back(r);
        c.slot_w.push_back((float) weights[i]);
    }
    c.S = (int) c.slot_row.size();
    if (!c.S) { std::fill(out, out + (size_t) rows * c.H, 0.f); return; }
    const bool persist = cfg.alloc == 1;
    v_size(c.x8, (size_t) rows * c.H, persist); v_size(c.xqa, (size_t) rows * c.H / KB, persist);
    v_size(c.xcorr, (size_t) rows * c.H / KB, persist);
    v_size(c.gu, (size_t) c.S * 2 * c.I, persist);
    v_size(c.a8, (size_t) c.S * c.I, persist); v_size(c.aqa, (size_t) c.S * c.I / KB, persist);
    v_size(c.acorr, (size_t) c.S * c.I / KB, persist);
    if (cfg.phases != 2) v_size(c.y, (size_t) c.S * c.H, persist);
    if (c.cap_rows < (size_t) rows) { c.row_state.reset(new std::atomic<int>[rows]); c.cap_rows = rows; }
    if (c.cap_groups < c.groups.size()) { c.slot_left.reset(new std::atomic<int>[c.groups.size()]); c.cap_groups = c.groups.size(); }
    for (int r = 0; r < rows; ++r) c.row_state[r].store(0, std::memory_order_relaxed);

    const int nt = std::max(1, threads);
    g_pool.ensure(nt);
    const int saved_spin = g_pool.spin_limit;
    if (cfg.spin) g_pool.spin_limit = 1 << 30;

    const int gunits = v_units(c.I / TILE, cfg.streams);
    c.item_group.clear(); c.item_tile.clear();
    for (int gi = 0; gi < (int) c.groups.size(); ++gi)
    {
        c.slot_left[gi].store(2 * gunits, std::memory_order_relaxed);
        for (int t = 0; t < 2 * gunits; ++t) { c.item_group.push_back(gi); c.item_tile.push_back(t); }
    }
    if (cfg.phases == 5) { c.next = 0; g_pool.run(vphase_quant_x, &c, nt); }
    c.next = 0; g_pool.run(vphase_gate_up, &c, nt);
    if (cfg.phases == 5) { c.next = 0; g_pool.run(vphase_act, &c, nt); }
    if (cfg.phases == 2)
    {
        c.next = 0; g_pool.run(vphase_down_direct, &c, nt);
    }
    else
    {
        const int dunits = v_units(c.H / TILE, cfg.streams);
        c.item_group.clear(); c.item_tile.clear();
        for (int gi = 0; gi < (int) c.groups.size(); ++gi)
            for (int t = 0; t < dunits; ++t) { c.item_group.push_back(gi); c.item_tile.push_back(t); }
        c.next = 0; g_pool.run(vphase_down, &c, nt);
        c.next = 0; g_pool.run(vphase_reduce, &c, nt);
    }
    g_pool.spin_limit = saved_spin;
}
