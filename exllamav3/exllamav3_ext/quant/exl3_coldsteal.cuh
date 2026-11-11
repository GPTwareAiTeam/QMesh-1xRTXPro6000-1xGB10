#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <string>
#include <tuple>

// ColdSteal: the P7 computes GX-bound (cold) expert picks itself, inside its own routed launch, whenever the expert
// is resident in one of the layer's S steal slots in P7 VRAM.
//
// The steal slots are extra entries of the layer's hot coop pointer / rate tables (local ids n_hot .. n_hot + S - 1,
// plain trellis layout from n_hot on: MoeCoopParams.plain_from), so a stolen pick costs one more expert inside the
// launch P7 runs anyway: no second launch, no wait, no add.
//
// Per layer, per MoE call (compute stream, before the GX request is issued, graph-capturable):
//   coldsteal_select   rewrites every cold pick of `sel`: a pick whose expert is resident (slot_of[e] >= 0) and
//                      among the s_max most-picked resident experts of this call becomes local id n_hot + slot and
//                      is blanked (-1) in the worker selection `sel_cpu`; every other cold pick becomes inactive for
//                      the P7 launch (the worker computes it). It publishes into host-mapped memory the call's
//                      sequence number (`seen`, written before slot_of is read) and a record of the call's
//                      non-resident cold experts (misses) and resident ones (hits), most-picked first.
//
// Off the critical path, a native proxy thread with its own non-blocking stream owns slot contents:
//   * admission: the most-picked miss of a layer's latest record is copied from pinned host RAM into a free slot,
//     then its rates, then slot_of[e] = slot -- all in one stream, so a select can only map an expert once its bytes
//     landed. Nothing ever waits on a copy.
//   * eviction (LRU, never an expert the latest record hit): slot_of[old] = -1 lands first; the slot is reused only
//     once the layer's `seen` exceeds the value V read after that: only selects with sequence <= V can have mapped the
//     old expert, and select V + 1 follows launch V in the compute stream.
//
// Slot / pool row byte layout (fixed offsets, so a slot's table pointers never change):
//   [g_suh g_svh u_suh u_svh d_suh d_svh | g_trellis | u_trellis | d_trellis], trellis regions `tstride` apart.

#define CS_MAX_PICKS 384
#define CS_REC_N 8                  // misses / hits per record
#define CS_LAYER_WORDS 32           // mapped int64 words per layer: [0] seen, [8 .. 8 + CS_REC_WORDS) record
#define CS_REC_OFF 8
#define CS_REC_WORDS 20             // [0] seq, [1] n_miss, [2..9] miss, [10] n_hit, [11..18] hit, [19] seq
#define CS_INACTIVE ((int64_t) 0x3fffffff)

void coldsteal_select
(
    at::Tensor& sel,                            // (rows, topk) int64 device physical ids, rewritten in place
    const c10::optional<at::Tensor>& sel_cpu,   // (>= rows * topk) int64 device worker selection; none = mask only
    const at::Tensor& slot_of,                  // (E,) int32 device: slot of each physical expert, -1 = not resident
    at::Tensor& dev_seq,                        // (1,) int64 device call counter of this layer
    at::Tensor& stats,                          // (4,) int64 device: calls, cold picks, stolen picks, stolen experts
    int64_t words,                              // host-mapped CS_LAYER_WORDS block of this layer
    int64_t n_hot,
    int64_t s_max
);

// CPU tier split: moves the least-shared ceil(frac * distinct) (<= max_cpu) worker-bound experts of this call from
// sel_gx (set to -1 there) into sel_cpu (cold ids, -1 elsewhere). In place, current stream, graph-capturable.
void coldsteal_cpu_split(at::Tensor& sel_gx, at::Tensor& sel_cpu, double frac, int64_t max_cpu);

int64_t coldsteal_host_alloc(int64_t nbytes, bool mapped);     // pinned (mapped: zeroed, device-addressable)
void coldsteal_host_free(int64_t ptr);
at::Tensor coldsteal_host_read(int64_t ptr, int64_t n);        // CPU copy of the first n int64 words

// Split policy (notes/SplitPolicy.md). Per call, of the cold experts of this call (nd distinct, h of them resident in
// a steal slot): s = most-picked resident ones stolen into the P7 launch (VRAM slots), z = most-picked non-resident
// ones the P7 computes straight from the pinned pool (zero-copy PCIe reads; launch table entries n_hot + S .. are
// pointed at their pool rows by this kernel), c = least-picked remaining ones to the host CPU tier (sel_t), the rest
// to the GX worker. (s, c, z) from `table` (int8 [LOAD_L][ROW_B][N+1][H+1][3] then [ROW_B][2] thresholds on the call's
// distinct hot experts: level 0 if <= t0, 1 if <= t1, else 2; clamped to the feasible range and the caps)
// or, with no table, drawn uniformly from the feasible range by a hash of (call seq, layer, salt). Graph-capturable:
// table and launch tables are device memory. With `ring` != 0, one CsPolRec per call is written there (host-mapped)
// and t1 by coldsteal_policy_end at the end of the layer's MoE; `pub` (host-mapped int64) counts finished records.
#define CS_POL_ROW_B 12             // rows 1..6, 7-8, 9-12, 13-16, 17-24, 25-32, 33+
#define CS_POL_N 63
#define CS_POL_H 8
#define CS_POL_V 3                  // s, c, z
#define CS_POL_LOAD_L 3             // P7 hot load levels
#define CS_POL_TABLE_CELLS (CS_POL_LOAD_L * CS_POL_ROW_B * (CS_POL_N + 1) * (CS_POL_H + 1))
#define CS_POL_TABLE_BYTES (CS_POL_TABLE_CELLS * CS_POL_V + CS_POL_ROW_B * 2)
struct CsPolRec
{
    uint64_t t0, t1;                // globaltimer ns: select start, layer MoE end
    int16_t f[16];                  // layer, rows, nd, h, s, c, hot_picks, hot_distinct, cold_picks, stolen_picks,
                                    // cpu_picks, explore, z, zc_picks, 0, 0
};
void coldsteal_policy
(
    at::Tensor& sel, at::Tensor& sel_cpu, const c10::optional<at::Tensor>& sel_t, const at::Tensor& slot_of,
    at::Tensor& dev_seq, at::Tensor& stats, int64_t words, int64_t n_hot, int64_t layer, int64_t s_cap, int64_t c_cap,
    const c10::optional<at::Tensor>& table, int64_t salt, int64_t ring, int64_t ring_mask, int64_t ring_head,
    at::Tensor& layer_cur, int64_t z_cap, int64_t S, const c10::optional<at::Tensor>& zc_tables,
    const c10::optional<at::Tensor>& zc_k2, int64_t zc_pool, int64_t zc_pool_k2, const c10::optional<at::Tensor>& zc_off,
    int64_t zc_stride
);
void coldsteal_policy_end(const at::Tensor& layer_cur, int64_t ring, int64_t ring_mask, int64_t pub);

int64_t coldsteal_proxy_create(int64_t device, int64_t max_out);
void coldsteal_proxy_add_layer
(
    int64_t handle,
    int64_t words,                      // host-mapped block (seen + record)
    int64_t slot_of,                    // device int32 (E,)
    int64_t k2,                         // device uint8 (3, n_hot + S) rate tables: the slot entries are written
    int64_t k2_stride,                  // n_hot + S
    int64_t slots,                      // device slot bytes, S * stride
    int64_t pool,                       // pinned rows, (E - n_hot) * stride, row r = physical expert n_hot + r
    int64_t pool_k2,                    // pinned uint8 (E - n_hot): row rate (k2), 0 = row not loaded
    int64_t E, int64_t n_hot, int64_t S, int64_t stride, int64_t scale_bytes, int64_t tstride,
    int64_t tunit                       // trellis bytes per half-bit of rate: a row at k2 holds 3 trellises of k2 * tunit
);
void coldsteal_proxy_launch(int64_t handle);
void coldsteal_proxy_set_active(int64_t handle, int64_t active);   // growable KV: slots [active, S) retired
int64_t coldsteal_proxy_retired(int64_t handle);                   // active count once retired slots drained, else -1
std::tuple<int64_t, int64_t, int64_t, std::string> coldsteal_proxy_stats(int64_t handle);  // copies, evictions, records, failure
void coldsteal_proxy_destroy(int64_t handle);   // joins; call without the GIL
