#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <string>
#include <vector>

// One-shot 2-rank P2P collectives for small payloads, one process per GPU (rank r = cuda:r).
//
// Each rank owns an arena (device memory exported by cudaIpc, or pinned shared host memory) laid out as:
//   [0, P2P_MAXB)          parity slot 0: payload pushed in by the peer
//   [P2P_MAXB, 2*MAXB)     parity slot 1
//   P2P_OFF_FLAGS          uint64 flags[2][P2P_MAXBLK]: flags[p][b] = last seq delivered to block b, parity p
//   P2P_OFF_STATE          uint64 state[P2P_MAXBLK]: per-block seq counters, advanced by the kernel itself
//   P2P_OFF_STATS          opt-in diagnostics (only written by the STATS=true kernel instantiation):
//                            uint64 tot[P2P_MAXBLK][P2P_STATS_TOT_FIELDS]     running totals per block
//                            P2PStatRec rec[P2P_MAXBLK][P2P_STATS_RING]       per-invocation ring, slot = (seq-1) % RING
// so a captured CUDA graph replays correctly (seq lives in device memory, not in the launch arguments).
//
// Protocol per collective (identical on both ranks, one kernel launch): push my payload into the peer's
// slot[p], fence.sys, release-store seq into the peer's flag[p][b]; acquire-spin on my own flag[p][b] >= seq;
// combine with the peer's payload. Parity p = seq & 1. Kernels on one stream never overlap, so a rank cannot
// overwrite the peer's slot[p] before the peer has finished the kernel that last read it.
//
// All collectives must be issued on the same stream, in the same order, with the same sizes on both ranks.

#ifndef P2P_MAXB
#define P2P_MAXB (4 * 1024 * 1024)
#endif
static_assert(P2P_MAXB > 0 && P2P_MAXB % 16 == 0, "P2P_MAXB must be a positive multiple of 16");
#define P2P_MAXBLK 8
#define P2P_OFF_FLAGS (2LL * P2P_MAXB)
#define P2P_OFF_STATE (2LL * P2P_MAXB + 256)
#define P2P_OFF_STATS (2LL * P2P_MAXB + 512)
#define P2P_STATS_RING 256
#define P2P_STATS_TOT_FIELDS 5  // calls, spins, push_ns, wait_ns, total_ns
#define P2P_STATS_REC_FIELDS 7  // seq, meta, spins, t_start, t_pushed, t_waited, t_end
#define P2P_OFF_STATS_RECS (P2P_OFF_STATS + P2P_MAXBLK * P2P_STATS_TOT_FIELDS * 8)
#define P2P_ARENA_BYTES (P2P_OFF_STATS_RECS + P2P_MAXBLK * P2P_STATS_RING * P2P_STATS_REC_FIELDS * 8)

// Diagnostic record layouts (all uint64, device-written by thread 0 of each block). Times are globaltimer ns.
// meta = mode | (gridDim.x << 8) | (units << 16)
struct P2PStatRec
{
    uint64_t seq, meta, spins, t_start, t_pushed, t_waited, t_end;
};

// Shared arena (transport c): /dev/shm object mapped by both processes, pinned + device-mapped for this GPU.
// Peer SM access is blocked in this container (IOMMU DMA-FQ + BAR1 256 MiB), so the exchange buffer lives in
// host memory both GPUs can access. Returns the device pointer for this process's GPU as int64.
int64_t p2p_shm_arena(const std::string& name, bool create);

// cudaIpc transport: the arena is cudaMalloc'd and zeroed on this GPU. Its IPC handle is exported as hex and
// opened by the peer process, which then writes into it over P2P. Requires peer access (see Python P2PComm).
int64_t p2p_dev_arena_alloc(int64_t device);
std::string p2p_dev_arena_handle(int64_t ptr);
int64_t p2p_dev_arena_open(const std::string& hex, int64_t local_device, int64_t peer_device);
// Compiled payload capacity in bytes per parity slot.
int64_t p2p_arena_maxb();


// fp32 sum over both ranks, in place: x <- x_rank0 + x_rank1. numel % 4 == 0, bytes <= P2P_MAXB.
// stats: record per-invocation device diagnostics into the local arena (opt-in; off = original kernel).
void p2p_all_reduce_f32(torch::Tensor x, int64_t rank, int64_t local, int64_t peer, int64_t nblk, bool stats);

// fp32 concat: out (numel 2*x.numel()) <- [x_rank0 | x_rank1]. out must not alias x.
void p2p_all_gather_f32(torch::Tensor x, torch::Tensor out, int64_t rank, int64_t local, int64_t peer, int64_t nblk,
                        bool stats);

// int64 broadcast from root: out <- x_root. numel even (16-byte units), bytes <= P2P_MAXB. out may alias x.
void p2p_bcast_i64(torch::Tensor x, torch::Tensor out, int64_t rank, int64_t root, int64_t local, int64_t peer,
                   int64_t nblk, bool stats);

// Diagnostics on the local arena. Both synchronize the device: call only after the collectives have drained.
// p2p_stats_read returns the raw uint64 region as int64: P2P_MAXBLK*P2P_STATS_TOT_FIELDS totals, then
// P2P_MAXBLK*P2P_STATS_RING*P2P_STATS_REC_FIELDS records (layout above). p2p_stats_reset zeroes it.
std::vector<int64_t> p2p_stats_read(int64_t local, int64_t device);
void p2p_stats_reset(int64_t local, int64_t device);
