/* qmesh_moe_cpp: the GX cold-expert forward with no Python on the request path (see the header).
 *
 * The sequence here is qmesh/exl3_moe_cuda.CudaExl3Moe.forward, step for step, with three
 * differences that are the point of the file:
 *
 *  - the request's x/ids/w are copied straight out of the pinned slab into persistent device
 *    buffers (cudaMemcpyAsync), and the answer is copied straight back into the pinned response
 *    slot. No numpy view, no .copy(), no .to(), no .cpu().numpy(), no tensor per element.
 *  - every buffer the kernels write is allocated once and reused (monotone grow-on-demand), so a
 *    served call runs no allocator code at all, and the bf16->half / weight conversions that
 *    `.half()` and `.float()` did are two kernels.
 *  - the alignment (align_block) is two kernels plus one small pinned count vector instead of
 *    sort/bincount/cumsum/repeat_interleave with two host read-backs.
 *
 * The math is the vendored cuda-exl3 kernels, called through the torch dispatcher exactly as the
 * Python path called them (`cuda_exl3_C::exl3_moe_*`), so a change in the vendor extension reaches
 * both paths at once. Numerics differ from the Python path in the epilogue of the down projection:
 * by default the routing weight is folded into that gemm's fp32 atomics, so the answer depends on
 * the order the blocks retire in and on block_m, and it is neither reproducible run to run nor
 * comparable per layer with the Python path's. SEAM_MOE_CPP_DET=1 runs the Python path's
 * deterministic branch instead (`exl3_moe_combine` into a half, one fp32 add per group), which is
 * the mode a byte comparison between the two paths is made in.
 *
 * The piece ladder is duplicated from worker_seam_server._pick_step, with the same constants and
 * the same SEAM_WS_* environment: the C hook is installed *instead of* the Python _forward, so it
 * has to make the same piece decision or the two paths would not be comparable. Keep them in
 * sync; the SEAM_WS_PICK line it prints is the same one, so a run that picks differently from the
 * Python path is visible in the log.
 */
#include "qmesh_moe_cpp.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int QMC_NL = 40;
constexpr int QMC_MAX_ROWS = 65536;
constexpr int QMC_MAX_EXPERTS = 384; /* the model's 384 routed experts: the per-group stride of the
                                      * count/offset vectors, and the padding bound of an alignment */
constexpr int kBlockMTiers[4] = {16, 32, 64, 128}; /* exl3_moe_cuda.BLOCK_M_TIERS */
constexpr int kBlockMRows = 24;                   /* exl3_moe_cuda.BLOCK_M_ROWS */
/* A bounds-only alignment may pad up to min(n_experts, m_valid) blocks, i.e. one block per pick's
 * worth of experts. That is free while m_valid is small; at rows*topk in the thousands it would be
 * hundreds of blocks nobody asked for, so the readback stays for those pieces (the prefill path,
 * where the sync is 5 % of a 54 ms call and the padding would not be). */
constexpr int kBoundMaxRows = 64;
constexpr int kRefuseMem = 4;
constexpr int kRefuseCompute = 5;

/* -- prof marks (SEAM_QMC_PROF) ------------------------------------------------------------------
 * A piece is bracketed by four marks -- entry, after the request's H2D copies, after the two
 * conversions, after the histogram/alignment (which ends at the plan readback's sync) -- and each
 * group by five more: the end of its scratch prep, the Hadamard, the w13 GEMM, the glu, the down
 * GEMM. A last mark after the response copy closes the piece. That splits the call into the spans
 * the Python path had (copy, quant, align, had, gemms, write) instead of folding the Hadamard and
 * the alignment into "the GEMMs", which is the question: how much of the seam window is device
 * work and how much is the host leaving the stream empty. Events exist only when the flag is on,
 * and only for the first kProfPieces pieces of a request (the SUM line says how many were timed).
 * Indices: the four piece marks, then kProfPerGroup marks per group, then the write mark. */
constexpr int kProfEntry = 0; /* run_piece entry */
constexpr int kProfCopy = 1;  /* after the three H2D copies of the request slot */
constexpr int kProfQuant = 2; /* after bf16->half and the routing weights */
constexpr int kProfAlign = 3; /* after hist/align and, off bound mode, the plan readback sync */
constexpr int kProfGBase = 4;
/* Per group, in index order: end of prep (fill/scatter), after had, after w13, after glu, after
 * down. The prep span runs from the previous boundary (the align mark for group 0), so it also
 * holds whatever host work came between the two. */
constexpr int kProfPerGroup = 5;
constexpr int kProfPerPiece = kProfGBase + kProfPerGroup * QMC_MAX_GROUPS + 1;
constexpr int kProfWrite = kProfPerPiece - 1; /* after the response D2H */
constexpr int kProfPieces = 16;

constexpr int prof_gi(int g, int k) { return kProfGBase + kProfPerGroup * g + k; }

thread_local std::string g_err;

void set_err(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void set_err(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_err = buf;
}

int env_int(const char* name, int fallback) {
  const char* v = getenv(name);
  if (!v || !*v) return fallback;
  return atoi(v);
}

/* -- /proc readings the piece picker and the memory refusal share with worker_seam_server -------- */
long meminfo_kib(const char* key) {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return -1;
  char line[256];
  const size_t klen = strlen(key);
  long out = -1;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, key, klen) == 0 && line[klen] == ':') {
      out = atol(line + klen + 1);
      break;
    }
  }
  fclose(f);
  return out;
}

int avail_mib() {
  const long kib = meminfo_kib("MemAvailable");
  return kib < 0 ? -1 : (int)(kib / 1024);
}

int swap_used_mib() {
  const long tot = meminfo_kib("SwapTotal"), fre = meminfo_kib("SwapFree");
  if (tot < 0 || fre < 0) return -1;
  return (int)((tot - fre) / 1024);
}

/* -- dispatcher ---------------------------------------------------------------------------------
 * Boxed calls keep this file independent of the extension's generated headers: the schema is the
 * one cuda_exl3 registered, and a signature drift fails loudly here instead of silently. */
c10::OperatorHandle op_handle(const char* name) {
  static std::mutex m;
  static std::unordered_map<std::string, c10::OperatorHandle> cache;
  std::lock_guard<std::mutex> lk(m);
  auto it = cache.find(name);
  if (it != cache.end()) return it->second;
  auto h = c10::Dispatcher::singleton().findSchemaOrThrow(name, "");
  cache.emplace(name, h);
  return h;
}

at::Tensor dev_tensor(void* p, at::IntArrayRef sizes, at::ScalarType dt, int dev) {
  return at::from_blob(p, sizes, at::TensorOptions().dtype(dt).device(at::kCUDA, dev));
}

/* Device-usable address of a request/response buffer handed to qmc_compute. On the served path
 * these are the seam's pinned slab (cudaHostAlloc mapped, qmesh_expert_seam.cu), whose device alias
 * cudaPointerGetAttributes returns; the benches pass device memory, used as is. Pageable host memory
 * is used directly only where the device says it can read it (GB10: coherent, pageableMemoryAccess);
 * anywhere else it is null and the call refuses rather than reading through a bad pointer. */
void* dev_alias(const void* p, int dev) {
  cudaPointerAttributes a{};
  if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
    cudaGetLastError();
    a.type = cudaMemoryTypeUnregistered;
  }
  if (a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged) return const_cast<void*>(p);
  if (a.type == cudaMemoryTypeHost) return a.devicePointer;
  int pageable = 0;
  cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev);
  return pageable ? const_cast<void*>(p) : nullptr;
}

c10::IValue iv_opt(const at::Tensor& t, bool has) { return has ? c10::IValue(t) : c10::IValue(); }
c10::IValue iv_i64(int64_t v) { return c10::IValue(v); }

void moe_had_in(const at::Tensor& x, const at::Tensor& out, const at::Tensor& suh,
                const at::Tensor& sorted_ids, const at::Tensor& expert_ids,
                const at::Tensor& n_rows, int block_m, int top_k, int m_valid) {
  std::vector<c10::IValue> a{c10::IValue(x), c10::IValue(out), c10::IValue(suh),
                             c10::IValue(sorted_ids), c10::IValue(expert_ids),
                             c10::IValue(n_rows), iv_i64(block_m), iv_i64(top_k), iv_i64(m_valid),
                             c10::IValue(true)};
  op_handle("cuda_exl3_C::exl3_moe_had_in").callBoxed(a);
}

void moe_glu(const at::Tensor& x, const at::Tensor& out, const at::Tensor& suh,
             const at::Tensor& expert_ids, const at::Tensor& n_rows, int block_m) {
  std::vector<c10::IValue> a{c10::IValue(x), c10::IValue(out), c10::IValue(suh),
                             c10::IValue(expert_ids), c10::IValue(n_rows), iv_i64(block_m)};
  op_handle("cuda_exl3_C::exl3_moe_glu_had_in").callBoxed(a);
}

at::Tensor moe_gemm(const at::Tensor& a_had, const at::Tensor& trellis, const at::Tensor& suh,
                    const at::Tensor& svh, const at::Tensor& expert_ids, const at::Tensor& n_rows,
                    const std::vector<int64_t>& group_n, int cb, int block_m, at::ScalarType dt,
                    const at::Tensor& sorted_ids, bool have_sids, const at::Tensor& topk_w,
                    bool have_w, int num_tokens, int top_k, const at::Tensor& out, bool have_out,
                    bool accumulate) {
  std::vector<c10::IValue> a{
      c10::IValue(a_had),
      c10::IValue(trellis),
      c10::IValue(suh),
      c10::IValue(svh),
      c10::IValue(expert_ids),
      c10::IValue(n_rows),
      c10::IValue(c10::List<int64_t>(group_n)),
      iv_i64(cb),
      iv_i64(block_m),
      c10::IValue(dt),
      iv_opt(sorted_ids, have_sids),
      iv_opt(topk_w, have_w),
      iv_i64(num_tokens),
      iv_i64(top_k),
      iv_opt(out, have_out),
      c10::IValue(accumulate)};
  // Boxed convention: the kernel drops its inputs off the stack and pushes its outputs, so after
  // the call the stack holds exactly the single returned tensor.
  op_handle("cuda_exl3_C::exl3_moe_gemm").callBoxed(a);
  return a.at(0).toTensor();
}

/* exl3_moe_combine(rows_out, sorted_ids, topk_weights, num_tokens, expert_ids, block_m): the
 * vendor's pick-order gather, one half output per (token, hidden) - the Python path's
 * deterministic combine, called instead of re-implemented so the two paths cannot drift. */
at::Tensor moe_combine(const at::Tensor& rows_out, const at::Tensor& sorted_ids,
                       const at::Tensor& topk_w, int num_tokens, const at::Tensor& expert_ids,
                       int block_m) {
  std::vector<c10::IValue> a{c10::IValue(rows_out), c10::IValue(sorted_ids),
                             c10::IValue(topk_w),    iv_i64(num_tokens),
                             c10::IValue(expert_ids), iv_i64(block_m)};
  op_handle("cuda_exl3_C::exl3_moe_combine").callBoxed(a);
  return a.at(0).toTensor();
}

/* exl3_moe_cuda.pick_block_m, including CUDA_EXL3_MOE_BLOCK_M: first minimum wins, as min(..., key=) */
int pick_block_m(int live, int n_experts) {
  static const int override_bm = env_int("CUDA_EXL3_MOE_BLOCK_M", 0);
  const double per = (double)live / (n_experts > 0 ? n_experts : 1);
  int best = kBlockMTiers[0];
  double best_cost = 1e300;
  for (int b : kBlockMTiers) {
    const double q = std::ceil(per / b);
    const double cost = q * b + kBlockMRows * q;
    if (cost < best_cost) {
      best_cost = cost;
      best = b;
    }
  }
  if (override_bm) best = override_bm;
  return best;
}

struct Group {
  qmc_group g{};
  at::Tensor t13, suh13, svh13, t2, suh2, svh2; /* registered once, alive for the process */
  int last_live = 0; /* live picks of the previous call: block_m for this one, no sync needed */
};

struct Layer {
  bool have = false;
  int n_groups = 0;
  Group grp[QMC_MAX_GROUPS];
  /* The alignment of the last piece served for this layer, for qmc_dump_align: only meaningful
   * right after a call, and only filled when the dump is on. */
  int dump_rows = 0;
  int dump_bm[QMC_MAX_GROUPS]{}, dump_nblk[QMC_MAX_GROUPS]{}, dump_total[QMC_MAX_GROUPS]{};
  /* SEAM_MOE_CPP_LAYER: the whole layer's descriptor for qmc_fused_layer, built at registration.
   * desc_ok is false when any group is outside what the layer kernels take (a fractional tile, a
   * group wider than 384 experts, H or I off the 128 grid, groups with different I); such a layer
   * stays on the per-group path. */
  qmc_layer_desc desc{};
  bool desc_ok = false;
};

/* Every device buffer a served call touches, allocated once and grown on demand. */
struct Buffers {
  int cap_rows = 0;
  int cap_sorted = 0;
  int cap_pad = 0; /* rows of the a13/inter/a2 trio currently allocated */
  int n_groups = 0;
  int n_cold_max = 0;

  /* No device staging of the request's x: the bf16->half kernel reads it straight from the pinned
   * slab (its device alias, see dev_alias), so x_h is the only rows*H*2 activation buffer. */
  void* x_h = nullptr;
  int32_t* ids = nullptr;
  uint16_t* w_bf = nullptr;
  float* w_f = nullptr;

  int32_t* counts = nullptr;   /* [n_groups * QMC_MAX_EXPERTS] */
  int32_t* cursor = nullptr;   /* [n_groups * QMC_MAX_EXPERTS] */
  int32_t* blk_off = nullptr;  /* [n_groups * QMC_MAX_EXPERTS] */
  int32_t* bad = nullptr;      /* [1] */
  int32_t* n_rows = nullptr;   /* [n_groups] */
  int32_t* nblk = nullptr;     /* [n_groups]: blocks per group, written by qmc_k_align */
  int32_t* live = nullptr;     /* [n_groups]: live picks per group, written by qmc_k_align */
  int32_t* sorted = nullptr;   /* [cap_sorted] */
  int32_t* eids = nullptr;     /* [n_groups * cap_sorted]: per-block expert id, one slot per group */

  void* a13 = nullptr;   /* half [2, cap_pad, H] */
  void* inter = nullptr; /* half [cap_pad, 2I] */
  void* a2 = nullptr;    /* half [1, cap_pad, I] */
  float* acc = nullptr; /* float [cap_rows, H] (two parity slots under SEAM_MOE_CPP_SIDE) */
  /* No out16: a 16-bit reply is rounded by one kernel straight into the pinned response slot. */

  int32_t* h_plan = nullptr; /* pinned [2 * n_groups]: nblk then live per group, one readback */
  int32_t* h_bad = nullptr;  /* pinned [1] */
  void* blk = nullptr;       /* int2 [cap_blk]: the layer path's (group, expert) per block */
  int cap_blk = 0;
  /* The layer path's deterministic combine: one fp32 slot per pick and its live flag
   * (qmc_fused_layer). Sized for cap_picks picks, grown with the piece. */
  int32_t* pick_live = nullptr; /* [cap_picks] */
  float* slots = nullptr;       /* [cap_picks, H] */
  int cap_picks = 0;
  int32_t* h_sorted = nullptr; /* pinned [n_groups * cap_sorted], only with SEAM_MOE_CPP_DUMP */
  int32_t* h_eids = nullptr;   /* pinned [n_groups * cap_sorted], only with SEAM_MOE_CPP_DUMP */

  ~Buffers() {
    auto cf = [](void*& p) {
      if (p) {
        cudaFree(p);
        p = nullptr;
      }
    };
    cf(x_h);
    cf((void*&)ids);
    cf((void*&)w_bf);
    cf((void*&)w_f);
    cf((void*&)counts);
    cf((void*&)cursor);
    cf((void*&)blk_off);
    cf((void*&)bad);
    cf((void*&)n_rows);
    cf((void*&)nblk);
    cf((void*&)live);
    cf((void*&)pick_live);
    cf((void*&)slots);
    cf(blk);
    cf((void*&)eids);
    cf(a13);
    cf(inter);
    cf(a2);
    cf((void*&)acc);
    auto ch = [](void*& p) {
      if (p) {
        cudaFreeHost(p);
        p = nullptr;
      }
    };
    ch((void*&)h_plan);
    ch((void*&)h_bad);
    ch((void*&)h_sorted);
    ch((void*&)h_eids);
  }
};

struct Ctx {
  int dev = 0;
  int hidden = 0;
  int topk = 0;
  int ws_rows = 0;
  bool ws_auto = true;
  int ws_guard = 96;
  int ws_hyst = 128;
  int swap_warn = 512;
  int fwd_floor = 0;
  int out_mode = 0; /* 0 = fp32 response slot, 1 = bf16, 2 = fp16 (SEAM_MOE_CPP_OUT) */
  int block_m = 0;  /* 0 = pick_block_m from the previous call's live count (see run_piece) */
  bool dump = false; /* SEAM_MOE_CPP_DUMP: mirror each group's alignment into host buffers */
  bool trace = false;
  bool bound = false; /* SEAM_MOE_CPP_BOUND: align from the picks' bound, no per-piece readback */
  int fused = 0; /* SEAM_MOE_CPP_FUSED: 1 = one launch per projection, 2 = require it (refuse the
                  * vendor fallback, so a green run is evidence the fused kernels actually ran) */
  bool h_acc = false; /* CUDA_EXL3_FP16_ACC: the vendor's fp16 accumulator, which fused refuses */
  bool det = false; /* SEAM_MOE_CPP_DET: the Python path's deterministic epilogue branch, op for op
                     * (see run_piece). Refuses the bounded alignment, whose epilogue is the other,
                     * order-dependent sum. */
  /* SEAM_MOE_CPP_LAYER: 1 = serve pieces of <= kBoundMaxRows rows with qmc_fused_layer (one plan
   * kernel, one w13 and one down launch for every group, no readback); 2 = require it (a piece of
   * that size the layer path cannot take is a refusal). Never under DET, fp16 accumulation or the
   * alignment dump, which need the per-group path. */
  int layer_path = 0;
  int64_t layer_pieces = 0;
  int ws_pick = -1;
  int n_groups_max = 0;
  std::mutex mu;
  Layer layers[QMC_NL];
  Buffers b;
  int64_t calls = 0;
  int64_t refused = 0;
  /* With SEAM_MOE_CPP_FUSED=1 the fused path is opportunistic, so a run says nothing about the fused
   * kernels unless it also says how much of the work they did: these two counters are what makes an
   * =1 measurement interpretable (a prefill-only run can fuse nothing at all if every group's
   * pick_block_m comes back 32). Printed every 4096 calls and on every refusal. */
  int64_t fused_groups = 0;
  int64_t vendor_groups = 0;
  int64_t stat_calls = 0;
  int64_t last_wall = 0;
  int64_t last_dev = 0;
  bool reported = false;
  /* Where an unregistered layer goes when this context is the proxy's hook (qmc_set_fallback):
   * fn is a qes_server_fn, fnctx what the proxy hands back as its first argument. Both NULL when
   * the caller keeps routing in Python, in which case an unregistered layer is a refusal. */
  void* fb_fn = nullptr;
  void* fb_ctx = nullptr;
  cudaEvent_t ev0 = nullptr, ev1 = nullptr;
  /* SEAM_QMC_PROF: per-piece cudaEvent marks (see run_piece and prof_report). prof_ev is empty
   * unless the flag is on, so the shipped path pays one loaded bool per mark site. */
  bool prof = false;
  int prof_min_rows = 512; /* SEAM_QMC_PROF_ROWS: pieces narrower than this (decode) are not timed */
  std::vector<cudaEvent_t> prof_ev; /* [kProfPieces * kProfPerPiece] */
  /* SEAM_MOE_CPP_SIDE: the response D2H leaves the compute stream. Per piece the compute stream
   * records side_join[p], the side stream waits for it and takes the copy, so the reply of piece p
   * crosses the bus while piece p+1's copies and gemms run. The answer is the same bytes from the
   * same buffer into the same slot; only the copy's stream changes, and ev_side_done joins the two
   * at the end of the call so qmc_compute's existing sync still means "the reply slot is complete".
   * side_join is sized per call to the call's own piece count, so no event is ever waited on after
   * being re-recorded. */
  bool side = false;
  cudaStream_t side_stream = nullptr;
  std::vector<cudaEvent_t> side_join;
  /* side_done[slot]: the side stream has finished reading accumulator slot `slot`, so the piece two
   * pieces later may write it (only two slots exist). */
  std::vector<cudaEvent_t> side_done; /* [2] */
  cudaEvent_t ev_side_done = nullptr;
};

bool cuda_alive() {
  /* A dead context fails this and keeps failing; a refusal for memory does not. Same test as
   * worker_seam_server._cuda_ok, from the C side. */
  void* p = nullptr;
  if (cudaMalloc(&p, 4) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  const cudaError_t e = cudaMemset(p, 0, 4);
  cudaFree(p);
  cudaGetLastError();
  return e == cudaSuccess;
}

int pick_step(Ctx& c, int rows) {
  if (c.ws_rows <= 0) return rows;
  if (!c.ws_auto) return std::min(c.ws_rows, rows);
  const int avail = avail_mib();
  const int swapped = swap_used_mib();
  const bool paging = swapped > c.swap_warn;
  const int ladder[2] = {2048, 1024}; /* worker_seam_server.WS_LADDER */
  const int kib_per_row = 235;        /* worker_seam_server.WS_KIB_PER_ROW */
  int pick = -1;
  if (!paging) {
    for (int n : ladder) {
      if (n > c.ws_rows) continue;
      const int slack = (c.ws_pick < 0 || n <= c.ws_pick) ? 0 : c.ws_hyst;
      /* ws_mib_for(rows, n): a piece larger than the request costs only the request. */
      const double need = (double)std::min(n, rows) * kib_per_row / 1024.0;
      if (avail < 0 || need + c.ws_guard + slack <= avail) {
        pick = n;
        break;
      }
    }
  }
  if (pick < 0) pick = std::min(c.ws_rows, ladder[1]);
  if (pick != c.ws_pick) {
    c.ws_pick = pick;
    fprintf(stderr,
            "SEAM_WS_PICK rows=%d ws=%d avail_mib=%d swapped_mib=%d need_mib=%d pieces=%d\n", rows,
            pick, avail, swapped,
            (int)((double)std::min(pick, rows) * kib_per_row / 1024.0 + c.ws_guard),
            (rows + pick - 1) / pick);
    fflush(stderr);
  }
  return std::min(pick, rows);
}

/* -- buffers ------------------------------------------------------------------------------------ */
bool ensure_buffers(Ctx& c, int rows, int total_pad) {
  Buffers& b = c.b;
  const int H = c.hidden, T = c.topk;
  auto grow = [](void** p, size_t bytes) {
    if (*p) {
      cudaFree(*p);
      *p = nullptr;
    }
    return cudaMalloc(p, bytes);
  };
  if (rows > b.cap_rows) {
    if (grow(&b.x_h, (size_t)rows * H * 2) != cudaSuccess) return false;
    if (grow((void**)&b.ids, (size_t)rows * T * 4) != cudaSuccess) return false;
    if (grow((void**)&b.w_bf, (size_t)rows * T * 2) != cudaSuccess) return false;
    if (grow((void**)&b.w_f, (size_t)rows * T * 4) != cudaSuccess) return false;
    /* SEAM_MOE_CPP_SIDE keeps two accumulator slots (see run_piece): the reply of piece p reads
     * slot p&1 while piece p+1's down gemm writes the other. One extra piece-sized buffer
     * (rows*H*4 = 20 MiB at 1024 rows), allocated only when the flag is on. */
    if (grow((void**)&b.acc, (size_t)rows * H * 4 * (c.side ? 2 : 1)) != cudaSuccess) return false;
    /* What this layout saves against the one before the merge (x_bf rows*H*2 staging copy of the
     * request, out16 rows*H*2 rounding buffer, both gone), printed once per growth. */
    fprintf(stderr,
            "SEAM_MOE_CPP_BUFFERS rows=%d x_h=%zu acc=%zu ids_w=%zu saved_vs_staged=%zu "
            "(x_bf %zu + out16 %zu)\n",
            rows, (size_t)rows * H * 2, (size_t)rows * H * 4 * (c.side ? 2 : 1),
            (size_t)rows * T * 10, (size_t)rows * H * 4, (size_t)rows * H * 2,
            (size_t)rows * H * 2);
    fflush(stderr);
    b.cap_rows = rows;
  }
  /* Sized for the largest group count of any registered layer: layers may disagree (a pack whose
   * bitrates do not cover every layer), and a per-layer flip would reallocate on every call. */
  const int n_groups = c.n_groups_max;
  if (n_groups > b.n_groups) {
    if (grow((void**)&b.counts, (size_t)n_groups * QMC_MAX_EXPERTS * 4) != cudaSuccess)
      return false;
    if (grow((void**)&b.cursor, (size_t)n_groups * QMC_MAX_EXPERTS * 4) != cudaSuccess)
      return false;
    if (grow((void**)&b.blk_off, (size_t)n_groups * QMC_MAX_EXPERTS * 4) != cudaSuccess)
      return false;
    if (grow((void**)&b.n_rows, (size_t)n_groups * 4) != cudaSuccess) return false;
    if (grow((void**)&b.nblk, (size_t)n_groups * 4) != cudaSuccess) return false;
    if (grow((void**)&b.live, (size_t)n_groups * 4) != cudaSuccess) return false;
    if (b.h_plan) cudaFreeHost(b.h_plan);
    b.h_plan = nullptr;
    if (cudaHostAlloc((void**)&b.h_plan, (size_t)2 * n_groups * 4, cudaHostAllocPortable) !=
        cudaSuccess)
      return false;
    b.n_groups = n_groups;
  }
  if (!b.bad) {
    if (grow((void**)&b.bad, 4) != cudaSuccess) return false;
    if (cudaHostAlloc((void**)&b.h_bad, 4, cudaHostAllocPortable) != cudaSuccess) return false;
  }
  /* Blocks are one per ceil(count/block_m), so an alignment is bounded by the picks plus one
   * padding block per expert, per group: cap >= total of every group. block_m is not known here
   * yet, so the bound takes the slowest tier; the trio itself is then grown to the exact total the
   * alignment reports. */
  const int cap = total_pad + (QMC_MAX_EXPERTS + 1) * kBlockMTiers[3];
  if (cap > b.cap_sorted) {
    if (grow((void**)&b.sorted, (size_t)cap * 4) != cudaSuccess) return false;
    if (grow((void**)&b.eids, (size_t)n_groups * cap * 4) != cudaSuccess) return false;
    if (b.h_sorted) cudaFreeHost(b.h_sorted);
    if (b.h_eids) cudaFreeHost(b.h_eids);
    b.h_sorted = b.h_eids = nullptr;
    if (c.dump) {
      if (cudaHostAlloc((void**)&b.h_sorted, (size_t)n_groups * cap * 4,
                        cudaHostAllocPortable) != cudaSuccess)
        return false;
      if (cudaHostAlloc((void**)&b.h_eids, (size_t)n_groups * cap * 4, cudaHostAllocPortable) !=
          cudaSuccess)
        return false;
    }
    b.cap_sorted = cap;
  }
  if (c.dump && !b.h_sorted) {
    if (cudaHostAlloc((void**)&b.h_sorted, (size_t)n_groups * b.cap_sorted * 4,
                      cudaHostAllocPortable) != cudaSuccess)
      return false;
    if (cudaHostAlloc((void**)&b.h_eids, (size_t)n_groups * b.cap_sorted * 4,
                      cudaHostAllocPortable) != cudaSuccess)
      return false;
  }
  return true;
}

bool grow_trio(Ctx& c, int total_pad, int inter) {
  Buffers& b = c.b;
  if (total_pad <= b.cap_pad) return true;
  auto grow = [](void** p, size_t bytes) {
    if (*p) {
      cudaFree(*p);
      *p = nullptr;
    }
    return cudaMalloc(p, bytes);
  };
  const size_t H = (size_t)c.hidden;
  if (grow(&b.a13, (size_t)2 * total_pad * H * 2) != cudaSuccess) return false;
  if (grow(&b.inter, (size_t)total_pad * 2 * (size_t)inter * 2) != cudaSuccess) return false;
  if (grow(&b.a2, (size_t)total_pad * (size_t)inter * 2) != cudaSuccess) return false;
  b.cap_pad = total_pad;
  return true;
}

/* SEAM_MOE_CPP_LAYER's bound on the layer plan's block count: every live (group, expert) cell
 * holds ceil(cnt/16) <= cnt blocks and at most min(picks, experts) cells are live, so the plan has
 * at most min(m_valid, sum of n_experts) blocks. */
int layer_max_blk(const Layer& L, int m_valid) {
  int ne = 0;
  for (int g = 0; g < L.n_groups; g++) ne += L.grp[g].g.n_experts;
  return std::min(m_valid, ne);
}

int reply_piece(Ctx& c, int rows, float* accp, float* outp, int piece, cudaStream_t s,
                cudaEvent_t* pe);
void maybe_stat(Ctx& c, bool refused);

/* -- one piece ---------------------------------------------------------------------------------- */
/* `first_piece` is only about the bad-id flag: bound mode clears it once per request (here, on the
 * first piece, after the buffers exist) and reads it once after qmc_compute's sync, so a bad id in
 * any piece of the request still refuses the whole request. */
int run_piece(Ctx& c, int layer, int rows, const uint8_t* xp, const int32_t* idp,
              const uint8_t* wp, float* outp, bool first_piece, int piece) {
  Layer& L = c.layers[layer];
  const int H = c.hidden, T = c.topk;
  const int m_valid = rows * T;
  cudaStream_t s = at::cuda::getCurrentCUDAStream(c.dev).stream();
  /* Marks go on the compute stream, so a span is device time including any hole the host left in
   * it -- which is the point: GEMM time and host latency need to be told apart. */
  const bool prof = c.prof && rows >= c.prof_min_rows && piece >= 0 && piece < kProfPieces;
  cudaEvent_t* pe = prof ? c.prof_ev.data() + (size_t)piece * kProfPerPiece : nullptr;
  auto mark = [&](int k) {
    if (pe) cudaEventRecord(pe[k], s);
  };
  mark(kProfEntry);

  if (!ensure_buffers(c, rows, m_valid)) {
    set_err("buffer alloc failed: %s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseMem;
  }
  Buffers& b = c.b;
  /* SEAM_MOE_CPP_SIDE: per-piece accumulator slot. The stride is cap_rows (the largest piece ever
   * seen), not this piece's rows, or a short last piece would put the two slots on top of each
   * other. */
  float* accp = c.side ? b.acc + (size_t)(piece & 1) * (size_t)b.cap_rows * H : b.acc;
  /* Two slots mean the third piece writes the slot the first piece's copy may still be reading, and
   * nothing orders the side stream against this one. Piece p's copy marks side_done[p&1]; piece
   * p+2 waits for it before its first write. By then the copy has had a whole piece of gemms to
   * finish, so the wait is a formality that makes the two-stream schedule correct rather than
   * likely. */
  if (c.side && c.side_done.size() == 2 && piece >= 2) {
    if (cudaStreamWaitEvent(s, c.side_done[(size_t)(piece & 1)], 0) != cudaSuccess)
      cudaGetLastError(); /* a lost ordering marker is a stall, not a wrong answer */
  }

  /* Request slot -> device. x is not copied: qmc_k_bf16_to_half reads the bf16 rows through the
   * slab's device alias (xp is that alias, see qmc_compute) and writes the half the gemms take.
   * ids and w (rows*T*6 B) keep their H2D, the routing-weight kernel follows. */
  if (cudaMemcpyAsync(b.ids, idp, (size_t)rows * T * 4, cudaMemcpyHostToDevice, s) != cudaSuccess ||
      cudaMemcpyAsync(b.w_bf, wp, (size_t)rows * T * 2, cudaMemcpyHostToDevice, s) != cudaSuccess) {
    set_err("H2D from the pinned slab failed: %s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  mark(kProfCopy);

  /* SEAM_MOE_CPP_LAYER: a decode-sized piece of a layer the layer kernels take is three launches
   * straight off the request bytes -- no conversion kernels, no per-group alignment, no readback.
   * The bad-id verdict waits for qmc_compute's sync exactly as in bound mode. */
  const bool layer_ok = c.layer_path != 0 && L.desc_ok && !c.det && !c.h_acc && !c.dump &&
                        rows <= kBoundMaxRows;
  if (c.layer_path == 2 && !layer_ok && rows <= kBoundMaxRows) {
    set_err("layer %d rows %d: layer path required but not applicable (desc_ok=%d det=%d h_acc=%d "
            "dump=%d)", layer, rows, (int)L.desc_ok, (int)c.det, (int)c.h_acc, (int)c.dump);
    return kRefuseCompute;
  }
  if (layer_ok) {
    const int max_blk = layer_max_blk(L, m_valid);
    if (max_blk > b.cap_blk) {
      if (b.blk) cudaFree(b.blk);
      b.blk = nullptr;
      b.cap_blk = 0;
      if (cudaMalloc(&b.blk, (size_t)max_blk * 8) != cudaSuccess) {
        set_err("layer plan alloc failed: %s", cudaGetErrorString(cudaGetLastError()));
        return kRefuseMem;
      }
      b.cap_blk = max_blk;
    }
    if (max_blk * 16 > b.cap_sorted) {
      set_err("layer plan needs %d rows, capacity %d", max_blk * 16, b.cap_sorted);
      return kRefuseMem;
    }
    if (!grow_trio(c, max_blk * 16, L.desc.inter)) {
      set_err("workspace alloc failed: %s", cudaGetErrorString(cudaGetLastError()));
      return kRefuseMem;
    }
    if (m_valid > b.cap_picks) {
      if (b.pick_live) cudaFree(b.pick_live);
      if (b.slots) cudaFree(b.slots);
      b.pick_live = nullptr;
      b.slots = nullptr;
      b.cap_picks = 0;
      if (cudaMalloc((void**)&b.pick_live, (size_t)m_valid * 4) != cudaSuccess ||
          cudaMalloc((void**)&b.slots, (size_t)m_valid * H * 4) != cudaSuccess) {
        set_err("layer slot alloc failed: %s", cudaGetErrorString(cudaGetLastError()));
        return kRefuseMem;
      }
      b.cap_picks = m_valid;
    }
    if (qmc_fused_layer(&L.desc, (const uint16_t*)xp, b.ids, b.w_bf, m_valid, T, H, max_blk, b.sorted, b.blk,
                        b.nblk, b.bad, first_piece ? 1 : 0, b.pick_live, b.slots, b.inter, accp, rows,
                        s) != 0) {
      set_err("layer %d rows %d: layer path launch refused", layer, rows);
      maybe_stat(c, true);
      return kRefuseCompute;
    }
    c.layer_pieces++;
    c.fused_groups += L.n_groups;
    maybe_stat(c, false);
    return reply_piece(c, rows, accp, outp, piece, s, pe);
  }

  qmc_k_bf16_to_half((const uint16_t*)xp, b.x_h, rows * H, s);
  qmc_k_weights(b.ids, b.w_bf, b.w_f, rows * T, s);
  mark(kProfQuant);

  /* Histogram and alignment, both on device. block_m is a function of the live picks per expert;
   * the live count of the previous call stands in for this one (a request moves it by a few
   * percent, and the tier is coarse), or the picks bound m_valid answers the first call of a
   * bucket. Nothing here reads the counts back, so the per-group arithmetic align_block did on the
   * host - offsets, repeat_interleave, the padded row count - is all in qmc_k_align, and the only
   * readback before the gemms is 8 bytes per group.
   *
   * Bound mode (SEAM_MOE_CPP_BOUND=1, rows <= kBoundMaxRows) drops that readback: block_m comes
   * from m_valid, the block count is bounded by min(n_experts, m_valid), and qmc_k_align marks
   * every block past the live ones with -1. The extra blocks cost one early return each in the
   * gemm (n_rows retires them before a byte of weights is read), so the piece is one enqueue with
   * no host wait. What moves to the end of the request is the bad-id verdict: the histogram still
   * flags an id outside [-1, n_cold) on device, but nothing reads the flag until qmc_compute's
   * own sync.
   *
   * A dump needs this piece's exact per-group numbers, so it keeps the readback. */
  const bool bound_mode = c.bound && !c.dump && !c.det && rows <= kBoundMaxRows;
  if (cudaMemsetAsync(b.counts, 0, (size_t)L.n_groups * QMC_MAX_EXPERTS * 4, s) != cudaSuccess ||
      ((!bound_mode || first_piece) && cudaMemsetAsync(b.bad, 0, 4, s) != cudaSuccess)) {
    set_err("histogram clear failed: %s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  for (int g = 0; g < L.n_groups; g++) {
    const qmc_group& q = L.grp[g].g;
    qmc_k_hist(b.ids, m_valid, q.n_cold, q.remap, q.n_experts,
               b.counts + (size_t)g * QMC_MAX_EXPERTS, b.bad, s);
  }
  int bm[QMC_MAX_GROUPS] = {0};
  int nblk_pad[QMC_MAX_GROUPS] = {0};
  for (int g = 0; g < L.n_groups; g++) {
    const qmc_group& q = L.grp[g].g;
    const int hint =
        bound_mode ? m_valid : (L.grp[g].last_live > 0 ? L.grp[g].last_live : m_valid);
    bm[g] = c.block_m > 0 ? c.block_m : pick_block_m(hint, q.n_experts);
    nblk_pad[g] = bound_mode ? (q.n_experts < m_valid ? q.n_experts : m_valid) : 0;
    qmc_k_align(b.counts + (size_t)g * QMC_MAX_EXPERTS, q.n_experts, bm[g], nblk_pad[g],
                b.blk_off + (size_t)g * QMC_MAX_EXPERTS,
                b.eids + (size_t)g * b.cap_sorted, b.nblk + g, b.live + g, s);
  }
  if (!bound_mode) {
    if (cudaMemcpyAsync(b.h_plan, b.nblk, (size_t)L.n_groups * 4, cudaMemcpyDeviceToHost, s) !=
            cudaSuccess ||
        cudaMemcpyAsync(b.h_plan + L.n_groups, b.live, (size_t)L.n_groups * 4,
                        cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaMemcpyAsync(b.h_bad, b.bad, 4, cudaMemcpyDeviceToHost, s) != cudaSuccess) {
      set_err("plan readback failed: %s", cudaGetErrorString(cudaGetLastError()));
      return kRefuseCompute;
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) {
      set_err("%s", cudaGetErrorString(cudaGetLastError()));
      return kRefuseCompute;
    }
    if (*b.h_bad) {
      set_err("request rejected: an id outside [-1, n_cold) for layer %d rows %d", layer, rows);
      return kRefuseCompute;
    }
  }
  mark(kProfAlign);

  const at::Tensor x_h = dev_tensor(b.x_h, {rows, H}, at::kHalf, c.dev);
  const at::Tensor w_f = dev_tensor(b.w_f, {rows, T}, at::kFloat, c.dev);
  /* In bound mode the accumulator is zeroed and every group adds into it, because an empty group
   * cannot be recognised without the readback; the gemm adds nothing when n_rows is 0. */
  bool wrote = bound_mode;
  int total_max = 0;
  int total[QMC_MAX_GROUPS] = {0}, live[QMC_MAX_GROUPS] = {0};
  for (int g = 0; g < L.n_groups; g++) {
    if (bound_mode) {
      total[g] = nblk_pad[g] * bm[g];
    } else {
      live[g] = b.h_plan[L.n_groups + g];
      total[g] = b.h_plan[g] * bm[g];
      L.grp[g].last_live = live[g];
    }
    if (total[g] > total_max) total_max = total[g];
  }
  if (bound_mode && cudaMemsetAsync(accp, 0, (size_t)rows * H * 4, s) != cudaSuccess) {
    set_err("zero fill failed: %s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  int rc = 0;
  if (total_max <= 0) {
    /* No group has a live pick: the layer's answer for these rows is zero. */
    if (cudaMemsetAsync(accp, 0, (size_t)rows * H * 4, s) != cudaSuccess) {
      set_err("zero fill failed: %s", cudaGetErrorString(cudaGetLastError()));
      return kRefuseCompute;
    }
  } else {
    if (total_max > b.cap_sorted) {
      set_err("alignment needs %d rows, capacity %d", total_max, b.cap_sorted);
      return kRefuseMem;
    }
    /* The trio is sized for the largest group of this piece. I is the same for every group of a
     * layer (one trellis tile width), so taking it from group 0 is exact. */
    if (!grow_trio(c, total_max, L.grp[0].g.inter)) {
      set_err("workspace alloc failed: %s", cudaGetErrorString(cudaGetLastError()));
      return kRefuseMem;
    }
    for (int g = 0; g < L.n_groups && rc == 0; g++) {
      const qmc_group& q = L.grp[g].g;
      if (!bound_mode && live[g] == 0) continue;
      int32_t* blk_off_g = b.blk_off + (size_t)g * QMC_MAX_EXPERTS;
      int32_t* eids_g = b.eids + (size_t)g * b.cap_sorted;
      if (cudaMemsetAsync(b.cursor + (size_t)g * QMC_MAX_EXPERTS, 0, (size_t)q.n_experts * 4, s) !=
              cudaSuccess ||
          cudaMemsetAsync(b.n_rows + g, 0, 4, s) != cudaSuccess) {
        set_err("alignment clear failed: %s", cudaGetErrorString(cudaGetLastError()));
        return kRefuseCompute;
      }
      qmc_k_fill_i32(b.sorted, m_valid, total[g], s);
      qmc_k_scatter(b.ids, m_valid, q.n_cold, q.remap, b.cursor + (size_t)g * QMC_MAX_EXPERTS,
                    blk_off_g, bm[g], b.sorted, b.n_rows + g, s);
      mark(prof_gi(g, 0));
      if (c.dump) {
        L.dump_rows = rows;
        L.dump_bm[g] = bm[g];
        L.dump_nblk[g] = b.h_plan[g];
        L.dump_total[g] = total[g];
        if (cudaMemcpyAsync(b.h_sorted + (size_t)g * b.cap_sorted, b.sorted, (size_t)total[g] * 4,
                            cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaMemcpyAsync(b.h_eids + (size_t)g * b.cap_sorted, eids_g,
                            (size_t)(bound_mode ? nblk_pad[g] : b.h_plan[g]) * 4,
                            cudaMemcpyDeviceToHost, s) != cudaSuccess) {
          set_err("alignment dump failed: %s", cudaGetErrorString(cudaGetLastError()));
          return kRefuseCompute;
        }
      }

      const at::Tensor sorted_t = dev_tensor(b.sorted, {total[g]}, at::kInt, c.dev);
      const int n_eids = bound_mode ? nblk_pad[g] : b.h_plan[g];
      const at::Tensor eids_t = dev_tensor(eids_g, {n_eids}, at::kInt, c.dev);
      const at::Tensor nr_t = dev_tensor(b.n_rows + g, {1}, at::kInt, c.dev);
      const at::Tensor a13 = dev_tensor(b.a13, {2, total[g], H}, at::kHalf, c.dev);
      const at::Tensor inter =
          dev_tensor(b.inter, {total[g], 2 * (int64_t)q.inter}, at::kHalf, c.dev);
      const at::Tensor a2 = dev_tensor(b.a2, {1, total[g], (int64_t)q.inter}, at::kHalf, c.dev);
      const at::Tensor acc = dev_tensor(accp, {rows, H}, at::kFloat, c.dev);

      /* Fused: the two Hadamard transforms run inside the GEMMs that read them, so neither a13 nor
       * a2 is ever written (qmesh_moe_fused.cu). Same tile (block_m=16) and same accumulators as
       * the vendor path; anything the fused kernels are not instantiated for -- another block_m, a
       * fractional bitrate, fp16 accumulation -- falls through to the four vendor ops below.
       * SEAM_MOE_CPP_DET outranks it: the fused stages write the folded fp32 atomic epilogue, so a
       * layer served by them would be neither reproducible nor comparable with the Python path,
       * which is the whole promise of the DET arm. DET therefore takes the vendor branch (whose
       * epilogue it replaces), and FUSED=2 under DET is a refusal rather than a silent vendor run. */
      const bool use_fused =
          c.fused != 0 && !c.det && !c.h_acc && bm[g] == 16 && total[g] % 16 == 0 &&
          q.words13 % 16 == 0 && q.words2 % 16 == 0 && H % 128 == 0 && q.inter % 128 == 0;
      /* SEAM_MOE_CPP_FUSED=2: the caller is measuring the fused kernels and a fallback would measure
       * something else, so what cannot be fused is refused with the reason it could not be. */
      if (c.fused == 2 && !use_fused) {
        set_err("layer %d group %d: fused required but not applicable (bm=%d total=%d words13=%d "
                "words2=%d h_acc=%d det=%d)", layer, g, bm[g], total[g], (int)q.words13,
                (int)q.words2, (int)c.h_acc, (int)c.det);
        rc = kRefuseCompute;
        break;
      }
      try {
        if (use_fused) {
          if (!wrote && cudaMemsetAsync(accp, 0, (size_t)rows * H * 4, s) != cudaSuccess) {
            set_err("zero fill failed: %s", cudaGetErrorString(cudaGetLastError()));
            rc = kRefuseCompute;
            break;
          }
          const int64_t bs13 = (int64_t)(H / 16) * (2 * q.inter / 16) * q.words13;
          const int64_t bs2 = (int64_t)(q.inter / 16) * (H / 16) * q.words2;
          const int mblocks = total[g] / bm[g];
          if (qmc_fused_w13(b.x_h, b.sorted, q.suh13, b.inter, q.trellis13, q.svh13, eids_g,
                            b.n_rows + g, H, 2 * q.inter, q.inter, 2, q.words13 / 16, q.cb, bm[g],
                            bs13, m_valid, T, mblocks, s) != 0 ||
              qmc_fused_down(b.inter, b.sorted, q.suh2, accp, q.trellis2, q.svh2, eids_g,
                             b.n_rows + g, b.w_f, q.inter, H, q.words2 / 16, q.cb, bm[g], bs2,
                             m_valid, T, mblocks, s) != 0) {
            /* Both stages report 2 for a refused launch and 1 for a shape with no instantiation;
             * either way the layer's numbers are not usable, so it is a refusal, not a result. */
            set_err("layer %d group %d: fused stage failed (bits=%d cb=%d bm=%d): no instantiation "
                    "or launch refused", layer, g, (int)(q.words13 / 16), (int)q.cb, bm[g]);
            rc = kRefuseCompute;
            break;
          }
          /* One launch per projection: the Hadamard runs inside w13 and the glu inside down, so
           * these four marks share a timestamp and the span lands in the had slot. */
          mark(prof_gi(g, 1));
          mark(prof_gi(g, 2));
          mark(prof_gi(g, 3));
          mark(prof_gi(g, 4));
        } else {
          /* Evidence for the served path, once per process: the cap the vendor gemm will see on
           * the first call that actually takes it (prefill block_m > 16, a group with no fused
           * instantiation, DET). Nonzero would be the 2026-11-05 split-k combine hazard. */
          static bool cap_printed = false;
          if (!cap_printed) {
            cap_printed = true;
            std::vector<c10::IValue> r;
            op_handle("cuda_exl3_C::exl3_get_moe_acc_cap").callBoxed(r);
            fprintf(stderr, "SEAM_MOE_CPP_VENDOR_FIRST layer=%d rows=%d g=%d bm=%d moe_acc_cap=%lld\n",
                    layer, rows, g, bm[g], r.empty() ? -1LL : (long long)r.at(0).toInt());
            fflush(stderr);
          }
          moe_had_in(x_h, a13, L.grp[g].suh13, sorted_t, eids_t, nr_t, bm[g], T, m_valid);
          mark(prof_gi(g, 1));
          moe_gemm(a13, L.grp[g].t13, L.grp[g].suh13, L.grp[g].svh13, eids_t, nr_t,
                   {(int64_t)q.inter, (int64_t)q.inter}, q.cb, bm[g], at::kHalf, sorted_t, true,
                   at::Tensor(), false, rows, T, inter, true, false);
          mark(prof_gi(g, 2));
          moe_glu(inter, a2, L.grp[g].suh2, eids_t, nr_t, bm[g]);
          mark(prof_gi(g, 3));
          if (c.det) {
            /* SEAM_MOE_CPP_DET: the Python path's deterministic branch (exl3_moe_cuda.py:482-491), the
             * only difference being the epilogue -- the down gemm unweighted into a half, the vendor's
             * pick-order combine into a second half, one fp32 add per group. The default epilogue folds
             * the routing weight into the down gemm's fp32 atomics instead, so its answer depends on
             * the order blocks retire in (and on block_m); this one depends on nothing but the group
             * order. That is what makes a per-layer byte comparison between the two paths mean
             * anything: measured on 2026-11-03 over 40 layers at rows 1/6/64/8192, the default arm
             * agrees with the Python path on 0/40 layers and this branch on 40/40 (max_abs exactly 0).
             * The "13/40 identical" figure often quoted next to this code is fused-vs-native, a
             * different comparison: both of those sum by fp32 atomicAdd in block-retirement order.
             * DET never takes the fused stages, whose epilogue is the folded one (see use_fused). */
            const at::Tensor routed =
                moe_gemm(a2, L.grp[g].t2, L.grp[g].suh2, L.grp[g].svh2, eids_t, nr_t, {(int64_t)H},
                         q.cb, bm[g], at::kHalf, sorted_t, true, at::Tensor(), false, rows, T,
                         at::Tensor(), false, false);
            const at::Tensor combined = moe_combine(routed, sorted_t, w_f, rows, eids_t, bm[g]);
            /* total = out.float() on the first contributing group and total + out.float() after: the
             * assign-vs-add distinction is -0.0's, and it is what the Python path does. accp, not
             * b.acc: with SEAM_MOE_CPP_SIDE the reply copy of the previous piece may still be reading
             * the other slot. */
            qmc_k_h2f_add(combined.data_ptr(), accp, rows * H, wrote ? 1 : 0, s);
          } else {
            moe_gemm(a2, L.grp[g].t2, L.grp[g].suh2, L.grp[g].svh2, eids_t, nr_t, {(int64_t)H}, q.cb,
                     bm[g], at::kFloat, sorted_t, true, w_f, true, rows, T, acc, true, wrote);
          }
          mark(prof_gi(g, 4));
        }
      } catch (const std::exception& e) {
        set_err("layer %d group %d raised: %s", layer, g, e.what());
        rc = kRefuseCompute;
        break;
      }
      if (use_fused)
        c.fused_groups++;
      else
        c.vendor_groups++;
      wrote = true;
      if (c.trace)
        /* blk is what the gemm grid ran over. In bound mode that is the padded count, not the
         * read-back plan, and live is -1 because this piece never learned it. */
        fprintf(stderr, "SEAM_MOE_CPP_PIECE layer=%d rows=%d g=%d E=%d bm=%d blk=%d pad=%d live=%d\n",
                layer, rows, g, q.n_experts, bm[g], bound_mode ? nblk_pad[g] : b.h_plan[g],
                total[g], bound_mode ? -1 : live[g]);
    }
    if (rc == 0 && !wrote) {
      if (cudaMemsetAsync(accp, 0, (size_t)rows * H * 4, s) != cudaSuccess) {
        set_err("zero fill failed: %s", cudaGetErrorString(cudaGetLastError()));
        return kRefuseCompute;
      }
    }
  }
  maybe_stat(c, rc != 0);
  if (rc != 0) return rc;
  return reply_piece(c, rows, accp, outp, piece, s, pe);
}

/* SEAM_MOE_CPP_FUSED_STATS every 4096 calls and on every refusal. A layer-path piece counts its
 * groups as fused (it runs the fused stages' body); `layer=` is how many pieces took that path, so
 * `layer>0` is the evidence SEAM_MOE_CPP_LAYER was live. */
void maybe_stat(Ctx& c, bool refused) {
  if (!(c.fused != 0 || c.layer_path != 0) || !(refused || c.calls - c.stat_calls >= 4096)) return;
  const int64_t tot = c.fused_groups + c.vendor_groups;
  fprintf(stderr,
          "SEAM_MOE_CPP_FUSED_STATS calls=%lld groups=%lld fused=%lld vendor=%lld fused_pct=%.1f "
          "knob=%d layer=%lld layer_knob=%d\n",
          (long long)c.calls, (long long)tot, (long long)c.fused_groups,
          (long long)c.vendor_groups, tot ? 100.0 * (double)c.fused_groups / (double)tot : 0.0,
          (int)c.fused, (long long)c.layer_pieces, c.layer_path);
  fflush(stderr);
  c.stat_calls = c.calls;
}

/* Answer -> pinned response slot. The Python path built this with .float().cpu().numpy(); it is
 * one D2H of the accumulator. With SEAM_MOE_CPP_OUT=bf16|fp16 the answer is rounded on device
 * first and only half the bytes cross the wire: the response is 5120 floats per row, two thirds
 * of a served request's payload, so this is the one place a wire format can be halved without
 * touching the kernels. The slot must then be half-sized - that is the seam's layout decision
 * (off_resp's stride in qmesh_expert_seam.cu and Seam.collect in remote_seam_host.py), not
 * this file's. */
int reply_piece(Ctx& c, int rows, float* accp, float* outp, int piece, cudaStream_t s,
                cudaEvent_t* pe) {
  Buffers& b = c.b;
  const int H = c.hidden;
  const size_t obytes = (size_t)rows * H * (c.out_mode ? 2 : 4);
  /* SEAM_MOE_CPP_SIDE: the reply write is enqueued on the side stream behind a marker of everything
   * the compute stream has done for this piece, so it overlaps the next piece instead of standing on
   * the compute stream. It reads accumulator slot piece&1 and writes only this piece's rows of the
   * response slot, so the 16-bit reply needs no second buffer to compose with SIDE. */
  cudaStream_t cs = s;
  if (c.side && c.side_stream && piece >= 0 && piece < (int)c.side_join.size()) {
    cudaError_t je = cudaEventRecord(c.side_join[(size_t)piece], s);
    if (je == cudaSuccess) je = cudaStreamWaitEvent(c.side_stream, c.side_join[(size_t)piece], 0);
    if (je == cudaSuccess)
      cs = c.side_stream;
    else
      cudaGetLastError(); /* a failed hand-off is a slow piece, not a wrong answer: keep it on s */
  }
  if (c.out_mode) {
    qmc_k_f32_to_half(accp, outp, rows * H, c.out_mode, cs);
    if (cudaGetLastError() != cudaSuccess) {
      set_err("reply rounding into the response slot failed");
      return kRefuseCompute;
    }
  } else if (cudaMemcpyAsync(outp, accp, obytes, cudaMemcpyDefault, cs) != cudaSuccess) {
    set_err("D2H to the response slot failed: %s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  if (cs == c.side_stream && c.side_done.size() == 2)
    /* The side stream has now read accumulator slot (piece&1) to the end: the piece two later waits
     * on this before writing it. */
    cudaEventRecord(c.side_done[(size_t)(piece & 1)], c.side_stream);
  if (pe) cudaEventRecord(pe[kProfWrite], s);
  return 0;
}

/* One elapsed time between two marks on the compute stream, or -1 when either was never recorded
 * (an empty layer skips its groups). */
float prof_span(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  if (cudaEventElapsedTime(&ms, a, b) != cudaSuccess) {
    cudaGetLastError();
    return -1.f;
  }
  return ms;
}

/* Printed after qmc_compute's own sync -- the first point in a call where every mark has a
 * timestamp -- so the report adds no wait of its own to the served path. `dev` is qmc_last_call_ns'
 * device span for the whole call (ev0 before the piece picker, ev1 after the last piece), so
 * `rest = dev - sum` is the part of the call no piece mark covers: the picker, the buffer growth of
 * the first call, and the report's own reading of the events. */
void prof_report(Ctx& c, int layer, int rows, int pieces, int step) {
  const int ng = c.layers[layer].n_groups;
  const int timed = std::min(pieces, kProfPieces);
  double s_copy = 0, s_quant = 0, s_align = 0, s_prep = 0, s_had = 0, s_w13 = 0, s_glu = 0,
         s_down = 0, s_write = 0, tot = 0;
  for (int p = 0; p < timed; p++) {
    cudaEvent_t* pe = c.prof_ev.data() + (size_t)p * kProfPerPiece;
    const float copy = std::max(prof_span(pe[kProfEntry], pe[kProfCopy]), 0.f);
    const float quant = std::max(prof_span(pe[kProfCopy], pe[kProfQuant]), 0.f);
    const float align = std::max(prof_span(pe[kProfQuant], pe[kProfAlign]), 0.f);
    double prep = 0, had = 0, w13 = 0, glu = 0, down = 0;
    cudaEvent_t prev = pe[kProfAlign];
    for (int g = 0; g < ng; g++) {
      prep += std::max(prof_span(prev, pe[prof_gi(g, 0)]), 0.f);
      had += std::max(prof_span(pe[prof_gi(g, 0)], pe[prof_gi(g, 1)]), 0.f);
      w13 += std::max(prof_span(pe[prof_gi(g, 1)], pe[prof_gi(g, 2)]), 0.f);
      glu += std::max(prof_span(pe[prof_gi(g, 2)], pe[prof_gi(g, 3)]), 0.f);
      down += std::max(prof_span(pe[prof_gi(g, 3)], pe[prof_gi(g, 4)]), 0.f);
      prev = pe[prof_gi(g, 4)];
    }
    const float write = std::max(prof_span(prev, pe[kProfWrite]), 0.f);
    const double total = (double)copy + quant + align + prep + had + w13 + glu + down + write;
    fprintf(stderr,
            "SEAM_QMC_PROF layer=%d piece=%d/%d rows=%d g=%d copy=%.3f quant=%.3f align=%.3f "
            "prep=%.3f had=%.3f w13=%.3f glu=%.3f down=%.3f write=%.3f total=%.3f\n",
            layer, p, pieces, rows, ng, copy, quant, align, prep, had, w13, glu, down, write,
            total);
    s_copy += copy;
    s_quant += quant;
    s_align += align;
    s_prep += prep;
    s_had += had;
    s_w13 += w13;
    s_glu += glu;
    s_down += down;
    s_write += write;
    tot += total;
  }
  const double dev = (double)c.last_dev / 1e6;
  fprintf(stderr,
          "SEAM_QMC_PROF_SUM layer=%d rows=%d step=%d pieces=%d timed=%d dev=%.3f wall_us=%.1f "
          "copy=%.3f quant=%.3f align=%.3f prep=%.3f had=%.3f w13=%.3f glu=%.3f down=%.3f "
          "write=%.3f sum=%.3f rest=%.3f\n",
          layer, rows, step, pieces, timed, dev, (double)c.last_wall / 1e3, s_copy, s_quant,
          s_align, s_prep, s_had, s_w13, s_glu, s_down, s_write, tot, dev - tot);
  fflush(stderr);
}
}  // namespace

extern "C" int qmc_open(int device, int hidden, int topk, int ws_rows, void** out) {
  if (!out) return -1;
  *out = nullptr;
  auto* c = new Ctx();
  c->dev = device;
  c->hidden = hidden;
  c->topk = topk;
  c->ws_rows = ws_rows;
  c->ws_auto = env_int("SEAM_WS_ROWS_AUTO", 1) != 0;
  c->ws_guard = env_int("SEAM_WS_GUARD_MIB", 96);
  c->ws_hyst = env_int("SEAM_WS_HYST_MIB", 128);
  c->swap_warn = env_int("SEAM_SWAP_WARN_MIB", 512);
  c->fwd_floor = env_int("SEAM_FWD_FLOOR_MIB", 0);
  c->block_m = env_int("SEAM_MOE_CPP_BLOCK_M", 0);
  c->dump = env_int("SEAM_MOE_CPP_DUMP", 0) != 0;
  c->trace = env_int("SEAM_MOE_CPP_TRACE", 0) != 0;
  /* Per-piece marks: off by default, and nothing is allocated for them when off. SEAM_QMC_PROF_ROWS
   * keeps decode and verify widths out of the log; a request at or above it is timed piece by
   * piece and reported after its own sync (prof_report). */
  c->prof = env_int("SEAM_QMC_PROF", 0) != 0;
  c->prof_min_rows = (int)env_int("SEAM_QMC_PROF_ROWS", 512);
  if (c->prof_min_rows < 1) c->prof_min_rows = 1;
  /* Alignment without the per-piece readback (see run_piece). Default off: on real routing it is
   * 3.3 % slower at 1 row and 5.1 % slower at 6 rows than the 8-byte readback, because padding every
   * group to min(n_experts, m_valid) costs more blocks than the sync saves at decode widths. On is a
   * measurement arm. */
  c->bound = env_int("SEAM_MOE_CPP_BOUND", 0) != 0;
  /* One plan kernel + one w13 + one down launch per decode piece, every group together, no
   * readback (see run_piece and qmc_fused_layer). Off by default until the A/B lands. */
  c->layer_path = (int)env_int("SEAM_MOE_CPP_LAYER", 0);
  /* One launch per projection instead of four (see qmesh_moe_fused.cu). Only the block_m=16 tile is
   * instantiated, and only fp32 accumulation: a fused kernel has no second pass to fall back on, so
   * CUDA_EXL3_FP16_ACC is refused here and the vendor path keeps it. Off by default until the
   * 40-layer dump comparison says the answer is identical. 2 = require: any layer the fused kernels
   * cannot take is a refusal rather than a silent run of the vendor path, which is what makes an
   * identity comparison between the two arms mean something. */
  c->fused = (int)env_int("SEAM_MOE_CPP_FUSED", 0);
  {
    /* The vendor reads this through exl3_env, which also honours the VLLM_EXL3_ alias; the seam
     * process only ever sets the CUDA_ name, and a set-but-empty or "0" value means off there too. */
    const char* ha = getenv("CUDA_EXL3_FP16_ACC");
    c->h_acc = ha && *ha && *ha != '0';
  }
  /* The Python path's deterministic epilogue (see run_piece): an unweighted half out of the down gemm,
   * the vendor's pick-order combine, one fp32 add per group. It is the only mode whose answer is the
   * same on every run and at every block_m, which is what makes a per-layer byte comparison between
   * the two paths meaningful; it also refuses SEAM_MOE_CPP_BOUND, whose epilogue is the other,
   * order-dependent sum. Off by default: the combine is an extra pass over rows*H halves and a
   * host-visible allocation, and the default epilogue is faster. */
  c->det = env_int("SEAM_MOE_CPP_DET", 0) != 0;
  /* Response format. Default fp32: the shipped seam slot is fp32 and off_resp's stride is
   * rows*HIDDEN*4 (qmesh_expert_seam.cu, and Seam.collect in remote_seam_host.py), so anything else
   * must be a deliberate, matching change on the client side. */
  {
    const char* om = getenv("SEAM_MOE_CPP_OUT");
    if (om && (!strcmp(om, "bf16") || !strcmp(om, "bfloat16")))
      c->out_mode = 1;
    else if (om && (!strcmp(om, "fp16") || !strcmp(om, "f16") || !strcmp(om, "half")))
      c->out_mode = 2;
    else if (om && *om && strcmp(om, "fp32") && strcmp(om, "f32"))
      fprintf(stderr, "SEAM_MOE_CPP_WARN unknown SEAM_MOE_CPP_OUT=%s, keeping fp32\n", om);
  }
  if (cudaSetDevice(device) != cudaSuccess) {
    set_err("cudaSetDevice(%d): %s", device, cudaGetErrorString(cudaGetLastError()));
    delete c;
    return -1;
  }
  if (cudaEventCreateWithFlags(&c->ev0, cudaEventDefault) != cudaSuccess ||
      cudaEventCreateWithFlags(&c->ev1, cudaEventDefault) != cudaSuccess) {
    set_err("event create failed: %s", cudaGetErrorString(cudaGetLastError()));
    delete c;
    return -1;
  }
  if (c->prof) {
    c->prof_ev.assign((size_t)kProfPieces * kProfPerPiece, nullptr);
    for (auto& e : c->prof_ev) {
      if (cudaEventCreateWithFlags(&e, cudaEventDefault) != cudaSuccess) {
        set_err("prof event create failed: %s", cudaGetErrorString(cudaGetLastError()));
        for (auto& x : c->prof_ev)
          if (x) cudaEventDestroy(x);
        delete c;
        return -1;
      }
    }
    fprintf(stderr, "SEAM_QMC_PROF_ON min_rows=%d pieces=%d per_piece=%d\n", c->prof_min_rows,
            kProfPieces, kProfPerPiece);
  }
  /* SEAM_MOE_CPP_SIDE: take the reply write off the compute stream and let it overlap the next
   * piece. Off by default (the shipped path must not change without a token-identity arm). A 16-bit
   * reply (SEAM_MOE_CPP_OUT or the seam's wire caps) is rounded straight into each piece's own rows
   * of the response slot, so it composes with SIDE (see run_piece). */
  c->side = env_int("SEAM_MOE_CPP_SIDE", 0) != 0;
  if (c->side) {
    bool ok = cudaStreamCreateWithFlags(&c->side_stream, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreateWithFlags(&c->ev_side_done, cudaEventDisableTiming) == cudaSuccess;
    c->side_done.assign(2, nullptr);
    for (size_t i = 0; ok && i < c->side_done.size(); i++)
      ok = cudaEventCreateWithFlags(&c->side_done[i], cudaEventDisableTiming) == cudaSuccess;
    if (!ok) {
      set_err("side stream create failed: %s", cudaGetErrorString(cudaGetLastError()));
      for (auto& e : c->side_done)
        if (e) cudaEventDestroy(e);
      c->side_done.clear();
      if (c->side_stream) cudaStreamDestroy(c->side_stream);
      c->side_stream = nullptr;
      if (c->ev_side_done) cudaEventDestroy(c->ev_side_done);
      c->ev_side_done = nullptr;
      c->side = false;
    } else {
      fprintf(stderr, "SEAM_MOE_CPP_SIDE_ON reply_copy_overlaps_next_piece=1 acc_slots=2\n");
    }
    fflush(stderr);
  }
  *out = c;
  return 0;
}

extern "C" int qmc_add_layer(void* ctx, int layer, int n_groups, const qmc_group* groups) {
  auto* c = (Ctx*)ctx;
  if (!c || layer < 0 || layer >= QMC_NL || n_groups <= 0 || n_groups > QMC_MAX_GROUPS || !groups) {
    set_err("qmc_add_layer: bad arguments (layer=%d n_groups=%d)", layer, n_groups);
    return -1;
  }
  if (n_groups > c->n_groups_max) c->n_groups_max = n_groups;
  /* The vendor gemm's MoE split-k is on by default (cap 8M elements) and its split epilogue has no
   * fused combine: a split down gemm with sorted_ids + weights writes unweighted rows at routed
   * offsets. qmesh/exl3_moe_cuda.py zeroes the cap at import, but this hook must not depend on who
   * imported the extension first (the bench loads the .so bare and its vendor arm was 0.27-0.86
   * max_rel off fp64, 2026-11-05). Zeroed here, once, and the value read back and printed so a
   * served worker's log proves it. A missing op (an old .so) is a refusal: the vendor path cannot
   * be shown safe. */
  {
    static int cap_state = -1; /* -1 unset, 0 ok, 1 failed */
    if (cap_state < 0) {
      try {
        std::vector<c10::IValue> a{iv_i64(0)};
        op_handle("cuda_exl3_C::exl3_set_moe_acc_cap").callBoxed(a);
        std::vector<c10::IValue> r;
        op_handle("cuda_exl3_C::exl3_get_moe_acc_cap").callBoxed(r);
        const int64_t cap = r.empty() ? -1 : r.at(0).toInt();
        /* One cap per loaded copy of the extension: a second copy (another path, a second dlopen)
         * would keep its own static and serve its own ops. Count distinct mapped cuda_exl3/_C
         * files; more than one refuses the registration (0 = mapped under another name: printed,
         * not refused, since the dispatcher op above did answer). */
        int copies = 0;
        {
          std::vector<std::string> seen;
          if (FILE* f = fopen("/proc/self/maps", "r")) {
            char line[4096];
            while (fgets(line, sizeof line, f)) {
              const char* p = strstr(line, "/cuda_exl3/_C");
              if (!p) continue;
              const char* s = strchr(line, '/');
              std::string path(s ? s : p);
              while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
              bool dup = false;
              for (auto& x : seen) dup |= x == path;
              if (!dup) seen.push_back(path);
            }
            fclose(f);
          }
          copies = (int)seen.size();
        }
        cap_state = cap == 0 && copies <= 1 ? 0 : 1;
        fprintf(stderr, "SEAM_MOE_CPP_ACC_CAP moe_acc_cap=%lld split_k=%s vendor_so_copies=%d\n",
                (long long)cap, cap == 0 ? "off" : "ON", copies);
      } catch (const std::exception& e) {
        cap_state = 1;
        fprintf(stderr, "SEAM_MOE_CPP_ACC_CAP moe_acc_cap=? err=%s\n", e.what());
      }
      fflush(stderr);
    }
    if (cap_state != 0) {
      set_err("qmc_add_layer: could not set the vendor MoE accumulator cap to 0");
      return -1;
    }
  }
  Layer& L = c->layers[layer];
  const int H = c->hidden;
  L.n_groups = n_groups;
  for (int g = 0; g < n_groups; g++) {
    const qmc_group& q = groups[g];
    L.grp[g].g = q;
    if (q.n_experts <= 0 || q.n_experts > QMC_MAX_EXPERTS || q.inter <= 0 || (q.inter & 15) ||
        q.words13 <= 0 || q.words2 <= 0 || (H & 15) || q.n_cold <= 0) {
      set_err("qmc_add_layer: layer %d group %d geometry (E=%d I=%d w13=%d w2=%d H=%d n_cold=%d)",
              layer, g, q.n_experts, q.inter, q.words13, q.words2, H, q.n_cold);
      return -1;
    }
    const int64_t E = q.n_experts, h16 = H / 16, i16 = q.inter / 16;
    const at::TensorOptions opt[6] = {
        at::TensorOptions().dtype(at::kShort).device(at::kCUDA, c->dev),
        at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, c->dev),
        at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, c->dev),
        at::TensorOptions().dtype(at::kShort).device(at::kCUDA, c->dev),
        at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, c->dev),
        at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, c->dev)};
    L.grp[g].t13 = at::from_blob((void*)q.trellis13, {E, h16, 2 * i16, q.words13}, opt[0]);
    L.grp[g].suh13 = at::from_blob((void*)q.suh13, {E, 2, H}, opt[1]);
    L.grp[g].svh13 = at::from_blob((void*)q.svh13, {E, 32 * i16}, opt[2]);
    L.grp[g].t2 = at::from_blob((void*)q.trellis2, {E, i16, h16, q.words2}, opt[3]);
    L.grp[g].suh2 = at::from_blob((void*)q.suh2, {E, 1, q.inter}, opt[4]);
    L.grp[g].svh2 = at::from_blob((void*)q.svh2, {E, H}, opt[5]);
  }
  /* The layer path's descriptor: same pointers and strides run_piece hands the per-group fused
   * stages (bs13/bs2 below are its own), and the same applicability test use_fused applies per
   * group, applied to the whole layer -- one non-conforming group keeps the layer per-group. */
  L.desc = qmc_layer_desc{};
  L.desc_ok = (H % 128 == 0);
  L.desc.n_groups = n_groups;
  L.desc.n_cold = groups[0].n_cold;
  L.desc.inter = groups[0].inter;
  for (int g = 0; g < n_groups; g++) {
    const qmc_group& q = groups[g];
    qmc_layer_group& d = L.desc.g[g];
    d.t13 = q.trellis13;
    d.suh13 = q.suh13;
    d.svh13 = q.svh13;
    d.t2 = q.trellis2;
    d.suh2 = q.suh2;
    d.svh2 = q.svh2;
    d.remap = q.remap;
    d.bs13 = (int64_t)(H / 16) * (2 * q.inter / 16) * q.words13;
    d.bs2 = (int64_t)(q.inter / 16) * (H / 16) * q.words2;
    d.n_experts = q.n_experts;
    d.bits13 = q.words13 / 16;
    d.bits2 = q.words2 / 16;
    d.cb = q.cb;
    if (q.words13 % 16 || q.words2 % 16 || d.bits13 < 2 || d.bits13 > 4 || d.bits2 < 2 ||
        d.bits2 > 4 || q.cb < 0 || q.cb > 2 || q.inter % 128 || q.inter != L.desc.inter ||
        q.n_cold != L.desc.n_cold)
      L.desc_ok = false;
  }
  L.have = true;
  return 0;
}

extern "C" int qmc_compute(void* ctx, int layer, int rows, const void* x_bf16, const int32_t* ids,
                           const void* w_bf16, float* out_f32) {
  auto* c = (Ctx*)ctx;
  if (!c || !x_bf16 || !ids || !w_bf16 || !out_f32) return kRefuseCompute;
  if (cudaSetDevice(c->dev) != cudaSuccess) {
    set_err("cudaSetDevice(%d): %s", c->dev, cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  if (layer < 0 || layer >= QMC_NL || !c->layers[layer].have) {
    /* Not a layer with cuda-exl3 stacks - the mixed-K pack (38/39) has none. When this context is
     * the proxy's hook, the server's own path answers it through the trampoline install() handed
     * to qmc_set_fallback; that is the shipped behaviour, and refusing here instead would fail a
     * request the server can still serve. A caller that kept the Python dispatch (no fallback
     * installed) never reaches this branch for such a layer. */
    if (c->fb_fn) {
      typedef int (*qes_fb_fn)(void*, int, int, const void*, const int32_t*, const void*, float*);
      return ((qes_fb_fn)c->fb_fn)(c->fb_ctx, layer, rows, x_bf16, ids, w_bf16, out_f32);
    }
    set_err("layer %d has no registered cuda-exl3 groups", layer);
    return kRefuseCompute;
  }
  if (rows <= 0 || rows > QMC_MAX_ROWS) {
    set_err("rows=%d out of range", rows);
    return kRefuseCompute;
  }
  std::lock_guard<std::mutex> lk(c->mu);
  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  c->calls++;

  /* The floor is the only thing here that needs the machine's memory, and the shipping pair leaves
   * SEAM_FWD_FLOOR_MIB unset: without this guard every request opened /proc/meminfo for a value
   * nobody read (4.0 us per call measured on the GB10 host, 2 % of a 64-row 203 us piece). With a
   * floor set the read stays where it was, once per call, before the piece picker. */
  int avail = -1;
  if (c->fwd_floor > 0) {
    avail = avail_mib();
    if (avail >= 0 && avail < c->fwd_floor) {
      c10::cuda::CUDACachingAllocator::emptyCache();
      avail = avail_mib();
    }
  }
  if (c->fwd_floor > 0 && avail >= 0 && avail < c->fwd_floor) {
    c->refused++;
    fprintf(stderr,
            "SEAM_COMPUTE_REFUSED layer=%d rows=%d why=memory need_mib=%d avail_mib=%d "
            "swapped_mib=%d calls=%lld refusals=%lld - answering nothing, the session ends, the "
            "server keeps listening\n",
            layer, rows, c->fwd_floor, avail, swap_used_mib(), (long long)c->calls,
            (long long)c->refused);
    fflush(stderr);
    return kRefuseMem;
  }

  cudaStream_t s = at::cuda::getCurrentCUDAStream(c->dev).stream();
  cudaEventRecord(c->ev0, s);
  int rc = 0;
  const int step = pick_step(*c, rows);
  /* SEAM_MOE_CPP_SIDE: one join event per piece of this call, so run_piece can hand the compute
   * stream's completion to the side stream without an event ever being re-recorded under a pending
   * wait. Grown once per shape and kept for the next call. */
  if (c->side && step > 0) {
    const size_t npieces = (size_t)((rows + step - 1) / step);
    while (c->side_join.size() < npieces) {
      cudaEvent_t e = nullptr;
      if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        break;
      }
      c->side_join.push_back(e);
    }
  }
  /* Bound mode leaves every bad-id flag on device until here: run_piece clears the flag on the
   * first piece (once the buffers exist) and this request's single sync, which the timing already
   * needed, carries the verdict back. */
  const bool req_bound = c->bound && !c->dump && std::min(step, rows) <= kBoundMaxRows;
  /* A layer-path piece defers its bad-id verdict the same way; it may be the short last piece
   * of a request whose earlier pieces read their flag back inline, so count them, not the mode. */
  const int64_t layer_before = c->layer_pieces;
  int piece = 0;
  /* x and the reply are read and written in place through the slab's device alias (no staging
   * copies, see run_piece): resolve both once per call. On the served path these are the seam's
   * cudaHostAllocMapped slab, so the alias always exists; anything else the device cannot address
   * is refused, never read through a host pointer. */
  const uint8_t* xd = (const uint8_t*)dev_alias(x_bf16, c->dev);
  uint8_t* od = (uint8_t*)dev_alias(out_f32, c->dev);
  if (!xd || !od) {
    set_err("request/response buffer not device-addressable (x=%p out=%p): pass pinned mapped or "
            "device memory", x_bf16, (void*)out_f32);
    rc = kRefuseCompute;
  }
  /* The reply slot's element size follows out_mode (4 B fp32, 2 B under the seam's 16-bit wire):
   * advancing a float* would put piece two of a 16-bit reply at twice its offset, past the rows the
   * client reads, while the rows it does read stay at their previous request's bytes. */
  const size_t out_elem = c->out_mode ? 2 : 4;
  for (int r0 = 0; r0 < rows && rc == 0; r0 += step) {
    const int n = std::min(step, rows - r0);
    rc = run_piece(*c, layer, n, xd + (size_t)r0 * c->hidden * 2, ids + (size_t)r0 * c->topk,
                   (const uint8_t*)w_bf16 + (size_t)r0 * c->topk * 2,
                   (float*)(od + (size_t)r0 * c->hidden * out_elem), r0 == 0, piece);
    ++piece;
  }
  const bool req_flag = req_bound || c->layer_pieces != layer_before;
  if (rc == 0 && req_flag &&
      cudaMemcpyAsync(c->b.h_bad, c->b.bad, 4, cudaMemcpyDeviceToHost, s) != cudaSuccess) {
    set_err("id flag readback failed: %s", cudaGetErrorString(cudaGetLastError()));
    rc = kRefuseCompute;
  }
  /* The reply of the last piece must be in the slot before this call returns: mark the side
   * stream's completion and make the compute stream (whose sync below is the one this call does)
   * wait for it. Empty side stream = a no-op. */
  if (c->side && c->side_stream) {
    if (cudaEventRecord(c->ev_side_done, c->side_stream) == cudaSuccess)
      cudaStreamWaitEvent(s, c->ev_side_done, 0);
  }
  cudaEventRecord(c->ev1, s);
  const cudaError_t se = cudaStreamSynchronize(s);
  if (rc == 0 && se != cudaSuccess) {
    set_err("stream sync: %s", cudaGetErrorString(se));
    rc = kRefuseCompute;
  }
  if (rc == 0 && req_flag && *c->b.h_bad) {
    set_err("request rejected: an id outside [-1, n_cold) for layer %d rows %d", layer, rows);
    rc = kRefuseCompute;
  }
  float dev_ms = 0.f;
  if (cudaEventElapsedTime(&dev_ms, c->ev0, c->ev1) != cudaSuccess) dev_ms = -1.f;
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  c->last_wall = (int64_t)(ts1.tv_sec - ts0.tv_sec) * 1000000000LL + (ts1.tv_nsec - ts0.tv_nsec);
  c->last_dev = (int64_t)(dev_ms * 1e6);
  if (c->prof && rc == 0 && rows >= c->prof_min_rows)
    prof_report(*c, layer, rows, piece, step);
  if (rc != 0) {
    c->refused++;
    const bool usable = cuda_alive();
    fprintf(stderr,
            "SEAM_COMPUTE_%s layer=%d rows=%d err=%s avail_mib=%d swapped_mib=%d calls=%lld "
            "refusals=%lld cuda_ok=%d\n",
            usable ? "REFUSED" : "FAIL", layer, rows, g_err.c_str(), avail_mib(), swap_used_mib(),
            (long long)c->calls, (long long)c->refused, (int)usable);
    fflush(stderr);
    /* worker_seam_server's fail-stop exits 70 on the same condition; here the caller sees the
     * refusal code and its own health probe decides. */
  }
  if (!c->reported) {
    c->reported = true;
    int nlayers = 0;
    for (auto& l : c->layers) nlayers += (int)l.have;
    fprintf(stderr,
            "SEAM_MOE_CPP ready=1 device=%d hidden=%d topk=%d ws_rows=%d ws_auto=%d layers=%d "
            "groups=%d out=%s\n",
            c->dev, c->hidden, c->topk, c->ws_rows, (int)c->ws_auto, nlayers,
            c->n_groups_max, c->out_mode == 1 ? "bf16" : (c->out_mode == 2 ? "fp16" : "fp32"));
    fflush(stderr);
  }
  return rc;
}

extern "C" int qmc_last_call_ns(void* ctx, int64_t* wall_ns, int64_t* dev_ns) {
  auto* c = (Ctx*)ctx;
  if (!c) return -1;
  if (wall_ns) *wall_ns = c->last_wall;
  if (dev_ns) *dev_ns = c->last_dev;
  return 0;
}

extern "C" int qmc_stats(void* ctx, int64_t* calls, int64_t* refused) {
  auto* c = (Ctx*)ctx;
  if (!c) return -1;
  if (calls) *calls = c->calls;
  if (refused) *refused = c->refused;
  return 0;
}

extern "C" int qmc_out_mode(void* ctx) {
  auto* c = (Ctx*)ctx;
  return c ? c->out_mode : -1;
}

extern "C" int qmc_set_out_mode(void* ctx, int mode) {
  auto* c = (Ctx*)ctx;
  if (!c || mode < 0 || mode > 2) return -1;
  std::lock_guard<std::mutex> lk(c->mu);
  const int old = c->out_mode;
  c->out_mode = mode;
  return old;
}

/* Alignment knobs. block_m = 0 keeps the automatic tier (previous call's live count); a positive
 * value freezes it, which is what a captured graph needs - the tier may not change shape inside a
 * capture. */
extern "C" int qmc_set_block_m(void* ctx, int block_m) {
  auto* c = (Ctx*)ctx;
  if (!c || block_m < 0) return -1;
  std::lock_guard<std::mutex> lk(c->mu);
  const int old = c->block_m;
  c->block_m = block_m;
  return old;
}

/* The address qes_server_set_fn installs. qmc_compute's arguments already are qes_server_fn's
 * (void* ctx, then exactly the proxy's six), so the install is a cast and not an adapter - and a
 * served request for a registered layer never re-enters the interpreter. */
extern "C" void* qmc_entry(void) { return (void*)(uintptr_t)&qmc_compute; }

/* Where qmc_compute sends a layer it holds no stacks for: fn is a qes_server_fn (the caller's
 * trampoline over Server.compute) and fnctx what the proxy passes back to it. Called under the
 * context's lock because it is set once at load, after the layer table is built. */
extern "C" int qmc_set_fallback(void* ctx, void* fn, void* fnctx) {
  auto* c = (Ctx*)ctx;
  if (!c) return -1;
  std::lock_guard<std::mutex> lk(c->mu);
  c->fb_fn = fn;
  c->fb_ctx = fnctx;
  return 0;
}

/* Debug: copy the alignment of the last piece served for `layer` back to the caller, so the bench
 * can compare it with align_block() for the same ids. Returns 0, or -1 with qmc_last_error() set;
 * *nblk/*total are 0 when the layer has no group `group` or nothing was served. */
extern "C" int qmc_dump_align(void* ctx, int layer, int group, int32_t* sorted, int32_t* eids,
                              int* nblk, int* total, int* block_m) {
  auto* c = (Ctx*)ctx;
  if (!c || !nblk || !total || !block_m) return -1;
  *nblk = *total = *block_m = 0;
  if (!c->dump) {
    set_err("qmc_dump_align: SEAM_MOE_CPP_DUMP is off");
    return -1;
  }
  if (layer < 0 || layer >= QMC_NL || group < 0 || group >= QMC_MAX_GROUPS) {
    set_err("qmc_dump_align: bad layer %d group %d", layer, group);
    return -1;
  }
  Layer& L = c->layers[layer];
  Buffers& b = c->b;
  if (group >= L.n_groups || L.dump_rows == 0) return 0;
  *nblk = L.dump_nblk[group];
  *total = L.dump_total[group];
  *block_m = L.dump_bm[group];
  if (sorted && *total > 0)
    memcpy(sorted, b.h_sorted + (size_t)group * b.cap_sorted, (size_t)*total * 4);
  if (eids && *nblk > 0)
    memcpy(eids, b.h_eids + (size_t)group * b.cap_sorted, (size_t)*nblk * 4);
  return 0;
}

extern "C" const char* qmc_last_error(void) { return g_err.c_str(); }

extern "C" void qmc_close(void* ctx) {
  auto* c = (Ctx*)ctx;
  if (!c) return;
  {
    std::lock_guard<std::mutex> lk(c->mu);
    for (auto& l : c->layers) {
      for (auto& g : l.grp) {
        g.t13 = at::Tensor();
        g.suh13 = at::Tensor();
        g.svh13 = at::Tensor();
        g.t2 = at::Tensor();
        g.suh2 = at::Tensor();
        g.svh2 = at::Tensor();
      }
    }
    if (c->ev0) cudaEventDestroy(c->ev0);
    if (c->ev1) cudaEventDestroy(c->ev1);
    c->ev0 = c->ev1 = nullptr;
    for (auto& e : c->prof_ev)
      if (e) cudaEventDestroy(e);
    c->prof_ev.clear();
    for (auto& e : c->side_join)
      if (e) cudaEventDestroy(e);
    c->side_join.clear();
    for (auto& e : c->side_done)
      if (e) cudaEventDestroy(e);
    c->side_done.clear();
    if (c->ev_side_done) cudaEventDestroy(c->ev_side_done);
    c->ev_side_done = nullptr;
    if (c->side_stream) cudaStreamDestroy(c->side_stream);
    c->side_stream = nullptr;
  }
  delete c; /* Buffers' destructor frees the rest */
}
