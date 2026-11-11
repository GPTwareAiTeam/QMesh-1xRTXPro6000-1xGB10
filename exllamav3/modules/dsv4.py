from __future__ import annotations
from typing_extensions import override
import math
import torch
import hashlib
import os
import json
import torch.nn.functional as F
from ..model.config import Config
from ..model.model_tp_alloc import TPAllocation
from .module import Module
from .linear import Linear
from .rmsnorm import RMSNorm
from ..ext import exllamav3_ext as ext
from ..util.rope import RopeStyle, yarn_inv_freq
from ..util.tensor import g_tensor_cache, get_for_device
from ..util.device_copy import to_device
from .quant.exl3 import LinearEXL3
from ..cache.dsa import DSV4LayerState, CacheLayer_dsa
from .multilinear import MultiLinear
from .attention_fn.dsa_triton import dsa_attn, dsa_indexer_scores, dsa_indexer_scores_rowmap, dsa_debug_bounds
from .attention_fn.bc_dsa import bc_dsa_enable, build_bc_dsa, build_bc_dsa_batch

# Multi-job decode via the batched path: one projection pass, one MULTIROW attention
# call and one grouped o_proj over all rows (per-slot state addressed through slot-id /
# per-job position arrays in-kernel). EXL3_DSV4_BATCH_EAGER=0 restores the loop
dsv4_batch_eager = os.environ.get("EXL3_DSV4_BATCH_EAGER", "1") != "0"
# Capture the batched step as one CUDA graph per (cache layer, B, S) after two warmup
# runs; EXL3_DSV4_BATCH_GRAPH=0 keeps the eager batched chain
dsv4_batch_graph = os.environ.get("EXL3_DSV4_BATCH_GRAPH", "1") != "0"
# bsz==1 single-token decode through the batched body (device-driven attention core, capturable in a
# piecewise segment). OFF by default: measured ~2.3 ms/token SLOWER than the eager core and not
# token-exact yet, so it stays experimental (EXL3_DSV4_B1_BATCH=1 to test it)
dsv4_b1_batch = os.environ.get("EXL3_DSV4_B1_BATCH", "0") != "0"
# No-q-head-norm models (DSV4.1) at bsz > 1: EXL3_DSV41_BATCH=1 takes the eager batched body (one projection pass and
# one attention call over all jobs) instead of the per-job loop. Its projections apply the q norm only where the
# model has one (_rope_qkv); the whole-batch BC graph stays off for these models (it applies q_ones unconditionally)
dsv41_batch = os.environ.get("EXL3_DSV41_BATCH", "0") == "1"
# Candidate pre-filter with a fixed-capacity device-side entry list instead of the eager
# nonzero() + boolean index (which costs a host sync per reading layer and cannot be captured).
# Same entry set, same order, same dsa_topk call, so the selection is bitwise identical;
# EXL3_DSV4_CAND_FIXED=0 restores the nonzero path (A/B only)
dsv4_cand_fixed = os.environ.get("EXL3_DSV4_CAND_FIXED", "1") != "0"
# Batched-step indexer key write: one projection over the step's whole new-entry run, with eager's
# row count and rope base (the per-entry loop rounds 1-2 fp16 ulps away from the eager keys). The
# run's BASE entry is read from the per-step device position -- a host base would be baked into the
# captured graph and replay stale for every later step of the same run length. Strict dump parity
# (bitwise, all 40 layers x 2 ranks) at consecutive multi-row steps, identical tokens and identical
# accept/reject counters. EXL3_DSV4_IDX_WK_GROUPED=0 restores the per-entry loop (A/B only)
dsv4_idx_wk_grouped = os.environ.get("EXL3_DSV4_IDX_WK_GROUPED", "1") != "0"
# Parity dump for the candidate paths: at one position, save the per-row selected entry ids
# (sorted, so only the selected set is compared) of every indexer reading layer. Temporary gate
# instrumentation, off unless EXL3_DSV4_SELDUMP is set
dsv4_seldump = os.environ.get("EXL3_DSV4_SELDUMP")
dsv4_seldump_pos = int(os.environ.get("EXL3_DSV4_SELDUMP_POS", "-1"))
# The dump must come from a decode slab (the multi-row candidate path), not from the prompt's last
# prefill chunk, which also sits above any position threshold and takes the eager union path
dsv4_seldump_maxseq = int(os.environ.get("EXL3_DSV4_SELDUMP_MAXSEQ", "17"))
dsv4_cand_dbg = os.environ.get("EXL3_DSV4_CANDDBG")
_cand_dbg_seen = set()
# Cross-rank selection diagnostic (read-only experiment, off by default; both env vars unset means
# every branch below is skipped). SEL=<pos0>: from the first decode slab at/after <pos0>, both
# ranks are forced onto rank 0's selected entry list through the params["backend"] all_reduce, so
# the attention that follows uses one selection instead of two rank-local ones. DBG=1: print
# md5-of-bytes of this layer's indexer inputs (x, q_idx, wts, pool[:ec]) once per (K, seq, pos0),
# which decides "ranks' inputs differ" vs "score kernel not rank-deterministic"
dsv4_xrank_sel = int(os.environ.get("EXL3_DSV4_XRANK_SEL", "-1"))
dsv4_xrank_dbg = os.environ.get("EXL3_DSV4_XRANK_DBG", "0") != "0"
_xrank_dbg_seen = set()
# Long-context selection inside the captured b1 body (single row, B == 1): 1 lets the body serve the
# top-k regime (ec > index_topk) with its bounded pool-scan top-k, 2 also serves the candidate
# pre-filter (ec > candidate_topk_blocks * block_size): the source layer refreshes the candidate mask
# in the captured step and the reading layers select over their compacted entries. 0 keeps the dense
# regime only (the body stays on the eager path above index_topk)
dsv4_b1_long = int(os.environ.get("EXL3_DSV4_B1_LONG", "0") or 0)
# The body's top-k normally runs the launch-time single-block kernel (t_seq = S); EXL3_DSV4_B1_SPLIT=1
# hands it the device bound instead (t_seq = 0) so a long single-row scan takes the span-split path the
# eager selection uses above 32768 entries
dsv4_b1_split = os.environ.get("EXL3_DSV4_B1_SPLIT", "0") == "1"

# The most recent selection over one pool inside one step, keyed by id(cache layer) -- the pool object,
# so several layers selecting over the same pool (index_source_layer_ids) and its readers all agree on
# it -- valued (step epoch, indices). Not params["dsv4_b_sel"]: Segment._capture runs the body on a
# shallow *copy* of params, so a publisher's write into a per-step key never reaches a reader's
# capture copy; the epoch object is set once per step in b1_prep and is shared by every copy, so a
# reader can still tell a same-step publish from a stale one.
_B_SEL = {}

# Opt-in per-round counters (EXL3_ROUND_PROF=1): which core served each step, so an anatomy run can tell
# graph replays from eager launches without perturbing anything (no-op when the probe is off). Lazy import
# because this module is loaded while exllamav3.generator is still initializing.
_rp_on = os.environ.get("EXL3_ROUND_PROF", "0") == "1"
_rp_note_fn = None

def _rpnote(name, value = 1):
    global _rp_note_fn
    if _rp_note_fn is None:
        try:
            from ..generator.roundprof import note as _n
        except Exception:
            _n = False
        _rp_note_fn = _n
    if _rp_note_fn:
        _rp_note_fn(name, value)

from ..constants import PAGE_SIZE

# Reference: transformers models/deepseek_v4 (paper §2)

def _ext_rope(x, inv_freq, position = 0, position_ids = None, positions = None):
    # In-place GPT-J rotation of a (bsz, seq, heads, rope_dim) tensor via ext.rope. x may be
    # a trailing-slice VIEW of wider heads. De-rotation (paper eq. 26) uses a negated inv_freq
    # table. attn_factor is 1.0 by V4 semantics (yarn mscale never applied to cos/sin).
    ext.rope(
        x, x, None, None, inv_freq, position, positions, position_ids,
        int(RopeStyle.GPTJ), 1.0, None, None, 1e-6, 0.0, 0.0, 0, 1, 0,
    )


# Diff instrumentation (EXL3_DSV4_DIFF_DUMP=<dir>): eager-only dumps of per-layer per-step
# tensors at the same program point in the scalar and batched paths. Off by default.
_dsv4_dump_dir = os.environ.get("EXL3_DSV4_DIFF_DUMP", "")
_dsv4_dump_n = {}

def _dsv4_dump(tag, layer, pos, ec, m, tensors, kl, bt_row):
    """tensors: name -> tensor (post-step values). Pool rows 0..ec-1 are read through bt_row."""
    if not _dsv4_dump_dir:
        return
    import os as _os
    d = {k: v.detach().float().cpu().clone() for k, v in tensors.items() if v is not None}
    if kl is not None and kl.kv_format == "fp4" and ec > 0 and bt_row is not None:
        e = torch.arange(ec, device = kl.pool_fp4.device)
        ri = bt_row[0, e // kl.epp].long() * kl.epp + e % kl.epp
        lat = ext.dsv4_fp4_dequant(kl.pool_fp4.view(-1, 256)[ri].contiguous(),
                                    kl.pool_fp4_scale.view(-1, 32)[ri].contiguous(), False)
        d["pool_c"] = lat[:, :kl.D_c].float().cpu()
        d["pool_r"] = lat[:, kl.D_c:].float().cpu()
        if kl.D_i:
            d["pool_idx"] = ext.dsv4_fp4_dequant(kl.pool_idx_fp4.view(-1, 64)[ri].contiguous(),
                                                kl.pool_idx_scale.view(-1, 4)[ri].contiguous(), True).float().cpu()
    if kl is not None and kl.pool_r is not None and ec > 0 and bt_row is not None:
        epp = kl.epp
        e = torch.arange(ec, device = kl.pool_r.device)
        ri = bt_row[0, e // epp].long() * epp + e % epp
        d["pool_r"] = kl.pool_r.view(-1, kl.D_r)[ri].float().cpu()
        if kl.quant:
            d["pool_s"] = kl.pool_s.view(-1, kl.G)[ri].float().cpu()
        else:
            d["pool_c"] = kl.pool_c.view(-1, kl.D_c)[ri].float().cpu()
        if kl.D_i:
            d["pool_idx"] = kl.pool_idx.view(-1, kl.D_i)[ri].float().cpu()
    _os.makedirs(_dsv4_dump_dir, exist_ok = True)
    import torch.distributed as _dist
    _rk = _dist.get_rank() if _dist.is_available() and _dist.is_initialized() else 0
    torch.save(d, f"{_dsv4_dump_dir}/r{_rk}_{tag}_L{layer}_p{pos}.pt")



# Indexer-selection parity dump (EXL3_DSV4_PDUMP=<dir>, EXL3_DSV4_PDUMP_POS=<pos>,
# EXL3_DSV4_PDUMP_WIDTH=<width>): the first step at or past that position writes each layer's raw
# scores / candidate entry list / mask / selected ids and the attention q/kv/out (plus the live pool
# rows, pool_idx/pool_r), from whichever path ran (eager _indexer_topk or the captured body), for an
# offline diff. EXL3_DSV4_PDUMP_WIDTH > 0 restricts the dump to steps of exactly that row count (0 =
# any), so one directory holds one width. Off by default, and never inside a capture (the body's
# Python only runs in the pre-capture warmup). Each dump is a .pt of row-major tensors plus a
# <file>.json sidecar recording width / position / rank / layer / tag.
_pdup_dir = os.environ.get("EXL3_DSV4_PDUMP", "")
_pdup_pos = int(os.environ.get("EXL3_DSV4_PDUMP_POS", "0") or 0)
_pdup_width = int(os.environ.get("EXL3_DSV4_PDUMP_WIDTH", "0") or 0)
# Diagnostic window form: dump EVERY step whose position lies in [POS, POS_END] instead of latching
# the first one, into a per-position subdirectory (<dir>/s<pos>), so the grader can pair positions
# one at a time. Off unless POS_END is set; dump-only and capture-safe.
_pdup_end = int(os.environ.get("EXL3_DSV4_PDUMP_POS_END", "0") or 0)
_pdup_done = set()
_pdup_manifest = set()

def _pdup(tag, layer, pos, width, tensors, pool = None, selection = None):
    if (not _pdup_dir or (_pdup_width and width != _pdup_width) or
        torch.cuda.is_current_stream_capturing()):
        return
    key = (layer, tag, width, pos) if _pdup_end else (layer, tag, width)
    if key in _pdup_done:
        return
    pos = int(pos)
    if pos < _pdup_pos or (_pdup_end and pos > _pdup_end):
        return
    _pdup_done.add(key)
    out_dir = _pdup_dir if not _pdup_end else f"{_pdup_dir}/s{pos}"
    os.makedirs(out_dir, exist_ok = True)
    if _pdup_width and out_dir not in _pdup_manifest:
        # One width per directory: record the strict filter next to the dumps so a directory's
        # contents are self-describing (the .pt dicts carry tensors only)
        _pdup_manifest.add(out_dir)
        mpath = f"{out_dir}/manifest.json"
        if not os.path.exists(mpath):
            with open(mpath, "w") as f:
                json.dump(dict(width = _pdup_width, position = _pdup_pos), f)
    d = {k: v.detach().to("cpu", copy = True) for k, v in tensors.items() if v is not None}
    if selection is not None:
        # Both paths dump one canonical layout so an offline diff compares values, not storage: the
        # pool-scan scores are cut at the step's live entry extent and masked past each row's own
        # causal bound, and the entry-scored (candidate) path gets the same per-row causal mask.
        # Candidate masks drop the blocks that are not live for any row of this width.
        rate, block = selection
        live = (pos + width) // rate
        vis = (pos + 1 + torch.arange(width)) // rate
        s = d.get("scores")
        if s is not None:
            ent = d.get("ent")
            if ent is not None and ent.dim() == 1 and s.dim() == 2 and s.shape[0] == 1 \
                    and ent.numel() == s.shape[1]:
                # The body stores a single row's entry list flat; eager stores it (1, cap)
                ent = ent.unsqueeze(0)
                d["ent"] = ent
            if ent is not None and ent.dim() == 2 and s.shape[:1] == ent.shape[:1]:
                s.masked_fill_(ent >= vis.unsqueeze(1), -float("inf"))
                s.masked_fill_(ent < 0, -float("inf"))
            elif ent is None:
                # Pool-scan scores are pool-entry indexed: cut at the live extent, then apply each
                # row's own causal bound (the b1 kernel may score the tile-padded tail).
                s = s[:, :live]
                cols = torch.arange(s.shape[1])
                s.masked_fill_(cols.unsqueeze(0) >= vis.unsqueeze(1), -float("inf"))
                d["scores"] = s
            # A flat entry list over a multi-row score matrix is already masked in entry space
        krow = d.get("krow")
        if krow is not None and krow.dim() == 1:
            d["krow"] = krow.unsqueeze(0)
        cand = d.get("cand")
        if cand is not None:
            nb = min(cand.shape[1], -(-live // block))
            cand = cand[:, :nb]
            cols = torch.arange(nb)
            cand.masked_fill_(cols.unsqueeze(0) * block >= vis.unsqueeze(1), True)
            d["cand"] = cand
    if pool is not None:
        kl, bt, live = pool
        ids = torch.arange(int(live), device = bt.device)
        rows = bt.reshape(-1)[ids // kl.epp].long() * kl.epp + ids % kl.epp
        if kl.kv_format == "fp4":
            lat = ext.dsv4_fp4_dequant(kl.pool_fp4.view(-1, 256)[rows].contiguous(),
                                        kl.pool_fp4_scale.view(-1, 32)[rows].contiguous(), False)
            d["pool_c"] = lat[:, :kl.D_c].cpu()
            d["pool_r"] = lat[:, kl.D_c:].cpu()
            if kl.pool_idx_fp4 is not None:
                d["pool_idx"] = ext.dsv4_fp4_dequant(kl.pool_idx_fp4.view(-1, 64)[rows].contiguous(),
                                                    kl.pool_idx_scale.view(-1, 4)[rows].contiguous(), True).cpu()
        else:
            if kl.pool_idx is not None:
                d["pool_idx"] = kl.pool_idx.view(-1, kl.D_i)[rows].detach().to("cpu", copy = True)
            d["pool_r"] = kl.pool_r.view(-1, kl.D_r)[rows].detach().to("cpu", copy = True)
            if kl.pool_c is not None:
                d["pool_c"] = kl.pool_c.view(-1, kl.D_c)[rows].detach().to("cpu", copy = True)
    rank = int(os.environ.get("RANK", "0"))
    path = f"{out_dir}/r{rank}s{pos}_L{layer}_{tag}.pt"
    torch.save(d, path)
    with open(path + ".json", "w") as f:
        json.dump(dict(width = width, position = pos, rank = rank, layer = layer, tag = tag), f)


_pdup_last = {}

def _pdup_reg(layer, tag, width, tensors, pool = None, selection = None):
    """Register static capture buffers, keyed by width to avoid stale graph buffers."""
    if _pdup_dir and (not _pdup_width or width == _pdup_width):
        _pdup_last[(layer, tag, width)] = (tensors, pool, selection)

def _pdup_host(layer, params, width):
    """Dump the buffers belonging to the width just replayed."""
    if not _pdup_dir or (_pdup_width and width != _pdup_width):
        return
    rsg = params.get("recurrent_states") or ()
    if not rsg:
        return
    pos = int(rsg[0].position)
    for (l, tag, w), (tensors, pool, selection) in list(_pdup_last.items()):
        if l == layer and w == width:
            _pdup("cap_" + tag, layer, pos, width, tensors, pool, selection)


_arange_cache = {}

def _cached_arange(device, n):
    """Static int32 [0, n): the captured body must not allocate a column index per step, and the
    buffer has to be fully written before the capture starts."""
    key = (str(device), n)
    t = _arange_cache.get(key)
    if t is None:
        _arange_cache[key] = t = torch.arange(n, device = device, dtype = torch.int32)
    return t

# EXL3_DSV4_SEQ_TOKENS: the longest context one sequence may reach. Per-row scratch of the batched indexer scan (the
# (rows, entries) score matrix) is sized by it instead of the whole pool: with a 5M-token growable cache shared by 8
# users and a 1M per-user cap that is 5x less (458 -> 92 MiB at 48 rows on a ratio-1 pool). 0 = the pool capacity.
_SEQ_TOKENS = int(os.environ.get("EXL3_DSV4_SEQ_TOKENS", "0") or 0)


def _seq_capacity(kl) -> int:
    """Pool entries one sequence can address: capacity, or the per-sequence token cap when set"""
    if _SEQ_TOKENS > 0:
        return min(kl.capacity, -(-_SEQ_TOKENS // kl.compress_rate))
    return kl.capacity


class DSV4CompressorState:
    """
    Cross-chunk state interface for one compressor (one entry name in HF terms). The
    compressor calls, in order:

    - entry_count (positioning)
    - get_buffer (rows carried from previous chunks)
    - store_rows (persist this chunk's projected rows)
    - advance_entries (windows emitted; consume rows)
    - get_overlap / set_overlap (Ca slice, overlapping scheme only).

    This base class is the simple in-memory implementation (whole-tensor buffers, no rewind); the
    cache layer provides a ring-backed subclass whose bookkeeping is derived from the absolute
    position so that rewind is pure cursor arithmetic.
    """

    def __init__(self):
        self._rows_kv = None       # (bsz, n, proj_width) fp32: rows not yet consumed
        self._rows_gate = None
        self._overlap = None       # (kv (bsz, m, hd), gate (bsz, m, hd)) fp32 or None
        self._count = 0

    @property
    def entry_count(self):
        return self._count

    def get_buffer(self):
        if self._rows_kv is None or self._rows_kv.shape[1] == 0:
            return None
        return self._rows_kv, self._rows_gate

    def store_rows(self, kv_new, gate_new):
        if self._rows_kv is None or self._rows_kv.shape[1] == 0:
            self._rows_kv, self._rows_gate = kv_new, gate_new
        else:
            self._rows_kv = torch.cat([self._rows_kv, kv_new], dim = 1)
            self._rows_gate = torch.cat([self._rows_gate, gate_new], dim = 1)

    def advance_entries(self, nw, m):
        self._count += nw
        if nw and self._rows_kv is not None:
            self._rows_kv = self._rows_kv[:, nw * m:].contiguous()
            self._rows_gate = self._rows_gate[:, nw * m:].contiguous()

    def get_overlap(self):
        return self._overlap

    def set_overlap(self, kv, gate):
        self._overlap = (kv.clone(), gate.clone())


class DSV4Compressor:
    """
    Torch-composed compressor shared by HCA (width = head_dim, non-overlapping) and CSA /
    indexer (width = 2 * head_dim: the Ca / Cb overlapping-window scheme). Stateless when
    state is None (complete windows within the chunk only, remainder discarded), stateful
    otherwise (buffer + overlap carried across chunks).

    Not a Module: owned by DSV4Attention, which registers the child Linears/norm.
    """

    # BC scratch row budget; must match BC_DSV4Compressor::MAX_QLEN (dsv4_compressor.h)
    BC_MAX_QLEN = 32

    def __init__(
        self,
        attn,
        key,
        head_dim,
        compress_rate,
        overlapping,
        qmap,
        select_hq_bits,
        wkv: Linear | None = None,
        wgate: Linear | None = None,
        norm: RMSNorm | None = None,
    ):
        cfg = attn.config
        proj_width = 2 * head_dim if overlapping else head_dim
        self.head_dim = head_dim
        self.rope_dim = attn.rope_head_dim if head_dim > attn.rope_head_dim else head_dim
        self.compress_rate = compress_rate
        self.overlapping = overlapping
        self.key = key
        self.wkv = wkv if wkv is not None else Linear(
            cfg, f"{key}.wkv", attn.hidden_size, proj_width, qmap = qmap, out_dtype = torch.half, trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.wgate = wgate if wgate is not None or compress_rate == 1 and not overlapping else Linear(
            cfg, f"{key}.wgate", attn.hidden_size, proj_width, qmap = qmap, out_dtype = torch.half, trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.norm = norm if norm is not None else RMSNorm(cfg, f"{key}.norm", attn.rms_norm_eps)
        self.ape = None  # (compress_rate, proj_width) fp32, loaded by DSV4Attention.load
        self.bc = None   # BC_DSV4Compressor companion (cached path), built lazily
        self.mg_pair = False   # 2-expert wkv/wgate exl3_mgemm available (set in _build_fused)
        self.fused_ready = False
        self.fused_inv_freq = None
        self.fused_norm_w = None


    def make_bc(self, inv_freq):
        # Arm the fused cached path. The BC companion itself is built lazily on the first
        # forward_fused call: at load time weight tensors may still be DEFERRED (stloader fills
        # them asynchronously), so any snapshot/copy taken here could capture zeros.
        self.fused_inv_freq = inv_freq


    def _build_fused(self):
        # Build the C++ bound class: whole forward (projections + fused compress kernels) in
        # one transition. Requires unpadded projections (always true for the real V4 dims);
        # quantized and fp16 checkpoints both supported.

        self.fused_ready = True

        # bf16 checkpoints: norm weights are O(1), bf16 -> fp16 is exact
        self.fused_norm_w = self.norm.weight.data
        if self.fused_norm_w.dtype != torch.half:
            self.fused_norm_w = self.fused_norm_w.half().contiguous()
        if self.wgate is None:
            return
        wkv_i, wgate_i = self.wkv.inner, self.wgate.inner
        if (
            self.wkv.out_features != self.wkv.out_features_unpadded or
            self.wgate.out_features != self.wgate.out_features_unpadded
        ):
            # padded projections: python fallback path
            return

        W = self.wkv.out_features
        device = self.ape.device

        kv_scratch = torch.empty((self.BC_MAX_QLEN, W), dtype = torch.half, device = device)           # TODO: tensor cache
        gate_scratch = torch.empty((self.BC_MAX_QLEN, W), dtype = torch.half, device = device)
        # 2 slabs: the batched wkv+wgate mgemm writes one transformed input per expert
        xh_scratch = torch.empty((2 * self.BC_MAX_QLEN, self.wkv.in_features), dtype = torch.half, device = device)

        args = dict(
            wkv_exl3 = None, wkv_fp16 = None, wgate_exl3 = None, wgate_fp16 = None,
            ape = self.ape, norm_w = self.fused_norm_w, rms_norm_eps = self.norm.rms_norm_eps,
            inv_freq = self.fused_inv_freq, m = self.compress_rate,
            kv_scratch = kv_scratch, gate_scratch = gate_scratch, xh_scratch = xh_scratch,
        )
        for tag, inner in [("wkv", wkv_i), ("wgate", wgate_i)]:
            if isinstance(inner, LinearEXL3):
                args[f"{tag}_exl3"] = inner.bc
            elif hasattr(inner, "weight") and inner.weight.dtype == torch.half:
                args[f"{tag}_fp16"] = ext.BC_LinearFP16(inner.weight, getattr(inner, "bias", None))
            else:
                return

        # Batched wkv+wgate projection: one 2-expert exl3_mgemm when formats match
        if (
            isinstance(wkv_i, LinearEXL3) and
            isinstance(wgate_i, LinearEXL3) and
            wkv_i.K == wgate_i.K and
            wkv_i.mcg == wgate_i.mcg and
            wkv_i.mul1 == wgate_i.mul1
        ):
            args["mg_trellis"] = torch.tensor(
                [wkv_i.trellis.data_ptr(), wgate_i.trellis.data_ptr()],
                dtype = torch.long, device = device
            )
            args["mg_suh"] = torch.tensor(
                [wkv_i.suh.data_ptr(), wgate_i.suh.data_ptr()],
                dtype = torch.long, device = device
            )
            args["mg_svh"] = torch.tensor(
                [wkv_i.svh.data_ptr(), wgate_i.svh.data_ptr()],
                dtype = torch.long, device = device
            )
            args["mg_indices"] = torch.arange(2, dtype = torch.long, device = device).unsqueeze(0)

        self.mg_pair = args.get("mg_suh") is not None
        if self.mg_pair:
            self.mg_pair_tables = (args["mg_trellis"], args["mg_suh"], args["mg_svh"], args["mg_indices"])
            self.mg_pair_K = wkv_i.K
            self.mg_pair_mcg = wkv_i.mcg
            self.mg_pair_mul1 = wkv_i.mul1
        self.bc = ext.BC_DSV4Compressor(**args)


    def unmake_bc(self):
        self.bc = None
        self.fused_ready = False
        self.fused_inv_freq = None
        self.fused_norm_w = None


    def forward_fused(self, x, params, buf_kv, buf_gate, ovl, dest_a, dest_b, position,
                      pool_bt = None, pool_epp = 0, stage_rel = False, position_tensor = None):
        """Cached-path forward (bsz 1): project + window-pool + norm + rope, writing emitted
        entries straight into the per-slot pools and updating the ring/snapshot state. One
        C++ transition via the BC companion when the chunk fits its scratch, else the two
        Linear forwards + the fused kernels."""
        bsz, seq, _ = x.shape
        if not self.fused_ready:
            self._build_fused()
        # The BC companion bypasses Linear.forward, which must run during conversion so the
        # capture/override machinery sees the projection inputs
        use_bc = self.bc is not None and seq <= self.BC_MAX_QLEN and \
            not any(k in params for k in ("capture", "quant_preserve", "ovr", "reconstruct"))
        if use_bc:
            self.bc.run(x[0], buf_kv, buf_gate, ovl, dest_a, dest_b, position, position_tensor, None,
                        pool_bt, pool_epp, stage_rel)
        else:
            kv = self.wkv.forward(x, params)[0]
            gate = self.wgate.forward(x, params)[0] if self.wgate is not None else kv
            ext.dsv4_compress(
                kv, gate, buf_kv, buf_gate, ovl, self.ape, self.fused_norm_w,
                self.norm.rms_norm_eps, self.fused_inv_freq, dest_a, dest_b, position,
                position_tensor, self.compress_rate, None, pool_bt, pool_epp, stage_rel,
            )


    def forward(self, x, params, inv_freq, state: DSV4CompressorState | None = None):
        """x (bsz, seq, hidden) half. Returns newly emitted compressed entries
        (bsz, n_windows, head_dim) half, roped at their window positions
        (w + entry_count) * compress_rate. With a state, sub-window remainders are buffered
        for the next chunk and the Ca overlap slice is carried; entry_count is advanced."""
        bsz, seq, _ = x.shape
        m = self.compress_rate
        kv_new = self.wkv.forward(x, params).float()
        gate_new = self.wgate.forward(x, params).float()
        kv, gate = kv_new, gate_new
        fwp = 0
        if state is not None:
            fwp = state.entry_count * m
            buf = state.get_buffer()
            if buf is not None:
                kv = torch.cat([buf[0], kv], dim = 1)
                gate = torch.cat([buf[1], gate], dim = 1)
            state.store_rows(kv_new, gate_new)
        usable = (kv.shape[1] // m) * m
        if usable == 0:
            return x.new_zeros((bsz, 0, self.head_dim))
        kv = kv[:, :usable].view(bsz, -1, m, kv.shape[-1])
        gate = gate[:, :usable].view(bsz, -1, m, gate.shape[-1]) + self.ape
        nw = kv.shape[1]
        if state is not None:
            state.advance_entries(nw, m)
        if self.overlapping:
            hd = self.head_dim
            new_kv = kv.new_zeros((bsz, nw, 2 * m, hd))
            new_gate = gate.new_full((bsz, nw, 2 * m, hd), -float("inf"))
            new_kv[:, :, m:] = kv[..., hd:]
            new_gate[:, :, m:] = gate[..., hd:]
            if nw > 1:
                new_kv[:, 1:, :m] = kv[:, :-1, :, :hd]
                new_gate[:, 1:, :m] = gate[:, :-1, :, :hd]
            if state is not None:
                ovl = state.get_overlap()
                if ovl is not None:
                    new_kv[:, 0, :m] = ovl[0]
                    new_gate[:, 0, :m] = ovl[1]
                # NOTE: the saved gate slice already carries the position bias (ape); it is
                # not re-added when restored into the next window's first half (HF semantics)
                state.set_overlap(kv[:, -1, :, :hd], gate[:, -1, :, :hd])
            kv, gate = new_kv, new_gate
        comp = (kv * gate.softmax(dim = 2)).sum(dim = 2)
        comp = self.norm.forward(comp.half(), params)
        wpos = (torch.arange(nw, device = x.device, dtype = torch.int) * m + fwp)
        _ext_rope(comp.view(bsz, nw, 1, self.head_dim)[..., -self.rope_dim:], inv_freq,
                  position_ids = wpos.unsqueeze(0).expand(bsz, -1).contiguous())
        return comp


    def modules(self):
        return [self.wkv, self.wgate, self.norm]


    def tp_export(self, plan, producer):
        # The compressor produces headless shared pool entries: fully replicated per rank
        return {
            "args": {
                "key": self.key,
                "head_dim": self.head_dim,
                "compress_rate": self.compress_rate,
                "overlapping": self.overlapping,
            },
            "wkv": self.wkv.tp_export(plan, producer),
            # V4.1's compress-rate-1 pool has no gate
            "wgate": self.wgate.tp_export(plan, producer) if self.wgate is not None else None,
            "norm": self.norm.tp_export(plan, producer),
            "ape": producer.send(self.ape),
        }


    @staticmethod
    def tp_import(local_context, exported, plan, attn):
        consumer = local_context["consumer"]
        def _imp(name):
            e = exported[name]
            return e["cls"].tp_import(local_context, e, plan) if e is not None else None
        comp = DSV4Compressor(
            attn,
            qmap = None,
            select_hq_bits = 0,
            wkv = _imp("wkv"),
            wgate = _imp("wgate"),
            norm = _imp("norm"),
            **exported["args"],
        )
        comp.ape = consumer.recv(exported["ape"], cuda = True)
        return comp


class DSV4Attention(Module):

    # Query rows per indexer score slab: the slab's score matrix is rows x context entries, the largest prefill
    # transient past ~300k tokens. 128 rows: 256 MiB at a 1M context (640 at 256 rows), +1.6% prefill time at 131k
    SCORE_SLAB = int(os.environ.get("EXL3_DSV4_SCORE_SLAB", "128"))

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        layer_type: str,  # "sliding" | "csa" | "hca"
        hidden_size: int,
        num_q_heads: int,
        head_dim: int,
        rope_head_dim: int,
        q_lora_rank: int,
        o_groups: int,
        o_lora_rank: int,
        sliding_window: int,
        compress_rate: int | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        rope_theta: float = 10000.0,
        compress_rope_theta: float = 160000.0,
        rope_scaling: dict | None = None,
        rms_norm_eps: float = 1e-6,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        qbits_key: str = "bits",
        select_hq_bits: int = 0,
        q_a: Linear | None = None,
        q_norm: RMSNorm | None = None,
        q_b: Linear | None = None,
        wkv: Linear | None = None,
        kv_norm: RMSNorm | None = None,
        wo_a: list | None = None,
        wo_b: Linear | None = None,
        idx_wq_b: Linear | None = None,
        idx_weights: Linear | None = None,
        tp_defer_compressors: bool = False,
        kv_source: int | None = None,
        kv_owner: DSV4Attention | None = None,
        indexer_mode: str | None = None,
        q_head_norm: bool = True,
        candidate_mode: str | None = None,
        candidate_topk_blocks: int = 0,
        candidate_block_size: int = 1,
    ):
        super().__init__(config = config, key = key, qmap = None)
        self.q_priority = 2 + select_hq_bits
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.sliding_window = sliding_window
        self.compress_rate = compress_rate
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.out_dtype = out_dtype
        self.sm_scale = head_dim ** -0.5

        # For the model-level allocator conventions (not a KV cache module yet -- M2)
        self.num_kv_heads = 1

        # TP bookkeeping; set before the zero-width early return below (forward() and the
        # TP import glue touch these on ranks that hold none of this layer's o_groups)
        self.tp_reduce = False
        self.tp_mode = False
        self.has_split_cache = False
        self.tp_recurrent_lookup = {}
        self.tp_cache_lookup = {}
        self.recurrent_layers = []
        self.cache_layers = []
        self.sinks = None  # (num_q_heads,) fp32
        self.compressor = None
        self.indexer = None
        self.idx_wq_b = None
        self.idx_weights = None
        self.idx_wk = self.idx_k_norm = None
        self.kv_source = None if layer_type == "sliding" else layer_idx if kv_source is None else kv_source
        self.kv_owner = kv_owner
        self.kv_mirror = False
        self.indexer_mode = indexer_mode or ("full" if layer_type == "csa" else None)
        self.q_head_norm = q_head_norm
        self.candidate_mode = candidate_mode
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.inv_freq_main = None
        self.inv_freq_compress = None
        self.wo_a_multi = None
        self.woa_indices = None
        self.woa_multi_ready = False
        self.x_fan = None
        self.q_fan = None
        self.x_fan_ready = False
        self.qb_multi = self.wob_multi = None
        self._one_idx = None
        self._fan_scratch = {}
        self._bgraph_state = {}

        if num_q_heads == 0:
            # Zero-width TP shard: no weights, no state, no caps (keeps the rank out of
            # the child's kv/recurrent module lists); forward() contributes zeros
            self.q_a = self.q_norm = self.q_b = self.wkv = self.kv_norm = None
            self.wo_a = []
            self.wo_b = None
            return

        self.q_a = q_a if q_a is not None else Linear(
            config,
            f"{key}.wq_a",
            hidden_size,
            q_lora_rank,
            qmap = qmap,
            out_dtype = torch.half,
            qbits_key = qbits_key,
            trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.q_norm = q_norm if q_norm is not None else RMSNorm(config, f"{key}.q_norm", rms_norm_eps)
        self.q_b = q_b if q_b is not None else Linear(
            config,
            f"{key}.wq_b",
            q_lora_rank,
            num_q_heads * head_dim,
            qmap = f"{key}.q_b",
            out_dtype = torch.half,
            qbits_key = qbits_key,
            trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.wkv = wkv if wkv is not None else Linear(
            config,
            f"{key}.wkv",
            hidden_size,
            head_dim,
            qmap = qmap,
            out_dtype = torch.half,
            qbits_key = qbits_key,
            trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.kv_norm = kv_norm if kv_norm is not None else RMSNorm(config, f"{key}.kv_norm", rms_norm_eps)

        group_width = num_q_heads * head_dim // o_groups
        self.wo_a = wo_a if wo_a is not None else [
            Linear(
                config,
                f"{key}.wo_a.slice.{g}",
                group_width,
                o_lora_rank,
                fkey = f"{key}.wo_a",
                frange = [g * o_lora_rank,
                (g + 1) * o_lora_rank],
                frange_dim = 0,
                qmap = f"{key}.o.{g}",
                out_dtype = torch.half,
                qbits_key = qbits_key,
                trim_padded_out = True,
                select_hq_bits = select_hq_bits,
            )
            for g in range(o_groups)
        ]
        self.wo_b = wo_b if wo_b is not None else Linear(
            config,
            f"{key}.wo_b",
            o_groups * o_lora_rank,
            hidden_size,
            qmap = f"{key}.o_b",
            out_dtype = torch.float,
            qbits_key = qbits_key,
            trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.idx_wq_b = idx_wq_b
        self.idx_weights = idx_weights
        if self.kv_source == layer_idx and not tp_defer_compressors:
            self.compressor = DSV4Compressor(
                self,
                f"{key}.compressor",
                head_dim,
                compress_rate,
                overlapping = (layer_type == "csa"),
                qmap = qmap,
                select_hq_bits = select_hq_bits
            )
        if layer_type == "csa" and not tp_defer_compressors:
            self.indexer = DSV4Compressor(
                self,
                f"{key}.indexer.compressor",
                index_head_dim,
                compress_rate,
                overlapping = True,
                qmap = qmap,
                select_hq_bits = select_hq_bits
            )
        if self.indexer_mode == "full" and self.idx_wq_b is None:
            self.idx_wq_b = Linear(
                config,
                f"{key}.indexer.wq_b",
                q_lora_rank,
                index_n_heads * index_head_dim,
                qmap = f"{key}.q_b",
                out_dtype = torch.half,
                qbits_key = qbits_key,
                trim_padded_out = True,
                select_hq_bits = select_hq_bits,
            )

            # Router-like scoring head: 4096 -> 64 (one logit per indexer head), unquantized
            self.idx_weights = Linear(
                config,
                f"{key}.indexer.weights_proj",
                hidden_size,
                index_n_heads,
                qmap = None,
                out_dtype = torch.half,
                pad_to = 1,
            )
        if self.indexer_mode == "full" and self.compressor is not None and self.indexer is None:
            self.idx_wk = Linear(config, f"{key}.indexer.wk", head_dim, index_head_dim, out_dtype = torch.half)
            self.idx_k_norm = RMSNorm(config, f"{key}.indexer.k_norm", rms_norm_eps)

        for m in [
            self.q_a,
            self.q_norm,
            self.q_b,
            self.wkv,
            self.kv_norm,
            *self.wo_a,
            self.wo_b,
            self.idx_wq_b,
            self.idx_wk,
            self.idx_k_norm,
            self.idx_weights
        ]:
            self.register_submodule(m)

        for comp in [
            self.compressor,
            self.indexer
        ]:
            if comp is not None:
                for m in comp.modules():
                    self.register_submodule(m)

        # The DSV4 bundle contexts take integer bitrates; keep these projections off the half-integer rates
        for m in self.modules:
            m.q_half_bits = False

        self.inv_freq_main = None
        self.inv_freq_compress = None

        # Batched wo_a slice projection (exl3_mgemm over the 8 group slices), built lazily
        # on first use; falls back to the per-slice loop for fp16 slices or mixed K
        self.wo_a_multi = None
        self.woa_indices = None
        self.woa_multi_ready = False

        # x-side / q_res-side projection fans (eager cached path), built lazily
        self.x_fan = None
        self.q_fan = None
        self.x_fan_ready = False
        self.qb_multi = self.wob_multi = None
        self._one_idx = None
        self._fan_scratch = {}
        self._bgraph_state = {}

        # Hybrid cache pattern: fixed-size per-slot recurrent state (SWA ring, compressor
        # sub-window buffers, overlap snapshots) plus, for CSA/HCA layers, paged pools that
        # alias the token page table (CacheLayer_dsa)
        self.caps.update({"recurrent_cache": True})
        self.layer_state_cls = DSV4LayerState
        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.cache_layers = []
        if self.kv_source == layer_idx:
            self.caps.update({"kv_cache": True})


    def cache_layer_type(self, default, kwargs: dict):
        """DSA pools are always CacheLayer_dsa; a quantized Cache request packs the pool's
        nope part at k_bits (the rope part and indexer keys stay fp16). Falls back to fp16
        pools when the nope width is not a multiple of 32. kv_format="fp4" stores whole
        latent rows and indexer keys in DeepSeek's trained FP4 format instead."""
        from ..cache import CacheLayer_quant
        kv_format = kwargs.get("kv_format", "fp16")
        if kv_format != "fp16":
            assert not issubclass(default, CacheLayer_quant), "kv_format and k_bits cannot be combined"
            return CacheLayer_dsa, {"kv_format": kv_format}
        if issubclass(default, CacheLayer_quant) and (self.head_dim - self.rope_head_dim) % 32 == 0:
            return CacheLayer_dsa, {"k_bits": kwargs["k_bits"], "v_bits": kwargs.get("v_bits")}
        return CacheLayer_dsa, {}


    @override
    def load(self, device: torch.device, **kwargs):
        # Captured graphs bind weights and cache storage, all of which load may replace.
        getattr(self, "_bc_dsa", {}).clear()
        getattr(self, "_bc_dsa_batch", {}).clear()
        super().load(device, **kwargs)
        stc = self.config.stc
        self.sinks = stc.get_tensor(f"{self.key}.attn_sink", device, no_defer = True, arena = False).float().contiguous()

        self.inv_freq_main = yarn_inv_freq(self.rope_head_dim, self.rope_theta, device)
        self.inv_freq_compress = yarn_inv_freq(
            self.rope_head_dim, self.compress_rope_theta, device, rope_scaling = self.rope_scaling)
        self.inv_freq_main_neg = -self.inv_freq_main
        self.inv_freq_compress_neg = -self.inv_freq_compress

        # Head norms fused into the qkv rope call: unweighted q head norm = ones weight # (same dtype
        # as the kv norm weight; the kernel requires matching dtypes). .data references only
        self.kv_norm_w = self.kv_norm.weight.data
        self.q_ones = torch.ones(self.head_dim, dtype = self.kv_norm_w.dtype, device = device)

        if self.idx_wk is not None:
            self.compressor.ape = torch.zeros((self.compress_rate, self.head_dim), dtype = torch.float, device = device)
            self.compressor.make_bc(torch.zeros_like(self.inv_freq_compress))
        elif self.compressor is not None:
            self.compressor.ape = stc.get_tensor(f"{self.compressor.key}.ape", device, no_defer = True, arena = False).float().contiguous()
            self.compressor.make_bc(self.inv_freq_compress)
        if self.indexer is not None:
            self.indexer.make_bc(self.inv_freq_compress)
            self.indexer.ape = stc.get_tensor(f"{self.indexer.key}.ape", device, no_defer = True, arena = False).float().contiguous()
        for rl in self.recurrent_layers:
            rl.alloc(device)
        for cl in self.cache_layers:
            cl.alloc(device)
        if self.kv_owner is not None:
            for cl in self.kv_owner.cache_layers:
                if cl.device not in (None, device) and device not in cl.mirrors:
                    self.kv_mirror = True
                    cl.alloc_mirror(device)


    @override
    def unload(self):
        if self.kv_mirror:
            for cl in self.kv_owner.cache_layers:
                cl.mirrors.pop(self.device, None)
            self.kv_mirror = False
        super().unload()
        for cl in self.cache_layers:
            cl.free()
        for rl in self.recurrent_layers:
            rl.free()
        self.sinks = None
        if self.compressor is not None:
            self.compressor.ape = None
            self.compressor.unmake_bc()
        if self.indexer is not None:
            self.indexer.ape = None
            self.indexer.unmake_bc()
        self.inv_freq_main = self.inv_freq_compress = None
        self.inv_freq_main_neg = self.inv_freq_compress_neg = None
        self.q_ones = self.kv_norm_w = None
        self.x_fan = None
        self.q_fan = None
        self.x_fan_ready = False
        self.qb_multi = self.wob_multi = None
        self._one_idx = None
        self._fan_scratch = {}
        self._bgraph_state = {}
        self._bc_dsa_batch = {}
        self._bc_dsa = {}
        self.wo_a_multi = None
        self.woa_indices = None
        self.woa_multi_ready = False


    @override
    def get_tensors(self):
        t = {f"{self.key}.attn_sink": self.sinks.contiguous()}
        if self.compressor is not None:
            t[f"{self.compressor.key}.ape"] = self.compressor.ape.contiguous()
        if self.indexer is not None:
            t[f"{self.indexer.key}.ape"] = self.indexer.ape.contiguous()
        return t


    @override
    def weights_numel(self):
        n = sum(m.weights_numel() for m in self.modules)
        n += self.num_q_heads
        for comp in [self.compressor, self.indexer]:
            if comp is not None:
                n += comp.compress_rate * (2 if comp.overlapping else 1) * comp.head_dim
        return n


    @override
    def optimizer_targets(self):
        return [t for m in self.modules for t in m.optimizer_targets()]


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        # Split unit = o_group (hpg heads + one wo_a slice + o_lora_rank rows of wo_b):
        # the grouped o_proj consumes whole 8-head slabs and the attention kernel stores
        # output group-major, so groups are the natural quantum. Everything KV-side is
        # shared-MQA and replicated per rank (q_a, wkv, norms, compressor, indexer, pools,
        # rings); only q_b / wo_a / wo_b / sinks split
        storage_split = self.q_b.storage_size() + self.wo_b.storage_size() + \
            sum(l.storage_size() for l in self.wo_a)
        storage_dev = self.q_a.storage_size() + self.wkv.storage_size()
        for l in (self.idx_wq_b, self.idx_weights):
            if l is not None:
                storage_dev += l.storage_size()
        for comp in (self.compressor, self.indexer):
            if comp is not None:
                # A V4.1 layer that reads another layer's pool keeps compressors without weights of their own
                storage_dev += sum(l.storage_size() for l in (comp.wkv, comp.wgate) if l is not None)
        for rl in self.recurrent_layers:
            storage_dev += rl.storage_size()
        for cl in self.cache_layers:
            storage_dev += cl.storage_size()
        overhead_d = self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = self.num_q_heads * self.head_dim * torch.half.itemsize  # q rows
        overhead_s += self.o_groups * self.o_lora_rank * torch.half.itemsize  # o intermediates
        recons = max(
            self.q_a.recons_size(),
            self.q_b.recons_size(),
            self.wo_b.recons_size(),
            max((l.recons_size() for l in self.wo_a), default = 0),
        )
        tpa = TPAllocation(
            key = self.key,
            channel_width = 1,
            channel_unit = "heads",
            storage_per_device = storage_dev,
            storage_to_split = storage_split,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = self.o_groups,
            limit_key = "attn",
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": DSV4Attention,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "layer_type": self.layer_type,
                "hidden_size": self.hidden_size,
                "head_dim": self.head_dim,
                "rope_head_dim": self.rope_head_dim,
                "q_lora_rank": self.q_a.out_features_unpadded,
                "o_lora_rank": self.o_lora_rank,
                "sliding_window": self.sliding_window,
                "compress_rate": self.compress_rate,
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_topk": self.index_topk,
                "rope_theta": self.rope_theta,
                "compress_rope_theta": self.compress_rope_theta,
                "rope_scaling": self.rope_scaling,
                "rms_norm_eps": self.rms_norm_eps,
                "out_dtype": self.out_dtype,
                "kv_source": self.kv_source,
                "indexer_mode": self.indexer_mode,
                "q_head_norm": self.q_head_norm,
                "candidate_mode": self.candidate_mode,
                "candidate_topk_blocks": self.candidate_topk_blocks,
                "candidate_block_size": self.candidate_block_size,
            },
            "kv_owner_key": self.kv_owner.key if self.kv_owner is not None else None,
            "num_q_heads": self.num_q_heads,
            "o_groups": self.o_groups,
            **{name: _export(getattr(self, name, None)) for name in (
                "q_a",
                "q_norm",
                "q_b",
                "wkv",
                "kv_norm",
                "wo_b",
                "idx_wq_b",
                "idx_weights",
                "idx_wk",
                "idx_k_norm",
            )},
            "wo_a": [l.tp_export(plan, producer) for l in self.wo_a],
            "sinks": producer.send(self.sinks),
            "compressor": self.compressor.tp_export(plan, producer) if self.compressor is not None else None,
            "indexer": self.indexer.tp_export(plan, producer) if self.indexer is not None else None,
            "recurrent_layers": [rl.tp_export(plan) for rl in self.recurrent_layers],
            "cache_layers": [cl.tp_export(plan) for cl in self.cache_layers],
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        kw = exported["kwargs"]
        key = kw["key"]
        head_dim = kw["head_dim"]
        o_lora_rank = kw["o_lora_rank"]
        full_heads = exported["num_q_heads"]
        full_groups = exported["o_groups"]
        hpg = full_heads // full_groups
        device = local_context["device"]
        first, last, unit = plan[key]
        assert unit == "heads"
        num_groups = last - first
        num_q_heads = num_groups * hpg

        # Column split of q_b by the local head range; row (input-dim) split of wo_b by
        # the local group blocks -- the trailing all-reduce sums the partial products.
        # Everything else is replicated (shared-KV MQA + headless compressor/indexer)
        q_b_split = (True, first * hpg * head_dim, last * hpg * head_dim) \
            if num_groups else None
        wo_b_split = (False, first * o_lora_rank, last * o_lora_rank) \
            if num_groups else None

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if num_groups and exported.get(name) else None

        def _import_split(name, split):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split(local_context, exported[name], plan, split) \
                if split and exported.get(name) else None

        wo_a = [
            e["cls"].tp_import(local_context, e, plan)
            for e in exported["wo_a"][first : last]
        ] if num_groups else []

        module = DSV4Attention(
            config = None,
            **kw,
            num_q_heads = num_q_heads,
            o_groups = num_groups,
            q_a = _import("q_a"),
            q_norm = _import("q_norm"),
            q_b = _import_split("q_b", q_b_split),
            wkv = _import("wkv"),
            kv_norm = _import("kv_norm"),
            wo_a = wo_a,
            wo_b = _import_split("wo_b", wo_b_split),
            idx_wq_b = _import("idx_wq_b"),
            idx_weights = _import("idx_weights"),
            tp_defer_compressors = True,
        )
        module.tp_mode = True
        # V4.1's pool owner with a full indexer and no indexer compressor keys the indexer from the pool itself;
        # the constructor makes these only next to a compressor, which TP attaches after construction
        module.idx_wk = _import("idx_wk")
        module.idx_k_norm = _import("idx_k_norm")
        # A layer reading another layer's pool links to that layer's rank-local instance, imported before it
        owner_key = exported.get("kv_owner_key")
        if owner_key is not None:
            module.kv_owner = next((sm for m in local_context["modules"] for sm in m if sm.key == owner_key), None)
            assert module.kv_owner is not None, f"{key}: no module {owner_key} in this TP worker to read the pool of"

        if num_groups:
            consumer = local_context["consumer"]
            # Per-head sinks: each rank keeps its local head range
            module.sinks = consumer.recv(
                exported["sinks"], cuda = True, slice_dim = 0,
                first = first * hpg, last = last * hpg,
            )
            if exported.get("compressor") is not None:
                module.compressor = DSV4Compressor.tp_import(
                    local_context, exported["compressor"], plan, module)
            if exported.get("indexer") is not None:
                module.indexer = DSV4Compressor.tp_import(
                    local_context, exported["indexer"], plan, module)
            for rl in exported["recurrent_layers"]:
                rli = rl["cls"](module, **rl["args"])
                module.recurrent_layers.append(rli)
                module.tp_recurrent_lookup[rl["args"]["cache_id"]] = rli
            if len(exported["cache_layers"]):
                module.has_split_cache = True
                for cl in exported["cache_layers"]:
                    cli = cl["cls"](None, module, **cl["args"])
                    module.cache_layers.append(cli)
                    module.tp_cache_lookup[cl["args"]["cache_id"]] = cli

        module.device = device
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
            module.tp_owner = module.tp_single_owner(local_context, key)

        module.load_local(device)
        torch.cuda.synchronize()
        return module


    def load_local(self, device, **kwargs):
        if self.num_q_heads == 0:
            return
        self.inv_freq_main = yarn_inv_freq(self.rope_head_dim, self.rope_theta, device)
        self.inv_freq_compress = yarn_inv_freq(
            self.rope_head_dim, self.compress_rope_theta, device, rope_scaling = self.rope_scaling)
        self.inv_freq_main_neg = -self.inv_freq_main
        self.inv_freq_compress_neg = -self.inv_freq_compress
        self.kv_norm_w = self.kv_norm.weight.data
        self.q_ones = torch.ones(self.head_dim, dtype = self.kv_norm_w.dtype, device = device)
        if self.compressor is not None:
            # A V4.1 pool owner that keys its indexer from the pool rotates nothing here, as in load()
            freq = torch.zeros_like(self.inv_freq_compress) if self.idx_wk is not None else self.inv_freq_compress
            self.compressor.make_bc(freq)
        if self.indexer is not None:
            self.indexer.make_bc(self.inv_freq_compress)
        for rl in self.recurrent_layers:
            rl.alloc(device)
        for cl in self.cache_layers:
            cl.alloc(device)


    def _rope_type(self):
        return self.inv_freq_main if self.layer_type == "sliding" else self.inv_freq_compress


    def _rope_type_neg(self):
        return self.inv_freq_main_neg if self.layer_type == "sliding" else self.inv_freq_compress_neg

    def _rope_qkv(self, q, kv, position, positions = None):
        rd = self.rope_head_dim
        if os.environ.get("EXL3_DSV41_FUSED_ROPE", "1") != "0":
            ext.rope(
                q, q, kv, kv, self._rope_type(), position, positions, None,
                int(RopeStyle.GPTJ), 1.0, self.q_ones if self.q_head_norm else None,
                self.kv_norm_w, self.rms_norm_eps, 0.0, 0.0, 0, 1, self.head_dim - rd,
            )
        else:
            _ext_rope(q[..., -rd:], self._rope_type(), position, positions = positions)
            ext.rope(
                kv, kv, None, None, self._rope_type(), position, positions, None,
                int(RopeStyle.GPTJ), 1.0, self.kv_norm_w, None,
                self.rms_norm_eps, 0.0, 0.0, 0, 1, self.head_dim - rd,
            )


    def _project_qkv(self, x, params, position, positions = None):
        """Project queries and keys, then apply their head norms and trailing RoPE."""
        bsz, seq, _ = x.shape
        q_res = self.q_norm.forward(self.q_a.forward(x, params), params, out_dtype = torch.half)
        q = self.q_b.forward(q_res, params).view(bsz, seq, self.num_q_heads, self.head_dim)
        kv = self.wkv.forward(x, params).view(bsz, seq, 1, self.head_dim)
        self._rope_qkv(q, kv, position, positions)
        return q_res, q, kv.view(bsz, seq, self.head_dim)


    def _build_x_fan(self):
        """x-side projection fan for the eager cached path: q_a / wkv / comp wkv+wgate /
        idx wkv+wgate as ONE per-matrix-N exl3_mgemm (uniform bits/format required, q_a
        widest). Mirrors the BC_DSV4Attention fan. A second fan pairs q_b with idx_wq_b
        (both consume q_res) for the top-k regime."""
        self.x_fan_ready = True
        # No-q-head-norm models (DSV4.1): the fused fan is argmax-stable but not bit-identical (max abs
        # logit diff 0.5-0.8, fp16 reduction order; qmesh/notes/DenseFuse.md), so it is opt-in there
        if os.environ.get("EXL3_DSV4_NO_XFAN", "0") != "0" or \
                (not self.q_head_norm and os.environ.get("EXL3_DSV41_XFAN", "0") == "0"):
            return
        device = torch.device(self.device)

        def mk_fan(lins):
            inner = [l.inner for l in lins]
            if not all(l.quant_type == "exl3" for l in lins):
                return None
            if not all(l.out_features == l.out_features_unpadded for l in lins):
                return None
            if len({(i.K, i.mcg, i.mul1) for i in inner}) != 1:
                return None
            ns = [l.out_features for l in lins]
            if max(ns) != ns[0]:
                return None    # first output is the dtype/max-width carrier
            return dict(
                trellis = torch.tensor([i.trellis.data_ptr() for i in inner], dtype = torch.long, device = device),
                suh = torch.tensor([i.suh.data_ptr() for i in inner], dtype = torch.long, device = device),
                svh = torch.tensor([i.svh.data_ptr() for i in inner], dtype = torch.long, device = device),
                n = torch.tensor(ns, dtype = torch.int32, device = device),
                idx = torch.arange(len(lins), dtype = torch.long, device = device).unsqueeze(0),
                ns = ns, K = inner[0].K, mcg = inner[0].mcg, mul1 = inner[0].mul1,
            )

        lins = [self.q_a, self.wkv]
        # Compressor with no gate projection (compress_rate == 1, non-overlapping): the fan's
        # compressor outputs need both kv and gate, so such layers keep the per-projection path
        if self.compressor is not None and self.compressor.wgate is None:
            self.x_fan = None
        else:
            if self.compressor is not None:
                self.compressor._build_fused() if not self.compressor.fused_ready else None
                lins += [self.compressor.wkv, self.compressor.wgate]
            if self.indexer is not None:
                self.indexer._build_fused() if not self.indexer.fused_ready else None
                lins += [self.indexer.wkv, self.indexer.wgate]
            self.x_fan = mk_fan(lins)
        self.q_fan = mk_fan([self.q_b, self.idx_wq_b]) if self.indexer is not None else None
        self._fan_scratch = {}

        from .multilinear import MultiLinear
        try:
            if self.q_fan is None and self.q_b.quant_type == "exl3":
                self.qb_multi = MultiLinear(self.device, [self.q_b])
            if self.wo_b.quant_type == "exl3":
                self.wob_multi = MultiLinear(self.device, [self.wo_b])
            self._one_idx = torch.zeros((1, 1), dtype = torch.long, device = device)
        except AssertionError:
            self.qb_multi = self.wob_multi = None


    def _mgemm1(self, mu, x2, out_dtype, tag):
        """Single-matrix exl3_mgemm (one 'expert'): capture-safe multi-row linear for the
        graphed batched body. x2 (1, R, k) -> (1, R, n)."""
        R = x2.shape[1]
        C = g_tensor_cache.get(x2.device, (1, R, mu.out_features), out_dtype, tag + "_c")
        ah = g_tensor_cache.get(x2.device, (1, R, x2.shape[-1]), torch.half, tag)
        ext.exl3_mgemm(
            x2, mu.ptrs_trellis, C, mu.ptrs_suh, ah, mu.ptrs_svh,
            self._one_idx, None, mu.K, -1, mu.mcg, mu.mul1, -1, -1, 0, 1, None, None)
        return C

    def _fan_outs(self, fan, tag, seq, shapes):
        """Per-(seq) fan output scratch + pointer array, cached (g_tensor_cache reuses the
        same storage per shape, so the pointer arrays stay valid)."""
        key = (tag, seq)
        ent = self._fan_scratch.get(key)
        if ent is None:
            device = torch.device(self.device)
            outs = [g_tensor_cache.get(device, (seq,) + sh, torch.half, f"dsv4_fan_{tag}{i}")
                    for i, sh in enumerate(shapes)]
            cptr = torch.tensor([o.data_ptr() for o in outs], dtype = torch.long, device = device)
            ahad = g_tensor_cache.get(device, (len(outs), seq, self.hidden_size if tag == "x"
                                               else self.q_a.out_features),
                                      torch.half, f"dsv4_fan_{tag}_ah")
            ent = (outs, cptr, ahad)
            self._fan_scratch[key] = ent
        return ent


    def _build_woa_multi(self):
        self.woa_multi_ready = True
        try:
            if all(l.quant_type == "exl3" for l in self.wo_a):
                self.wo_a_multi = MultiLinear(self.device, self.wo_a)
                self.woa_indices = torch.arange(
                    self.o_groups, dtype = torch.long, device = self.device).unsqueeze(0)
        except AssertionError:
            self.wo_a_multi = None    # mixed K/format across slices: per-slice loop


    def _project_o_grouped(self, o, params, out_dtype, mgemm_out = False):
        """
        Grouped output projection: o (G, bsz, seq, hpg * head_dim) fp16 contiguous,
        rope slice already de-rotated. Short rows: ONE exl3_mgemm over the G slices (each
        group is an "expert" with its own input slice A[g]); at seq == 1 the expert-major
        output (G, 1, n) is memory-identical to the concatenated (1, G * n) row, so the cat
        is a free view. Long rows / fp16 / conversion: per-slice wo_a loop + cat.
        """
        G = self.o_groups
        bsz, seq = o.shape[1], o.shape[2]
        if not self.woa_multi_ready and self.device is not None:
            self._build_woa_multi()

        use_mg = (
            self.wo_a_multi is not None and bsz == 1 and seq <= 32
            and not any(k in params for k in ("capture", "quant_preserve", "ovr", "reconstruct"))
        )
        if use_mg:
            mu = self.wo_a_multi
            A = o[:, 0]                                   # (G, seq, hpg * head_dim)
            ah = g_tensor_cache.get(o.device, tuple(A.shape), torch.half, "dsv4_woa_had")
            # Static out / transposed-out buffers both paths: the kernel writes every element, and
            # a fresh allocation + a permute().reshape() clone per layer costs 3 launches a layer
            C = g_tensor_cache.get(o.device, (G, seq, mu.out_features), torch.half, "dsv4_woa_c")
            ext.exl3_mgemm(
                A, mu.ptrs_trellis, C, mu.ptrs_suh, ah, mu.ptrs_svh,
                self.woa_indices, None, mu.K, -1, mu.mcg, mu.mul1,
                -1, -1, 0, 1, None, None
            )
            if seq == 1:
                o2 = C.view(1, 1, G * mu.out_features)
            else:
                o2 = g_tensor_cache.get(o.device, (seq, G * mu.out_features), torch.half,
                                        "dsv4_woa_t")
                o2.view(seq, G, mu.out_features).copy_(C.transpose(0, 1))
                o2 = o2.unsqueeze(0)
            if mgemm_out and self.wob_multi is not None and seq <= 32:
                return self._mgemm1(self.wob_multi, o2.contiguous(),
                                    out_dtype or self.out_dtype,
                                    f"dsv4_wob1_L{self.layer_idx}")
            return self.wo_b.forward(o2, params, out_dtype = out_dtype or self.out_dtype)

        o = torch.cat([self.wo_a[g].forward(o[g], params) for g in range(self.o_groups)], dim = -1)
        return self.wo_b.forward(o, params, out_dtype = out_dtype or self.out_dtype)


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        if self.num_q_heads == 0:
            # Zero-width TP shard: contribute nothing, keep the collective aligned
            y = torch.zeros_like(x, dtype = out_dtype or self.out_dtype)
            if self.tp_reduce:
                self.tp_collect(params["backend"], y, False)
            return y
        mode = params.get("attn_mode", "flash_attn_nc")
        if mode == "flash_attn":
            y = self._forward_cached(x, params, out_dtype)
        else:
            assert mode == "flash_attn_nc", f"DSV4Attention: unsupported attn_mode {mode}"
            y = self._forward_nc(x, params, out_dtype)
        if self.tp_reduce:
            self.tp_collect(params["backend"], y)
        return y


    def _forward_nc(self, x, params, out_dtype):
        """
        Stateless single-shot pass (HF cache = None semantics: every complete compressor
        window in the chunk is compressed, the remainder discarded). Same kernels as the
        cached path:

        - fused compressor into throwaway scratch pools
        - indexer top-k index lists,
        - dsa_attn with in-chunk window indices and the fused derot/grouped epilogue

        No masks or eager attention are ever materialized.
        """
        bsz, seq, _ = x.shape
        device = x.device
        position = params.get("position", 0)

        q_res, q, kv = self._project_qkv(x, params, position)

        # Window rows come from the chunk itself (win_floor == q_pos0: no prior rows in nc)
        w = self.sliding_window

        m = self.compress_rate if self.kv_source is not None else 1
        T = seq // m if self.kv_source is not None else 0
        hpg = self.num_q_heads // self.o_groups
        hd = self.head_dim
        D_c, D_r = hd - self.rope_head_dim, self.rope_head_dim

        if self.compressor is not None:
            cap = max(T, 1)
            rows = min(seq, PAGE_SIZE + m)
            pool_c = g_tensor_cache.get(device, (cap, D_c), torch.half, "dsv4_nc_pool_c")
            pool_r = g_tensor_cache.get(device, (cap, D_r), torch.half, "dsv4_nc_pool_r")
            ring_kv = g_tensor_cache.get(device, (rows, self.compressor.wkv.out_features_unpadded),
                                         torch.half, "dsv4_nc_ring_kv")
            ring_gate = g_tensor_cache.get(device, ring_kv.shape, torch.half, "dsv4_nc_ring_gate")
            ovl = g_tensor_cache.get(device, (1, 2, m, hd), torch.float, "dsv4_nc_ovl") \
                if self.layer_type == "csa" else None
            bt = torch.arange(-(-cap // PAGE_SIZE), dtype = torch.int32, device = device).unsqueeze(0)
        else:
            pool_c = x.new_empty((1, D_c), dtype = torch.half)
            pool_r = x.new_empty((1, D_r), dtype = torch.half)
            bt = torch.zeros((1, 1), dtype = torch.int32, device = device)

        outs = []
        for b in range(bsz):
            indices = None
            k_len = 0
            sh = params.setdefault("dsv4_shared", {}).setdefault(b, {})
            if "quant_preserve" in params:
                params["quant_preserve"]["dsv4_shared"] = params["dsv4_shared"]
            if self.idx_wk is not None:
                stage = torch.empty((cap, hd), dtype = torch.half, device = device)
                self.compressor.forward_fused(x[b:b + 1], params, ring_kv, ring_gate, None, stage, None, 0)
                pool_idx = self._indexer_keys(stage[:T], params, 0) if T else None
                pool_c, pool_r = stage[:, :D_c].contiguous(), stage[:, D_c:].contiguous()
                sh["pool"] = pool_c, pool_r, pool_idx, bt
            elif self.compressor is not None:
                # Window positions count from 0 in the stateless path (fwp = 0), matching
                # the HF cache = None reference; causal bounds use the absolute positions
                self.compressor.forward_fused(
                    x[b:b + 1], params, ring_kv, ring_gate, ovl, pool_c, pool_r, 0)
                if self.layer_type == "csa":
                    idx_hd = self.index_head_dim
                    pool_idx = g_tensor_cache.get(device, (max(T, 1), idx_hd), torch.half, "dsv4_nc_pool_idx")
                    iring_kv = g_tensor_cache.get(device, (rows, self.indexer.wkv.out_features_unpadded), torch.half, "dsv4_nc_iring_kv")
                    iring_gate = g_tensor_cache.get(device, iring_kv.shape, torch.half, "dsv4_nc_iring_gate")
                    iovl = g_tensor_cache.get(device, (1, 2, m, idx_hd), torch.float, "dsv4_nc_iovl")
                    self.indexer.forward_fused(x[b:b + 1], params, iring_kv, iring_gate, iovl, pool_idx, None, 0)
            elif self.kv_source is not None:
                if sh["pool"][0].device != device:
                    sh["pool"] = tuple(None if t is None else to_device(t, device) for t in sh["pool"])
                pool_c, pool_r, pool_idx, bt = sh["pool"]
            if self.indexer_mode is not None and T > self.index_topk:
                if self.indexer_mode == "full":
                    sh["topk"] = self._indexer_topk(
                        x[b:b + 1], params, q_res[b:b + 1], pool_idx[:T], T, position, shared = sh)
                elif sh["topk"][0].device != device:
                    sh["topk"] = to_device(sh["topk"][0], device), sh["topk"][1]
                indices, k_len = sh["topk"]

            out = dsa_attn(
                q[b].half().contiguous(), pool_c, pool_r, bt, sinks = self.sinks,
                kv_chunk = kv[b].contiguous(), win_len = w, win_floor = position,
                indices = indices, k_len = k_len, pool_len = T, q_pos0 = position,
                compress_rate = m, scale = self.sm_scale,
                derot_inv_freq = self._rope_type_neg(), groups = self.o_groups, group_major = True,
                out = torch.empty((self.o_groups, seq, hpg * hd), dtype = torch.half, device = device),
                nc_chunk = bool(params.get("nc_chunk", False)),
            )
            outs.append(self._project_o_grouped(out.unsqueeze(1), params, out_dtype))
        return torch.cat(outs, dim = 0) if bsz > 1 else outs[0]


    def _indexer_keys(self, lat, params, e0):
        n = lat.shape[0]
        k = self.idx_k_norm.forward(self.idx_wk.forward(lat.unsqueeze(0), params), params, out_dtype = torch.half)
        positions = e0 if isinstance(e0, torch.Tensor) else torch.arange(n, dtype = torch.int, device = lat.device) + e0
        pos = (positions * self.compress_rate).unsqueeze(0)
        if os.environ.get("EXL3_DSV4_FUSED_INDEXER_ROPE", "1") != "0":
            kr = k.view(1, n, 1, -1)[..., -self.rope_head_dim:]
            lr = lat.view(1, n, 1, -1)[..., -self.rope_head_dim:]
            ext.rope(
                kr, kr, lr, lr, self.inv_freq_compress, 0, None, pos,
                int(RopeStyle.GPTJ), 1.0, None, None, 1e-6, 0.0, 0.0, 0, 1, 0,
            )
        else:
            for t in (k, lat):
                _ext_rope(t.view(1, n, 1, -1)[..., -self.rope_head_dim:], self.inv_freq_compress, position_ids = pos)
        return k[0]


    def _indexer_topk(
        self,
        x,
        params,
        q_res,
        idx_pool,
        ec,
        pos0,
        q_idx_pre = None,
        block_table = None,
        epp = 0,
        shared = None,
        idx_scales = None,
    ):
        """Lightning-indexer scoring + top-k selection over the indexer key pool (ec valid
        rows). The indexer query rope uses the compress table at the query positions == this
        layer's own cos/sin (CSA layers rope with the compress table). Causal bounds and the
        head-weight scale live in the scoring kernel; the -1-padded int32 index list comes
        from the pack kernel. Returns (indices (seq, K_pad) int32, k_len)."""
        _, seq, _ = x.shape
        if q_idx_pre is not None:
            q_idx = q_idx_pre.view(1, seq, self.index_n_heads, self.index_head_dim)
        else:
            q_idx = self.idx_wq_b.forward(q_res, params).view(1, seq, self.index_n_heads, self.index_head_dim).contiguous()
        _ext_rope(q_idx[..., -self.rope_head_dim:], self.inv_freq_compress, position = pos0)
        wts = self.idx_weights.forward(x, params)
        bs = self.candidate_block_size
        mode = self.candidate_mode if ec > self.candidate_topk_blocks * bs else None
        if dsv4_cand_dbg and (self.kv_source, seq, mode) not in _cand_dbg_seen:
            _cand_dbg_seen.add((self.kv_source, seq, mode))
            print(f"CANDDBG rank={os.environ.get('RANK','x')} L{self.kv_source} seq={seq} pos0={pos0} "
                  f"ec={ec} mode={mode} fixed={dsv4_cand_fixed} cap={self.candidate_topk_blocks * bs}",
                  flush = True)
        if mode == "source":
            shared["cand"] = torch.ones((seq, ec // bs + 1), dtype = torch.bool, device = x.device)
        elif mode == "use":
            shared["cand"] = to_device(shared["cand"], x.device)
        k = min(self.index_topk, ec)
        K_pad = -(-k // 32) * 32
        indices = torch.empty((seq, K_pad), dtype = torch.int32, device = x.device)

        def slab(r0, r1):
            sel, ninf = indices[r0:r1], -float("inf")
            if mode == "use":
                # A row selects inside its candidate blocks: score only the blocks any row of the slab kept
                drop = shared["cand"][r0:r1]
                vis = (pos0 + 1 + torch.arange(r0, r1, device = x.device)) // self.compress_rate
                S = r1 - r0
                if dsv4_cand_fixed and S <= 17:
                    # Device-side compaction into fixed-capacity entry lists: the same ascending
                    # entry ids as the nonzero path below, -1 padded, so every shape here is static
                    # (capturable) and no host sync is taken. A multi-row slab compacts and scores
                    # each row's own list -- the eager union list only exists because its width had
                    # to be shared, and one row's own list is exactly the union restricted to the
                    # blocks that row kept (fewer columns scored than the union path, which scores
                    # every union column for every row)
                    cap = self.candidate_topk_blocks * bs
                    rows_max = 17   # widest non-prefill slab (a chunk slab stays on the union path)
                    ent = g_tensor_cache.get(x.device, (rows_max, cap), torch.int32, "dsv4_cand_ent_rows")[:S]
                    if block_table is None:
                        pages = None
                    else:
                        # int32 throughout: the block table is int32, and a .long() here makes
                        # every row's id arithmetic promote (one to + _to_copy + copy_ per row)
                        pages = block_table.reshape(-1) if block_table.dtype == torch.int32 \
                            else block_table.reshape(-1).to(torch.int32)
                    if S == 1:
                        ext.dsa_compact_ent((~drop).any(dim = 0), ent[0], bs, ec)
                        entc = ent.clamp(min = 0)
                        rows_ent = (entc if pages is None
                                    else pages[entc // epp] * epp + entc % epp)
                    elif pages is None:
                        ext.dsa_compact_ent_rows(~drop, ent, bs, ec, pos0 + 1 + r0, self.compress_rate)
                        rows_ent = ent.clamp(min = 0)
                    else:
                        # Row s of the slab is query position pos0 + r0 + s, so its entry bound
                        # min((pos0 + 1 + r) // compress_rate, ec) is per row, computed in-kernel.
                        # The same kernel emits each kept entry's pool row through the block table,
                        # which otherwise costs an aten gather plus an index-dtype promotion per row
                        rows_ent = g_tensor_cache.get(x.device, (rows_max, cap), torch.int32,
                                                      "dsv4_cand_rows")[:S]
                        ext.dsa_compact_ent_rows_map(~drop, ent, rows_ent, pages, epp, bs, ec,
                                                     pos0 + 1 + r0, self.compress_rate)
                    # Row-contiguous (S, cap) score backing: cap is the kernel's S_stride, so this is
                    # a single Triton compile at any context length and the rows are scored in place
                    scores = g_tensor_cache.get(x.device, (rows_max, cap), torch.half, "dsv4_cand_scores")[:S]
                    if S > 1 and pages is not None:
                        # One launch for every row: the kernel reads each row's own pool rows through its
                        # map (-1 past the row's bound -> -inf), the same scores the per-row loop below
                        # writes, without its S gathers + S launches + 2S fills (host-bound at verify widths)
                        dsa_indexer_scores_rowmap(q_idx[0, r0:r1], wts[0, r0:r1], idx_pool, rows_ent, scores,
                                                  idx_scales = idx_scales)
                    else:
                        for i in range(S):
                            s_i = dsa_indexer_scores(q_idx[0, r0 + i:r0 + i + 1], wts[0, r0 + i:r0 + i + 1],
                                                     idx_pool[rows_ent[i]],
                                                     cap * self.compress_rate, self.compress_rate, cap,
                                                     idx_scales = idx_scales[rows_ent[i]].contiguous() if idx_scales is not None else None,
                                                     scores = scores[i:i + 1])
                            s_i.masked_fill_(ent[i] >= vis[i], ninf)
                            s_i.masked_fill_(ent[i] < 0, ninf)
                    ext.dsa_topk(scores, sel, min(k, cap), None, 0)
                    sel.copy_(torch.where(sel < 0, sel, ent.gather(1, sel.clamp(min = 0))))
                    krow = idx_pool[block_table.reshape(-1)[(ec - 1) // epp].long() * epp + (ec - 1) % epp] \
                        if _pdup_dir and block_table is not None and epp and ec > 0 and \
                        self.idx_wk is not None and seq == 1 else None
                    _pdup("eager_use", self.layer_idx, pos0, seq, dict(
                        scores = scores, sel = sel, ent = ent, cand = drop, q = q_idx[0, r0:r1], w = wts[0, r0:r1],
                        krow = krow),
                        selection = (self.compress_rate, bs))
                    return
                ent = ((~drop).any(dim = 0).nonzero() * bs + torch.arange(bs, device = x.device)).flatten()
                ent = ent[ent < ec]
                rows = ent if block_table is None else block_table.reshape(-1)[ent // epp].long() * epp + ent % epp
                scores = dsa_indexer_scores(q_idx[0, r0:r1], wts[0, r0:r1], idx_pool[rows], ent.numel() * self.compress_rate,
                                            self.compress_rate, ent.numel(),
                                            idx_scales = idx_scales[rows].contiguous() if idx_scales is not None else None)
                scores.masked_fill_(drop[:, ent // bs], ninf)
                scores.masked_fill_(ent >= vis[:, None], ninf)
                ext.dsa_topk(scores, sel, min(k, ent.numel()), None, 0)
                sel.copy_(torch.where(sel < 0, sel, ent.int()[sel.clamp(min = 0)]))
                _pdup("eager_use", self.layer_idx, pos0, seq, dict(
                    scores = scores, sel = sel, ent = ent, cand = drop, q = q_idx[0, r0:r1], w = wts[0, r0:r1]),
                    selection = (self.compress_rate, bs))
                return
            scores = dsa_indexer_scores(q_idx[0, r0:r1], wts[0, r0:r1], idx_pool, pos0 + r0, self.compress_rate, ec,
                                        block_table = block_table, epp = epp, idx_scales = idx_scales)
            if mode == "source":
                nf = ec // bs
                full, tail = scores[:, :nf * bs].unflatten(1, (nf, bs)), scores[:, nf * bs:]
                tail_max = F.pad(tail, (0, bs), value = ninf).amax(dim = -1, keepdim = True)
                blk = torch.cat((full.amax(dim = -1), tail_max), dim = 1)
                last = ((pos0 + 1 + torch.arange(r0, r1, device = x.device)) // self.compress_rate - 1).clamp_(min = 0)
                top = blk.scatter_(1, last.unsqueeze(1) // bs, float("inf")).topk(self.candidate_topk_blocks, dim = 1)
                shared["cand"][r0:r1].scatter_(1, top.indices, top.values == ninf)
            ext.dsa_topk(scores, sel, k, None, 0)
            krow = None
            if block_table is not None and epp and ec > 0 and self.idx_wk is not None and seq == 1:
                # Same single newly compressed key row the captured body's dump carries
                krow = idx_pool[block_table.reshape(-1)[(ec - 1) // epp].long() * epp + (ec - 1) % epp]
            _pdup("eager_" + (mode or "none"), self.layer_idx, pos0, seq, dict(
                scores = scores, sel = sel, cand = shared["cand"][r0:r1] if mode == "source" else None,
                q = q_idx[0, r0:r1], w = wts[0, r0:r1], krow = krow),
                selection = (self.compress_rate, bs))

        # Row slabs bound the transient score matrix, as in MLAttention._indexer_topk: selection is
        # per row. No slab is left with the few rows that take the decode kernels
        cuts = list(range(0, max(seq - 16, 1), self.SCORE_SLAB))
        for r0, r1 in zip(cuts, cuts[1:] + [seq]):
            slab(r0, r1)
        if dsv4_xrank_dbg and seq <= 17 and (self.kv_source, seq, pos0) not in _xrank_dbg_seen:
            _xrank_dbg_seen.add((self.kv_source, seq, pos0))
            def _sig(t):
                b = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
                return hashlib.md5(b).hexdigest()[:12]
            print(f"XRDBG rank={os.environ.get('RANK', 'x')} L{self.layer_idx} K{self.kv_source} "
                  f"seq={seq} pos0={pos0} ec={ec} x={_sig(x)} qres={_sig(q_res)} q={_sig(q_idx)} "
                  f"w={_sig(wts)} pool={_sig(idx_pool[:ec])} sel_sum={int(indices.long().sum())}",
                  flush = True)
        if dsv4_xrank_sel >= 0 and seq <= 17 and pos0 >= dsv4_xrank_sel:
            # Both ranks must issue the same collective sequence: pos0/seq agree by construction
            rk = int(os.environ.get("RANK", "0"))
            t = indices.to(torch.float32) if rk == 0 else torch.zeros_like(indices, dtype = torch.float32)
            params["backend"].all_reduce(t, rk == 0)
            indices.copy_(t.to(torch.int32))

        if dsv4_seldump and pos0 >= dsv4_seldump_pos and seq <= dsv4_seldump_maxseq:
            rk = os.environ.get("RANK", "x")
            # EVERY=1 keeps one file per position (path-local selections step by step), default
            # keeps the first one only
            every = os.environ.get("EXL3_DSV4_SELDUMP_EVERY", "0") != "0"
            name = f"{dsv4_seldump}.{rk}.L{self.layer_idx}.K{self.kv_source}"
            path = (f"{name}.p{pos0}.pt" if every else f"{name}.pt")
            if not os.path.exists(path):
                torch.save({"sel": indices.sort(dim = 1).values.cpu(), "k": k, "ec": ec, "pos0": pos0}, path)
        return indices, k


    def autosplit_extra_measure(self, params):
        """Indexer score matrix at maximum context, which the (1, chunk)-at-context-0 measuring
        forward does not reach: allocated (and dropped) here for the device budget"""
        if os.environ.get("EXL3_AUTOSPLIT_WORSTCASE", "1") == "0":
            return
        cache = params.get("cache")
        if cache is None or self.device is None or self.indexer_mode != "full":
            return
        rows = params["batch_shape"][1]
        ec = cache.layers[self.kv_source, params.get("layer_instance") or 0].capacity
        if ec <= self.index_topk:
            return   # cache too small to ever reach the top-k regime
        slab = min(rows, self.SCORE_SLAB + 16)
        n = slab * max(128, 1 << (ec - 1).bit_length())
        # Attention and indexer queries, live while it is scored
        n += rows * (self.num_q_heads * self.head_dim + self.index_n_heads * self.index_head_dim)
        bs = self.candidate_block_size
        if self.candidate_mode == "source" and ec > self.candidate_topk_blocks * bs:
            # Block maxima of a slab, live twice while the tail block is appended, and the candidate mask
            n += 2 * slab * (ec // bs + 1) + rows * (ec // bs + 1) // 2
        elif self.candidate_mode == "use" and ec > self.candidate_topk_blocks * bs:
            # Candidate mask, one byte per block, when its source is on another device; then, per slab, the
            # keys of the candidate entries, the mask over them and their ids
            n += rows * (ec // bs + 1) // 2 + ec * (self.index_head_dim + slab // 2 + 8)
        t = torch.empty((n,), dtype = torch.half, device = self.device)
        del t

    def _cand_regime(self, ec):
        """True when this layer selects inside its candidate blocks at entry bound `ec` (the eager
        source/use branch), and the dial lets the body serve it."""
        return (dsv4_b1_long >= 2 and self.candidate_mode is not None
                and ec > self.candidate_topk_blocks * self.candidate_block_size)

    def _cand_mask(self, device, kl, rows = 1):
        """Per-pool, per-query dropped-block mask with capture-stable storage."""
        bs = self.candidate_block_size
        shape = (kl.capacity // bs + 2,) if rows == 1 else (rows, kl.capacity // bs + 2)
        return g_tensor_cache.get(device, shape, torch.bool, f"dsv4_b_cand_{id(kl)}")

    def _cand_source(self, device, kl, scores, a_ec, s_max, a_pos, m):
        """Publish the eager source selection independently for every query row."""
        bs, ninf = self.candidate_block_size, -float("inf")
        R, nf = scores.shape[0], s_max // bs
        if R == 1:
            bounds = a_ec
        else:
            bounds = (a_pos[0] + 1 + _cached_arange(device, R)) // m
        scores.masked_fill_(_cached_arange(device, s_max)[None, :] >= bounds[:, None], ninf)
        blk = scores.view(R, nf, bs).amax(dim = -1)
        blk.scatter_(1, ((bounds - 1).clamp(min = 0) // bs).view(-1, 1).long(), float("inf"))
        top = blk.topk(self.candidate_topk_blocks, dim = 1)
        cand = self._cand_mask(device, kl, R)
        cand.fill_(True)
        if R == 1:
            cand.scatter_(0, top.indices[0], top.values[0] == ninf)
        else:
            cand.scatter_(1, top.indices, top.values == ninf)

    def _cand_use(self, device, kl, bt_st, qi, wts, indices, m, a_ec, a_pos):
        """Compact and score each query's own candidate blocks, then map to pool ids."""
        from .attention_fn.dsa_triton import dsa_indexer_scores
        bs, ninf = self.candidate_block_size, -float("inf")
        cap = self.candidate_topk_blocks * bs
        R, Hi, Di = qi.shape
        cand = self._cand_mask(device, kl, R)
        if R > 1:
            nb = cand.shape[1]
            bounds = (a_pos[0] + 1 + _cached_arange(device, R)) // m
            keep = (~cand) & (_cached_arange(device, nb)[None, :] * bs < bounds[:, None])
            ent = g_tensor_cache.get(device, (R, cap), torch.int32, "dsv4_b_ent_rows")
            scores = g_tensor_cache.get(device, (R, cap), torch.half, "dsv4_b_scores_cap")
            for row in range(R):
                ext.dsa_compact_ent(keep[row], ent[row], bs, kl.capacity)
                ent[row].masked_fill_(ent[row] >= bounds[row], -1)
                entc = ent[row].clamp(min = 0)
                pool_rows = bt_st[0][(entc // kl.epp).long()].long() * kl.epp + (entc % kl.epp).long()
                if kl.kv_format == "fp4":
                    keys = kl.pool_idx_fp4.view(-1, 64)[pool_rows]
                    scales = kl.pool_idx_scale.view(-1, 4)[pool_rows]
                else:
                    keys, scales = kl.pool_idx.view(-1, Di)[pool_rows], None
                dsa_indexer_scores(qi[row:row + 1], wts[row:row + 1], keys,
                                   cap * m, m, cap, scores = scores[row:row + 1], idx_scales = scales)
                scores[row].masked_fill_(ent[row] < 0, ninf)
            ext.dsa_topk(scores, indices, min(self.index_topk, cap), None, 0)
            indices.copy_(torch.where(indices < 0, indices, ent.gather(1, indices.clamp(min = 0))))
            return scores, ent
        # Entry list bound: the eager compaction is told the live entry count ec (a host int there) and
        # truncates every kept block to it. Here ec is a device value: drop the blocks that start at or
        # past it, then send the partial block's entries at and past it back to the -1 padding. The kept
        # blocks are the ones the mask below leaves, in ascending order, so those entries are the last
        # real ones and the list, every column of the score matrix and the ent dump match eager's
        nb = cand.numel()
        keep = g_tensor_cache.get(device, (nb,), torch.bool, f"dsv4_b_keep_{id(kl)}")
        torch.lt(_cached_arange(device, nb) * bs, a_ec[0], out = keep)
        keep.logical_and_(~cand)
        ent = g_tensor_cache.get(device, (cap,), torch.int32, "dsv4_b_ent")
        ext.dsa_compact_ent(keep, ent, bs, kl.capacity)
        ent.masked_fill_(ent >= a_ec[0], -1)
        entc = ent.clamp(min = 0)
        rows = bt_st[0][(entc // kl.epp).long()].long() * kl.epp + (entc % kl.epp).long()
        if kl.kv_format == "fp4":
            keys = kl.pool_idx_fp4.view(-1, 64)[rows]
            scales = kl.pool_idx_scale.view(-1, 4)[rows]
        else:
            keys, scales = kl.pool_idx.view(-1, Di)[rows], None
        scores = g_tensor_cache.get(device, (R, cap), torch.half, "dsv4_b_scores_cap")
        dsa_indexer_scores(qi, wts, keys, cap * m, m, cap, scores = scores, idx_scales = scales)
        scores.masked_fill_(cand[entc // bs].unsqueeze(0), ninf)
        scores.masked_fill_((ent >= a_ec[0]).unsqueeze(0), ninf)
        scores.masked_fill_((ent < 0).unsqueeze(0), ninf)
        ext.dsa_topk(scores, indices, min(self.index_topk, cap), None, 0)
        indices.copy_(torch.where(indices < 0, indices, ent[indices.clamp(min = 0)]))
        return scores, ent

    def _batch_ok_b1(self, kl, pos_end, single = False):
        """Layers the batched B=1 body can express at this step: sliding layers at any length, and
        compressed pools only once they are past index_topk -- below that bound the eager path is the
        exact dense read (indices = None) and the body has no second, indexed regime for the same rows.
        Above it both select the bounded pool-scan top-k: EXL3_DSV4_B1_LONG >= 1 for a plain layer, >= 2
        for a candidate layer, whose selection must stay inside its blocks (the eager source/use branch).
        A pool reader follows its owner, which publishes the selection in the body. Both for a single row
        only. Quantized pools stay on _forward_cached_one (host decision, before any state moves)."""
        if self.kv_source is None:
            return self.candidate_mode is None
        if kl.quant:
            return False
        if self.kv_owner is not None and not self.kv_owner._batch_ok_b1(kl, pos_end, single):
            # A pool reader only enters the body together with its owner: the owner publishes the
            # selection in the body, while an eager owner publishes in params["dsv4_shared"] only
            return False
        ec = pos_end // self.compress_rate
        if ec <= self.index_topk:
            return False
        if self.candidate_mode is not None and ec > self.candidate_topk_blocks * self.candidate_block_size:
            return single and dsv4_b1_long >= 2
        return single and dsv4_b1_long >= 1

    def b1_prep(self, params, seq = 1):
        """Stage host state for the captured decode body, including multi-row verification.

        Return None before touching state when the body cannot serve the whole step. Ring
        maintenance remains on the eager path; the graph key includes the input shape and regime.
        """
        # One epoch object per step, shared by the live params and every Segment capture's copy
        params.setdefault("dsv4_b_epoch", object())
        if not dsv4_b1_batch or not self.num_q_heads:
            return None
        if params.get("attn_mode") != "flash_attn" or params.get("prefill"):
            return None
        rsg = params.get("recurrent_states")
        if not rsg:
            return None
        rs = rsg[0]
        B = len(rsg)
        if seq > 1 and (B != 1 or not dsv4_cand_fixed):
            return None
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        rsl = self._get_rsl(rs, layer_instance)
        for state in rsg:
            if self._get_rsl(state, layer_instance) is not rsl or state.position - state.window_beg + seq > rsl.ring_rows:
                return None
        kl = None
        if self.kv_source is not None:
            if "block_table" not in params:
                return None
            owner = self.kv_owner if self.kv_owner is not None else self
            kl = owner.tp_cache_lookup[rs.cache] if self.tp_mode \
                else rs.cache.layers[(self.kv_source, layer_instance[1])]
            if self.kv_owner is not None and kl.device != self.device:
                return None
        if any(not self._batch_ok_b1(kl, state.position + seq, B == 1) for state in rsg):
            return None
        if seq > 1 and self.kv_source is not None:
            # A single captured branch cannot represent a change of selection regime within
            # the verification window. Let the eager path handle that boundary.
            first_ec = (rs.position + 1) // self.compress_rate
            last_ec = (rs.position + seq) // self.compress_rate
            if first_ec <= self.index_topk or self._cand_regime(first_ec) != self._cand_regime(last_ec):
                return None
        m = self.compress_rate if self.kv_source is not None else 1
        w = self.sliding_window
        # Selection width: the padded top-k for layers that own a pool selection (indexer_mode set,
        # V4.1) or build one from a per-layer indexer compressor (V4 "csa"); window-only layers
        # never index and keep the one-warp minimum
        kp = -(-self.index_topk // 32) * 32 \
            if self.indexer_mode is not None or self.indexer is not None else 32
        gkey = ("b1", id(rsl), id(kl), B)
        st = self._bgraph_state.get(gkey)
        if st is None:
            st = dict(
                pins = [torch.empty((6, B), dtype = torch.int32, pin_memory = True) for _ in range(8)],
                pin_i = 0,
                arr = torch.empty((6, B), dtype = torch.int32, device = self.device),
                bt_st = torch.zeros((B, kl.num_pages if kl is not None else 1), dtype = torch.int32,
                                    device = self.device),
            )
            self._bgraph_state[gkey] = st
        pin = st["pins"][st["pin_i"]]
        st["pin_i"] = (st["pin_i"] + 1) % len(st["pins"])
        pos = [state.position for state in rsg]
        floor = [state.position - min(w - 1, state.position - state.window_beg, state.position) for state in rsg]
        ec = [(state.position + seq) // m if self.kv_source is not None else 0 for state in rsg]
        # Per-row selection count: a layer below index_topk keeps every causal entry, so its k_len
        # is the live entry count, not the padded width (the top-k output is -1 padded either way)
        pin.copy_(torch.tensor([pos, floor, [state.window_beg for state in rsg], ec,
                                [state.slot for state in rsg], [min(kp, e) for e in ec]],
                               dtype = torch.int32))
        st["arr"].copy_(pin, non_blocking = True)
        if kl is not None:
            bt = get_for_device(params, "block_table", self.device)
            npr = min(bt.shape[1], kl.num_pages)
            st["bt_st"][:, :npr].copy_(bt[:B, :npr])
        # Host entry window of this step's new compressed entries: the captured body projects them in
        # one call, the way the eager path does, so `nnew` is part of the graph key (a step whose run
        # length differs gets its own capture rather than a stale shape). A single-row step never reads it
        # (_batch_body's S == 1 branch masks the write on device), so it is 0 there and keys nothing
        nnew = (ec[0] - pos[0] // m) if (kl is not None and seq > 1) else 0
        if os.environ.get("EXL3_DSV4_B1_DBG") and kl is not None:
            print(f"B1PREP L{self.layer_idx} pos={pos[0]} seq={seq} m={m} ec={ec[0]} "
                  f"e0={pos[0] // m} n={nnew}", flush=True)
        return (rsl, kl, m, w, kp, self._cand_regime(ec[0]), nnew), st["arr"], st["bt_st"]



    def _forward_cached(self, x, params, out_dtype):
        """attn_mode flash_attn: kernel-based path over the per-job ring state and the paged
        pools. Each batch row appends to its slot's ring and, through its block-table row,
        to its pool pages, then attends over [sliding ring ++ selected/dense pool entries]
        via dsa_attn. Block-table rows correspond 1:1 to batch rows (generator batches and
        batch_shape identity tables alike)."""
        rsg = params["recurrent_states"]
        bsz = x.shape[0]
        assert len(rsg) >= bsz
        if _dsv4_dump_dir and bsz == 1:
            _dsv4_dump("X", self.layer_idx, rsg[0].position, 0, 1, {"x": x}, None, None)
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        kl = bt = None
        if self.kv_source is not None:
            # In TP workers rs.cache is an opaque id; resolve to this rank's replicated
            # pool layer before anything dereferences it (a V4.1 reader's pool is its owner's)
            owner = self.kv_owner if self.kv_owner is not None else self
            kl = owner.tp_cache_lookup[rsg[0].cache] if self.tp_mode \
                else rsg[0].cache.layers[(self.kv_source, layer_instance[1])]
            if "block_table" in params:
                bt = get_for_device(params, "block_table", self.device)
            else:
                # Direct-forward use without a page table (tests/benchmarks): slot-
                # partitioned identity, one row per batch row via each state's slot
                assert not self.tp_mode, "DSV4Attention: TP forward requires a block table"
                sbt = kl.slot_bt(rsg[0].cache.num_slots)
                bt = sbt[[rsg[i].slot for i in range(bsz)]]
            if dsa_debug_bounds:
                bmax, bmin = int(bt.max()), int(bt.min())
                assert 0 <= bmin and (bmax < kl.num_pages or bmax == kl.trash_page), \
                    f"DSA block table content OOB: [{bmin}, {bmax}] vs {kl.num_pages} pages " \
                    f"(layer {self.layer_idx})"
            if self.kv_owner is not None and kl.device != self.device:
                src, kl = kl, kl.mirrors.get(self.device, kl)
                bt = to_device(bt, self.device)
                if self.kv_mirror:
                    m, pages = self.compress_rate, params.get("block_table", bt)
                    for i in range(bsz):
                        kl.pull(src, pages[i], rsg[i].position // m, (rsg[i].position + x.shape[1]) // m)
        if bsz == 1:
            if dsv4_b1_batch and x.shape[1] == 1 and self._batch_ok_b1(kl, rsg[0].position + 1, True):
                return self._forward_cached_batch(x, params, rsg, layer_instance, out_dtype, kl, bt)
            return self._forward_cached_one(
                x, params, rsg[0], self._get_rsl(rsg[0], layer_instance), out_dtype,
                kl = kl, bt_row = bt[:1] if bt is not None else None)
        if not dsv4_batch_eager or not (self.q_head_norm or dsv41_batch):
            # Per-job loop: each job dispatches into its per-slot whole-step graph (or the
            # eager core). Measured faster than the batched-eager path below at bsz 2-8 --
            # graph replays beat batched GEMVs + per-job eager cores. The batched path is
            # the capture substrate for the planned bszN whole-batch graphs
            outs = [
                self._forward_cached_one(
                    x[i:i + 1], params, rsg[i],
                    self._get_rsl(rsg[i], layer_instance),
                    out_dtype, copy_static = True,
                    kl = kl, bt_row = bt[i:i + 1] if bt is not None else None)
                for i in range(bsz)
            ]
            return torch.cat(outs, dim = 0)
        return self._forward_cached_batch(x, params, rsg, layer_instance, out_dtype, kl, bt)


    def _get_rsl(self, rs, layer_instance):
        """Per-slot layer state: direct on the parent, via the id-keyed lookup in TP
        workers (rs is a DSV4ExportedState there and rs.cache an opaque id)."""
        if self.tp_mode:
            return self.tp_recurrent_lookup[rs.cache]
        return rs.cache.get_recurrent_layer(layer_instance)


    def _forward_cached_batch(self, x, params, rsg, layer_instance, out_dtype, kl, bt):
        """Multi-job cached path: the projections that don't touch per-slot state (x fan or
        Linears, q_norm, q_b / q-fan, fused-norm rope with per-job base positions) run ONCE
        over all rows, the per-slot attention core (ring/pools/indexer/attention) runs per
        job on row slices, and the grouped o_proj runs once over the collected outputs.
        Falls back to the per-job loop when the rows can't share a projection pass."""
        B, S, _ = x.shape
        eligible = (
            S <= 16 and x.dtype == torch.half and x.is_contiguous()
            and not any(k in params for k in ("capture", "quant_preserve", "ovr", "reconstruct"))
        )
        if not eligible:
            outs = [
                self._forward_cached_one(x[i:i + 1], params, rsg[i],
                                         self._get_rsl(rsg[i], layer_instance),
                                         out_dtype, copy_static = True,
                                         kl = kl, bt_row = bt[i:i + 1] if bt is not None else None)
                for i in range(B)
            ]
            return torch.cat(outs, dim = 0)

        device = x.device
        R = B * S
        rsl = self._get_rsl(rsg[0], layer_instance)
        for rs in rsg[1:B]:
            assert self._get_rsl(rs, layer_instance) is rsl, \
                "batched DSA path requires all jobs in one cache"
        m = self.compress_rate if self.kv_source is not None else 1
        w = self.sliding_window
        kp = -(-self.index_topk // 32) * 32 \
            if self.indexer_mode is not None or self.indexer is not None else 32

        # Host prepass (never captured): ring shifts BEFORE attention (the body then
        # addresses the post-shift ring via the effective ring_beg; appends are in-body,
        # after attention reads), then the per-job state rows into a pinned staging write
        pos_l, floor_l, beg_l, ec_l, slot_l = [], [], [], [], []
        for i in range(B):
            rs = rsg[i]
            pos0 = rs.position
            slot = rs.slot
            offset = pos0 - rs.window_beg
            shift = 0
            if offset + S > rsl.ring_rows:
                need = offset + S - rsl.ring_rows
                shift = -(-need // PAGE_SIZE) * PAGE_SIZE
                ring_j = rsl.ring[slot]
                ring_j[:rsl.ring_rows - shift].copy_(ring_j[shift:].clone())
                rs.wshift = shift
            ec = (pos0 + S) // m if self.kv_source is not None else 0
            assert self.kv_source is None or ec <= bt.shape[1] * kl.epp, \
                f"DSA pool overflow: entry {ec} beyond block table ({bt.shape[1]} pages)"
            pos_l.append(pos0)
            floor_l.append(pos0 - min(w - 1, offset, pos0))
            beg_l.append(rs.window_beg + shift)
            ec_l.append(ec)
            slot_l.append(slot)

        # Whole-batch BC graph (house pattern): all per-job state device-driven through the
        # owner's (6, MAX_B) array and block-table static, the input pointer is the only
        # patched parameter. Declines (no fan / non-exl3 projections) fall through to the
        # eager batched body below. The candidate regime keeps out: the BC selection scans the
        # full pool, which only matches eager while every causal entry is kept
        cand_regime = R == 1 and self._cand_regime(ec_l[0])
        if dsv4_batch_graph and B <= 8 and S <= 16 and R <= 32 and self.q_head_norm and not cand_regime:
            if not hasattr(self, "_bc_dsa_batch"):
                self._bc_dsa_batch = {}
            bcd = self._bc_dsa_batch.get(id(rsl))
            if bcd is None:
                bcd = build_bc_dsa_batch(self, rsl, kl)
                self._bc_dsa_batch[id(rsl)] = bcd if bcd is not None else False
            if bcd:
                y = bcd.run(x, B, S, pos_l, floor_l, beg_l, ec_l, slot_l, bt)
                if y is not None:   # None: the graph kernels do not fit this device's shared memory
                    if _rp_on: _rpnote("batchbc_graph_hit")
                    return y
                if _rp_on: _rpnote("batchbc_graph_decline")

        # Eager batched body (capture reference / fallback): per-job state in a device
        # array, same kernels as the graphs
        gkey = (id(rsl), B, S)
        st = self._bgraph_state.get(gkey)
        if st is None:
            st = dict(
                # Pinned staging is a RING: the host may run several steps ahead of the
                # GPU, so a single pin would be rewritten while a previous step's async
                # H2D from it is still queued
                pins = [torch.empty((6, B), dtype = torch.int32, pin_memory = True)
                        for _ in range(8)],
                pin_i = 0,
                arr = torch.empty((6, B), dtype = torch.int32, device = device),
                bt_st = torch.zeros((B, kl.num_pages), dtype = torch.int32, device = device)
                    if kl is not None else None,
            )
            self._bgraph_state[gkey] = st
        pin = st["pins"][st["pin_i"]]
        st["pin_i"] = (st["pin_i"] + 1) % len(st["pins"])
        pin.copy_(torch.tensor([pos_l, floor_l, beg_l, ec_l, slot_l,
                                [min(kp, e) for e in ec_l]], dtype = torch.int32))
        st["arr"].copy_(pin, non_blocking = True)
        if kl is not None:
            npr = min(bt.shape[1], kl.num_pages)
            st["bt_st"][:, :npr].copy_(bt[:B, :npr])
        if _rp_on: _rpnote("batchbody_S%d" % S)
        # Host entry window for the whole step: the run the eager path projects in one call
        nnew = (ec_l[0] - pos_l[0] // m) if (kl is not None and B == 1) else None
        return self._batch_body(x, st["arr"], B, S, R, rsl, kl, st["bt_st"], m, w, kp,
                                params, out_dtype, cand_regime = cand_regime, nnew = nnew)

    def _batch_body(self, x, arr, B, S, R, rsl, kl, bt_st, m, w, kp, params, out_dtype,
                    cand_regime = False, nnew = None):
        """Eager batched step, the capture reference for BC_DSV4BatchAttention: everything
        is device-driven (per-job state from `arr`, slot/block-table indirection
        in-kernel). `cand_regime` selects the candidate pre-filter branch of the indexer selection,
        which the host half (b1_prep) decides for the captured piecewise path."""
        from .attention_fn.dsa_triton import dsa_attn, _dsa_indexer_fewq_kernel, _dsa_indexer_kernel
        import triton
        device = x.device
        a_pos, a_floor, a_beg, a_ec, a_slots, a_klen = arr.unbind(0)

        q_res, q, kv, comp_kv, comp_gate, idx_kv, idx_gate, q_idx = \
            self._project_batch(x, params, None, a_pos = a_pos)

        indices = None
        # Newly compressed indexer key row (single-row step only); set inside the compressor block
        krow = None
        if self.compressor is not None:
            comp, idx = self.compressor, self.indexer
            fp4 = kl.kv_format == "fp4"
            if kl.quant or fp4:
                # Packed pool: per-job staging rows, then quantize + scatter through each
                # job's block-table row
                stage = g_tensor_cache.get(device, (B, S // m + 1, self.head_dim), torch.half, "dsv4_b_stage")
                ext.dsv4_compress(
                    comp_kv, comp_gate, rsl.comp_buf_kv, rsl.comp_buf_gate,
                    rsl.comp_ovl, comp.ape, comp.fused_norm_w, comp.norm.rms_norm_eps,
                    comp.fused_inv_freq, stage, None,
                    0, a_pos, m, a_slots, None, 0, True)
                if kl.quant:
                    ext.dsv4_pool_quant_scatter(
                        stage, kl.pool_c_view(), kl.pool_s.view(-1, kl.G), kl.pool_r.view(-1, kl.D_r),
                        bt_st, 0, a_pos, m, S, kl.epp)
            else:
                ext.dsv4_compress(
                    comp_kv, comp_gate, rsl.comp_buf_kv, rsl.comp_buf_gate,
                    rsl.comp_ovl, comp.ape, comp.fused_norm_w, comp.norm.rms_norm_eps,
                    comp.fused_inv_freq, kl.pool_c.view(-1, kl.D_c), kl.pool_r.view(-1, kl.D_r),
                    0, a_pos, m, a_slots, bt_st, kl.epp, False)
            if self.layer_type == "csa":
                idx_stage = (g_tensor_cache.get(device, (B, S // m + 1, kl.D_i), torch.half, "dsv4_b_idx_stage")
                             if fp4 else None)
                ext.dsv4_compress(
                    idx_kv, idx_gate, rsl.idx_buf_kv, rsl.idx_buf_gate,
                    rsl.idx_ovl, idx.ape, idx.fused_norm_w, idx.norm.rms_norm_eps,
                    idx.fused_inv_freq, idx_stage if fp4 else kl.pool_idx.view(-1, kl.D_i), None,
                    0, a_pos, m, a_slots, None if fp4 else bt_st, 0 if fp4 else kl.epp, fp4)
                if fp4:
                    ext.dsv4_fp4_scatter(idx_stage, kl.pool_idx_fp4.view(-1, 64), kl.pool_idx_scale.view(-1, 4),
                                         bt_st, 0, a_pos, m, S, kl.epp, True)
            if fp4 and self.idx_wk is not None:
                entries = (a_pos[:, None] // m + _cached_arange(device, S // m + 1)[None, :]).reshape(-1)
                keys = self._indexer_keys(stage.view(-1, self.head_dim), params, entries.to(torch.int))
                ext.dsv4_fp4_scatter(keys.view(B, S // m + 1, kl.D_i),
                                     kl.pool_idx_fp4.view(-1, 64), kl.pool_idx_scale.view(-1, 4),
                                     bt_st, 0, a_pos, m, S, kl.epp, True)
            elif self.idx_wk is not None:
                pool_r_flat = kl.pool_r.view(-1, kl.D_r)
                idx_flat = kl.pool_idx.view(-1, kl.D_i)
                if S == 1:
                    # Preserve the existing single-row projection and write behavior.
                    e = (a_ec - 1).clamp(min = 0)
                    row = bt_st.gather(1, (e // kl.epp).long().unsqueeze(1)).squeeze(1).long() * kl.epp + (e % kl.epp).long()
                    lat = torch.cat((kl.pool_c.view(-1, kl.D_c)[row], pool_r_flat[row]), dim = -1)
                    keys = self._indexer_keys(lat, params, e)
                    new = (a_ec > a_pos // m).view(-1, 1)
                    pool_r_flat.index_copy_(0, row, torch.where(new, lat[:, kl.D_c:], pool_r_flat[row]))
                    idx_flat.index_copy_(0, row, torch.where(new, keys, idx_flat[row]))
                elif dsv4_idx_wk_grouped and nnew and B == 1 and not kl.quant:
                    # Exact eager write: project the step's whole new-entry run in ONE call. The base
                    # entry of the run comes from the per-step device position, never from the host:
                    # a host base baked here would replay stale for every later step sharing this
                    # graph (same run length), shifting the write window by the position delta. Only
                    # the run LENGTH is a host value, and it is part of the segment key.
                    e = _cached_arange(device, nnew) + a_pos[0] // m
                    safe_e = e.clamp(max = bt_st.shape[1] * kl.epp - 1)
                    row = bt_st[0].gather(0, (safe_e // kl.epp).long()).long() * kl.epp + (safe_e % kl.epp).long()
                    lat = torch.cat((kl.pool_c.view(-1, kl.D_c)[row], pool_r_flat[row]), dim = -1)
                    keys = self._indexer_keys(lat, params, e.to(torch.int))
                    pool_r_flat.index_copy_(0, row, lat[:, kl.D_c:])
                    idx_flat.index_copy_(0, row, keys)
                else:
                    # Fixed iteration bounds keep the writes graph-capture safe. Process
                    # each job/window separately so masked padding cannot race a real
                    # write to the same physical row.
                    for b in range(B):
                        for j in range(S // m + 1):
                            e = a_pos[b:b + 1] // m + j
                            new = (e < a_ec[b:b + 1]).view(1, 1)
                            safe_e = e.clamp(max = bt_st.shape[1] * kl.epp - 1)
                            row = bt_st[b].gather(0, (safe_e // kl.epp).long()).long() * kl.epp + (safe_e % kl.epp).long()
                            if kl.quant:
                                # The quantized pool cannot reconstruct the eager latent.
                                lat = stage[b, j:j + 1].clone()
                            else:
                                lat = torch.cat((kl.pool_c.view(-1, kl.D_c)[row], pool_r_flat[row]), dim = -1)
                            keys = self._indexer_keys(lat, params, e)
                            pool_r_flat.index_copy_(0, row, torch.where(new, lat[:, kl.D_c:], pool_r_flat[row]))
                            idx_flat.index_copy_(0, row, torch.where(new, keys, idx_flat[row]))
                if _pdup_dir:
                    krow = idx_flat[row]
            if fp4:
                ext.dsv4_fp4_scatter(stage, kl.pool_fp4.view(-1, 256), kl.pool_fp4_scale.view(-1, 32),
                                     bt_st, 0, a_pos, m, S, kl.epp, False)

        # Pool selection: every layer the eager path scans for (indexer_mode "full": V4.1 layers that
        # own an indexer head, whether or not they own the pool they index -- index_source_layer_ids
        # includes readers of an earlier layer's pool, which select for themselves -- plus a V4 "csa"
        # indexer compressor, whose mode defaults to "full"). Layers that read a selection instead
        # (indexer_mode "shared") reuse the most recent one over the same pool below
        # Eager only builds a selection once its pool passes index_topk (dense_pool below); mirror
        # that in the dumped field set so both paths carry the same keys
        # Dump field set only: the instrumented positions sit in the top-k regime (ec >> index_topk)
        # on both paths, and reading a_ec back here would sync inside the captured step
        sel_live = bool(_pdup_dir)
        if self.indexer_mode == "full":
            # Unified selection for ALL rows: for jobs still under index_topk the bounded
            # top-k IS the causal identity set (ascending) -- no dense/gathered regime split.
            # Above index_topk the same bounded top-k over the pool scan selects, and in the
            # candidate regime the source layer refreshes the candidate mask from those block
            # maxima while the reading layers select over their compacted entries
            Hi, Di = self.index_n_heads, self.index_head_dim
            qi = q_idx.view(B, S, Hi, Di)
            ext.rope(
                qi[..., -self.rope_head_dim:], qi[..., -self.rope_head_dim:],
                None, None, self.inv_freq_compress, 0, a_pos, None,
                int(RopeStyle.GPTJ), 1.0, None, None, 1e-6, 0.0, 0.0, 0, 1, 0)
            wts = g_tensor_cache.get(device, (R, Hi), torch.half, "dsv4_b_wts")
            # Same call the eager path's LinearFP16.forward makes for an unquantized head
            # (cuBLAS matmul into a half out): the BC_LinearFP16 kernel is not bitwise equal to
            # cuBLAS here and the head weights feed the top-k, so mirror eager exactly
            iw = self.idx_weights.inner
            torch.matmul(x.view(R, -1), iw.weight, out = wts)
            if iw.bias is not None:
                wts += iw.bias
            indices = g_tensor_cache.get(device, (R, kp), torch.int32, "dsv4_b_idx")
            sc, ent = None, None

            if cand_regime and self.candidate_mode == "use":
                sc, ent = self._cand_use(device, kl, bt_st, qi.view(R, Hi, Di), wts, indices, m, a_ec, a_pos)
                scores = sc
            else:
                bs = self.candidate_block_size
                # The score stride and the tile alignment both follow the kv tile, which the
                # ladder may shrink on small-smem devices (smem.py)
                from .attention_fn.smem import pick_config, shared_bytes, halving_ladder
                tiled = S > 4
                kernel = _dsa_indexer_kernel if tiled else _dsa_indexer_fewq_kernel
                def i_args(cfg):
                    bm, bn = cfg
                    s_max = -(-_seq_capacity(kl) // bn) * bn
                    if cand_regime and s_max % bs:
                        s_max = -(-s_max // bs) * bs
                    scores = g_tensor_cache.get(device, (R, s_max), torch.half, "dsv4_b_scores")
                    args = (qi.view(R, Hi, Di), wts.view(R, Hi),
                            kl.pool_idx_fp4.view(-1, 64) if kl.kv_format == "fp4" else kl.pool_idx.view(-1, Di),
                            scores, a_ec, R, a_pos, a_ec, bt_st, bt_st.stride(0))
                    consts = dict(
                        H_i = Hi, D_i = Di, S_stride = s_max, compress_rate = m,
                        scale = Di ** -0.5 * Hi ** -0.5, BLOCK_N = bn,
                        SEQ = S, MULTIROW = 1, EPP = kl.epp,
                        DEBUG_BOUNDS = 1 if dsa_debug_bounds else 0,
                        DEBUG_PAGES = kl.num_pages if dsa_debug_bounds else 0)
                    if kl.kv_format == "fp4":
                        consts.update(idx_scales = kl.pool_idx_scale.view(-1, 4), FP4 = True)
                    if tiled:
                        consts["BLOCK_M"] = bm
                    else:
                        consts["H_pad"] = max(triton.next_power_of_2(Hi), 16)
                    return s_max, scores, args, consts
                with torch.cuda.device(device):
                    ladder = ([(64, 128)] + [(bm, bn) for bn in halving_ladder(128)
                              for bm in halving_ladder(64) if (bm, bn) != (64, 128)]) if tiled else \
                             [(1, bn) for bn in halving_ladder(128)]
                    def probe(cfg):
                        _, _, args, consts = i_args(cfg)
                        return shared_bytes(kernel, args, **consts, num_warps = 8, num_stages = 2)
                    bm, bn = pick_config(device, "dsv4_indexer_tiled" if tiled else "dsv4_indexer_fewq",
                                         (Hi, Di, m, S, kl.epp, kl.capacity), ladder, probe)
                    s_max, scores, args, consts = i_args((bm, bn))
                    grid = (triton.cdiv(S, bm), triton.cdiv(s_max, bn), B) if tiled else \
                           (R, triton.cdiv(s_max, bn))
                    kernel[grid](*args, **consts, num_warps = 8, num_stages = 2)
                if cand_regime:
                    self._cand_source(device, kl, scores, a_ec, s_max, a_pos, m)
                # Per-row scan bound from the state array: each row reads only its own
                # causal region, so the score buffer needs no -inf backfill
                ext.dsa_topk(scores, indices, self.index_topk, a_ec,
                             0 if dsv4_b1_split and R == 1 else S)
            # Tag by the branch that actually served the step, so the dump pairs with the eager one
            b1_mode = ("use" if self.candidate_mode == "use" else "source") if cand_regime else "none"

            if _pdup_dir and sel_live:
                tensors = dict(
                    scores = scores, sel = indices, ent = ent,
                    cand = self._cand_mask(device, kl, R) if cand_regime else None,
                    q = qi.view(R, Hi, Di), w = wts,
                    krow = krow if (S == 1 and self.idx_wk is not None) else None)
                # Register static buffers for the host after replay.
                sel_args = (m, self.candidate_block_size)
                _pdup_reg(self.layer_idx, b1_mode, S, tensors, selection = sel_args)
                _pdup("b1_" + b1_mode, self.layer_idx, a_pos[0], S, tensors, selection = sel_args)
            # Readers of this pool (indexer_mode "shared") take this selection in the same step: the
            # selector precedes every one of its readers in layer order, and each selector of a pool
            # overwrites the entry, so a reader picks up the most recent selection -- the same
            # last-writer-wins rule the eager sh["topk"] follows. No k_len is needed -- a selection
            # only runs once ec > index_topk, i.e. min(index_topk, ec) == index_topk
            _B_SEL[id(kl)] = (params.get("dsv4_b_epoch"), indices)
        elif self.indexer_mode == "shared":
            epoch, sel = _B_SEL.get(id(kl), (None, None))
            if sel is None or epoch is not params.get("dsv4_b_epoch"):
                raise RuntimeError(
                    f"DSV4Attention: layer {self.layer_idx} reads pool {self.kv_source} before a "
                    "selector published a selection in this step")
            indices = sel

        # Paged pools: the split kernel reads one block-table row per job from the fixed
        # (B, num_pages) static
        qc = None
        if self.kv_source is not None:
            bt = bt_st
            epp = kl.epp
            pool_c = kl.pool_c_view()
            pool_r = (g_tensor_cache.get(device, (1, self.rope_head_dim), torch.half, "dsv4_b_pr0")
                      if kl.kv_format == "fp4" else kl.pool_r.view(-1, self.rope_head_dim))
            qc = kl.qc()
        else:
            epp = 256
            pool_c = g_tensor_cache.get(device, (1, self.head_dim - self.rope_head_dim),
                                        torch.half, "dsv4_b_pc0")
            pool_r = g_tensor_cache.get(device, (1, self.rope_head_dim), torch.half, "dsv4_b_pr0")
            bt = g_tensor_cache.get(device, (1, 1), torch.int32, "dsv4_b_bt0")

        hpg = self.num_q_heads // self.o_groups
        out = dsa_attn(
            q.view(R, self.num_q_heads, self.head_dim).half().contiguous(),
            pool_c, pool_r, bt, sinks = self.sinks,
            ring = rsl.ring, kv_chunk = kv.reshape(R, self.head_dim),
            win_len = w, indices = indices,
            compress_rate = m, scale = self.sm_scale,
            derot_inv_freq = self._rope_type_neg(), groups = self.o_groups, group_major = True,
            page_size = epp, qc = qc,
            out = g_tensor_cache.get(device, (self.o_groups, R, hpg * self.head_dim),
                                     torch.half, "dsv4_b_out"),
            multirow = dict(
                q_pos = a_pos, win_floor = a_floor, ring_beg = a_beg, pool_len = a_ec,
                k_len = a_klen, slot_ids = a_slots,
                ring_stride = rsl.ring_rows * self.head_dim, seq = S,
            ),
            # dsa_attn auto-splits a single row into 16 flash-decoding partials once the scanned
            # region passes 256 rows, else 8 -- and its multirow branch defaults to 8. The splits
            # decide how the fp32 partials are combined, so an 8-split body would differ from the
            # 16-split eager call in the last bits: force the value eager computes for this layer
            # (indices is None only for the window-only layers, whose eager estimate is win_len)
            n_splits = 16 if (w if w > 0 else 0) + (self.index_topk if indices is not None else 0) > 256 else 8,
        )

        if _dsv4_dump_dir:
            _dsv4_dump("B", self.layer_idx, int(a_pos[0]), int(a_ec[0]), m,
                       {"q": q, "kv": kv, "out": out, "ck": comp_kv, "cg": comp_gate}, kl,
                       bt_st[:1] if bt_st is not None else None)
        if _pdup_dir:
            ts = dict(out = out.transpose(0, 1).reshape(R, -1), kv = kv.reshape(R, -1),
                      q = q.reshape(R, -1), indices = indices if sel_live else None)
            pool = (kl, bt_st[:1], a_ec[0]) if self.kv_source is not None else None
            _pdup_reg(self.layer_idx, "attn", S, ts, pool)
            _pdup("b1_attn", self.layer_idx, a_pos[0], S, ts, pool)
        # Ring appends (shift already applied in the prepass): batched, slot-indexed
        ext.dsv4_ring_append(kv.reshape(R, self.head_dim), rsl.ring, a_pos, a_beg, a_slots)
        return self._project_o_grouped(out.unsqueeze(1), params, out_dtype,
                                       mgemm_out = True).view(B, S, -1)

    def _project_batch(self, x, params, rsg, a_pos = None):
        """One projection pass over all B * S rows for the batched cached path. Returns
        (q_res (1, R, q_lora), q (B, S, H, hd) roped, kv (B, S, 1, hd) roped+normed,
        comp_kv/gate (R, Wc), idx_kv/gate (R, Wi), q_idx (1, R, Hi * Di) or None), or None
        to decline (mixed-format fans handle their own decline; plain Linears always work)."""
        B, S, _ = x.shape
        rows = B * S
        if not self.x_fan_ready:
            self._build_x_fan()
        xf = x.view(1, rows, -1)
        comp_kv = comp_gate = idx_kv = idx_gate = q_idx = None
        # Layers that run an indexer scan need the indexer query: indexer_mode "full" (V4.1 layers with
        # their own indexer head -- with idx_wk only if they also own the pool they index -- or a V4
        # "csa" layer, whose mode defaults to "full"). Readers ("shared") reuse a published selection
        need_q_idx = self.indexer_mode == "full" and self.idx_wq_b is not None

        if self.x_fan is not None and rows <= 32:
            f = self.x_fan
            shapes = [(self.q_a.out_features,), (self.head_dim,)]
            if self.compressor is not None:
                shapes += [(self.compressor.wkv.out_features,)] * 2
            if self.indexer is not None:
                shapes += [(self.indexer.wkv.out_features,)] * 2
            fouts, cptr, ahad = self._fan_outs(f, "x", rows, shapes)
            ext.exl3_mgemm(
                xf, f["trellis"], fouts[0].view(1, rows, -1), f["suh"], ahad, f["svh"],
                f["idx"], None, f["K"], -1, f["mcg"], f["mul1"], -1, -1, 0, 1,
                f["n"], cptr)
            q_res = g_tensor_cache.get(x.device, (rows, self.q_a.out_features),
                                       torch.half, "dsv4_b_qres")
            ext.rms_norm(fouts[0], self.q_norm.weight, q_res, self.q_norm.rms_norm_eps,
                         self.q_norm.constant_bias, self.q_norm.constant_scale,
                         False, False)
            q_res = q_res.view(1, rows, -1)
            kv = fouts[1]
            if self.compressor is not None:
                comp_kv, comp_gate = fouts[2], fouts[3]
            if self.indexer is not None:
                idx_kv, idx_gate = fouts[4], fouts[5]
        else:
            q_res = self.q_norm.forward(self.q_a.forward(xf, params), params, out_dtype = torch.half)
            kv = self.wkv.forward(xf, params)[0]
            if self.compressor is not None:
                if not self.compressor.fused_ready:
                    self.compressor._build_fused()
                comp = self.compressor
                if comp.mg_pair:
                    # Same 2-expert exl3_mgemm the eager BC compressor runs (wkv, wgate share
                    # the input), so the projected rows match eager bit-for-bit
                    mg_tr, mg_suh, mg_svh, mg_idx = comp.mg_pair_tables
                    ah = g_tensor_cache.get(x.device, (2, rows, self.hidden_size), torch.half, "dsv4_comp_ah")
                    cc = g_tensor_cache.get(x.device, (2, rows, comp.wkv.out_features), torch.half, "dsv4_comp_mg")
                    ext.exl3_mgemm(xf, mg_tr, cc, mg_suh, ah, mg_svh, mg_idx, None,
                                   comp.mg_pair_K, -1, comp.mg_pair_mcg, comp.mg_pair_mul1,
                                   -1, -1, 0, 1, None, None)
                    comp_kv, comp_gate = cc[0], cc[1]
                else:
                    comp_kv = comp.wkv.forward(xf, params)[0]
                    # Gateless compressor: compress_ratios == 1 means non-overlapping windows,
                    # so dsv4_compress's gate softmax runs over a single element and any
                    # same-shaped tensor yields kv unchanged -- reuse comp_kv (the kernel's
                    # gate_new is not optional, and the shape check compares it to kv_new)
                    comp_gate = comp.wgate.forward(xf, params)[0] if comp.wgate is not None \
                        else comp_kv
            if self.indexer is not None:
                if not self.indexer.fused_ready:
                    self.indexer._build_fused()
                idx_kv = self.indexer.wkv.forward(xf, params)[0]
                idx_gate = self.indexer.wgate.forward(xf, params)[0]

        if need_q_idx and self.q_fan is not None and rows <= 32:
            f2 = self.q_fan
            o2, cptr2, ahad2 = self._fan_outs(f2, "q", rows, [
                (self.num_q_heads * self.head_dim,),
                (self.index_n_heads * self.index_head_dim,)
            ])
            ext.exl3_mgemm(
                q_res, f2["trellis"], o2[0].view(1, rows, -1), f2["suh"], ahad2, f2["svh"],
                f2["idx"], None, f2["K"], -1, f2["mcg"], f2["mul1"], -1, -1, 0, 1,
                f2["n"], cptr2)
            q = o2[0]
            q_idx = o2[1].view(1, rows, -1)
        elif self.qb_multi is not None and rows <= 32:
            q = self._mgemm1(self.qb_multi, q_res, torch.half, "dsv4_qb1_ah")
            # qb_multi stands in when q_b and idx_wq_b can't share a fan, so the indexer query
            # still needs its own projection
            if need_q_idx:
                q_idx = self.idx_wq_b.forward(q_res, params)
        else:
            q = self.q_b.forward(q_res, params)
            if need_q_idx:
                q_idx = self.idx_wq_b.forward(q_res, params)

        q = q.view(B, S, self.num_q_heads, self.head_dim)
        kv = kv.view(B, S, 1, self.head_dim)
        positions = a_pos if a_pos is not None else \
            torch.tensor([rs.position for rs in rsg[:B]], dtype = torch.int32,
                         device = x.device)
        self._rope_qkv(q, kv, 0, positions)
        return q_res, q, kv, comp_kv, comp_gate, idx_kv, idx_gate, q_idx


    def _forward_cached_one(
        self,
        x,
        params,
        rs,
        rsl,
        out_dtype,
        copy_static = False,
        pre = None,
        return_o = False,
        kl = None,
        bt_row = None,           # (1, npr) i32 device block-table row of this job
    ):
        _, seq, _ = x.shape

        # Whole-step graph path (EXL3_BC_DSA=1); not used when the batched path already
        # projected this job's rows (pre)
        if pre is None and self.q_head_norm and \
                bc_dsa_enable and seq <= 16 and x.dtype == torch.half and x.is_contiguous():
            if not hasattr(self, "_bc_dsa"):
                self._bc_dsa = {}
            key = (id(rsl), rs.slot)
            bcd = self._bc_dsa.get(key)
            if bcd is None:
                bcd = build_bc_dsa(self, rs, rsl, kl)
                self._bc_dsa[key] = bcd if bcd is not None else False
            if bcd:
                y = bcd.run(x, rs, rsl, bt_row)
                if y is not None:
                    # y is a shared static, overwritten by the next slot's replay; batch
                    # rows are assembled only after all slots have run
                    if _rp_on: _rpnote("bc_graph_hit")
                    return y.clone() if copy_static else y
                if _rp_on: _rpnote("bc_graph_decline")
        if _rp_on: _rpnote("eager_core_S%d" % seq)
        device = x.device
        pos0 = rs.position
        slot = rs.slot

        if not self.x_fan_ready:
            self._build_x_fan()

        converting = any(k in params for k in ("capture", "quant_preserve", "ovr", "reconstruct"))
        use_fan = self.x_fan is not None and seq <= 32 and not converting
        ec = (pos0 + seq) // self.compress_rate if self.kv_source is not None else 0
        topk_regime = self.indexer_mode is not None and ec > self.index_topk
        q_idx_pre = None
        fouts = None
        if pre is not None:
            # Batched path already ran the shared projection pass (rope applied, per-job
            # positions); the compress section consumes the row slices directly
            q_res, q, kv = pre["q_res"], pre["q"], pre["kv"]
            fouts = [None, None, pre["comp_kv"], pre["comp_gate"], pre["idx_kv"], pre["idx_gate"]]
            q_idx_pre = pre["q_idx"]
            use_fan = True
        elif use_fan:

            # One per-matrix-N mgemm covers the whole x-side projection fan (q_a, wkv and
            # both compressors' kv/gate); a second one pairs q_b with idx_wq_b over q_res
            # in the top-k regime. Head norms fold into the rope kernel as in _project_qkv
            f = self.x_fan
            shapes = [(self.q_a.out_features,), (self.head_dim,)]
            if self.compressor is not None:
                shapes += [(self.compressor.wkv.out_features,)] * 2
            if self.indexer is not None:
                shapes += [(self.indexer.wkv.out_features,)] * 2
            fouts, cptr, ahad = self._fan_outs(f, "x", seq, shapes)
            ext.exl3_mgemm(
                x,
                f["trellis"],
                fouts[0].view(1, seq, -1),
                f["suh"],
                ahad,
                f["svh"],
                f["idx"],
                None,
                f["K"],
                -1,
                f["mcg"], f["mul1"],
                -1, -1, 0, 1,
                f["n"],
                cptr
            )
            q_res = self.q_norm.forward(fouts[0].view(1, seq, -1), params, out_dtype = torch.half)
            kv = fouts[1].view(1, seq, 1, self.head_dim)

            if topk_regime and self.q_fan is not None:
                f2 = self.q_fan
                o2, cptr2, ahad2 = self._fan_outs(f2, "q", seq, [
                    (self.num_q_heads * self.head_dim,),
                    (self.index_n_heads * self.index_head_dim,)
                ])
                ext.exl3_mgemm(
                    q_res, f2["trellis"], o2[0].view(1, seq, -1), f2["suh"], ahad2, f2["svh"],
                    f2["idx"], None, f2["K"], -1, f2["mcg"], f2["mul1"], -1, -1, 0, 1,
                    f2["n"], cptr2
                )
                q = o2[0].view(1, seq, self.num_q_heads, self.head_dim)
                q_idx_pre = o2[1]
            else:
                q = self.q_b.forward(q_res, params).view(1, seq, self.num_q_heads, self.head_dim)

            self._rope_qkv(q, kv, pos0)
            kv = kv.view(1, seq, self.head_dim)
        else:
            q_res, q, kv = self._project_qkv(x, params, pos0)

        # Window sources for the kernel: this chunk's kv rows plus prior rows read from the
        # ring at abs - window_beg; the kernel derives all per-query addressing from the
        # positions, so no temp/index tensors are built. The ring itself is updated AFTER
        # attention: the shift/rebase branches move rows the kernel reads in place
        w = self.sliding_window
        n_prev = min(w - 1, pos0 - rs.window_beg, pos0)
        win_floor = pos0 - n_prev
        ring = rsl.ring[slot]
        ring_beg = rs.window_beg

        indices = None
        k_len = 0
        pool_len = 0
        dense_m = 1
        epp = 256
        if self.compressor is not None:
            m = self.compress_rate
            dense_m = m
            epp = kl.epp
            assert ec <= bt_row.shape[1] * epp, \
                f"DSA pool overflow: entry {ec} beyond block table ({bt_row.shape[1]} pages)"

            # Fused compressor step: projections + window pooling + norm + rope, entries
            # written into the paged pools through the job's block table, ring/snapshot
            # state updated. With the fan the projections are already done: feed the
            # compress kernels directly
            fp4 = kl.kv_format == "fp4"
            pool_r_flat = None if fp4 else kl.pool_r.view(-1, kl.D_r)
            idx_stage = (g_tensor_cache.get(device, (seq // m + 1, kl.D_i), torch.half, "dsv4_idx_stage")
                         if fp4 and self.layer_type == "csa" else None)
            if kl.quant or fp4 or self.idx_wk is not None:
                # Packed pool: compress into transient staging rows (this step's entries),
                # then quantize + scatter through the block table
                stage = g_tensor_cache.get(device, (seq // m + 1, self.head_dim), torch.half, "dsv4_stage")
                dest_a, dest_b, dbt, depp, rel = stage, None, None, 0, True
            else:
                dest_a, dest_b, dbt, depp, rel = kl.pool_c.view(-1, kl.D_c), pool_r_flat, bt_row, epp, False
            if use_fan:
                comp = self.compressor
                ext.dsv4_compress(
                    fouts[2], fouts[3], rsl.comp_buf_kv[slot], rsl.comp_buf_gate[slot],
                    rsl.comp_ovl[slot] if rsl.comp_ovl is not None else None,
                    comp.ape, comp.fused_norm_w, comp.norm.rms_norm_eps, comp.fused_inv_freq,
                    dest_a, dest_b, pos0, None, m, None, dbt, depp, rel)
                if self.layer_type == "csa":
                    idx = self.indexer
                    ext.dsv4_compress(
                        fouts[4], fouts[5], rsl.idx_buf_kv[slot], rsl.idx_buf_gate[slot],
                        rsl.idx_ovl[slot], idx.ape, idx.fused_norm_w, idx.norm.rms_norm_eps,
                        idx.fused_inv_freq, idx_stage if fp4 else kl.pool_idx.view(-1, kl.D_i), None, pos0,
                        None, m, None, None if fp4 else bt_row, 0 if fp4 else epp, fp4)
            else:
                self.compressor.forward_fused(
                    x, params, rsl.comp_buf_kv[slot], rsl.comp_buf_gate[slot],
                    rsl.comp_ovl[slot] if rsl.comp_ovl is not None else None,
                    dest_a, dest_b, pos0, dbt, depp, stage_rel = rel)
                if self.layer_type == "csa":
                    self.indexer.forward_fused(
                        x, params, rsl.idx_buf_kv[slot], rsl.idx_buf_gate[slot],
                        rsl.idx_ovl[slot], idx_stage if fp4 else kl.pool_idx.view(-1, kl.D_i), None, pos0,
                        None if fp4 else bt_row, 0 if fp4 else epp, stage_rel = fp4)
            if os.environ.get("EXL3_DSV4_B1_DBG"):
                print(f"EAGER L{self.layer_idx} pos={pos0} seq={seq} m={m} ec={ec} "
                      f"e0={pos0 // m} n={ec - pos0 // m}", flush=True)
            if self.idx_wk is not None and ec > pos0 // m:
                e0 = pos0 // m
                keys = self._indexer_keys(stage[:ec - e0], params, e0)
                if fp4:
                    ext.dsv4_fp4_scatter(keys, kl.pool_idx_fp4.view(-1, 64), kl.pool_idx_scale.view(-1, 4),
                                         bt_row, pos0, None, m, seq, epp, True)
                else:
                    rows = torch.arange(e0, ec, device = device)
                    rows = bt_row[0, rows // epp].long() * epp + rows % epp
                if not fp4 and not kl.quant and os.environ.get("EXL3_DSV41_FUSED_POOL_SCATTER", "1") != "0":
                    ext.dsv4_pool_scatter(rows, keys, stage[:ec - e0],
                                          kl.pool_idx.view(-1, kl.D_i), kl.pool_c.view(-1, kl.D_c), pool_r_flat)
                elif not fp4:
                    kl.pool_idx.view(-1, kl.D_i)[rows] = keys
                    if not kl.quant:
                        kl.pool_c.view(-1, kl.D_c)[rows] = stage[:ec - e0, :kl.D_c]
                        pool_r_flat[rows] = stage[:ec - e0, kl.D_c:]
            if kl.quant:
                ext.dsv4_pool_quant_scatter(
                    stage, kl.pool_c_view(), kl.pool_s.view(-1, kl.G), pool_r_flat,
                    bt_row, pos0, None, m, seq, epp)
            if fp4:
                ext.dsv4_fp4_scatter(stage, kl.pool_fp4.view(-1, 256), kl.pool_fp4_scale.view(-1, 32),
                                     bt_row, pos0, None, m, seq, epp, False)
                if idx_stage is not None and self.idx_wk is None:
                    ext.dsv4_fp4_scatter(idx_stage, kl.pool_idx_fp4.view(-1, 64), kl.pool_idx_scale.view(-1, 4),
                                         bt_row, pos0, None, m, seq, epp, True)
            pool_len = ec
        elif self.kv_source is not None:
            assert kl.device == device, \
                f"DSV4Attention: shared pool on another device (layer {self.layer_idx}, " \
                f"source {self.kv_source}, pool {kl.device}, attn {device}, " \
                f"owner {self.kv_owner.layer_idx if self.kv_owner is not None else None})"
            dense_m, epp, pool_len = self.compress_rate, kl.epp, ec

        if topk_regime:
            # Selection is only non-trivial once the pool exceeds index_topk: below that,
            # top-k keeps every entry under the causal bound, which is exactly DENSE_POOL
            # mode. Indexer scoring chain is skipped (key pool is still maintained for later)
            sh = params.setdefault("dsv4_shared", {}).setdefault(slot, {})
            if self.indexer_mode == "full":
                sh["topk"] = self._indexer_topk(
                    x, params, q_res, kl.pool_idx_fp4.view(-1, 64) if kl.kv_format == "fp4" else kl.pool_idx.view(-1, kl.D_i),
                    ec, pos0, q_idx_pre, block_table = bt_row, epp = epp, shared = sh,
                    idx_scales = kl.pool_idx_scale.view(-1, 4) if kl.kv_format == "fp4" else None)
            elif sh["topk"][0].device != device:
                sh["topk"] = to_device(sh["topk"][0], device), sh["topk"][1]
            indices, k_len = sh["topk"]

        qc = None
        if self.kv_source is not None:
            pool_c = kl.pool_c_view()
            pool_r = (g_tensor_cache.get(device, (1, kl.D_r), torch.half, "dsv4_pool_r0")
                      if kl.kv_format == "fp4" else kl.pool_r)
            qc = kl.qc()
            bt = bt_row
        else:
            pool_c = g_tensor_cache.get(device, (1, self.head_dim - self.rope_head_dim), torch.half, "dsv4_pool_c0")
            pool_r = g_tensor_cache.get(device, (1, self.rope_head_dim), torch.half, "dsv4_pool_r0")
            bt = torch.zeros((1, 1), dtype = torch.int32, device = device)

        # eq. 26 de-rotation and the group-major store for the grouped o_proj are fused into
        # the kernel epilogue: output is (G, seq, hpg * D), fp16
        hpg = self.num_q_heads // self.o_groups
        # The chunk's attention-output workspace goes through get_scratch, not get: an exact-shape entry
        # is pinned for the life of the process, and a 32k prefill with an adaptive chunk creates a new
        # row count every few chunks. Measured 2026-11-01 on the prod pair: one 32k prefill left 0.627 GiB
        # of dsv4_attn_out entries (2560+2304+2048+1536+1280+256 rows, 93% of all statics) and that was
        # almost exactly the +700 MiB by which live allocation grew during the prefill - which is what
        # shrank the grant from 2560 to 256 rows. get_scratch keeps decode-class sizes and sends larger
        # ones through the caching allocator, power-of-two rounded, so the next chunk size reuses it.
        out_ws = g_tensor_cache.get_scratch(device, self.o_groups * seq * hpg * self.head_dim,
                                            torch.half, "dsv4_attn_out", seq) \
            .view(self.o_groups, seq, hpg * self.head_dim)
        out = dsa_attn(
            q[0].half().contiguous(), pool_c, pool_r, bt, sinks = self.sinks,
            ring = ring, kv_chunk = kv[0], win_len = self.sliding_window,
            win_floor = win_floor, ring_beg = ring_beg,
            indices = indices, k_len = k_len, pool_len = pool_len, q_pos0 = pos0,
            compress_rate = dense_m, scale = self.sm_scale,
            derot_inv_freq = self._rope_type_neg(), groups = self.o_groups, group_major = True,
            page_size = epp, qc = qc,
            out = out_ws,
            # Image chunk (vision): bidirectional over the chunk, own window into history
            nc_chunk = bool(params.get("nc_chunk", False)),
        )
        if _dsv4_dump_dir:
            _dsv4_dump("S", self.layer_idx, pos0, ec, dense_m,
                       {"q": q, "kv": kv, "out": out, "ck": fouts[2] if fouts is not None else None,
                        "cg": fouts[3] if fouts is not None else None}, kl,
                       bt_row if kl is not None else None)
        if _pdup_dir:
            # Second dump stage: identical attention inputs but a different output would put the
            # divergence in the attn kernel invocation rather than in the selection above
            _pdup("eager_attn", self.layer_idx, pos0, seq, dict(
                out = out.transpose(0, 1).reshape(seq, -1), kv = kv.reshape(seq, -1),
                q = q.reshape(seq, -1), indices = indices if topk_regime else None),
                (kl, bt_row, pool_len) if self.kv_source is not None else None)

        # Ring update after attention: the shift/rebase branches move rows the kernel
        # reads in place (the kernel addresses the PRE-update ring via ring_beg). Keep the
        # trailing window resident for the next forward: in-place append while the chunk
        # fits; SWA-style page shift for small appends near the ring end; a window rebase
        # for chunks that overflow the ring outright. All layers compute the same wshift
        # for the same forward, so setting it is idempotent
        offset = pos0 - rs.window_beg
        pos_end = pos0 + seq
        if offset + seq <= rsl.ring_rows:
            ring[offset : offset + seq].copy_(kv[0])
        elif seq < PAGE_SIZE:
            need = offset + seq - rsl.ring_rows
            shift = -(-need // PAGE_SIZE) * PAGE_SIZE
            ring[:rsl.ring_rows - shift].copy_(ring[shift:].clone())
            rs.wshift = shift
            ring[offset - shift : offset - shift + seq].copy_(kv[0])
        else:
            new_beg = max(pos_end - (w - 1), 0) // PAGE_SIZE * PAGE_SIZE
            n_keep = pos_end - new_beg
            n_from_kv = min(n_keep, seq)
            n_from_ring = n_keep - n_from_kv          # rows below pos0 still needed
            assert 0 < n_keep <= rsl.ring_rows
            if n_from_ring > 0:
                src0 = pos0 - n_from_ring - ring_beg
                ring[:n_from_ring].copy_(ring[src0 : src0 + n_from_ring].clone())
            ring[n_from_ring : n_keep].copy_(kv[0, seq - n_from_kv:])
            rs.wshift = new_beg - rs.window_beg

        if return_o:
            return out
        return self._project_o_grouped(out.unsqueeze(1), params, out_dtype)

