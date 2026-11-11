#!/usr/bin/env python3
"""cuda-exl3 as the GX cold-expert path: restack a layer's cold experts per bitrate group.

The production cold path (exllamav3's `BlockSparseMLP.forward` with external routing) plans tiers
per expert and runs one launch per tier per expert group. cuda-exl3 wants the opposite shape: the
experts of one bitrate stacked into a single trellis tensor, so that one launch per bitrate covers
every expert that shares a codebook. That is the whole trade this module makes:

    had_in(x, suh13)  ->  gemm(w13, shards [I, I])  ->  glu_had_in(suh2)  ->  gemm(w2, shards [H])

per bitrate group, with the final gemm fusing the routing weight and the gather back to tokens
(atomics; `CUDA_EXL3_DETERMINISTIC=1` swaps that for `exl3_moe_combine`).

Layout built here for a group of E experts (experts of one layer share a tile width across
gate/up/down, so the group key is the trellis word count):

    trellis13  (E, H/16, 2*I/16, W)   gate tiles then up tiles along N
    suh13      (E, 2, H)
    svh13      (E, 2*I)
    trellis2   (E, I/16, H/16, W)
    suh2       (E, 1, I)
    svh2       (E, H)

`forward` takes the same half-precision token block the server already builds, the clamped
int64 routing ids (misses -> expert 0 with weight 0, exactly like the incumbent path) and the
routing weights, and returns fp32 (M, H). The down projection's routing-weighted epilogue adds
straight into one fp32 accumulator, so the bitrate groups cost one output buffer and no
elementwise sum: the slab gets the layer's answer with a single rounding, at the end.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys

import torch

CUDA_EXL3_SRC = os.environ.get("CUDA_EXL3_SRC", "/src/exllamav3/vendor/cuda-exl3/src")
if CUDA_EXL3_SRC not in sys.path:
    sys.path.insert(0, CUDA_EXL3_SRC)
import cuda_exl3.ops as cx_ops  # noqa: E402,F401  (loads the extension)

_C = torch.ops.cuda_exl3_C

TRACE = os.environ.get("CUDA_EXL3_MOE_TRACE", "0") != "0"
CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
DETERMINISTIC = os.environ.get("CUDA_EXL3_DETERMINISTIC", "0") != "0"
BLOCK_M_TIERS = (16, 32, 64, 128)
# One extra pass over an expert's weights costs about as many padded rows as this (layer 5,
# GX2, 512..8192 rows): it is what puts the optimum at 128 at 4096 rows but 64 at 8192.
BLOCK_M_ROWS = 24
BLOCK_M_OVERRIDE = int(os.environ.get("CUDA_EXL3_MOE_BLOCK_M", "0"))
# Split-k needs the rows*n_total fp32 accumulator; it is only used when rows is small and the
# upstream knob warns that the accumulator epilogue can read out of bounds when the grid is below
# the SM count (upstream PR #7). The group gemms here are never starved - 268 experts over 3
# groups is thousands of blocks - so the accumulator is capped to zero and every launch is
# unsplit. Set CUDA_EXL3_MOE_ACC_CAP to re-enable it for an experiment.
# The guard is for importers that only want the pack helpers (_Shard/_Stage/_pack_names) on a host
# without the vendor extension (the P7 ColdSteal byte source): there no op exists to cap.
if int(os.environ.get("CUDA_EXL3_MOE_ACC_CAP", "0")) == 0 and hasattr(_C, "exl3_set_moe_acc_cap"):
    _C.exl3_set_moe_acc_cap(0)


def _has_acc_out() -> bool:
    """Does this build of the extension take the shared-accumulator (out, accumulate) pair?"""
    try:
        return "accumulate" in str(_C.exl3_moe_gemm.default._schema)
    except Exception:  # a .so from before the fold: fall back to summing three outputs
        return False

HAS_ACC_OUT = _has_acc_out()


def _bits_frac(words: int) -> tuple[int, bool]:
    """Trellis words per tile -> (whole bits, fractional-bit tile)."""
    return words // 16, (words % 16) != 0


def _codebook(line) -> int:
    """Codebook id for an exllamav3 EXL3 linear, from the flags the module already carries."""
    inner = getattr(line, "inner", line)
    if getattr(inner, "mul1", False):
        return CB_MUL1
    if getattr(inner, "mcg", False):
        return CB_MCG
    return CB_3INST


def _inner(line):
    inner = getattr(line, "inner", line)
    for name in ("trellis", "suh", "svh"):
        if not hasattr(inner, name):
            raise AttributeError(f"exl3_moe_cuda: {type(line).__name__} has no .{name} "
                                 f"(not an EXL3 linear?)")
    return inner


class RestackBroken(RuntimeError):
    """The streaming restack failed after source tensors were released (see CudaExl3Moe)."""


class Group:
    """One bitrate: the stacked tensors of every expert that shares its tile width."""

    __slots__ = ("bits", "frac", "cb", "trellis13", "suh13", "svh13",
                 "trellis2", "suh2", "svh2", "n_experts", "source", "remap")

    def __init__(self, bits, frac, cb):
        self.bits, self.frac, self.cb = bits, frac, cb
        self.n_experts = 0

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.trellis13, self.suh13, self.svh13, self.trellis2, self.suh2, self.svh2))


def _stack_group(gates, ups, downs, device, free_source: bool = False) -> Group:
    g0 = _inner(gates[0])
    d0 = _inner(downs[0])
    words = int(g0.trellis.shape[2])
    bits, frac = _bits_frac(words)
    grp = Group(bits, frac, _codebook(gates[0]))
    e_n = len(gates)
    kt, nt = int(g0.trellis.shape[0]), int(g0.trellis.shape[1])
    k2t, n2t = int(d0.trellis.shape[0]), int(d0.trellis.shape[1])
    grp.n_experts = e_n
    grp.trellis13 = torch.empty((e_n, kt, 2 * nt, words), dtype=torch.int16, device=device)
    grp.suh13 = torch.empty((e_n, 2, 16 * kt), dtype=torch.half, device=device)
    grp.svh13 = torch.empty((e_n, 32 * nt), dtype=torch.half, device=device)
    grp.trellis2 = torch.empty((e_n, k2t, n2t, int(d0.trellis.shape[2])), dtype=torch.int16,
                               device=device)
    grp.suh2 = torch.empty((e_n, 1, 16 * k2t), dtype=torch.half, device=device)
    grp.svh2 = torch.empty((e_n, 16 * n2t), dtype=torch.half, device=device)
    for e, (g, u, d) in enumerate(zip(gates, ups, downs)):
        gi, ui, di = _inner(g), _inner(u), _inner(d)
        grp.trellis13[e, :, :nt] = gi.trellis
        grp.trellis13[e, :, nt:] = ui.trellis
        grp.suh13[e, 0] = gi.suh
        grp.suh13[e, 1] = ui.suh
        grp.svh13[e, :32 * nt // 2] = gi.svh
        grp.svh13[e, 32 * nt // 2:] = ui.svh
        grp.trellis2[e] = di.trellis
        grp.suh2[e, 0] = di.suh
        grp.svh2[e] = di.svh
        if free_source:
            # Streaming: this expert's bytes are inside the stack now (every copy above is a
            # synchronous device-to-device copy on the default stream, so nothing is in flight), and
            # holding the source until the whole layer is stacked is what doubles the conversion's
            # footprint. The locals go too, so the allocator can reuse the block immediately.
            for inner in (gi, ui, di):
                inner.trellis = inner.suh = inner.svh = None
            del gi, ui, di
    return grp


def _live_mib(ffn) -> float:
    """Bytes of this layer's cold experts as they sit now, MiB."""
    total = 0
    for e in range(len(ffn.gates)):
        for line in (ffn.gates[e], ffn.ups[e], ffn.downs[e]):
            i = _inner(line)
            for t in (i.trellis, i.suh, i.svh):
                if t is not None:
                    total += t.numel() * t.element_size()
    return total / 2**20


def _group_mib(ffn) -> tuple[int, float]:
    """(group count, largest group MiB) for a layer's bitrate split."""
    by_key: dict[tuple[int, int, int], int] = {}
    for e, g in enumerate(ffn.gates):
        key = (int(_inner(g).trellis.shape[2]), int(_inner(ffn.downs[e]).trellis.shape[2]),
               _codebook(g))
        sz = 0
        for line in (g, ffn.ups[e], ffn.downs[e]):
            i = _inner(line)
            sz += sum(t.numel() * t.element_size() for t in (i.trellis, i.suh, i.svh))
        by_key[key] = by_key.get(key, 0) + sz
    return len(by_key), max(by_key.values(), default = 0) / 2**20


def transient_mib(ffn) -> float:
    """Largest single group the restack allocates before any source of that group is freed, MiB.

    This is what a conversion actually has to find free, since the layer it converts is already
    resident: `_stack_group` takes one group's sources (already outside MemAvailable) and allocates
    the group's six stacked tensors on top, then frees those sources before the next group starts.
    Measured on GX2 2026-11-01: 175 ms per 2.7-3.0 GiB layer, largest group 1.03-1.37 GiB, and the
    steady-state MemAvailable fell by only ~0.9 GiB per converted layer.
    """
    return _group_mib(ffn)[1]


def peak_mib(ffn) -> float:
    """Absolute residency at the conversion's peak, MiB: live cold bytes plus the largest group.

    Only the second term is new allocation (see `transient_mib`); the first is already resident and
    already excluded from MemAvailable, so a gate has to compare MemAvailable against
    `transient_mib`. Kept for reports: it is the figure that explains a box going over its head
    when the gate is bypassed (GX1 2026-11-01 16:44 needed 5.4 GiB of a 1.32 GiB budget).
    """
    return _live_mib(ffn) + _group_mib(ffn)[1]


def align_block(local_ids: torch.Tensor, block_m: int, num_experts: int, m_valid: int):
    """vLLM's `moe_align_block_size`, in torch: bucket the routed rows by expert and pad every
    bucket to whole `block_m` blocks.

    `local_ids` is (M, T) of this group's expert index per pick, -1 where the pick belongs to
    another bitrate group (that row is that group's to compute, and is left out here). The
    returned `sorted_ids` maps a padded row position to a flat token-major pick `idx < m_valid`,
    which is what the kernels gather from; the pad sentinel is `m_valid` and not -1 because the
    kernels test `sorted_ids[row] < m_valid` for a live row, so a negative sentinel would read as
    a live row pointing before the token block.
    """
    flat = local_ids.reshape(-1)
    live = flat >= 0
    rows_idx = torch.nonzero(live, as_tuple=False).squeeze(1)
    n = int(rows_idx.numel())
    if n == 0:
        return None, None, None, block_m
    ids_live = flat[rows_idx]
    order = torch.argsort(ids_live, stable=True)
    counts = torch.bincount(ids_live, minlength=num_experts)
    nz = counts > 0
    bpe = torch.zeros_like(counts)
    bpe[nz] = (counts[nz] + block_m - 1) // block_m
    blk_off = torch.cumsum(bpe, 0) - bpe
    nblk = int(blk_off[-1] + bpe[-1])
    total = nblk * block_m
    ent = ids_live[order]
    off = torch.cumsum(counts, 0) - counts
    pos = blk_off[ent] * block_m + (torch.arange(n, device=flat.device) - off[ent])
    sorted_ids = torch.full((total,), m_valid, dtype=torch.int32, device=flat.device)
    sorted_ids[pos] = rows_idx[order].to(torch.int32)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=flat.device), bpe)
    return sorted_ids, expert_ids, int(pos.max()) + 1, block_m


def pick_block_m(rows: int, n_experts: int) -> int:
    """Per-expert row-block size, from the two costs a block size trades against.

    Both projections read an expert's whole trellis slice once per block, so the traffic is
    `ceil(per/bm)` weight passes, and each expert's run is padded up to `ceil(per/bm)*bm` rows of
    compute. `BLOCK_M_ROWS` weighs one weight pass against that many padded rows; it was fitted
    to layer 5 on GX2 (measured minima: per=11.5 -> 16, 46 -> 64, 91 -> 128, 182 -> 64) and is
    what makes the answer 128 at 4096 rows and 64 at both 2048 and 8192.
    """
    per = rows / max(n_experts, 1)
    bm = min(BLOCK_M_TIERS,
             key = lambda b: (math.ceil(per / b) * b + BLOCK_M_ROWS * math.ceil(per / b)))
    if BLOCK_M_OVERRIDE:
        bm = BLOCK_M_OVERRIDE
    if bm not in BLOCK_M_TIERS:
        raise ValueError(f"exl3_moe_cuda: block_m {bm} is not a compiled tile "
                         f"({BLOCK_M_TIERS}; 256 only with the h_acc path)")
    return bm


class CudaExl3Moe:
    """A cold layer's experts, restacked per bitrate, forwarded by one launch per projection."""

    def __init__(self, ffn, device=None, drop_source: bool = True):
        self.device = device or next(iter(ffn.gates)).device
        self.top_k = int(getattr(ffn, "num_experts_per_tok", 0) or 0)
        self.num_experts = int(len(ffn.gates))
        by_key: dict[tuple[int, int, int], list[int]] = {}
        for e, g in enumerate(ffn.gates):
            gi, ui, di = _inner(g), _inner(ffn.ups[e]), _inner(ffn.downs[e])
            key = (int(gi.trellis.shape[2]), int(di.trellis.shape[2]), _codebook(g))
            if (int(ui.trellis.shape[2]), _codebook(ffn.ups[e])) != (key[0], key[2]):
                raise ValueError(f"expert {e}: gate/up tile widths differ "
                                 f"({key[0]} vs {ui.trellis.shape[2]})")
            by_key.setdefault(key, []).append(e)
        # Geometry has to be read before any source tensor goes away.
        g0 = _inner(ffn.gates[0])
        d0 = _inner(ffn.downs[0])
        self.hidden = 16 * int(g0.trellis.shape[0])
        self.inter = 16 * int(g0.trellis.shape[1])
        del g0, d0
        if self.top_k == 0:
            self.top_k = int(getattr(ffn, "num_experts_per_tok", 6) or 6)
        if drop_source:
            # The module-wide holders go first: BC_BlockSparseMLP keeps a shared_ptr per expert BC
            # (block_sparse_mlp.py:771-773) and the multilinear bundles keep fusions built over the
            # same experts, so nothing is freed while those are alive. Dropping them before the
            # stack, and each expert's own BC_LinearEXL3 reference with them, is what makes the
            # conversion cost "live cold bytes + one group" instead of twice the layer: `_stack_group`
            # then frees every expert the moment its bytes are in the stack.
            ffn.bc = None
            for name in ("multi_gate", "multi_up", "multi_down"):
                if getattr(ffn, name, None) is not None:
                    setattr(ffn, name, None)
            for e in range(self.num_experts):
                for line in (ffn.gates[e], ffn.ups[e], ffn.downs[e]):
                    _inner(line).bc = None
        self.groups: list[Group] = []
        for key, idx in sorted(by_key.items()):
            try:
                grp = _stack_group([ffn.gates[e] for e in idx], [ffn.ups[e] for e in idx],
                                   [ffn.downs[e] for e in idx], self.device,
                                   free_source = drop_source)
            except Exception as e:
                # Once drop_source has run, this layer's per-expert copies and the module's fused
                # bundles are gone, so *any* failure - the first group included - leaves nothing
                # the mixed-K path could answer with. Say that instead of letting
                # worker_seam_server._cx_moe fall back to a module whose experts were released.
                if drop_source:
                    raise RestackBroken(f"group {key} ({len(idx)} experts) of layer with "
                                        f"{self.num_experts} experts: {type(e).__name__}: {e}") from e
                raise
            grp.source = idx
            self.groups.append(grp)
        # ids arrive as this layer's cold-expert indices; each group needs its own 0..E-1 labels,
        # with -1 marking the picks another group owns.
        for grp in self.groups:
            remap = torch.full((self.num_experts,), -1, dtype=torch.int64, device=self.device)
            remap[torch.tensor(grp.source, dtype=torch.int64, device=self.device)] = torch.arange(
                len(grp.source), dtype=torch.int64, device=self.device)
            grp.remap = remap

    # -- geometry -------------------------------------------------------------------------------
    def nbytes(self) -> int:
        return sum(g.nbytes() for g in self.groups)

    def describe(self) -> str:
        return " ".join(f"bits{g.bits}{'+frac' if g.frac else ''}:{g.n_experts}"
                        for g in self.groups)

    @classmethod
    def from_pack(cls, path: str, layer: int, cold_ids, device = None, fadvise: bool = True):
        """Stack a layer's cold experts straight out of its pack shard, without per-expert modules.

        The restack path (`__init__`) needs the layer's experts resident as one module per expert
        before it can copy them into the stacked layout, so a conversion on a nearly full box has
        to find `transient_mib` (1.5-1.8 GiB) of free memory that exists only for the duration of
        the copy, and whatever the allocator cannot hand back afterwards stays gone. By layers
        37-39 of a 113.65 GiB pack on a 121.6 GiB box that transient does not exist and the layer
        is refused (`SEAM_MOE_NOFIT`). Here the stacked tensors are the *only* copy: they are
        allocated first and each expert's trellis/suh/svh is read from the shard into its slice,
        so the layer costs its cold bytes plus one staging buffer (<= 12 MiB) and the decision
        cannot depend on how much of the box is already loaded.

        Geometry and the codebook come from the shard's header (the shape of `w1.trellis`, the
        presence of `mul1`/`mcg`), which is the same information the module path reads off the
        loaded linears; `qmesh/tests/cx_direct_check.py` compares the two byte for byte.

        Expert `e` of the checkpoint stores its gate as `w1`, its up projection as `w3` and its
        down projection as `w2` (exllamav3's own key mapping), in the layout the kernels here
        consume: trellis (in/16, out/16, words) int16, suh half (in), svh half (out).
        A member's `source` is its position in `cold_ids`, exactly as `__init__` labels the module's
        experts: the ids this layer receives are local to the cold set (`_check_ids` bounds them by
        len(cold)), and `remap` is indexed by them. The shard's keys, by contrast, are the
        checkpoint's global expert ids, which is why the two are carried separately.
        """
        dev = torch.device(device or "cuda")
        shard = _Shard(path, fadvise = fadvise)
        self = cls.__new__(cls)
        self.device = dev
        self.num_experts = len(cold_ids)
        try:
            by_key: dict[tuple[int, int, int], list[tuple[int, int, int, int, int, int]]] = {}
            for local_i, e in enumerate(cold_ids):
                g, u, d = _pack_names(layer, e)
                kt, nt, w13 = shard.shape(g + ".trellis")
                _, nt_u, w13u = shard.shape(u + ".trellis")
                k2t, n2t, w2 = shard.shape(d + ".trellis")
                if w13u != w13:
                    raise ValueError(f"expert {e}: gate/up tile widths differ ({w13} vs {w13u})")
                cb = shard.codebook(g)
                if shard.codebook(u) != cb or shard.codebook(d) != cb:
                    raise ValueError(f"expert {e}: gate/up/down codebooks differ")
                by_key.setdefault((int(w13), int(w2), cb), []).append(
                    (int(local_i), int(e), int(kt), int(nt), int(k2t), int(n2t)))
            # Geometry of the whole layer, from any expert (they share hidden/inter).
            first = by_key[sorted(by_key)[0]][0]
            self.hidden = 16 * first[2]
            self.inter = 16 * first[3]
            self.groups: list[Group] = []
            stage = _Stage(dev)
            for key, members in sorted(by_key.items()):
                _, _, gkt, gnt, gk2t, gn2t = members[0]
                e_n = len(members)
                grp = Group(*_bits_frac(key[0]), key[2])
                grp.n_experts = e_n
                grp.trellis13 = torch.empty((e_n, gkt, 2 * gnt, key[0]),
                                            dtype = torch.int16, device = dev)
                grp.suh13 = torch.empty((e_n, 2, 16 * gkt), dtype = torch.half, device = dev)
                grp.svh13 = torch.empty((e_n, 32 * gnt), dtype = torch.half, device = dev)
                grp.trellis2 = torch.empty((e_n, gk2t, gn2t, key[1]),
                                           dtype = torch.int16, device = dev)
                grp.suh2 = torch.empty((e_n, 1, 16 * gk2t), dtype = torch.half, device = dev)
                grp.svh2 = torch.empty((e_n, 16 * gn2t), dtype = torch.half, device = dev)
                for i, (src, expert, skt, snt, sk2t, sn2t) in enumerate(members):
                    if (skt, snt, sk2t, sn2t) != (gkt, gnt, gk2t, gn2t):
                        raise ValueError(f"expert {src} (checkpoint {expert}): tile geometry "
                                         f"differs inside one bitrate group")
                    g, u, d = _pack_names(layer, expert)
                    shard.read_into(g + ".trellis", grp.trellis13[i, :, :gnt], stage)
                    shard.read_into(u + ".trellis", grp.trellis13[i, :, gnt:], stage)
                    shard.read_into(g + ".suh", grp.suh13[i, 0], stage)
                    shard.read_into(u + ".suh", grp.suh13[i, 1], stage)
                    shard.read_into(g + ".svh", grp.svh13[i, :16 * gnt], stage)
                    shard.read_into(u + ".svh", grp.svh13[i, 16 * gnt:], stage)
                    shard.read_into(d + ".trellis", grp.trellis2[i], stage)
                    shard.read_into(d + ".suh", grp.suh2[i, 0], stage)
                    shard.read_into(d + ".svh", grp.svh2[i], stage)
                grp.source = [m[0] for m in members]
                self.groups.append(grp)
            for grp in self.groups:
                remap = torch.full((self.num_experts,), -1, dtype=torch.int64, device = dev)
                remap[torch.tensor(grp.source, dtype = torch.int64, device = dev)] = torch.arange(
                    len(grp.source), dtype = torch.int64, device = dev)
                grp.remap = remap
        finally:
            shard.close()
        return self

    # -- forward --------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x half (M, H), ids int64 (M, T) this layer's cold experts, w half (M, T) -> fp32 (M, H).

        Each group does its own alignment over the picks it owns and its own gemm/combine. The
        down gemm adds routing-weighted rows into `acc` (the epilogue's atomics), so the groups
        land on one fp32 (M, H) buffer in sequence and nothing sums afterwards; a build without
        that entry point falls back to three fp16 outputs summed here.
        """
        ids = ids.to(self.device)
        M, T = ids.shape
        m_valid = M * T
        wf = w.float().contiguous()
        # One layer-wide accumulator, zeroed by the first group that runs into it (accumulate=False)
        # and added to by the rest. Allocated before the first restack so a missing 168 MiB shows
        # up as a normal OOM here rather than inside a gemm.
        acc = (torch.empty((M, self.hidden), dtype=torch.float32, device=x.device)
               if HAS_ACC_OUT and not DETERMINISTIC else None)
        total = None
        wrote = False
        # Optional per-phase attribution (CUDA_EXL3_MOE_TRACE=1):
        # each mark is an event on the caller's stream, so consecutive marks bound one op.
        trace = [] if TRACE else None

        def mark(label, blk=0, nrows=0):
            if trace is not None:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                trace.append((label, blk, nrows, ev))

        for grp in self.groups:
            if grp.n_experts == 0:
                continue
            mark(f"align:{grp.bits}{'f' if grp.frac else ''}")
            loc = grp.remap[ids]
            live = int((loc >= 0).sum())
            if live == 0:
                continue
            block_m = pick_block_m(live, grp.n_experts)
            sid, eids, n_rows, block_m = align_block(loc, block_m, grp.n_experts, m_valid)
            if eids is None:
                continue
            rows = int(eids.numel()) * block_m
            n_rows = torch.tensor([n_rows], dtype=torch.int32, device=self.device)
            tag = f"{grp.bits}{'f' if grp.frac else ''}"
            mark(f"had_in:{tag}", block_m, rows)
            # skip_padding leaves the padded rows of a13 unwritten, so both gemms have to be
            # told about sorted_ids or they would fetch them; with it, a padded row is never
            # staged and never written back, which is where a decode-shaped call saves most.
            a13 = torch.empty((2, rows, self.hidden), dtype=torch.half, device=x.device)
            _C.exl3_moe_had_in(x, a13, grp.suh13, sid, eids, n_rows, block_m, T, m_valid, True)
            mark(f"gemm13:{tag}", block_m, rows)
            inter = _C.exl3_moe_gemm(a13, grp.trellis13, grp.suh13, grp.svh13, eids, n_rows,
                                     [self.inter, self.inter], grp.cb, block_m, torch.half,
                                     sid, None, M, T)
            del a13
            mark(f"glu:{tag}", block_m, rows)
            a2 = torch.empty((1, rows, self.inter), dtype=torch.half, device=x.device)
            _C.exl3_moe_glu_had_in(inter, a2, grp.suh2, eids, n_rows, block_m)
            del inter
            mark(f"gemm2:{tag}", block_m, rows)
            if DETERMINISTIC:
                routed = _C.exl3_moe_gemm(a2, grp.trellis2, grp.suh2, grp.svh2, eids, n_rows,
                                          [self.hidden], grp.cb, block_m, torch.half,
                                          sid, None, M, T)
                mark(f"combine:{tag}", block_m, rows)
                out = _C.exl3_moe_combine(routed, sid, wf, M, eids, block_m)
                del routed
                mark(f"sum:{tag}", block_m, rows)
                total = out.float() if total is None else total + out.float()
                wrote = True
            elif acc is not None:
                # Straight into the layer's accumulator: the epilogue adds each routed row into
                # its token's row, which is the same sum this used to do in a separate pass.
                out = _C.exl3_moe_gemm(a2, grp.trellis2, grp.suh2, grp.svh2, eids, n_rows,
                                       [self.hidden], grp.cb, block_m, torch.float,
                                       sid, wf, M, T, acc, wrote)
                total = out
                wrote = True
            else:
                out = _C.exl3_moe_gemm(a2, grp.trellis2, grp.suh2, grp.svh2, eids, n_rows,
                                       [self.hidden], grp.cb, block_m, torch.half,
                                       sid, wf, M, T)
                mark(f"sum:{tag}", block_m, rows)
                total = out.float() if total is None else total + out.float()
                wrote = True
            del out, a2
        mark("end")
        if trace is not None:
            self.trace = trace
        if total is None:
            return torch.zeros((M, self.hidden), dtype=torch.float32, device=x.device)
        return total


# -- reading the cold pack shard -------------------------------------------------------------
_ST_DTYPE = {
    "F16": torch.half, "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64,
    "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def _pack_names(layer: int, expert: int) -> tuple[str, str, str]:
    """(gate, up, down) tensor-name prefixes of one cold expert in a `cold-LNN.safetensors` shard.

    exllamav3's own mapping (`integration/transformers.py: EXPERT_PROJ_NAMES`): w1 gate, w3 up,
    w2 down. The stacked layout wants gate first and up second along N, exactly as the pack
    stores them.
    """
    base = f"layers.{int(layer)}.ffn.experts.{int(expert)}"
    return base + ".w1", base + ".w3", base + ".w2"


class _Stage:
    """One reusable pinned host buffer for the shard reads (pageable if the target is the CPU)."""

    def __init__(self, device):
        self.device = device
        self.buf = None

    def get(self, nbytes: int) -> torch.Tensor:
        if self.buf is None or self.buf.numel() < nbytes:
            self.buf = torch.empty(nbytes, dtype = torch.uint8,
                                   pin_memory = self.device.type == "cuda")
        return self.buf[:nbytes]


class _Shard:
    """Header-only view of one pack shard plus a pread-into-`dst` reader.

    Not a safetensors mmap: mmapping 113.65 GiB of shards would put the page cache back on the
    unified pool this design is trying to keep clear. Each tensor is one `preadv` into the
    staging buffer and one copy into its slice, and the pages it touched are handed back with
    `POSIX_FADV_DONTNEED` immediately, so a layer's read costs its bytes and not its bytes plus
    a cached copy of the shard.
    """

    def __init__(self, path: str, fadvise: bool = True):
        self.path = path
        self.fadvise = fadvise
        self.fd = os.open(path, os.O_RDONLY)
        try:
            head = os.pread(self.fd, 8, 0)
            if len(head) != 8:
                raise ValueError(f"{path}: not a safetensors shard (short header length)")
            n = struct.unpack("<Q", head)[0]
            hdr = json.loads(os.pread(self.fd, n, 8).decode("utf-8"))
            self.data = 8 + n
            self.info = {k: (v["dtype"], tuple(v["shape"]),
                             int(v["data_offsets"][0]), int(v["data_offsets"][1]))
                         for k, v in hdr.items() if k != "__metadata__"}
        except BaseException:  # noqa: BLE001 - a malformed shard must not leak a fd per layer
            os.close(self.fd)
            self.fd = -1
            raise

    def shape(self, key: str) -> tuple:
        try:
            return self.info[key][1]
        except KeyError:
            raise KeyError(f"{self.path}: no tensor {key!r}") from None

    def codebook(self, prefix: str) -> int:
        """Which codebook this linear uses, from the flags the loader would set on it."""
        if prefix + ".mul1" in self.info:
            return CB_MUL1
        if prefix + ".mcg" in self.info:
            return CB_MCG
        return CB_3INST

    def read_into(self, key: str, dst: torch.Tensor, stage: _Stage) -> int:
        tag, shape, begin, end = self.info[key]
        n = end - begin
        buf = stage.get(n)
        got = os.preadv(self.fd, [buf.numpy()], self.data + begin)
        if got != n:
            raise IOError(f"{self.path}: short read for {key!r}: {got} of {n} bytes")
        # Blocking: the pinned buffer is reused for the next tensor as soon as this returns.
        dst.copy_(buf.view(_ST_DTYPE[tag]).view(shape))
        if self.fadvise:
            os.posix_fadvise(self.fd, self.data + begin, n, os.POSIX_FADV_DONTNEED)
        return n

    def close(self) -> None:
        if getattr(self, "fd", -1) >= 0:
            os.close(self.fd)
            self.fd = -1
