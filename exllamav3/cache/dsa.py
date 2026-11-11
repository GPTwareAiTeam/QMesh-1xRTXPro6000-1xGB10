from __future__ import annotations
from dataclasses import dataclass
import torch
from ..constants import PAGE_SIZE
from ..util.device_copy import to_device
from ..util import pad_bucket
from .cache import Cache, CacheLayer
from .recurrent import new_checkpoint_handle, mp_cache_recurrent_stash, mp_cache_recurrent_unstash, host_copy
"""
Cache state for DSA (DeepSeek-V4-style hybrid sparse attention) layers, split along the
framework's paged/recurrent line:

 - The compressed pool, rope pool and indexer-key pool are PAGED cache layers
   (CacheLayer_dsa): append-only, entry-indexed by position // m, immutable once written --
   semantically a KV cache growing at rate 1/m. Pool entries alias the token page table:
   token page i holds entries [i * epp, (i + 1) * epp), epp = PAGE_SIZE // m, addressed
   through the job's ordinary block table. Capacity therefore follows the Cache's
   max_num_tokens budget exactly (any mix of jobs fits iff total tokens fit), and prefix
   sharing, eviction, defragmentation and the CPU tier apply to pool pages for free.

 - Everything else is fixed-shape per-slot recurrent state (DSV4LayerState), with all
   compressor bookkeeping derived from absolute position (entry_count = position // m,
   buffer fill = position % m, overlap exists iff entry_count >= 1), so rewind is pure
   cursor arithmetic:

    - SWA ring: shifting linear buffer of raw roped K=V rows, page-aligned window_beg
      carried on the job state (identical mechanism to SWALayerState; window +
      overprovision slack gives guaranteed_rollback = PAGE_SIZE).
    - Compressor sub-window buffers: rings of the last (PAGE_SIZE + m) PROJECTED (kv, gate)
      rows, indexed by absolute token position % ring size. Rewind = cursor move; the rows
      for the new partial window are still present for any rollback <= PAGE_SIZE.
    - CSA overlap (previous window's Ca slice): ring of the last few window-boundary
      snapshots indexed by (entry number - 1) % depth, depth sized for PAGE_SIZE worth of
      windows.

Checkpoints (stash) carry only the recurrent residue; pool persistence across
pause/resume is governed by the same page-anchoring logic as K/V pages on hybrid
recurrent models.
"""

import os


def kv_grow() -> bool:
    """EXL3_KV_GROW=1: DSA pools reserve virtual memory for max_num_tokens and map physical pages on demand
    (notes/GrowKV.md). Read at call time so tests can toggle it."""
    return os.environ.get("EXL3_KV_GROW", "0") == "1"


class CacheLayer_dsa(CacheLayer):
    """
    Paged pool storage for one CSA/HCA DSAttention layer: pool_c (compressed KV, nope part),
    pool_r (rope part) and, for CSA, pool_idx (indexer keys), each shaped
    (num_pages, PAGE_SIZE // m, D) so that one token page holds the pool entries produced by
    exactly that page's tokens. Tensors are page-major, so the defragmenter's rotation and
    the CPU tier's per-page slabs apply unchanged.

    With k_bits > 0 the nope part is stored packed in the cache-quant format (32-value
    groups, H32-rotated domain, group scales; the layout CacheLayer_quant / _MLA_quant use,
    read online by the shared Triton loaders inside the DSA attention kernels): pool_q holds
    the int32 words and pool_s the fp16 group scales. The rope part (pool_r) and the indexer
    keys (pool_idx) stay fp16: they are small and feed every positional score / selection.
    Requires D_c to be a multiple of 32 (448 on DeepSeek-V4).

    With kv_format="fp4", complete 512-wide latent rows use packed E2M1 and one E4M3
    scale per 16 values. Indexer rows use packed E2M1 and one E8M0 scale per 32 values.
    This is a separate format from k_bits and includes the positional columns.
    """

    def __init__(
        self,
        config: Config | None,
        attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int = 0,
        v_bits: int | None = None,
        compand_a: float = 0.0,
        kv_format: str = "fp16",
        **kwargs
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)
        assert compand_a == 0.0, "compander is not supported by the online-dequant loaders"
        assert kv_format in ("fp16", "fp4"), "unknown DSA KV format"
        assert kv_format != "fp4" or not k_bits, "FP4 and k_bits cannot be combined"
        self.kv_format = kv_format
        self.k_bits = k_bits
        self.v_bits = v_bits
        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."
        m = attention.compress_rate
        assert m and PAGE_SIZE % m == 0, f"compress_rate {m} must divide {PAGE_SIZE}"
        self.compress_rate = m
        self.epp = PAGE_SIZE // m
        self.num_pages = max_num_tokens // PAGE_SIZE
        self.capacity = self.num_pages * self.epp
        # Pad-to-bucket decode (util/pad_bucket.py): one page past the page table's range takes the pool
        # and indexer writes of pad rows. The page table never hands it out (it owns [0, num_pages)), and
        # num_pages / capacity keep their meaning, so nothing a real row computes changes
        self.trash_page = self.num_pages if pad_bucket.enabled() else None
        self.alloc_pages = self.num_pages + (1 if self.trash_page is not None else 0)
        D = attention.head_dim
        D_r = attention.rope_head_dim
        self.D_c = D - D_r
        self.D_r = D_r
        self.D_i = attention.index_head_dim if getattr(attention, "indexer_mode", None) == "full" else 0
        if kv_format == "fp4":
            assert D == 512 and self.D_i in (0, 128), "FP4 DSA requires 512-wide latents and 128-wide index keys"
        if k_bits:
            assert 2 <= k_bits <= 8, "quantized DSA pool must be from 2 to 8 bits"
            assert self.D_c % 32 == 0, "quantized DSA pool requires head_dim - rope_dim to be a multiple of 32"
        self.G = self.D_c // 32
        self.pool_c = None
        self.pool_q = None
        self.pool_s = None
        self.pool_r = None
        self.pool_idx = None
        self.pool_fp4 = self.pool_fp4_scale = None
        self.pool_idx_fp4 = self.pool_idx_scale = None
        self.device = None
        self._slot_bt = None
        self.mirrors = {}

    @property
    def quant(self) -> bool:
        return self.k_bits > 0

    def alloc(self, device: torch.device):
        # Captured BC DSA graphs bind this pool's storage; drop them before (re)allocating it so a
        # replay can never read freed or replaced memory
        getattr(self.attention, "_bc_dsa", {}).clear()
        getattr(self.attention, "_bc_dsa_batch", {}).clear()
        device = torch.device(device)
        self.device = device
        self.mirrors = {}
        P = self.alloc_pages
        shapes = self._pool_shapes(P)
        if kv_grow():
            # Growable pools (notes/GrowKV.md): each pool tensor is a view of its own reserved virtual range sized for
            # all P pages; physical memory is mapped page-prefix-wise by ensure_pages, so only the pages the page
            # table has handed out cost VRAM. The base addresses never move (captured graphs stay valid)
            from .vmm import VmmRange
            assert self.trash_page is None, "EXL3_KV_GROW does not support pad-to-bucket decode (EXL3_PW_BUCKETS)"
            self.ranges = {}
            for name, (shape, dtype) in shapes.items():
                r = VmmRange(device, shape[0] * shape[1] * shape[2] * torch.empty((), dtype = dtype).element_size())
                self.ranges[name] = r
                setattr(self, name, r.view(dtype, shape))
            self.backed_pages = 0
            return
        self.ranges = None
        self.backed_pages = P
        for name, (shape, dtype) in shapes.items():
            setattr(self, name, torch.zeros(shape, dtype = dtype, device = device))

    def _pool_shapes(self, P: int) -> dict:
        """name -> ((P, epp, width), dtype) of every pool tensor this layer's format allocates"""
        s = {}
        if self.kv_format == "fp4":
            s["pool_fp4"] = ((P, self.epp, 256), torch.uint8)
            s["pool_fp4_scale"] = ((P, self.epp, 32), torch.uint8)
            if self.D_i:
                s["pool_idx_fp4"] = ((P, self.epp, 64), torch.uint8)
                s["pool_idx_scale"] = ((P, self.epp, 4), torch.uint8)
            return s
        if self.k_bits:
            s["pool_q"] = ((P, self.epp, self.G * self.k_bits), torch.int32)
            s["pool_s"] = ((P, self.epp, self.G), torch.half)
        else:
            s["pool_c"] = ((P, self.epp, self.D_c), torch.half)
        s["pool_r"] = ((P, self.epp, self.D_r), torch.half)
        if self.D_i:
            s["pool_idx"] = ((P, self.epp, self.D_i), torch.half)
        return s

    def page_bytes(self) -> int:
        """VRAM one page costs across this layer's pool tensors"""
        return sum(sh[1] * sh[2] * torch.empty((), dtype = dt).element_size()
                   for sh, dt in self._pool_shapes(1).values())

    def ensure_pages(self, n: int) -> int:
        """Grow mode: back pages [0, n) of every pool tensor (zero-filled); returns the VRAM bytes newly mapped"""
        if self.ranges is None or n <= self.backed_pages:
            return 0
        n = min(n, self.alloc_pages)
        got = 0
        for name, r in self.ranges.items():
            t = getattr(self, name)
            got += r.ensure(n * t.stride(0) * t.element_size())
        self.backed_pages = n
        return got

    def trim_pages(self, n: int) -> int:
        """Grow mode: release physical memory above page n (granule-rounded per tensor); the caller guarantees no
        kernel in flight reads pages >= n. Returns the VRAM bytes released."""
        if self.ranges is None or n >= self.backed_pages:
            return 0
        got = 0
        for name, r in self.ranges.items():
            t = getattr(self, name)
            got += r.trim(n * t.stride(0) * t.element_size())
        self.backed_pages = n
        return got

    def backed_bytes(self) -> int:
        return sum(r.mapped for r in self.ranges.values()) if self.ranges else self.storage_size()

    def free(self):
        getattr(self.attention, "_bc_dsa", {}).clear()
        getattr(self.attention, "_bc_dsa_batch", {}).clear()
        self.device = None
        self.pool_c = self.pool_q = self.pool_s = self.pool_r = self.pool_idx = None
        self.pool_fp4 = self.pool_fp4_scale = self.pool_idx_fp4 = self.pool_idx_scale = None
        for r in (getattr(self, "ranges", None) or {}).values():
            r.free()
        self.ranges = None
        self._slot_bt = None
        self.mirrors = {}

    def alloc_mirror(self, device: torch.device):
        """Second copy of the pools for the layers that read them on another device: paged like
        the pools (copy_page, get_tensors) and filled by pull()."""
        mirror = CacheLayer_dsa(
            self.config, self.attention, self.cache_id, self.max_num_tokens, self.k_bits, self.v_bits,
            kv_format = self.kv_format
        )
        mirror.alloc(device)
        self.mirrors[device] = mirror

    def pull(self, source: CacheLayer_dsa, pages, e0: int, e1: int):
        """Copy entries [e0, e1) of one sequence from source; pages is its block table row."""
        if e1 <= e0:
            return
        for p in range(e0 // self.epp, (e1 - 1) // self.epp + 1):
            i, a, b = int(pages[p]), max(e0 - p * self.epp, 0), min(e1 - p * self.epp, self.epp)
            for dst, src in zip(self.get_tensors(), source.get_tensors()):
                dst[i, a:b].copy_(to_device(src[i, a:b], self.device))

    def pool_c_view(self) -> torch.Tensor:
        """Flat pool: FP16 nope rows, H32 words, or packed complete FP4 latent rows."""
        if self.kv_format == "fp4":
            return self.pool_fp4.view(-1, 256)
        if self.k_bits:
            return self.pool_q.view(-1, self.G * self.k_bits)
        return self.pool_c.view(-1, self.D_c)

    def qc(self):
        """Return scales and format: positive bits for H32, -1 for FP4, or None."""
        if self.kv_format == "fp4":
            return self.pool_fp4_scale.view(-1, 32), -1
        if self.k_bits:
            return self.pool_s.view(-1, self.G), self.k_bits
        return None

    def get_kv(self, cache_seqlens, block_table, sliding_window = -1):
        return None, None

    def update_kv(self, cache_seqlens, block_table, k, v, length):
        pass

    def update_kv_direct(self, cache_seqlens, block_table, k, v, length):
        pass

    def copy_page(self, source: CacheLayer_dsa, from_page: int, to_page: int, num_tokens: int):
        ne = min(num_tokens // self.compress_rate, self.epp)
        if ne <= 0:
            return
        if self.kv_format == "fp4":
            assert source.kv_format == self.kv_format
            for dst, src in zip((self.pool_fp4, self.pool_fp4_scale, self.pool_idx_fp4, self.pool_idx_scale),
                                (source.pool_fp4, source.pool_fp4_scale, source.pool_idx_fp4, source.pool_idx_scale)):
                if dst is not None:
                    dst[to_page, :ne].copy_(src[from_page, :ne], non_blocking = True)
            for device, mirror in self.mirrors.items():
                mirror.copy_page(source.mirrors.get(device, source), from_page, to_page, num_tokens)
            return
        if self.k_bits:
            self.pool_q[to_page, :ne].copy_(source.pool_q[from_page, :ne], non_blocking = True)
            self.pool_s[to_page, :ne].copy_(source.pool_s[from_page, :ne], non_blocking = True)
        else:
            self.pool_c[to_page, :ne].copy_(source.pool_c[from_page, :ne], non_blocking = True)
        self.pool_r[to_page, :ne].copy_(source.pool_r[from_page, :ne], non_blocking = True)
        if self.pool_idx is not None:
            self.pool_idx[to_page, :ne].copy_(source.pool_idx[from_page, :ne], non_blocking = True)
        for device, mirror in self.mirrors.items():
            mirror.copy_page(source.mirrors.get(device, source), from_page, to_page, num_tokens)

    def get_tensors(self):
        return [t for t in [self.pool_c, self.pool_q, self.pool_s, self.pool_r, self.pool_idx,
                            self.pool_fp4, self.pool_fp4_scale, self.pool_idx_fp4, self.pool_idx_scale] if t is not None] + \
            [t for mirror in self.mirrors.values() for t in mirror.get_tensors()]

    def storage_size(self):
        rows = self.alloc_pages * self.epp
        if self.kv_format == "fp4":
            return rows * (288 + (68 if self.D_i else 0))
        if self.k_bits:
            return rows * (self.G * self.k_bits * torch.int32.itemsize +
                           (self.G + self.D_r + self.D_i) * torch.half.itemsize)
        return rows * (self.D_c + self.D_r + self.D_i) * torch.half.itemsize

    def overhead_size(self):
        return 0

    def slot_bt(self, num_slots: int) -> torch.Tensor:
        """
        Slot-partitioned identity block table for direct-forward use without a page table
        (tests, benchmarks): slot s owns pages [s * pps, (s + 1) * pps),
        pps = num_pages // num_slots, reproducing isolated per-slot pool semantics.
        """
        if self._slot_bt is None or self._slot_bt.shape[0] != num_slots:
            pps = self.num_pages // num_slots
            assert pps > 0, "CacheLayer_dsa: cache too small for slot-partitioned fallback"
            self._slot_bt = torch.arange(num_slots * pps, dtype = torch.int32,
                                         device = self.device).view(num_slots, pps)
        return self._slot_bt

    def tp_export(self, plan):
        # Pools are shared-KV (MQA) state: replicated per rank, shapes derived from the
        # (replicated) compressor fields of the shard module passed as `attention` on import
        return {
            "cls": CacheLayer_dsa,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
                "kv_format": self.kv_format,
            }
        }


@dataclass
class DSV4ExportedState:
    cache: int
    slot: int
    position: int
    window_beg: int
    serial: int
    wshift: int = 0
    last_history: int = 0


class DSV4State:
    """
    Job-level state (recurrent_state_cls): position bookkeeping shared by every DSA layer
    of the model. Mirrors SWAState's contract; all tensor work lives in the layer states.
    """

    exported = False
    guaranteed_rollback = PAGE_SIZE

    _serial = 0    # distinguishes successive jobs on the same slot (BC block-table refresh)

    def __init__(
        self,
        cache: Cache,
        slot: int,
        position: int,
        clear: bool = True,
        stashed: dict | None = None,
        test_state: bool = False,
    ):
        assert test_state or position == 0 or stashed is not None
        self.cache = cache
        self.slot = slot
        self.serial = DSV4State._serial
        DSV4State._serial += 1
        self.position = position
        self.last_history = 0
        self.window_beg = position // PAGE_SIZE * PAGE_SIZE
        self.wshift = 0
        if stashed is not None:
            self.unstash(stashed)
        self.checkpoint_size = sum(
            l.get_checkpoint_size() for l in cache.get_all_recurrent_layers().values()
        )
        if clear and stashed is None:
            for l in cache.get_all_recurrent_layers().values():
                l.clear(slot)

    def free(self):
        self.cache.release_state(self)

    def rewind(self, num_tokens: int):
        assert num_tokens <= self.rollback_capacity(), \
            f"DSV4State: rewind {num_tokens} exceeds capacity {self.rollback_capacity()}"
        self.position -= num_tokens
        self.last_history = 0

    def rollback_capacity(self):
        return self.position - self.window_beg

    def post_advance(self):
        self.window_beg += self.wshift
        self.wshift = 0

    def stash(self):
        stashed = {
            "position": self.position,
            "window_beg": self.window_beg,
            "checkpoint_size": self.checkpoint_size,
        }
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                stashed[k] = l.stash(self.slot, self.position)
        else:
            cp_handle = new_checkpoint_handle()
            self.cache.model.tp_dispatch_all(
                mp_cache_recurrent_stash, (id(self.cache), cp_handle, self.slot, self.position))
            stashed["tp_handle"] = cp_handle
        return stashed

    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        self.window_beg = stashed["window_beg"]
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                l.unstash(self.slot, stashed[k], self.position)
        else:
            cp_handle = stashed["tp_handle"]
            self.cache.model.tp_dispatch_all(
                mp_cache_recurrent_unstash, (id(self.cache), cp_handle, self.slot, self.position))

    def tp_export(self):
        return DSV4ExportedState(
            cache = id(self.cache),
            slot = self.slot,
            position = self.position,
            window_beg = self.window_beg,
            serial = self.serial,
        )

    def tp_readback(self, exported: DSV4ExportedState):
        # The forward decides the per-step ring shift; post_advance() applies it parent-side
        self.wshift = exported.wshift

    def reset(self):
        self.position = 0
        self.window_beg = 0
        self.wshift = 0
        self.last_history = 0


class DSV4LayerState:
    """Per-layer, per-cache recurrent state tensors for one DSAttention module. Allocated on
    meta at construction, materialized by the module's load. Component set depends on layer
    type: every layer has the SWA ring; CSA/HCA add compressor sub-window rings; CSA adds
    the indexer rings and overlap snapshots. The pools live in the paged CacheLayer_dsa."""

    def __init__(self, module, max_batch_size: int, max_history: int, cache_id: int):
        self.module = module
        self.cache_id = cache_id
        self.max_batch_size = max_batch_size
        self.max_history = max_history
        self.device = None

        D = module.head_dim
        D_r = module.rope_head_dim
        self.layer_type = module.layer_type
        self.window = module.sliding_window
        overp = 2 * PAGE_SIZE
        self.ring_rows = -(-(self.window + overp) // PAGE_SIZE) * PAGE_SIZE

        mk = lambda *shape, dtype = torch.half: torch.zeros(shape, dtype = dtype, device = "meta")
        B = max_batch_size
        self.D_c = D - D_r
        self.D_r = D_r
        self.ring = mk(B, self.ring_rows, D)

        self.comp_buf_kv = self.comp_buf_gate = None
        self.idx_buf_kv = self.idx_buf_gate = None
        self.comp_ovl = self.idx_ovl = None

        if module.compressor is not None:
            m = module.compress_rate
            w = module.compressor.wkv.out_features_unpadded
            self.buf_rows = PAGE_SIZE + m
            self.comp_buf_kv = mk(B, self.buf_rows, w)
            self.comp_buf_gate = mk(B, self.buf_rows, w)
            if self.layer_type == "csa":
                # overlap snapshots: one per emitted window; PAGE_SIZE//m + slack of them.
                # fp32: the saved gate slice carries the fp32 position bias
                self.ovl_depth = PAGE_SIZE // m + 2
                self.comp_ovl = mk(B, self.ovl_depth, 2, m, D, dtype = torch.float)
                D_i = module.index_head_dim
                wi = module.indexer.wkv.out_features_unpadded
                self.idx_buf_kv = mk(B, self.buf_rows, wi)
                self.idx_buf_gate = mk(B, self.buf_rows, wi)
                self.idx_ovl = mk(B, self.ovl_depth, 2, m, D_i, dtype = torch.float)


    def _tensors(self):
        return [t for t in [
            self.ring,
            self.comp_buf_kv, self.comp_buf_gate, self.idx_buf_kv, self.idx_buf_gate,
            self.comp_ovl, self.idx_ovl,
        ] if t is not None]


    def _set_tensors(self, ts):
        names = ["ring", "comp_buf_kv", "comp_buf_gate",
                 "idx_buf_kv", "idx_buf_gate", "comp_ovl", "idx_ovl"]
        it = iter(ts)
        for n in names:
            if getattr(self, n) is not None:
                setattr(self, n, next(it))


    def get_checkpoint_size(self):
        # Ring slice + buffers + overlaps (upper bound; stash() copies up to ring_rows rows of
        # the ring, fewer at low positions). Pools are paged cache state and not part of checkpoints
        return sum(t[0].numel() * t.element_size() for t in self._tensors())


    def storage_size(self):
        return sum(t.numel() * t.element_size() for t in self._tensors())


    def alloc(self, device):
        getattr(self.module, "_bc_dsa", {}).clear()
        getattr(self.module, "_bc_dsa_batch", {}).clear()
        self.device = device
        self._set_tensors([torch.zeros_like(t, device = device) for t in self._tensors()])


    def free(self):
        getattr(self.module, "_bc_dsa", {}).clear()
        getattr(self.module, "_bc_dsa_batch", {}).clear()
        self.device = None
        self._set_tensors([torch.zeros_like(t, device = "meta") for t in self._tensors()])


    def clear(self, idx):
        for t in self._tensors():
            t[idx].zero_()


    def rewind(self, slot, last_history, num_tokens):
        pass  # all bookkeeping is position-derived; stale rows are overwritten on re-advance


    def stash(self, slot, position):
        out = [host_copy(self.ring[slot, :min(self.ring_rows, position)])]
        if self.comp_buf_kv is not None:
            out.append(host_copy(self.comp_buf_kv[slot]))
            out.append(host_copy(self.comp_buf_gate[slot]))
            if self.idx_buf_kv is not None:
                out.append(host_copy(self.idx_buf_kv[slot]))
                out.append(host_copy(self.idx_buf_gate[slot]))
                out.append(host_copy(self.comp_ovl[slot]))
                out.append(self.idx_ovl[slot].cpu())
        return out


    def unstash(self, slot, stashed, position):
        it = iter(stashed)
        ring = next(it)
        self.ring[slot].zero_()  # never leave stale rows below the restored window
        self.ring[slot, :ring.shape[0]].copy_(ring)
        if self.comp_buf_kv is not None:
            self.comp_buf_kv[slot].copy_(next(it))
            self.comp_buf_gate[slot].copy_(next(it))
            if self.idx_buf_kv is not None:
                self.idx_buf_kv[slot].copy_(next(it))
                self.idx_buf_gate[slot].copy_(next(it))
                self.comp_ovl[slot].copy_(next(it))
                self.idx_ovl[slot].copy_(next(it))


    def tp_export(self, plan):
        return {
            "cls": DSV4LayerState,
            "args": {
                "cache_id": self.cache_id,
                "max_history": self.max_history,
                "max_batch_size": self.max_batch_size,
            },
        }


class DSparkLayerState:
    """
    Window ring for one DSpark drafter block's main-kv rows: one (head_dim,) fp16 row per TARGET position,
    derived from the trunk's tap states by update_kv_from_target. The drafter only reads the trailing
    sliding_window rows below the block it drafts, so each state slot keeps ring_rows rows (window plus two
    pages of rollback room) instead of a context-sized paged plane.

    Addressing is modular: position p lives at ring row p % ring_rows of its slot. ring_rows is a whole number
    of pages, so token page k of a sequence always maps to ring page k % pages_per_slot of that slot, and a
    slot's block table (DSparkState.block_table) is constant. The drafter's paged kernels (dspark_write_rows,
    dsa_attn NC_BLOCK) therefore read and write the ring unchanged, and a captured graph's block table only
    changes when a different slot takes the row. One extra page past the slots (trash_page) takes the rows of a
    batched write that belong to no sequence.
    """

    def __init__(self, module, max_batch_size: int, max_history: int, cache_id: int):
        self.module = module
        self.cache_id = cache_id
        self.max_batch_size = max_batch_size
        self.max_history = max_history
        self.ring_rows = module.dspark_ring_rows
        assert self.ring_rows % PAGE_SIZE == 0
        self.pages_per_slot = self.ring_rows // PAGE_SIZE
        self.width = module.head_dim
        self.trash_page = max_batch_size * self.pages_per_slot
        self.ring = torch.empty((self.trash_page + 1, PAGE_SIZE, self.width),
                                dtype = torch.half, device = "meta")
        self.device = None

    def get_checkpoint_size(self):
        return self.ring_rows * self.width * torch.half.itemsize

    def storage_size(self):
        return self.ring.numel() * self.ring.element_size()

    def alloc(self, device):
        self.device = device
        self.ring = torch.zeros_like(self.ring, device = device)

    def free(self):
        self.device = None
        self.ring = torch.empty_like(self.ring, device = "meta")

    def clear(self, idx: int):
        p = self.pages_per_slot
        self.ring[idx * p : (idx + 1) * p].zero_()

    def rewind(self, slot: int, last_history: int, num_tokens: int):
        pass   # position bookkeeping only (DSparkState)

    def _rows(self, slot: int, lo: int, hi: int) -> torch.Tensor:
        """Flat ring row indices of positions [lo, hi) of a slot (hi - lo <= ring_rows, so all distinct)"""
        pos = torch.arange(lo, hi, dtype = torch.long, device = self.device)
        return slot * self.ring_rows + pos % self.ring_rows

    def stash(self, slot: int, lo: int, position: int):
        return host_copy(self.ring.view(-1, self.width)[self._rows(slot, lo, position)])

    def unstash(self, slot: int, stashed: torch.Tensor, position: int):
        n = stashed.shape[0]
        assert n <= self.ring_rows and stashed.shape[1] == self.width, \
            "DSpark ring checkpoint shape does not match the destination slot"
        self.ring.view(-1, self.width)[self._rows(slot, position - n, position)] = \
            stashed.to(self.device, non_blocking = True)

    def tp_export(self, plan):
        raise NotImplementedError("Tensor-parallel loading is not supported for DSpark layers")


class DSparkState:
    """
    Job-level state of a DSpark drafter (its cache's recurrent_state_cls), the draft side of a PairedState.
    position is the next target position whose main-kv row the drafter has not received. Rows of positions
    [valid_from(), position) are intact in every layer's ring: writes land at position % ring_rows, so a write
    up to `written` has overwritten everything below written - ring_rows, and a checkpoint restore only brings
    back the rows it stored (from `floor`). The drafter reads the sliding_window rows below position, so a
    rewind must leave that many intact rows (rollback_capacity).
    """

    exported = False
    guaranteed_rollback = PAGE_SIZE

    def __init__(
        self,
        cache: Cache,
        slot: int,
        position: int,
        clear: bool = True,
        stashed: dict | None = None,
        test_state: bool = False,
    ):
        assert test_state or position == 0 or stashed is not None
        self.cache = cache
        self.slot = slot
        self.position = position
        self.last_history = 0
        layers = cache.get_all_recurrent_layers()
        first = next(iter(layers.values()))
        self.ring_rows = first.ring_rows
        self.pages_per_slot = first.pages_per_slot
        self.trash_page = first.trash_page
        self.window = first.module.sliding_window
        self.floor = 0
        self.written = position
        if stashed is not None:
            self.unstash(stashed)
        self.checkpoint_size = sum(l.get_checkpoint_size() for l in layers.values())
        if clear and stashed is None:
            for l in layers.values():
                l.clear(slot)

    def free(self):
        self.cache.release_state(self)

    def block_table(self, width: int) -> torch.Tensor:
        """(1, width) int32 host block table into this slot's ring: token page k -> ring page
        slot * pages_per_slot + k % pages_per_slot"""
        p = self.pages_per_slot
        return (self.slot * p + torch.arange(width, dtype = torch.int32) % p).unsqueeze(0)

    def trash_table(self, width: int) -> torch.Tensor:
        """(1, width) int32 host block table whose every page is the layers' trash page: rows written through it
        land in no slot"""
        return torch.full((1, width), self.trash_page, dtype = torch.int32)

    def valid_from(self) -> int:
        return max(self.floor, self.written - self.ring_rows)

    def rollback_capacity(self) -> int:
        lo = self.valid_from()
        if lo == 0:
            return self.position
        return max(0, self.position - lo - self.window)

    def rewind(self, num_tokens: int):
        assert num_tokens <= self.rollback_capacity(), \
            f"DSparkState: rewind {num_tokens} exceeds capacity {self.rollback_capacity()}"
        self.position -= num_tokens
        self.last_history = 0

    def post_advance(self):
        pass

    def wrote(self, pos0: int, accepted: int, rows_written: int):
        """Rows were written at [pos0, pos0 + rows_written); the first `accepted` of them are this sequence's
        (the rest are a batch's padding, overwritten before they are read)"""
        assert self.position == pos0, \
            f"DSparkState: rows written at {pos0}, state at {self.position}"
        self.written = max(self.written, pos0 + rows_written)
        self.position = pos0 + accepted
        self.last_history = 0

    def stash(self):
        lo = max(self.valid_from(), self.position - self.ring_rows)
        stashed = {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size,
        }
        for k, l in self.cache.get_all_recurrent_layers().items():
            stashed[k] = l.stash(self.slot, lo, self.position)
        return stashed

    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        n = None
        for k, l in self.cache.get_all_recurrent_layers().items():
            t = stashed[k]
            assert n is None or t.shape[0] == n, "DSpark ring checkpoint layers disagree on their row count"
            n = t.shape[0]
            l.unstash(self.slot, t, self.position)
        self.floor = self.position - n
        self.written = self.position

    def reset(self):
        self.position = 0
        self.last_history = 0
        self.floor = 0
        self.written = 0
