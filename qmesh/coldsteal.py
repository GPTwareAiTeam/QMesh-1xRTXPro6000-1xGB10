"""ColdSteal host half: P7-resident steal slots for GX-bound (cold) experts, folded into the routed launch.

Device side: `exllamav3_ext/quant/exl3_coldsteal.{cuh,cu}` (protocol in the header) and the ColdSteal tables of
`BC_BlockSparseMLP`. Per armed layer this module owns

  * S steal slots in P7 VRAM, appended to the layer's coop pointer / rate tables as local ids n_hot .. n_hot + S - 1;
  * `slot_of`, the device map expert -> slot (-1 = not resident) the select reads and the proxy writes;
  * a pinned host copy of every cold expert of the layer (the pool), laid out like a slot;
  * a host-mapped block (`seen` + the latest call's hit/miss record) the select writes and the proxy reads.

Per MoE call, `select()` runs one kernel between routing and the GX request: resident cold picks (at most s_max
experts) become slot picks of the P7 launch and are blanked out of the worker selection; the rest stay with the
worker. Nothing waits: slots are filled and evicted by the native proxy on its own stream, and an expert becomes
stealable only after its bytes landed.

Env (read at load):
  QMESH_COLDSTEAL=1                arm it (off: nothing here is instantiated, the release path is unchanged)
  QMESH_COLDSTEAL_SRC=<dir>        `cold-LNN.safetensors` shards (the GX cold pack, plain trellis layout)
  QMESH_COLDSTEAL_SLOTS=8          steal slots per layer (13.3 MB each at 3 bit)
  QMESH_COLDSTEAL_SMAX=3           most experts stolen per layer per call
  QMESH_COLDSTEAL_MAX_OUT=4        admissions in flight (one H2D link)
  QMESH_COLDSTEAL_MAX_ROWS=24      wider calls (prefill) are left entirely to the worker
"""
from __future__ import annotations

import atexit
import os
import re
import threading
import time

import torch

# Region order inside a slot / pool row: scales first, then the three trellises at a fixed stride
SCALES = (("gate", ".suh"), ("gate", ".svh"), ("up", ".suh"), ("up", ".svh"), ("down", ".suh"), ("down", ".svh"))
TRELLIS = ("gate", "up", "down")
K2_OK = (4, 6, 8)           # 2 / 3 / 4 bit rows (mixed-K kernel rates with a plain-layout instance)
WORDS = 32                  # CS_LAYER_WORDS


def enabled() -> bool:
    return os.environ.get("QMESH_COLDSTEAL", "0") == "1"


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _ext():
    from exllamav3.ext import exllamav3_ext as ext
    return ext


def _align(n, a = 256):
    return (n + a - 1) // a * a


class _Shard:
    """Header view of one `cold-LNN.safetensors` plus pread into pinned memory."""

    _DT = {"I16": 2, "F16": 2, "I32": 4, "BF16": 2, "F32": 4}

    def __init__(self, path):
        import json, struct
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        n = struct.unpack("<Q", os.pread(self.fd, 8, 0))[0]
        hdr = json.loads(os.pread(self.fd, n, 8))
        self.base = 8 + n
        self.info = {k: (v["dtype"], tuple(v["shape"]), v["data_offsets"][0], v["data_offsets"][1])
                     for k, v in hdr.items() if k != "__metadata__"}

    def nbytes(self, key):
        _, _, b, e = self.info[key]
        return e - b

    def read_into(self, key, view):
        _, _, b, e = self.info[key]
        n = e - b
        assert view.numel() == n, (key, view.numel(), n)
        got = os.preadv(self.fd, [view.numpy()], self.base + b)
        if got != n:
            raise IOError(f"{self.path}: short read of {key}: {got}/{n}")

    def close(self):
        os.close(self.fd)


class LayerSteal:
    """One layer's slots, tables, pool and select state."""

    def __init__(self, m, shard, layer):
        ext = _ext()
        dev = m.device
        self.key = m.key
        self.n_hot = n_hot = m.cpu_split_first
        self.E = E = m.num_experts
        self.S = S = _env_int("QMESH_COLDSTEAL_SLOTS", 8)
        self.s_max = _env_int("QMESH_COLDSTEAL_SMAX", 3)
        self.max_rows = _env_int("QMESH_COLDSTEAL_MAX_ROWS", 24)
        perm = m._split_perm or list(range(E))
        H, I = m.multi_up.in_features, m.multi_up.out_features
        assert m.multi_down.in_features == I and m.multi_down.out_features == H
        # Trellis bytes of one projection at k2 half-bits: (in/16) * (out/16) tiles of 16 * K int16 = in * out * k2 / 16
        self.tunit = H * I // 16
        names = {}
        rates = {}
        for e_phys in range(n_hot, E):
            base = f"layers.{layer}.ffn.experts.{perm[e_phys]}"
            k = names[e_phys] = {"gate": base + ".w1", "up": base + ".w3", "down": base + ".w2"}
            if all(k[p] + ".trellis" in shard.info for p in TRELLIS):
                tb = shard.nbytes(k["up"] + ".trellis")
                k2 = tb // self.tunit
                if k2 * self.tunit == tb and k2 in K2_OK and all(shard.nbytes(k[p] + ".trellis") == tb for p in TRELLIS):
                    rates[e_phys] = k2
        if not rates:
            raise ValueError(f"no readable cold expert in {shard.path}")
        tstride = _align(max(rates.values()) * self.tunit)    # the layer's widest cold rate
        scale_sizes = []
        probe = names[next(iter(rates))]
        for proj, suf in SCALES:
            scale_sizes.append(shard.nbytes(probe[proj] + suf))
        scale_bytes = _align(sum(scale_sizes))
        self.stride = stride = scale_bytes + 3 * tstride
        self.scale_bytes, self.tstride = scale_bytes, tstride
        n_cold = E - n_hot

        # Pool: every cold expert of the layer, pinned, slot layout
        # Mapped (device-addressable) when the split policy may compute experts straight from it (zero-copy)
        zc = policy_enabled() and _env_int("QMESH_SPLIT_ZCAP", 0) > 0
        self.pool_ptr = ext.coldsteal_host_alloc(n_cold * stride, zc)
        pool = _pinned_view(self.pool_ptr, n_cold * stride)
        self.pool_k2_ptr = ext.coldsteal_host_alloc(max(n_cold, 64), True)   # mapped: the policy kernel reads it
        pool_k2 = _pinned_view(self.pool_k2_ptr, n_cold)
        pool_k2.zero_()
        for r in range(n_cold):
            k2 = rates.get(n_hot + r)
            if k2 is None:
                continue
            k = names[n_hot + r]
            tb = k2 * self.tunit
            row = pool[r * stride:(r + 1) * stride]
            off = 0
            for (proj, suf), n in zip(SCALES, scale_sizes):
                shard.read_into(k[proj] + suf, row[off:off + n])
                off += n
            for j, proj in enumerate(TRELLIS):
                o = scale_bytes + j * tstride
                shard.read_into(k[proj] + ".trellis", row[o:o + tb])
            pool_k2[r] = k2
        self.pool = pool
        self.pool_k2 = pool_k2
        self.loaded = int((pool_k2 > 0).sum())
        # Device address of the pinned pool (UVA: the host address; cudaHostAllocMapped makes it device-readable)
        self.pool_dev = self.pool_ptr if zc else 0

        # Slots and the folded tables
        # Slots on a reserved virtual range when the KV cache grows (EXL3_KV_GROW, notes/GrowKV.md): retiring the top
        # slots unmaps their memory for the KV pools; the base (baked into the folded tables) never moves
        from exllamav3.cache.dsa import kv_grow
        if kv_grow():
            from exllamav3.cache.vmm import VmmRange
            self.vmm = VmmRange(dev, S * stride)
            self.vmm.ensure(S * stride)
            self.slots = self.vmm.view(torch.uint8, (S * stride,))
        else:
            self.vmm = None
            self.slots = torch.empty(S * stride, dtype = torch.uint8, device = dev)
        assert self.slots.data_ptr() % 256 == 0
        sp = self.slots.data_ptr()
        mg = m.multi_gate
        hot = (mg.ptrs_trellis, mg.ptrs_suh, mg.ptrs_svh,
               m.multi_up.ptrs_trellis, m.multi_up.ptrs_suh, m.multi_up.ptrs_svh,
               m.multi_down.ptrs_trellis, m.multi_down.ptrs_suh, m.multi_down.ptrs_svh)
        offs = [0]
        for n in scale_sizes[:-1]:
            offs.append(offs[-1] + n)
        # row order of the tables: (g_t, g_suh, g_svh, u_t, u_suh, u_svh, d_t, d_suh, d_svh)
        slot_off = [scale_bytes + 0 * tstride, offs[0], offs[1],
                    scale_bytes + 1 * tstride, offs[2], offs[3],
                    scale_bytes + 2 * tstride, offs[4], offs[5]]
        # Zero-copy entries (QMESH_SPLIT_ZCAP, split policy): Z more table columns after the S slots, pointed per call by
        # the policy kernel at pool rows (the P7 then reads those experts over PCIe from pinned RAM)
        Z = self.Z = _env_int("QMESH_SPLIT_ZCAP", 0) if policy_enabled() else 0
        self.zc_off = torch.tensor(slot_off, dtype = torch.long, device = dev)
        tables = torch.empty((9, n_hot + S + Z), dtype = torch.long)
        for i in range(9):
            tables[i, :n_hot] = hot[i][:n_hot].cpu()
            for s in range(S):
                tables[i, n_hot + s] = sp + s * stride + slot_off[i]
            for z in range(Z):          # until a call points them: the first slot's row (never selected before then)
                tables[i, n_hot + S + z] = sp + slot_off[i]
        self.tables = tables.to(dev)
        k2 = torch.full((3, n_hot + S + Z), 6, dtype = torch.uint8)
        for i, multi in enumerate((mg, m.multi_up, m.multi_down)):
            k2[i, :n_hot] = (multi.k2[:n_hot].cpu() if multi.k2 is not None
                             else torch.full((n_hot,), int(round(2 * multi.K)), dtype = torch.uint8))
        self.k2 = k2.to(dev)
        self.slot_of = torch.full((E,), -1, dtype = torch.int32, device = dev)
        self.dev_seq = torch.zeros(1, dtype = torch.long, device = dev)
        self.stats = torch.zeros(4, dtype = torch.long, device = dev)
        self.layer = layer
        self.cur = torch.zeros(1, dtype = torch.long, device = dev)       # policy log: ring index of this layer's call
        self.words = ext.coldsteal_host_alloc(WORDS * 8, True)
        self.armed = False
        self.bc = None
        self.manager = None

    def kwargs(self):
        return dict(cs_tables = self.tables, cs_k2 = self.k2)

    def attach(self, bc):
        self.bc = bc
        self.armed = bool(getattr(bc, "cs", False))
        if not self.armed:
            print(f" !! ColdSteal {self.key}: the block class refused the steal tables; stealing disabled", flush = True)
        return self.armed

    def select(self, sel, sel_cpu):
        """Rewrite `sel` in place for the P7 launch (stolen picks -> slot ids, other cold picks inactive) and blank
        the stolen picks in `sel_cpu`. With `sel_cpu` None (a call no worker sees) only the masking is done."""
        if not self.armed:
            return False
        if not self.manager.started:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("ColdSteal: the first select must run eagerly (it starts the proxy)")
            self.manager.start()
        rows = sel.shape[0]
        if rows > self.max_rows or sel.numel() > 384:
            s_max = 0
        else:
            s_max = self.s_max
        _ext().coldsteal_select(sel, sel_cpu, self.slot_of, self.dev_seq, self.stats, self.words, self.n_hot, s_max)
        return True

    def policy_select(self, sel, sel_cpu, sel_t):
        """Three-unit split (QMESH_SPLIT_POLICY): steal s, send c to the CPU tier (sel_t, None = no tier), rest to
        the GX, (s, c) from the policy table or the explore draw. Calls wider than max_rows steal and offload
        nothing (as select())."""
        if not self.armed:
            return False
        if not self.manager.started:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("ColdSteal: the first select must run eagerly (it starts the proxy)")
            self.manager.start()
        pol = policy()
        rows = sel.shape[0]
        wide = rows > self.max_rows or sel.numel() > 384
        z = 0 if wide else self.Z
        _ext().coldsteal_policy(sel, sel_cpu, sel_t, self.slot_of, self.dev_seq, self.stats, self.words, self.n_hot,
                                self.layer, 0 if wide else pol.s_cap, 0 if wide else pol.c_cap, pol.table, pol.salt,
                                pol.ring, pol.mask, pol.head, self.cur,
                                z, self.S, self.tables if z else None, self.k2 if z else None,
                                self.pool_dev if z else 0, self.pool_k2_ptr if z else 0,
                                self.zc_off if z else None, self.stride)
        return True

    def policy_end(self):
        """Timestamp the end of this layer's MoE for the call policy_select logged (no-op unless the policy runs
        with the log on)."""
        if not policy_enabled():
            return
        pol = policy()
        if pol.ring:
            _ext().coldsteal_policy_end(self.cur, pol.ring, pol.mask, pol.pub)

    def counters(self):
        c = self.stats.tolist()
        return {"calls": c[0], "cold_picks": c[1], "stolen_picks": c[2], "stolen_experts": c[3]}

    def free(self):
        ext = _ext()
        if self.words:
            ext.coldsteal_host_free(self.words)
            self.words = 0
        if self.pool_ptr:
            self.pool = None
            ext.coldsteal_host_free(self.pool_ptr)
            self.pool_ptr = 0
        if self.pool_k2_ptr:
            self.pool_k2 = None
            ext.coldsteal_host_free(self.pool_k2_ptr)
            self.pool_k2_ptr = 0
        if getattr(self, "vmm", None) is not None:
            self.slots = None
            self.vmm.free()
            self.vmm = None


def _pinned_view(ptr, n):
    """uint8 CPU tensor over pinned memory allocated by coldsteal_host_alloc (lifetime owned by the caller)."""
    import ctypes
    buf = (ctypes.c_uint8 * n).from_address(ptr)
    return torch.frombuffer(buf, dtype = torch.uint8, count = n)


class Manager:
    """Process-wide owner of the layers and the proxy (one H2D link)."""

    def __init__(self):
        self.layers = []
        self.handle = 0
        self.started = False
        self.lock = threading.Lock()
        atexit.register(self.close)

    def arm(self, m):
        def refuse(why):
            print(f" -- ColdSteal off for {m.key}: {why}", flush = True)
            return None
        if self.started:
            return refuse("proxy already running (layers must arm at load)")
        if not m.gated or m.multi_gate is None:
            return refuse("gateless experts")
        if getattr(m, "cpu_split_first", None) in (None, 0) or m.num_local_experts != m.cpu_split_first:
            return refuse("not a static split layer")
        if getattr(m, "_split_dynamic", False) or getattr(m, "_split_spares", 0):
            return refuse("dynamic placement")
        if getattr(m, "routing_first", 0) != 0:
            return refuse("routing_first != 0")
        for name, multi in (("gate", m.multi_gate), ("up", m.multi_up), ("down", m.multi_down)):
            if not multi.mul1 or multi.mcg:
                return refuse(f"{name} experts are not mul1")
        if m.ups[0].inner is not None and m.ups[0].inner.bias is not None:
            return refuse("expert biases")
        mm = re.search(r"layers\.(\d+)\.", m.key)
        src = os.environ.get("QMESH_COLDSTEAL_SRC", "")
        if not mm or not src:
            return refuse("no QMESH_COLDSTEAL_SRC")
        layer = int(mm.group(1))
        path = os.path.join(src, f"cold-L{layer:02d}.safetensors")
        if not os.path.isfile(path):
            return refuse(f"no shard {path}")
        shard = _Shard(path)
        try:
            steal = LayerSteal(m, shard, layer)
        finally:
            shard.close()
        steal.manager = self
        self.layers.append(steal)
        print(f" -- ColdSteal {m.key}: {steal.n_hot} hot, {steal.S} slots x {steal.stride / 2**20:.1f} MB, "
              f"pool {steal.loaded}/{steal.E - steal.n_hot} cold rows, s_max {steal.s_max}", flush = True)
        return steal

    def start(self):
        """Start the proxy (idempotent; the first MoE call does it)."""
        if self.started:
            return
        with self.lock:
            if self.started or not self.layers:
                return
            ext = _ext()
            dev = self.layers[0].slots.device
            self.handle = ext.coldsteal_proxy_create(dev.index or 0, _env_int("QMESH_COLDSTEAL_MAX_OUT", 4))
            for L in self.layers:
                ext.coldsteal_proxy_add_layer(
                    self.handle, L.words, L.slot_of.data_ptr(), L.k2.data_ptr(), L.n_hot + L.S + L.Z,
                    L.slots.data_ptr(), L.pool_ptr, L.pool_k2.data_ptr(),
                    L.E, L.n_hot, L.S, L.stride, L.scale_bytes, L.tstride, L.tunit)
            ext.coldsteal_proxy_launch(self.handle)
            self.started = True

    def close(self):
        if self.handle:
            ext = _ext()
            torch.cuda.synchronize()
            copies, evictions, records, why = ext.coldsteal_proxy_stats(self.handle)
            tot = {"calls": 0, "cold_picks": 0, "stolen_picks": 0, "stolen_experts": 0}
            for L in self.layers:
                for k, v in L.counters().items():
                    tot[k] += v
            ext.coldsteal_proxy_destroy(self.handle)
            self.handle = 0
            share = tot["stolen_picks"] / max(1, tot["cold_picks"])
            print(f" -- ColdSteal proxy: {len(self.layers)} layers, {copies} copies, {evictions} evictions, "
                  f"{records} records, calls {tot['calls']}, cold picks {tot['cold_picks']}, stolen "
                  f"{tot['stolen_picks']} ({100 * share:.1f} %), failure={why!r}", flush = True)
        for L in self.layers:
            L.free()
        self.layers.clear()

    # ---- slot lender for the growable KV cache (notes/GrowKV.md) -----------------------------------------------------
    # The page table borrows VRAM from the ColdSteal slots, top slot first, in all layers at once: one slot's worth is
    # S_vmm granule-rounded bytes per layer. A lend blocks until the proxy has drained the retired slots (no launch can
    # read them), then unmaps them; give_back maps them again and the proxy resumes admitting into them.

    def _slot_bytes(self) -> int:
        """Physical bytes released per retired slot across all layers (granule-rounded per layer)"""
        tot = 0
        for L in self.layers:
            if L.vmm is None:
                return 0
            tot += L.vmm.round_up(L.S * L.stride) - L.vmm.round_up((L.S - 1) * L.stride)
        return tot

    def active_slots(self) -> int:
        return getattr(self, "_active", self.layers[0].S if self.layers else 0)

    def lendable_bytes(self) -> int:
        if not self.layers or self.layers[0].vmm is None:
            return 0
        return self.active_slots() * self._slot_bytes()

    def lent_bytes(self) -> int:
        if not self.layers or self.layers[0].vmm is None:
            return 0
        return (self.layers[0].S - self.active_slots()) * self._slot_bytes()

    def _set_active(self, n: int):
        ext = _ext()
        if self.handle:
            ext.coldsteal_proxy_set_active(self.handle, n)
            t0 = time.time()
            while ext.coldsteal_proxy_retired(self.handle) != n:
                if time.time() - t0 > 30:
                    raise RuntimeError(f"ColdSteal: retired slots did not drain to {n} active in 30 s")
                time.sleep(0.0005)
        self._active = n

    def lend(self, nbytes: int) -> int:
        """Retire top slots in every layer until at least nbytes are released; returns slots retired. The device must
        be idle for the released memory (the caller maps it into the KV pools right after)."""
        per = self._slot_bytes()
        if per <= 0 or nbytes <= 0:
            return 0
        k = min(self.active_slots(), -(-nbytes // per))
        n = self.active_slots() - k
        self._set_active(n)
        torch.cuda.synchronize()          # no launch issued before the retire may still read the slots
        for L in self.layers:
            L.vmm.trim(n * L.stride)
        print(f" -- ColdSteal: {k} slot(s) lent to the KV cache, {n} active", flush = True)
        return k

    def give_back(self, nbytes: int) -> int:
        """Restore lent slots that nbytes of freed KV memory cover; returns slots restored"""
        per = self._slot_bytes()
        if per <= 0 or nbytes < per:
            return 0
        S = self.layers[0].S
        k = min(S - self.active_slots(), nbytes // per)
        if k <= 0:
            return 0
        n = self.active_slots() + k
        for L in self.layers:
            L.vmm.ensure(n * L.stride)
        torch.cuda.synchronize()          # the zero fill lands before the proxy may copy into the slots
        self._set_active(n)
        print(f" -- ColdSteal: {k} slot(s) returned by the KV cache, {n} active", flush = True)
        return k


_manager = None
_manager_lock = threading.Lock()


def manager():
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = Manager()
        return _manager


# ---- three-unit split policy (notes/SplitPolicy.md) -----------------------------------------------------------------
#
#   QMESH_SPLIT_POLICY=explore|table   off (unset): ColdSteal's fixed s_max select, CPU tier by QMESH_CPU_TIER_FRAC
#   QMESH_SPLIT_TABLE=<file.npy>       qmesh/split_table_fmt.py layout (P7 load, rows, distinct cold, resident -> s, c, z)
#   QMESH_SPLIT_SCAP / _CCAP / _ZCAP   caps on s (steal), c (CPU), z (P7 zero-copy from pinned RAM; 0 = off)
#   QMESH_SPLIT_SALT                   explore draw seed
#   QMESH_SPLIT_LOG=<file>             per-call records (CsPolRec, 48 bytes each) appended there by a drain thread

REC_BYTES = 48


class Policy:
    def __init__(self):
        self.mode = os.environ.get("QMESH_SPLIT_POLICY", "")
        assert self.mode in ("explore", "table"), f"QMESH_SPLIT_POLICY={self.mode!r}"
        self.s_cap = _env_int("QMESH_SPLIT_SCAP", 8)
        self.c_cap = _env_int("QMESH_SPLIT_CCAP", 8)
        self.salt = _env_int("QMESH_SPLIT_SALT", 1)
        self.table = None
        dev = torch.device("cuda", torch.cuda.current_device())
        if self.mode == "table":
            from qmesh import split_table_fmt as fmt
            self.table = torch.zeros(fmt.NBYTES, dtype = torch.int8, device = dev)
            self.load_table(os.environ["QMESH_SPLIT_TABLE"])
        self._log_start()

    def load_table(self, path):
        """Copy a table file into the live device table in place (captured graphs read it at replay)."""
        import numpy as np
        assert self.table is not None, "load_table needs QMESH_SPLIT_POLICY=table"
        from qmesh import split_table_fmt as fmt
        t = np.load(path)
        assert t.dtype == np.int8 and t.size == fmt.NBYTES, (t.dtype, t.size, fmt.NBYTES)
        # The table is created during model load under inference mode (an inference tensor): the in-place update must
        # run there too, or torch refuses it on a live switch
        with torch.inference_mode():
            self.table.copy_(torch.from_numpy(np.ascontiguousarray(t)).view(-1).to(self.table.device))
        torch.cuda.synchronize()

    def _log_start(self):
        ext = _ext()
        self.ring = self.mask = self.head = self.pub = 0
        log = os.environ.get("QMESH_SPLIT_LOG")
        if log:
            n = 1 << 20                                      # 48 MB, ~5 s of B8 decode at 40 layers
            self.mask = n - 1
            self.ring = ext.coldsteal_host_alloc(n * REC_BYTES, True)
            self.head = ext.coldsteal_host_alloc(64, True)
            self.pub = ext.coldsteal_host_alloc(64, True)
            self._drain_start(log, n)

    def _drain_start(self, path, n):
        """Copy finished records to `path`. pub counts finished records in ring order; a record is published by its
        layer's end kernel, which runs after every earlier layer's (one stream), so [done, pub) is complete."""
        import ctypes
        ring = (ctypes.c_uint8 * (n * REC_BYTES)).from_address(self.ring)
        pub = ctypes.c_int64.from_address(self.pub)
        self._stop = False

        def drain():
            import time
            done = 0
            with open(path, "ab") as f:
                while not self._stop:
                    p = pub.value
                    if p - done > n:
                        print(f" !! split log: drain fell {p - done - n} records behind, skipped", flush = True)
                        done = p - n
                    while done < p:
                        a = done & self.mask
                        b = min(p - done, n - a)
                        f.write(bytes(ring[a * REC_BYTES:(a + b) * REC_BYTES]))
                        done += b
                    f.flush()
                    time.sleep(0.05)

        self._thread = threading.Thread(target = drain, daemon = True)
        self._thread.start()
        atexit.register(self.close)

    def close(self):
        import time
        if self.ring and not self._stop:
            time.sleep(0.2)
            self._stop = True
            self._thread.join(timeout = 2)


_policy = None


def policy_enabled() -> bool:
    return os.environ.get("QMESH_SPLIT_POLICY", "") in ("explore", "table")


def policy():
    global _policy
    with _manager_lock:
        if _policy is None:
            _policy = Policy()
        return _policy
