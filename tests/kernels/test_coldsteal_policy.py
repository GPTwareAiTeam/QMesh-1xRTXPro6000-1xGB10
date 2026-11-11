"""coldsteal_policy (exl3_coldsteal.cu, notes/SplitPolicy.md): the three-unit split of one call's cold picks.

Every cold pick lands on exactly one unit (P7 steal slot, CPU tier, GX); stolen experts are the s most-picked resident
ones, CPU experts the c least-picked of the rest; s and c respect feasibility and the caps; the table mode applies the
table cell of (rows bucket, distinct cold, resident) clamped to the feasible range; the explore draw is reproducible
for a seed and covers the whole feasible range; the log record carries the call's state and the end timestamp."""
import ctypes
import itertools

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from qmesh import split_table_fmt as fmt

N_HOT, E, TOPK = 100, 384, 6
INACTIVE = 0x3fffffff
REC = 48
S_SLOTS = 8
POOL_STRIDE = 256


class Layer:
    def __init__(self, dev, resident, ring_n = 64):
        self.slot_of = torch.full((E,), -1, dtype=torch.int32)
        for s, e in enumerate(resident):
            self.slot_of[e] = s
        self.slot_of = self.slot_of.to(dev)
        self.dev_seq = torch.zeros(1, dtype=torch.long, device=dev)
        self.stats = torch.zeros(4, dtype=torch.long, device=dev)
        self.cur = torch.zeros(1, dtype=torch.long, device=dev)
        self.words = ext.coldsteal_host_alloc(32 * 8, True)
        self.ring_n = ring_n
        self.ring = ext.coldsteal_host_alloc(ring_n * REC, True)
        self.head = ext.coldsteal_host_alloc(64, True)
        self.pub = ext.coldsteal_host_alloc(64, True)

    def free(self):
        for p in (self.words, self.ring, self.head, self.pub, *getattr(self, "extra", [])):
            ext.coldsteal_host_free(p)

    def zc_args(self, z_cap):
        if not z_cap:
            return (0, S_SLOTS, None, None, 0, 0, None, 0)
        return (z_cap, S_SLOTS, self.zt, self.zk, self.pool, self.pool_k2, self.zoff, POOL_STRIDE)

    def enable_zc(self, dev, z_cap):
        """Zero-copy state: launch tables with n_hot + S + z_cap columns, a mapped pool of (E - n_hot) rows"""
        n_cs = N_HOT + S_SLOTS + z_cap
        self.zt = torch.zeros((9, n_cs), dtype=torch.long, device=dev)
        self.zk = torch.zeros((3, n_cs), dtype=torch.uint8, device=dev)
        self.zoff = torch.arange(9, dtype=torch.long, device=dev) * 16
        self.pool = ext.coldsteal_host_alloc((E - N_HOT) * POOL_STRIDE, True)
        self.pool_k2 = ext.coldsteal_host_alloc(E - N_HOT, True)
        k2 = (ctypes.c_uint8 * (E - N_HOT)).from_address(self.pool_k2)
        for r in range(E - N_HOT):
            k2[r] = 4 + 2 * (r % 3)
        self.extra = [self.pool, self.pool_k2]

    def run(self, sel, cpu_tier, s_cap, c_cap, table = None, salt = 1, layer = 3, z_cap = 0):
        dev = sel.device
        sel = sel.clone()
        rel = sel - N_HOT
        sel_cpu = torch.where((rel >= 0) & (rel < E - N_HOT), rel, torch.full_like(rel, -1)).contiguous()
        sel_t = torch.full_like(sel_cpu, 7) if cpu_tier else None
        ext.coldsteal_policy(sel, sel_cpu.view(-1), None if sel_t is None else sel_t.view(-1), self.slot_of,
                             self.dev_seq, self.stats, self.words, N_HOT, layer, s_cap, c_cap, table, salt,
                             self.ring, self.ring_n - 1, self.head, self.cur, *self.zc_args(z_cap))
        ext.coldsteal_policy_end(self.cur, self.ring, self.ring_n - 1, self.pub)
        torch.cuda.synchronize()
        idx = int(self.cur.item()) % self.ring_n
        raw = (ctypes.c_uint8 * REC).from_address(self.ring + idx * REC)
        t0, t1 = (int.from_bytes(bytes(raw[k:k + 8]), "little") for k in (0, 8))
        f = [int.from_bytes(bytes(raw[16 + 2 * k:18 + 2 * k]), "little", signed=True) for k in range(16)]
        return sel.cpu(), sel_cpu.cpu(), None if sel_t is None else sel_t.cpu(), (t0, t1, f)


def check_partition(orig, sel, sel_cpu, sel_t, slot_of_cpu):
    """Every cold pick on exactly one unit; returns (stolen experts, cpu experts) as sets with their pick counts"""
    stolen, cpu, gx = {}, {}, {}
    for i, e in enumerate(orig.view(-1).tolist()):
        if not (N_HOT <= e < E):
            assert sel.view(-1)[i].item() == e
            continue
        s, w = sel.view(-1)[i].item(), sel_cpu.view(-1)[i].item()
        t = -1 if sel_t is None else sel_t.view(-1)[i].item()
        units = [s != INACTIVE, w >= 0, t >= 0]
        assert sum(units) == 1, (i, e, s, w, t)
        if units[0]:
            assert s == N_HOT + slot_of_cpu[e], (e, s)
            stolen[e] = stolen.get(e, 0) + 1
        elif units[1]:
            assert w == e - N_HOT
            gx[e] = gx.get(e, 0) + 1
        else:
            assert t == e - N_HOT
            cpu[e] = cpu.get(e, 0) + 1
    # An expert lives on one unit only
    assert not (set(stolen) & set(cpu)) and not (set(stolen) & set(gx)) and not (set(cpu) & set(gx))
    return stolen, cpu, gx


def routing(gen, rows, hot_frac = 0.4, cold_pool = 30):
    sel = torch.empty((rows, TOPK), dtype=torch.long)
    for r in range(rows):
        k_hot = int(torch.binomial(torch.tensor(float(TOPK)), torch.tensor(hot_frac), generator=gen).item())
        hot = torch.randperm(N_HOT, generator=gen)[:k_hot]
        cold = N_HOT + torch.randperm(cold_pool, generator=gen)[:TOPK - k_hot]
        sel[r] = torch.cat([hot, cold])
    return sel


@pytest.mark.gpu
def test_explore_partition_and_ranking(device):
    gen = torch.Generator().manual_seed(5)
    resident = [N_HOT + i for i in range(0, 30, 3)][:8]
    L = Layer(device, resident)
    slot_of_cpu = L.slot_of.cpu().tolist()
    seen = set()
    try:
        for it in range(300):
            rows = (1, 2, 4, 6, 12, 24)[it % 6]
            orig = routing(gen, rows)
            sel, sel_cpu, sel_t, (t0, t1, f) = L.run(orig.to(device), True, 8, 8, salt=11)
            stolen, cpu, gx = check_partition(orig, sel, sel_cpu, sel_t, slot_of_cpu)
            counts = {}
            for e in orig.view(-1).tolist():
                if e >= N_HOT:
                    counts[e] = counts.get(e, 0) + 1
            res = [e for e in counts if slot_of_cpu[e] >= 0]
            nd, h, s, c = len(counts), len(res), len(stolen), len(cpu)
            assert 0 <= s <= min(h, 8) and 0 <= c <= min(nd - s, 8)
            # Stolen = the s most-picked resident (count desc, id asc)
            order = sorted(res, key=lambda e: (-counts[e], e))
            assert set(stolen) == set(order[:s])
            # CPU = the c least-picked of the remaining (count asc, id desc)
            rest = sorted((e for e in counts if e not in stolen), key=lambda e: (counts[e], -e))
            assert set(cpu) == set(rest[:c])
            # Record: layer, rows, nd, h, s, c, ... explore flag; end after start
            assert f[:6] == [3, rows, nd, h, s, c] and f[11] == 1
            assert f[9] == sum(stolen.values()) and f[10] == sum(cpu.values()) and f[8] == sum(counts.values())
            assert t1 >= t0 > 0
            seen.add((rows, nd, h, s, c))
        # The draw covers the feasible range: some call with h >= 2 saw s = 0, s = h, c = 0 and c > 0
        assert any(k[3] == 0 for k in seen) and any(k[3] == k[2] and k[2] >= 2 for k in seen)
        assert any(k[4] == 0 for k in seen) and any(k[4] >= 2 for k in seen)
    finally:
        L.free()


@pytest.mark.gpu
def test_explore_reproducible_and_no_tier(device):
    gen = torch.Generator().manual_seed(9)
    resident = [N_HOT + i for i in range(8)]
    calls = [routing(gen, r, cold_pool=12) for r in (1, 6, 6, 12, 24, 4)]
    out = []
    for _ in range(2):
        L = Layer(device, resident)
        try:
            out.append([L.run(c.to(device), False, 3, 8, salt=77)[:2] for c in calls])
        finally:
            L.free()
    for (a_sel, a_cpu), (b_sel, b_cpu) in zip(*out):
        assert torch.equal(a_sel, b_sel) and torch.equal(a_cpu, b_cpu)
    # No CPU tier: c = 0, s <= s_cap
    L = Layer(device, resident)
    slot_of_cpu = L.slot_of.cpu().tolist()
    try:
        for c in calls:
            sel, sel_cpu, sel_t, (_, _, f) = L.run(c.to(device), False, 3, 8, salt=77)
            stolen, cpu, _ = check_partition(c, sel, sel_cpu, None, slot_of_cpu)
            assert not cpu and f[5] == 0 and len(stolen) <= 3
    finally:
        L.free()


@pytest.mark.gpu
def test_table_mode(device):
    """Table cells by (P7 hot load level, rows bucket, nd, h): the level comes from the call's distinct hot experts vs
    the per-rows-bucket thresholds; the cell's (s, c) is applied, clamped to the feasible range"""
    gen = torch.Generator().manual_seed(21)
    resident = [N_HOT + i for i in range(8)]
    cells, thr = fmt.empty()
    # rows 6: level 0 (light) nd 3..14 -> steal 2 + CPU 1, level 2 (heavy) nd 3..9 -> steal 1; rows 1: nd 2..4 -> CPU 1 at
    # every level; thresholds rows 6: <= 4 hot distinct light, <= 12 mid
    cells[0, fmt.row_bucket(6), 3:15, :, 0] = 2
    cells[0, fmt.row_bucket(6), 3:15, :, 1] = 1
    cells[2, fmt.row_bucket(6), 3:10, :, 0] = 1
    cells[:, fmt.row_bucket(1), 2:5, :, 1] = 1
    thr[fmt.row_bucket(6)] = (4, 12)
    thr[fmt.row_bucket(1)] = (100, 100)
    t = torch.from_numpy(fmt.pack(cells, thr)).to(device)
    L = Layer(device, resident)
    slot_of_cpu = L.slot_of.cpu().tolist()
    hits = set()
    try:
        for i in range(300):
            rows = (1, 6)[i % 2]
            orig = routing(gen, rows, hot_frac=(0.0, 0.4, 0.9)[(i // 2) % 3], cold_pool=14)
            sel, sel_cpu, sel_t, (_, _, f) = L.run(orig.to(device), True, 8, 8, table=t)
            stolen, cpu, _ = check_partition(orig, sel, sel_cpu, sel_t, slot_of_cpu)
            nd, h, hot_distinct = f[2], f[3], f[7]
            b = int(fmt.row_bucket(rows))
            lv = 0 if hot_distinct <= thr[b, 0] else (1 if hot_distinct <= thr[b, 1] else 2)
            want_s, want_c, _ = cells[lv, b, min(nd, fmt.POL_N), min(h, fmt.POL_H)]
            assert len(stolen) == min(want_s, h) and len(cpu) == min(want_c, nd - len(stolen))
            assert f[11] == 0
            if want_s or want_c:
                hits.add((rows, lv))
        assert {(6, 0), (6, 2), (1, 0)}.issubset(hits), hits
    finally:
        L.free()


@pytest.mark.gpu
def test_graph_follows_table_edits(device):
    """The table is read from device memory at replay: a captured graph picks up a new table without recapture"""
    import numpy as np
    resident = [N_HOT + i for i in range(8)]
    L = Layer(device, resident)
    table = torch.from_numpy(fmt.pack(*fmt.empty())).to(device)
    orig = torch.tensor([[1, 2, N_HOT, N_HOT + 1, N_HOT + 2, N_HOT + 20]], dtype=torch.long, device=device)
    sel = orig.clone()
    rel = sel - N_HOT
    sel_cpu = torch.where(rel >= 0, rel, torch.full_like(rel, -1)).contiguous()
    sel_t = torch.empty_like(sel_cpu)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(s):
            g.capture_begin()
            sel.copy_(orig)
            sel_cpu.copy_(torch.where(rel >= 0, rel, torch.full_like(rel, -1)))
            ext.coldsteal_policy(sel, sel_cpu.view(-1), sel_t.view(-1), L.slot_of, L.dev_seq, L.stats, L.words,
                                 N_HOT, 0, 8, 8, table, 1, L.ring, L.ring_n - 1, L.head, L.cur, *L.zc_args(0))
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        g.replay(); torch.cuda.synchronize()
        assert (sel_t >= 0).sum().item() == 0 and (sel != INACTIVE).sum().item() == 2      # nothing taken
        cell = ((0 * fmt.ROW_B + int(fmt.row_bucket(1))) * (fmt.POL_N + 1) + 4) * (fmt.POL_H + 1) + 3  # level 0 (thresholds 255), nd 4, h 3
        table[cell * fmt.POL_V] = 2
        table[cell * fmt.POL_V + 1] = 1
        g.replay(); torch.cuda.synchronize()
        assert (sel[0, 2:] < INACTIVE).sum().item() == 2 and (sel_t >= 0).sum().item() == 1
    finally:
        L.free()


@pytest.mark.gpu
def test_zero_copy(device):
    """z most-picked non-resident experts go to launch entries n_hot + S + rank, whose table columns the kernel points at
    the experts' pool rows (base + row * stride + off[q]) with the row's rate; the rest keep the steal/CPU/GX split"""
    gen = torch.Generator().manual_seed(31)
    resident = [N_HOT + i for i in range(0, 16, 4)]
    L = Layer(device, resident)
    L.enable_zc(device, 4)
    slot_of_cpu = L.slot_of.cpu().tolist()
    seen_z = set()
    try:
        for it in range(200):
            rows = (1, 4, 6, 12)[it % 4]
            orig = routing(gen, rows, cold_pool=24)
            sel, sel_cpu, sel_t, (_, _, f) = L.run(orig.to(device), True, 8, 8, salt=5, z_cap=4)
            counts = {}
            for e in orig.view(-1).tolist():
                if e >= N_HOT:
                    counts[e] = counts.get(e, 0) + 1
            miss = sorted((e for e in counts if slot_of_cpu[e] < 0), key=lambda e: (-counts[e], e))
            z = f[12]
            assert 0 <= z <= min(4, len(miss)) and f[13] == sum(counts[e] for e in miss[:z])
            zt, zk = L.zt.cpu(), L.zk.cpu()
            for rank, e in enumerate(miss[:z]):
                col = N_HOT + S_SLOTS + rank
                picks = [i for i, v in enumerate(orig.view(-1).tolist()) if v == e]
                for i in picks:
                    assert sel.view(-1)[i].item() == col and sel_cpu.view(-1)[i].item() == -1
                    assert sel_t.view(-1)[i].item() == -1
                row = e - N_HOT
                for q in range(9):
                    assert zt[q, col].item() == L.pool + row * POOL_STRIDE + q * 16
                assert all(zk[q, col].item() == 4 + 2 * (row % 3) for q in range(3))
            # every other cold pick still on exactly one of steal / CPU / GX
            for i, e in enumerate(orig.view(-1).tolist()):
                if e >= N_HOT and e not in miss[:z]:
                    units = [N_HOT <= sel.view(-1)[i].item() < N_HOT + S_SLOTS, sel_cpu.view(-1)[i].item() >= 0,
                             sel_t.view(-1)[i].item() >= 0]
                    assert sum(units) == 1
            seen_z.add(z)
        assert {0, 1, 2}.issubset(seen_z)
    finally:
        L.free()
