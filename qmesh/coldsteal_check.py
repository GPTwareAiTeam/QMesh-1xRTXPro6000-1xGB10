"""ColdSteal GPU check on one real layer (P7, no GX): the folded launch, the device select and the proxy.

    python3 -u qmesh/coldsteal_check.py --hot <P7 hot pack dir> --cold <dir of cold-LNN.safetensors> --layer 5

Builds the layer's hot tables from the hot pack (paired layout, mixed rates) and S steal slots, then:
  1. fold      every cold expert of a random selection forced resident: the folded launch (hot + slot picks) must equal
               hot-only launch + slot-only launch of the same experts (fp32 sums of the same per-expert outputs, so
               equal up to the order of the final fold, checked at 1e-5 relative), at rows 1 / 4 / 6 / 24
  2. select    on real routing rows: stolen picks are exactly the resident ones (top s_max by count), blanked in sel_cpu,
               every cold pick is computed once (P7 xor worker); graph capture + replay of the select works
  3. proxy     drives decode-like calls; after warm-up the stolen share is > 0, the slot contents equal the pool rows
               (byte compare), timing of the launch at 0 / 1 / 2 / 3 stolen experts per call
Prints CS_CHECK lines; exits non-zero on the first failure.
"""
import argparse
import json
import os
import statistics
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext
from qmesh import coldsteal as cs

DEV = torch.device("cuda:0")
H, I = 5120, 2304
TOPK = 6


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load_hot(path, layer, hot_ids):
    """Device tensors of the layer's hot experts (checkpoint id order of `hot_ids`), their rates"""
    from safetensors import safe_open
    out, rates = [], []
    with safe_open(path, framework = "pt", device = "cpu") as f:
        for e in hot_ids:
            b = f"layers.{layer}.ffn.experts.{e}"
            ex = {p: {t: f.get_tensor(f"{b}.{w}.{t}").to(DEV) for t in ("trellis", "suh", "svh")}
                  for p, w in (("g", "w1"), ("u", "w3"), ("d", "w2"))}
            out.append(ex)
            rates.append(ex["u"]["trellis"].shape[-1] // 16)
    return out, rates


class FakeMulti:
    def __init__(self, experts, rates, p):
        self.ptrs_trellis = torch.tensor([e[p]["trellis"].data_ptr() for e in experts], dtype = torch.long, device = DEV)
        self.ptrs_suh = torch.tensor([e[p]["suh"].data_ptr() for e in experts], dtype = torch.long, device = DEV)
        self.ptrs_svh = torch.tensor([e[p]["svh"].data_ptr() for e in experts], dtype = torch.long, device = DEV)
        self.k2 = torch.tensor([2 * r for r in rates], dtype = torch.uint8, device = DEV)
        self.K = float(rates[0])
        self.mul1, self.mcg = True, False
        self.in_features = H if p != "d" else I
        self.out_features = I if p != "d" else H


class FakeModule:
    def __init__(self, layer, experts, rates, n_hot, E, perm):
        self.key = f"model.layers.{layer}.mlp"
        self.device = DEV
        self.cpu_split_first = n_hot
        self.num_experts = E
        self.num_local_experts = n_hot
        self._split_perm = perm
        self.multi_gate = FakeMulti(experts, rates, "g")
        self.multi_up = FakeMulti(experts, rates, "u")
        self.multi_down = FakeMulti(experts, rates, "d")
        self.gated = True


class Scratch:
    def __init__(self, slots, rows):
        self.gu_g = torch.empty((slots, 1, I), dtype = torch.half, device = DEV)
        self.gu_u = torch.empty_like(self.gu_g)
        self.act = torch.empty((slots, 1, I), dtype = torch.half, device = DEV)
        self.d = torch.empty((slots, 1, H), dtype = torch.float, device = DEV)
        self.had_g = torch.empty((slots, H), dtype = torch.half, device = DEV)
        self.had_u = torch.empty_like(self.had_g)
        n = ext.exl3_moe_coop_ctr_len(slots, rows, I, H) if hasattr(ext, "exl3_moe_coop_ctr_len") else \
            slots * (I // 128) * 2 + rows * (H // 128) + 4 * slots + 8
        self.ctr = torch.zeros((n,), dtype = torch.int, device = DEV)


def launch(tabs, k2, x, sel, rw, out, sc, max_expert, paired, plain_from = -1):
    ext.exl3_moe_coop(
        x, sel, rw, 0, max_expert, H,
        tabs[0], tabs[1], tabs[2], tabs[3], tabs[4], tabs[5], tabs[6], tabs[7], tabs[8],
        None, None, None, 3.0, 3.0, 3.0, False, True, 0, 10.0, True,
        sc.had_g, sc.had_u, sc.gu_g, sc.gu_u, sc.act, sc.d, sc.ctr, out, None, None,
        k2_g = k2[0], k2_u = k2[1], k2_d = k2[2], paired = paired, plain_from = plain_from)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", required = True)
    ap.add_argument("--cold", required = True)
    ap.add_argument("--layer", type = int, default = 5)
    ap.add_argument("--route", default = "")
    args = ap.parse_args()
    L = args.layer
    pl = json.load(open(os.path.join(args.hot, "placement.mix-mix4460-nq.json")))
    hot_ids = pl["layers"][str(L)]["hot"]
    warm = pl["layers"][str(L)]["warm"]
    perm = list(hot_ids) + list(warm)
    n_hot, E = len(hot_ids), len(perm)
    hdr, _ = header(os.path.join(args.hot, f"layers.{L}.safetensors"))
    paired = (hdr.get("__metadata__") or {}).get("qmesh_trellis_layout") == "coop_pair_v1"
    experts, rates = load_hot(os.path.join(args.hot, f"layers.{L}.safetensors"), L, hot_ids)
    m = FakeModule(L, experts, rates, n_hot, E, perm)
    os.environ.setdefault("QMESH_COLDSTEAL_SLOTS", "8")
    shard = cs._Shard(os.path.join(args.cold, f"cold-L{L:02d}.safetensors"))
    t0 = time.time()
    st = cs.LayerSteal(m, shard, L)
    shard.close()
    print(f"CS_CHECK load layer={L} n_hot={n_hot} E={E} paired={paired} pool={st.loaded} rows "
          f"stride={st.stride} in {time.time() - t0:.1f}s", flush = True)
    S = st.S

    # ---- 1. fold: force 8 cold experts resident by direct copy (as the proxy would) ----
    rng = np.random.default_rng(1)
    cold_rows = [r for r in range(E - n_hot) if st.pool_k2[r] > 0]
    pick = rng.choice(cold_rows, S, replace = False)
    for s, r in enumerate(pick):
        st.slots[s * st.stride:(s + 1) * st.stride].copy_(st.pool[r * st.stride:(r + 1) * st.stride])
        for i in range(3):
            st.k2[i, n_hot + s] = int(st.pool_k2[r])
        st.slot_of[n_hot + r] = s
    torch.cuda.synchronize()
    tabs = [st.tables[i] for i in range(9)]
    k2 = [st.k2[i] for i in range(3)]
    # Separate slot-only tables (plain rows only) for the reference launch
    tabs_s = [t[n_hot:].contiguous() for t in tabs]
    k2_s = [k[n_hot:].contiguous() for k in k2]
    # hot-only tables
    tabs_h = [t[:n_hot].contiguous() for t in tabs]
    k2_h = [k[:n_hot].contiguous() for k in k2]
    bc_fail = 0
    for rows in (1, 4, 6, 24):
        g = torch.Generator().manual_seed(rows)
        x = (torch.randn((rows, H), generator = g) * 0.5).half().to(DEV)
        # 3 hot + 3 slot picks per row
        hsel = torch.stack([torch.randperm(n_hot, generator = g)[:3] for _ in range(rows)])
        ssel = torch.stack([torch.randperm(S, generator = g)[:3] for _ in range(rows)])
        rw = (torch.rand((rows, TOPK), generator = g) * 0.3 + 0.05).half()
        sel_f = torch.cat([hsel, ssel + n_hot], 1).to(DEV)
        rw = rw.to(DEV)
        sc = Scratch(rows * TOPK, rows)
        o_h = torch.empty((rows, H), dtype = torch.float, device = DEV)
        o_s = torch.empty_like(o_h)
        launch(tabs_h, k2_h, x, hsel.to(DEV).contiguous(), rw[:, :3].contiguous(), o_h, sc, n_hot, paired)
        launch(tabs_s, k2_s, x, ssel.to(DEV).contiguous(), rw[:, 3:].contiguous(), o_s, sc, S, False)
        o_f = torch.empty_like(o_h)
        launch(tabs, k2, x, sel_f.contiguous(), rw, o_f, sc, n_hot + S, paired, n_hot)
        # an inactive cold pick (the select's CS_INACTIVE) must contribute nothing
        sel_i = sel_f.clone(); sel_i[:, 5] = 0x3fffffff
        rw_i = rw.clone()
        o_i = torch.empty_like(o_h)
        launch(tabs, k2, x, sel_i.contiguous(), rw_i, o_i, sc, n_hot + S, paired, n_hot)
        sel_r = sel_f[:, :5].contiguous(); rw_r = rw[:, :5].contiguous()
        o_r = torch.empty_like(o_h)
        launch(tabs, k2, x, sel_r, rw_r, o_r, sc, n_hot + S, paired, n_hot)
        torch.cuda.synchronize()
        ref = o_h + o_s
        scale = ref.abs().max().item()
        rel = (o_f - ref).abs().max().item() / scale
        rel_i = (o_i - o_r).abs().max().item() / scale
        ok = rel < 1e-5 and rel_i < 1e-5 and bool(torch.isfinite(o_f).all())
        print(f"CS_CHECK fold rows={rows} max_rel={rel:.3e} inactive_rel={rel_i:.3e} {'PASS' if ok else 'FAIL'}",
              flush = True)
        bc_fail += 0 if ok else 1
    if bc_fail:
        sys.exit(1)

    # ---- 1b. slot experts against an fp64 reference of the cold pack's own tensors ----
    import math
    hb = torch.ones((1, 1), dtype = torch.float64)
    while hb.shape[0] < 128:
        hb = torch.cat((torch.cat((hb, hb), 1), torch.cat((hb, -hb), 1)), 0)
    hb = (hb / math.sqrt(128)).to(DEV)
    had = lambda v: (v.view(v.shape[0], -1, 128) @ hb).view(v.shape[0], -1)
    off = [0]
    shard = cs._Shard(os.path.join(args.cold, f"cold-L{L:02d}.safetensors"))
    for s_, r in enumerate(pick[:4]):
        e = perm[n_hot + r]
        k2v = int(st.pool_k2[r]); K = k2v / 2
        row = st.pool[r * st.stride:(r + 1) * st.stride]
        base = f"layers.{L}.ffn.experts.{e}"
        mk = {p_: base + w for p_, w in (("gate", ".w1"), ("up", ".w3"), ("down", ".w2"))}
        mul1 = torch.empty(4, dtype = torch.uint8); shard.read_into(mk["up"] + ".mul1", mul1)
        def tens(proj, j, din, dout):
            sizes = [shard.nbytes(mk[p_] + suf) for p_, suf in cs.SCALES]
            o = sum(sizes[:2 * j])
            suh = row[o:o + din * 2].view(torch.float16).to(DEV).double()
            svh = row[o + din * 2:o + din * 2 + dout * 2].view(torch.float16).to(DEV).double()
            to = st.scale_bytes + j * st.tstride
            tr = row[to:to + k2v * st.tunit].view(torch.int16).view(din // 16, dout // 16, 8 * k2v).to(DEV)
            W = torch.empty((din, dout), dtype = torch.half, device = DEV)
            ext.reconstruct(W, tr, float(K), False, True)
            return suh, svh, W.double()
        g, u, d = tens("gate", 0, H, I), tens("up", 1, H, I), tens("down", 2, I, H)
        lin = lambda x, t: had(had(x * t[0]) @ t[2]) * t[1]
        rows = 4
        gx = torch.Generator().manual_seed(77 + s_)
        x = (torch.randn((rows, H), generator = gx) * 0.5).half().to(DEV)
        a_ = torch.nn.functional.silu(lin(x.double(), g)).clamp(max = 10.0) * lin(x.double(), u).clamp(-10.0, 10.0)
        ref = lin(a_, d) * 0.25
        sel = torch.full((rows, 1), s_, dtype = torch.long, device = DEV)
        rw = torch.full((rows, 1), 0.25, dtype = torch.half, device = DEV)
        o = torch.empty((rows, H), dtype = torch.float, device = DEV)
        sc = Scratch(rows, rows)
        launch(tabs_s, k2_s, x, sel, rw, o, sc, S, False)
        torch.cuda.synchronize()
        rel = ((o.double() - ref).abs().max() / ref.abs().max()).item()
        ok = rel < 3e-3
        print(f"CS_CHECK ref slot={s_} expert={e} k2={k2v} mul1={mul1.view(torch.int32).item() & 0xffffffff:#x} "
              f"max_rel_vs_fp64={rel:.3e} {'PASS' if ok else 'FAIL'}", flush = True)
        if not ok:
            sys.exit(1)
    shard.close()

    # ---- 2. select on real routing ----
    st.armed = True
    st.manager = type("NoProxy", (), {"started": True})()     # sections 1-2 keep the forced slots: no proxy yet
    sel_rows = []
    if args.route:
        z = np.load(args.route)
        ids = z["ids"][:, L, :TOPK].astype(np.int64)
        ph = z["phase"]
        dec = ids[ph == 1][:4000]
        inv = {e: i for i, e in enumerate(perm)}
        sel_rows = np.vectorize(inv.get)(dec)
    else:
        sel_rows = rng.integers(0, E, size = (4000, TOPK))
    resident = {int(n_hot + r): s for s, r in enumerate(pick)}
    bad = 0
    for it in range(50):
        rows = (1, 4, 6, 8, 12, 24)[it % 6]
        base = torch.tensor(sel_rows[it * 24: it * 24 + rows], dtype = torch.long)
        sel = base.clone().to(DEV)
        sel_cpu = torch.where(base >= n_hot, base - n_hot, torch.full_like(base, -1)).reshape(-1).to(DEV)
        before_cpu = sel_cpu.clone()
        st.select(sel, sel_cpu)
        torch.cuda.synchronize()
        out, oc = sel.cpu(), sel_cpu.cpu().view(rows, TOPK)
        # every cold pick: exactly one of (P7 slot id) / (worker id)
        cnt = {}
        for e in base.flatten().tolist():
            if e >= n_hot and e in resident:
                cnt[e] = cnt.get(e, 0) + 1
        ranked = sorted(cnt, key = lambda e: (-cnt[e], e))[:st.s_max]
        for i in range(rows):
            for k in range(TOPK):
                e = int(base[i, k])
                if e < n_hot:
                    if int(out[i, k]) != e or int(oc[i, k]) != -1:
                        bad += 1
                    continue
                stolen = e in ranked
                want_p7 = (n_hot + resident[e]) if stolen else ext_inactive()
                if int(out[i, k]) != want_p7 or (int(oc[i, k]) == -1) != stolen:
                    bad += 1
    print(f"CS_CHECK select calls=50 mismatches={bad} {'PASS' if not bad else 'FAIL'} counters={st.counters()}",
          flush = True)
    if bad:
        sys.exit(1)
    # graph capture + replay
    sel = torch.tensor(sel_rows[:6], dtype = torch.long, device = DEV)
    sel_cpu = torch.empty(36, dtype = torch.long, device = DEV)
    src = sel.clone()
    gr = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        sel.copy_(src); st.select(sel, sel_cpu)
        torch.cuda.synchronize()
        with torch.cuda.graph(gr, stream = s):
            sel.copy_(src)
            st.select(sel, sel_cpu)
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    print(f"CS_CHECK graph replay ok counters={st.counters()}", flush = True)

    # ---- 3. proxy on a fresh layer state: decode-like stream ----
    st.slot_of.fill_(-1)
    for i in range(3):
        st.k2[i, n_hot:] = 6
    mgr = cs.Manager()
    mgr.layers.append(st)
    st.manager = mgr                    # the first select below starts the proxy, as in the model
    n_calls = 0
    stolen0 = st.counters()["stolen_picks"]
    t0 = time.time()
    for it in range(1500):
        rows = 4
        base = torch.tensor(sel_rows[(it * rows) % 3900:(it * rows) % 3900 + rows], dtype = torch.long)
        sel = base.to(DEV)
        sel_cpu = torch.where(base >= n_hot, base - n_hot, torch.full_like(base, -1)).reshape(-1).to(DEV)
        st.select(sel, sel_cpu)
        torch.cuda.synchronize()
        time.sleep(0.0005)              # ~ one layer's worth of other work between calls
        n_calls += 1
    c = st.counters()
    copies, evictions, records, why = ext.coldsteal_proxy_stats(mgr.handle)
    so = st.slot_of.cpu()
    mismatch = 0
    for e in range(n_hot, E):
        s_ = int(so[e])
        if s_ >= 0:
            r = e - n_hot
            tb = int(st.pool_k2[r]) * st.tunit
            a = st.slots[s_ * st.stride: s_ * st.stride + st.scale_bytes + tb].cpu()
            b = st.pool[r * st.stride: r * st.stride + st.scale_bytes + tb]
            mismatch += 0 if torch.equal(a, b) else 1
    share = (c["stolen_picks"] - stolen0) / max(1, c["cold_picks"])
    print(f"CS_CHECK proxy calls={n_calls} copies={copies} evictions={evictions} records={records} "
          f"stolen_share={share:.3f} resident={(so >= 0).sum().item()} slot_mismatch={mismatch} failure={why!r} "
          f"{'PASS' if mismatch == 0 and not why and copies > 0 else 'FAIL'} ({time.time() - t0:.1f}s)", flush = True)
    mgr.close()
    if mismatch or why or not copies:
        sys.exit(1)


def ext_inactive():
    return 0x3fffffff


if __name__ == "__main__":
    main()
