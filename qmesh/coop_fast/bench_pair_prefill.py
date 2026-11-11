"""
Paired vs plain routed-expert trellis layout on one real P7 hot layer: the decode coop
kernel at rows 1 / 4 / 6 / 8 and the prefill tiers at 2,048 / 8,192 token rows, each arm on the same weights (plain
tensors from the pack, paired copies made with qmesh.pair_layout.pair_permute), outputs checked bit-identical between
arms on every call shape.

Prefill tiers (as BlockSparseMLP.forward routes a chunk on P7): experts with <= FUSED_ROWS (256 with the mul1 row
tiles) assignments run in the fused MoE kernel (ext.exl3_moe, 16 / 32 / 64-row tiles per expert range), heavier ones
in the batched reconstruct tier (ext.reconstruct_had_batch + hgemm, moe_batch_recon.py), one reconstruct per
projection per expert; the bench times both on the layer's real experts with top-6 routing drawn uniformly over the
layer's local hot experts (the router of the hot pack picks among these).

    PYTHONPATH=<tree> python3 bench_pair_prefill.py --pack <hot-pack> --layer 3 \
        --out out/x.json
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../..")))
sys.path.insert(0, HERE)
from qmesh import pair_layout  # noqa: E402

DEV = torch.device("cuda:0")
TOPK = 6
SWIGLU_LIMIT = 10.0
CAP = 256           # fused tier row cap with the mul1 32 / 64-row tiles (FUSED_ROWS_WIDE)
MTILE_T1, MTILE_T2 = 16, 32


def load_layer(pack, layer, expected_layout = "plain"):
    """Routed experts of one layer file: [{p: (trellis, suh, svh)}], rates [K]; p in g (w1) / u (w3) / d (w2)"""
    path = os.path.join(pack, f"layers.{layer}.safetensors")
    pat = re.compile(rf"^layers\.{layer}\.ffn\.experts\.(\d+)\.w1\.trellis$")
    with safe_open(path, framework = "pt", device = "cpu") as f:
        meta = f.metadata() or {}
        assert pair_layout.file_layout(meta) == expected_layout, f"unexpected pack layout: {pair_layout.file_layout(meta)}"
        ids = sorted(int(m.group(1)) for k in f.keys() if (m := pat.match(k)))
        ex, rates = [], []
        for e in ids:
            base = f"layers.{layer}.ffn.experts.{e}"
            d = {p: tuple(f.get_tensor(f"{base}.{w}.{t}").to(DEV) for t in ("trellis", "suh", "svh"))
                 for p, w in (("g", "w1"), ("u", "w3"), ("d", "w2"))}
            ex.append(d)
            rates.append(d["u"][0].shape[-1] / 16)
    return ex, rates


def paired_copy(ex, rates):
    return [{p: ((pair_layout.pair_permute(t, int(rates[i])) if pair_layout.pairable(rates[i]) else t), s, v)
             for p, (t, s, v) in e.items()} for i, e in enumerate(ex)]


def tables(ex):
    return {p: tuple(torch.tensor([e[p][j].data_ptr() for e in ex], dtype = torch.long, device = DEV) for j in range(3))
            for p in ("g", "u", "d")}


def k2_tables(rates):
    k2 = torch.tensor([int(round(2 * r)) for r in rates], dtype = torch.uint8, device = DEV)
    return {"g": k2, "u": k2.clone(), "d": k2.clone()}


def timed(fn, reps, warm = 10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing = True) for _ in range(reps + 1)]
    torch.cuda._sleep(int(50e-3 * 2.0e9))
    ev[0].record()
    for i in range(reps):
        fn()
        ev[i + 1].record()
    torch.cuda.synchronize()
    ts = [ev[i].elapsed_time(ev[i + 1]) * 1e3 for i in range(reps)]
    return statistics.median(ts)


# ---------------------------------------------------------------------------------------------------------------------
# Decode: exl3_moe_coop at bsz rows

def coop_case(ext, ex, rates, k2, bsz, paired, gen, inputs = None):
    E = len(ex)
    H = ex[0]["g"][0].shape[0] * 16
    I = ex[0]["g"][0].shape[1] * 16
    if inputs is None:
        x = (torch.randn((bsz, H), generator = gen) * 0.5).half().to(DEV)
        sel = torch.stack([torch.randperm(E, generator = gen)[:TOPK] for _ in range(bsz)]).to(DEV)
        rw = (torch.rand((bsz, TOPK), generator = gen) * 0.3 + 0.05).half().to(DEV)
    else:
        x, sel, rw = inputs
    slots = bsz * TOPK
    smax = max(slots, min(256, 8 * (TOPK + 1)))
    gu_g = torch.empty((smax, 1, I), dtype = torch.half, device = DEV)
    gu_u = torch.empty_like(gu_g)
    act = torch.empty((smax, 1, I), dtype = torch.half, device = DEV)
    d = torch.empty((smax, 1, H), dtype = torch.float, device = DEV)
    had_g = torch.empty((smax, H), dtype = torch.half, device = DEV)
    had_u = torch.empty_like(had_g)
    ctr = torch.zeros((smax * (I // 128) + bsz * (H // 128) + 2 + (smax + 1) + smax + smax * (I // 128) + 2,),
                      dtype = torch.int, device = DEV)
    out = torch.zeros((bsz, H), dtype = torch.float, device = DEV)
    tabs = tables(ex)

    def fn():
        ext.exl3_moe_coop(x, sel, rw, -1, -1, H, *tabs["g"], *tabs["u"], *tabs["d"], None, None, None,
                          4.0, 4.0, 4.0, False, True, 0, SWIGLU_LIMIT, True,
                          had_g, had_u, gu_g, gu_u, act, d, ctr, out, None, None,
                          k2_g = k2["g"], k2_u = k2["u"], k2_d = k2["d"], paired = paired)
    return fn, out, len(set(sel.flatten().tolist()))


# ---------------------------------------------------------------------------------------------------------------------
# Prefill: routing of a chunk, fused tier (exl3_moe) and batched reconstruct tier

def chunk_routing(E, rows, gen):
    sel = torch.stack([torch.randperm(E, generator = gen)[:TOPK] for _ in range(rows)]).to(DEV)
    w = (torch.rand((rows, TOPK), generator = gen) * 0.3 + 0.05).half().to(DEV)
    flat_e = sel.reshape(-1)
    order = torch.argsort(flat_e, stable = True)
    flat_t = torch.arange(rows, device = DEV).repeat_interleave(TOPK)
    count = torch.bincount(flat_e, minlength = E + 1)
    A = flat_e.numel()
    inv = torch.empty_like(order).scatter_(0, order, torch.arange(A, device = DEV))
    return dict(flat_e = flat_e, inv = inv, token_sorted = flat_t[order], weight_sorted = w.reshape(-1)[order],
                count = count, start = torch.cumsum(count, 0) - count)


def fused_case(ext, ex, rates, k2, rt, rows, paired, gen, inputs = None):
    """Fused tier over the experts with <= CAP assignments (row-tile tiers as BlockSparseMLP launches them),
    deterministic slot + gather accumulation (FUSED_DET, the shipped mode)"""
    E = len(ex)
    H = ex[0]["g"][0].shape[0] * 16
    I = ex[0]["g"][0].shape[1] * 16
    y = inputs if inputs is not None else (torch.randn((rows, H), generator = gen) * 0.05).half().to(DEV)
    cnt = rt["count"].clone()
    cnt_l = cnt.tolist()
    fused = [c if 0 < c <= CAP else 0 for c in cnt_l[:E]]
    kind = torch.tensor([1 if c else 0 for c in fused] + [0], dtype = torch.long, device = DEV)
    fc = cnt * kind
    base = torch.cumsum(fc, 0) - fc
    scratch = torch.empty((max(int(fc.sum()), 1), H), dtype = torch.float, device = DEV)
    out = torch.zeros((rows, H), dtype = torch.float, device = DEV)
    conc = ext.exl3_moe_max_concurrency(0)
    bufs = [torch.empty((conc, CAP, dim), dtype = torch.half, device = DEV) for dim in (H, H, I, I)]
    tabs = tables(ex)
    t1 = sum(1 for c in fused if MTILE_T1 < c <= MTILE_T2)
    t2 = sum(1 for c in fused if c > MTILE_T2)
    t0 = sum(1 for c in fused if 0 < c <= MTILE_T1)
    tiers = [(t2, MTILE_T2 + 1, CAP, 64), (t1, MTILE_T1 + 1, MTILE_T2, 32), (t0, 1, MTILE_T1, 16)]

    def fn():
        out.zero_()
        for n_act, lo, hi, mt in tiers:
            if not n_act:
                continue
            ext.exl3_moe(y, out, cnt, rt["token_sorted"], rt["weight_sorted"], *bufs, 0, 4.0, 4.0, 4.0,
                         *tabs["g"], *tabs["u"], *tabs["d"], False, True, False, True, False, True,
                         SWIGLU_LIMIT, n_act, scratch, base, lo, hi, mt, k2["g"], k2["u"], k2["d"], paired = paired)
        ext.exl3_moe_gather(out, scratch, rt["flat_e"], rt["inv"], rt["start"], base, kind, rt["weight_sorted"])
    heavy = [e for e in range(E) if cnt_l[e] > CAP]
    return fn, out, dict(fused_experts = t0 + t1 + t2, tiers = [t0, t1, t2], heavy_experts = len(heavy))


def recon_case(ext, ex, rates, rt, paired):
    """Weight-reconstruct work of the batched tier: every heavy expert's three projections reconstructed with both
    Hadamards folded (reconstruct_had_batch, one launch per projection over the heavy experts of one rate). The hgemm
    that follows reads the dense weights and is layout-independent, so it is left out"""
    E = len(ex)
    cnt_l = rt["count"].tolist()
    heavy = [e for e in range(E) if cnt_l[e] > CAP] or list(range(min(E, 8)))
    by_rate = {}
    for e in heavy:
        by_rate.setdefault(rates[e], []).append(e)
    jobs = []
    for K, es in by_rate.items():
        for p in ("g", "u", "d"):
            t0 = ex[es[0]][p][0]
            k, n = t0.shape[0] * 16, t0.shape[1] * 16
            W = torch.empty((len(es), k, n), dtype = torch.half, device = DEV)
            ptr = lambda j: torch.tensor([ex[e][p][j].data_ptr() for e in es], dtype = torch.long, device = DEV)
            jobs.append((W, ptr(0), ptr(1), ptr(2), float(K)))

    def fn():
        for W, pt, ps, pv, K in jobs:
            ext.reconstruct_had_batch(W, pt, ps, pv, K, False, True, paired = paired)
    return fn, [j[0] for j in jobs], dict(heavy_experts = len(heavy), recon_launches = len(jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required = True)
    ap.add_argument("--layer", type = int, default = 3)
    ap.add_argument("--out", required = True)
    ap.add_argument("--decode-rows", type = int, nargs = "+", default = [1, 4, 6, 8])
    ap.add_argument("--prefill-rows", type = int, nargs = "+", default = [2048, 8192])
    ap.add_argument("--reps", type = int, default = 300)
    ap.add_argument("--prefill-reps", type = int, default = 30)
    a = ap.parse_args()
    # Full extension of this tree (exl3_moe / reconstruct / coop).
    from exllamav3.ext import exllamav3_ext as ext
    assert getattr(ext, "HAS_COOP_PAIR", False), "extension without paired support"
    ex, rates = load_layer(a.pack, a.layer)
    exp = paired_copy(ex, rates)
    k2 = k2_tables(rates)
    res = dict(meta = dict(args = vars(a), gpu = torch.cuda.get_device_name(0), experts = len(ex),
                           rates = {str(r): rates.count(r) for r in sorted(set(rates))}), decode = [], prefill = [])
    print(res["meta"], flush = True)

    for bsz in a.decode_rows:
        outs, t = [], {}
        for name, e_set, pr in (("plain", ex, False), ("paired", exp, True)):
            fn, out, distinct = coop_case(ext, e_set, rates, k2, bsz, pr, torch.Generator().manual_seed(1000 + bsz))
            fn(); torch.cuda.synchronize(); outs.append(out.clone())
            t[name] = timed(fn, a.reps)
        r = dict(rows = bsz, distinct = distinct, plain_us = t["plain"], paired_us = t["paired"],
                 delta = t["paired"] / t["plain"] - 1, bit_identical = bool(torch.equal(outs[0], outs[1])))
        res["decode"].append(r)
        print("[decode]", json.dumps(r), flush = True)
        json.dump(res, open(a.out, "w"), indent = 1)

    for rows in a.prefill_rows:
        rt = chunk_routing(len(ex), rows, torch.Generator().manual_seed(2000 + rows))
        for tier, mk in (("fused", lambda e_set, pr: fused_case(ext, e_set, rates, k2, rt, rows, pr,
                                                                torch.Generator().manual_seed(3000 + rows))),
                         ("recon", lambda e_set, pr: recon_case(ext, e_set, rates, rt, pr))):
            outs, t, info = [], {}, None
            for name, e_set, pr in (("plain", ex, False), ("paired", exp, True)):
                fn, out, info = mk(e_set, pr)
                fn(); torch.cuda.synchronize()
                outs.append([o.clone() for o in out] if isinstance(out, list) else out.clone())
                t[name] = timed(fn, a.prefill_reps, warm = 3)
            same = all(torch.equal(x, y) for x, y in zip(outs[0], outs[1])) if isinstance(outs[0], list) \
                else bool(torch.equal(outs[0], outs[1]))
            r = dict(rows = rows, tier = tier, plain_ms = t["plain"] / 1e3, paired_ms = t["paired"] / 1e3,
                     delta = t["paired"] / t["plain"] - 1, bit_identical = same, **info)
            res["prefill"].append(r)
            print("[prefill]", json.dumps(r), flush = True)
            json.dump(res, open(a.out, "w"), indent = 1)
    print("done", flush = True)


if __name__ == "__main__":
    main()
