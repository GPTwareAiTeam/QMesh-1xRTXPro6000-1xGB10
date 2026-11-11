"""
Paired routed-expert trellis layout (qmesh/pair_layout.py, quant/trellis_pair.cuh, notes/CoopKernel2.md §11): every
routed-expert entry point given `paired = True` on pair_permute'd trellises must produce bit-identical outputs to the
plain call on the original trellises (the permutation only moves words; decode and arithmetic order are unchanged):
reconstruct / reconstruct_slice / reconstruct_had_slice / reconstruct_batch / reconstruct_had_batch, the fused MoE
kernel exl3_moe (16 / 32 / 64-row tiles, N = 128 / 256, uniform and per-expert mixed rates) and the decode coop kernel
exl3_moe_coop (uniform and mixed). Rates outside 2 / 3 / 4 bpw stay plain inside a paired launch. Plus the CPU side:
the permutation round-trips exactly and the in-place byte form equals the tensor form.
"""
import os
import sys

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_experts, rand_scale, rand_trellis
from testlib.moe import expert_ptr_tables, experts_to, moe_buffers, ptr_table, routing_layout

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "../../..")))
from qmesh import pair_layout  # noqa: E402

PAIRABLE = [2, 3, 4]


def paired_experts(ex, Ks):
    """Copy of an expert set {p: [(trellis, suh, svh)]} with every pairable trellis permuted; Ks[p][e] = rate"""
    return {p: [((pair_layout.pair_permute(t, int(Ks[p][e])) if pair_layout.pairable(Ks[p][e]) else t), s, v)
                for e, (t, s, v) in enumerate(ex[p])] for p in ex}


@pytest.mark.nogpu
@pytest.mark.parametrize("K", PAIRABLE)
def test_permutation_round_trip(K):
    g = torch.Generator().manual_seed(K)
    for ks, nt in ((320, 144), (144, 320), (2, 2), (8, 6)):
        t = rand_trellis(ks * 16, nt * 16, K, g)
        p = pair_layout.pair_permute(t, K)
        assert p.shape == t.shape and p.dtype == t.dtype
        assert torch.equal(pair_layout.pair_unpermute(p, K), t)
        b = t.clone().view(torch.uint8).view(-1)
        pair_layout.pair_permute_bytes_(b, ks, nt, K)
        assert torch.equal(b.view(torch.int16).view(t.shape), p)


@pytest.mark.parametrize("K", PAIRABLE)
@pytest.mark.parametrize("cb", [0, 1, 2])
def test_reconstruct_paired(device, K, cb):
    g = torch.Generator().manual_seed(100 + 10 * K + cb)
    k, n = 512, 768
    t = rand_trellis(k, n, K, g, device)
    p = pair_layout.pair_permute(t, K)
    mcg, mul1 = cb == 1, cb == 2
    suh, svh = rand_scale(k, g, device), rand_scale(n, g, device)

    def both(fn):
        a = torch.empty((k, n), dtype = torch.half, device = device)
        b = torch.empty_like(a)
        fn(a, t, False)
        fn(b, p, True)
        torch.cuda.synchronize()
        assert torch.equal(a, b)

    both(lambda w, tr, pr: ext.reconstruct(w, tr, float(K), mcg, mul1, pr))
    both(lambda w, tr, pr: ext.reconstruct_had_slice(w, tr, suh, svh, float(K), mcg, mul1, 0, pr))
    # Column slices start on 128-column (8-tile) boundaries, i.e. on whole paired chunks
    for fn in (lambda w, tr, pr: ext.reconstruct_slice(w, tr, float(K), mcg, mul1, 256, pr),
               lambda w, tr, pr: ext.reconstruct_had_slice(w, tr, suh, svh[256:512], float(K), mcg, mul1, 256, pr)):
        a = torch.empty((k, 256), dtype = torch.half, device = device)
        b = torch.empty_like(a)
        fn(a, t, False)
        fn(b, p, True)
        torch.cuda.synchronize()
        assert torch.equal(a, b)


@pytest.mark.parametrize("K", PAIRABLE)
def test_reconstruct_batch_paired(device, K):
    g = torch.Generator().manual_seed(200 + K)
    B, k, n = 3, 256, 384
    ts = [rand_trellis(k, n, K, g, device) for _ in range(B)]
    ps = [pair_layout.pair_permute(t, K) for t in ts]
    suh = [rand_scale(k, g, device) for _ in range(B)]
    svh = [rand_scale(n, g, device) for _ in range(B)]
    a = torch.empty((B, k, n), dtype = torch.half, device = device)
    b = torch.empty_like(a)
    ext.reconstruct_batch(a, ptr_table(ts, device), float(K), False, True)
    ext.reconstruct_batch(b, ptr_table(ps, device), float(K), False, True, True)
    torch.cuda.synchronize()
    assert torch.equal(a, b)
    ext.reconstruct_had_batch(a, ptr_table(ts, device), ptr_table(suh, device), ptr_table(svh, device), float(K), False, True)
    ext.reconstruct_had_batch(b, ptr_table(ps, device), ptr_table(suh, device), ptr_table(svh, device), float(K), False, True,
                              True)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


def test_reconstruct_paired_plain_rate(device):
    """A paired launch at a rate outside 2-4 bpw reads the plain layout (the pack keeps those rates plain)"""
    g = torch.Generator().manual_seed(300)
    t = rand_trellis(256, 256, 5, g, device)
    a = torch.empty((256, 256), dtype = torch.half, device = device)
    b = torch.empty_like(a)
    ext.reconstruct(a, t, 5.0, False, True)
    ext.reconstruct(b, t, 5.0, False, True, True)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


def _moe_det(y, lay, bufs, tabs, E, K, cap, m_tile, k2 = None, paired = False):
    """Fused MoE through the deterministic slot + gather path (bit-reproducible: every assignment's slot is written
    by its expert's GEMM alone, then summed in k order)"""
    T, H = y.shape
    kind = ((lay.expert_count > 0) & (lay.expert_count <= cap)).long()
    kind[E:] = 0
    fc = lay.expert_count * kind
    base = torch.cumsum(fc, 0) - fc
    scratch = torch.full((max(int(fc.sum()), 1), H), float("nan"), dtype = torch.float, device = y.device)
    out = torch.zeros((T, H), dtype = torch.float, device = y.device)
    ext.exl3_moe(y, out, lay.expert_count, lay.token_sorted, lay.weight_sorted, *bufs, 0, K, K, K,
                 *tabs["g"], *tabs["u"], *tabs["d"], False, True, False, True, False, True, 0.0, -1,
                 scratch, base, 1, cap, m_tile,
                 k2["g"] if k2 else None, k2["u"] if k2 else None, k2["d"] if k2 else None, paired = paired)
    ext.exl3_moe_gather(out, scratch, lay.flat_e, lay.inv, lay.expert_start, base, kind, lay.weight_sorted)
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("K", PAIRABLE)
@pytest.mark.parametrize("dims", [(256, 512), (512, 256)])   # (H, I): N = 256 and N = 128 shapes
@pytest.mark.parametrize("m_tile", [16, 32, 64])
def test_fused_moe_paired(device, K, dims, m_tile):
    H, I = dims
    E, TOPK, T, CAP = 6, 2, 160, 128
    gen = torch.Generator().manual_seed(400 + K * 7 + m_tile)
    ex = experts_to(rand_experts(E, H, I, K, gen), device)
    exp = paired_experts(ex, {p: [K] * E for p in ex})
    y = (torch.randn((T, H), generator = gen) * 0.05).half().to(device)
    sel = torch.stack([torch.randperm(E, generator = gen)[:TOPK] for _ in range(T)]).to(device)
    w = torch.rand((T, TOPK), generator = gen).half().to(device)
    lay = routing_layout(sel, w, E)
    bufs = moe_buffers(H, I, CAP, device)
    a = _moe_det(y, lay, bufs, expert_ptr_tables(ex, device), E, K, CAP, m_tile)
    b = _moe_det(y, lay, bufs, expert_ptr_tables(exp, device), E, K, CAP, m_tile, paired = True)
    assert torch.equal(a, b)
    assert torch.equal(a, _moe_det(y, lay, bufs, expert_ptr_tables(ex, device), E, K, CAP, m_tile))


def test_fused_moe_paired_mixed(device):
    """Per-expert rates (runtime-K instance): 2 / 3 / 4 bpw experts paired, 2.5 / 5 bpw plain in the same launch"""
    H, I, E, TOPK, T, CAP = 256, 256, 5, 2, 96, 128
    rates = [2.0, 3.0, 4.0, 2.5, 5.0]
    gen = torch.Generator().manual_seed(500)
    ex = {}
    for p, (k, n) in (("g", (H, I)), ("u", (H, I)), ("d", (I, H))):
        ex[p] = [(rand_trellis(k, n, rates[e], gen, device), rand_scale(k, gen, device), rand_scale(n, gen, device))
                 for e in range(E)]
    exp = paired_experts(ex, {p: rates for p in ex})
    k2 = {p: torch.tensor([int(round(2 * r)) for r in rates], dtype = torch.uint8, device = device) for p in ex}
    y = (torch.randn((T, H), generator = gen) * 0.05).half().to(device)
    sel = torch.stack([torch.randperm(E, generator = gen)[:TOPK] for _ in range(T)]).to(device)
    w = torch.rand((T, TOPK), generator = gen).half().to(device)
    lay = routing_layout(sel, w, E)
    bufs = moe_buffers(H, I, CAP, device)
    # Scalar K is a placeholder when the k2 tables are given (runtime-K instance), as in test_moe_fused_mixed.py
    a = _moe_det(y, lay, bufs, expert_ptr_tables(ex, device), E, 4.0, CAP, 16, k2 = k2)
    b = _moe_det(y, lay, bufs, expert_ptr_tables(exp, device), E, 4.0, CAP, 16, k2 = k2, paired = True)
    assert torch.equal(a, b)


def _coop(ex, Ks, x, sel, rw, H, I, paired, k2 = None):
    bsz, topk = sel.shape
    slots = bsz * topk
    smax = 4 * slots
    dev = x.device
    gu_g = torch.empty((smax, 1, I), dtype = torch.half, device = dev)
    gu_u = torch.empty_like(gu_g)
    act = torch.empty((smax, 1, I), dtype = torch.half, device = dev)
    d = torch.empty((smax, 1, H), dtype = torch.float, device = dev)
    out = torch.full((bsz, H), float("nan"), dtype = torch.float, device = dev)
    ctr = torch.zeros((2 * smax * (I // 128) + 8 * (H // 128) + 2 * smax + 5,), dtype = torch.int, device = dev)
    had_g = torch.empty((slots, H), dtype = torch.half, device = dev)
    had_u = torch.empty_like(had_g)
    tabs = expert_ptr_tables(ex, dev)
    kw = dict(k2_g = k2["g"], k2_u = k2["u"], k2_d = k2["d"]) if k2 else {}
    ext.exl3_moe_coop(x, sel, rw, -1, -1, H, *tabs["g"], *tabs["u"], *tabs["d"], None, None, None,
                      Ks, Ks, Ks, False, True, 0, 0.0, True,
                      had_g, had_u, gu_g, gu_u, act, d, ctr, out, None, None, paired = paired, **kw)
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("K", PAIRABLE)
@pytest.mark.parametrize("bsz", [1, 4, 6, 8])
def test_coop_paired(device, K, bsz):
    # The DSV4.1 routed shapes' k ranges (Hi / 16 = 320, I / 16 = 144) are slice-pair aligned; smaller dims of the
    # same parity are too (H 512, I 256)
    H, I, E, TOPK = 512, 256, 16, 6
    gen = torch.Generator().manual_seed(600 + K * 10 + bsz)
    ex = experts_to(rand_experts(E, H, I, K, gen), device)
    exp = paired_experts(ex, {p: [K] * E for p in ex})
    x = (torch.randn((bsz, H), generator = gen) * 0.5).half().to(device)
    sel = torch.stack([torch.randperm(E, generator = gen)[:TOPK] for _ in range(bsz)]).to(device)
    rw = (torch.rand((bsz, TOPK), generator = gen) * 0.3 + 0.05).half().to(device)
    assert torch.equal(_coop(ex, float(K), x, sel, rw, H, I, False), _coop(exp, float(K), x, sel, rw, H, I, True))


@pytest.mark.parametrize("bsz", [1, 6])
def test_coop_paired_mixed(device, bsz):
    H, I, E, TOPK = 512, 256, 10, 6
    rates = [2.0, 3.0, 4.0, 2.5, 3.5] * 2
    gen = torch.Generator().manual_seed(700 + bsz)
    ex = {}
    for p, (k, n) in (("g", (H, I)), ("u", (H, I)), ("d", (I, H))):
        ex[p] = [(rand_trellis(k, n, rates[e], gen, device), rand_scale(k, gen, device), rand_scale(n, gen, device))
                 for e in range(E)]
    exp = paired_experts(ex, {p: rates for p in ex})
    k2 = {p: torch.tensor([int(round(2 * r)) for r in rates], dtype = torch.uint8, device = device) for p in ex}
    x = (torch.randn((bsz, H), generator = gen) * 0.5).half().to(device)
    sel = torch.stack([torch.randperm(E, generator = gen)[:TOPK] for _ in range(bsz)]).to(device)
    rw = (torch.rand((bsz, TOPK), generator = gen) * 0.3 + 0.05).half().to(device)
    a = _coop(ex, 4.0, x, sel, rw, H, I, False, k2)
    b = _coop(exp, 4.0, x, sel, rw, H, I, True, k2)
    assert torch.equal(a, b)


def test_coop_paired_refuses_odd_k_ranges(device):
    """A shape whose warp k ranges split a slice pair is refused, not run: Hi / 16 = 24 slices over the 8 warps of the
    narrow bsz-1 gate/up tile gives 3 slices per warp"""
    H, I, E = 384, 256, 4
    gen = torch.Generator().manual_seed(800)
    ex = experts_to(rand_experts(E, H, I, 3, gen), device)
    x = (torch.randn((1, H), generator = gen) * 0.5).half().to(device)
    sel = torch.tensor([[0, 1]], device = device)
    rw = torch.full((1, 2), 0.25, dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "slice-pair"):
        _coop(ex, 3.0, x, sel, rw, H, I, True)
