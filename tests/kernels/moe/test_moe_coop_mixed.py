"""
Per-expert mixed-bitrate fused decode MoE (ext.exl3_moe_coop with k2 tables): each local expert of a projection has
its own EXL3 rate K in {2, 2.5, 3, 3.5, 4} (mul1 codebook), dispatched per block inside the *_mixed kernels.
Checked against a per-expert dense reference (testlib.moe.exl3_linear_ref at the expert's own K), plus bit
reproducibility. Uniform-K layers stay on the templated kernels (tests/kernels/moe/test_moe_coop.py).
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import act_ref, rand_scale, rand_trellis
from testlib.moe import exl3_linear_ref, ptr_table

ACTS = {"silu": 0, "gelu": 1, "relu2": 2, "swiglu_oai": 3}
RATES = [2.0, 2.5, 3.0, 3.5, 4.0]          # instantiated set (half-bit k2 = 4..8)


def k2_of(K: float) -> int:
    return int(round(2 * K))               # 2*bits + half


class MixedProj:
    """One projection of E experts, expert e at rate Ks[e] (mul1)"""

    def __init__(self, k, n, Ks, bias, gen, device):
        self.Ks = list(Ks)
        E = len(Ks)
        self.trellis = [rand_trellis(k, n, Ks[e], gen, device) for e in range(E)]
        self.suh = [rand_scale(k, gen, device) for _ in range(E)]
        self.svh = [rand_scale(n, gen, device) for _ in range(E)]
        self.bias = [(torch.randn((n,), generator = gen) * 0.1).half().to(device) for _ in range(E)] if bias else None
        self.tabs = (ptr_table(self.trellis, device), ptr_table(self.suh, device), ptr_table(self.svh, device))
        self.bias_tab = ptr_table(self.bias, device) if bias else None
        self.k2 = torch.tensor([k2_of(K) for K in Ks], dtype = torch.uint8, device = device)

    def ref(self, x, e):
        y = exl3_linear_ref(x, self.trellis[e], self.suh[e], self.svh[e], self.Ks[e], "mul1").float()
        if self.bias is not None:
            y = y + self.bias[e].float()
        return y


def run_mixed(device, seed, Kg, Ku, Kd, act = "silu", gated = True, bias = False, limit = 0.0,
              H = 256, Hi = 256, I = 256, Ho = 256, H_out = 256, E = 6, topk = 3, bsz = 5,
              min_e = -1, max_e = -1, shared = None, tol = 3e-3):
    """Kg / Ku / Kd: per-expert rate lists (length E) for gate, up, down"""
    gen = torch.Generator().manual_seed(seed)
    gate = MixedProj(Hi, I, Kg, bias, gen, device) if gated else None
    up = MixedProj(Hi, I, Ku, bias, gen, device)
    down = MixedProj(I, Ho, Kd, bias, gen, device)
    x = (torch.randn((bsz, H), generator = gen) * 0.05).half().to(device)
    n_global = E if min_e < 0 else E + 4
    sel = torch.stack([torch.randperm(n_global, generator = gen)[:topk] for _ in range(bsz)]).to(device)
    rw = (torch.rand((bsz, topk), generator = gen) * 0.9 + 0.1).half().to(device)
    if bsz * topk > 1:
        rw[0, 0] = 0.0
    first = 0 if min_e < 0 else min_e

    xp = torch.zeros((bsz, Hi), dtype = torch.half, device = device)
    xp[:, :H] = x
    ref = torch.zeros((bsz, H_out), dtype = torch.float, device = device)
    for b in range(bsz):
        for k in range(topk):
            e = int(sel[b, k]); w = float(rw[b, k])
            if min_e >= 0 and not (min_e <= e < max_e):
                continue
            el = e - first
            u = up.ref(xp[b:b + 1], el)
            g = gate.ref(xp[b:b + 1], el) if gated else u
            a = act_ref(act, gated, g, u, limit).half()
            d = down.ref(a, el)
            ref[b] += w * d[0, :H_out]
    sh_out = sh_w = None
    if shared is not None:
        sh_out = torch.randn((bsz, H), generator = gen).float().to(device)
        if shared == "gated":
            sh_w = (torch.randn((H,), generator = gen) * 0.05).half().to(device)
            ref += torch.sigmoid((x.float() * sh_w.float()).sum(1, keepdim = True)) * sh_out
        else:
            ref += sh_out

    slots = bsz * topk
    smax = 4 * slots
    gu_g = torch.empty((smax, 1, I), dtype = torch.half, device = device)
    gu_u = torch.empty((smax, 1, I), dtype = torch.half, device = device)
    act_out = torch.empty((smax, 1, I), dtype = torch.half, device = device)
    d_out = torch.empty((smax, 1, Ho), dtype = torch.float, device = device)
    out = torch.full((8, H_out), float("nan"), dtype = torch.float, device = device)
    ctr = torch.zeros((smax * (I // 128) + 8 * (Ho // 128) + 2 * smax + 3,), dtype = torch.int, device = device)
    had_g = torch.empty((slots, Hi), dtype = torch.half, device = device)
    had_u = torch.empty_like(had_g)
    gtabs = gate.tabs if gated else up.tabs
    gk2 = gate.k2 if gated else up.k2
    gbias = gate.bias_tab if gated else None
    # The mixed path reads the k2 tables; the uniform scalar K arguments are ignored there (passed as the
    # first expert's rate). Tables must be set for every projection of a mixed layer.
    kg0, ku0, kd0 = Kg[0], Ku[0], Kd[0]

    def run():
        ext.exl3_moe_coop(
            x, sel, rw, min_e, max_e, Hi,
            *gtabs, *up.tabs, *down.tabs,
            gbias, up.bias_tab, down.bias_tab,
            kg0, ku0, kd0, False, True,
            ACTS[act], limit, gated,
            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out[:bsz],
            sh_out, sh_w, gk2, up.k2, down.k2,
        )
        torch.cuda.synchronize()
        return out[:bsz].clone()

    o1 = run()
    o2 = run()
    assert torch.equal(o1, o2), "not bit-reproducible"
    assert torch.isfinite(o1).all()
    scale = ref.abs().max().item()
    err = (o1 - ref).abs().max().item()
    assert err <= tol * max(scale, 1e-12), f"max err {err:.3e} vs scale {scale:.3e} (Kg {Kg} Ku {Ku} Kd {Kd})"


@pytest.mark.parametrize("act", ["silu", "gelu", "relu2", "swiglu_oai"])
def test_mixed_per_expert_acts(device, act):
    Ks = [RATES[i % 5] for i in range(6)]
    run_mixed(device, 31, Ks, Ks, Ks, act = act, limit = 1.0 if act == "swiglu_oai" else 0.0)


def test_mixed_all_rates_one_layer(device):
    # every instantiated rate appears in gate, up and down (independent per projection)
    run_mixed(device, 32, [2.0, 2.5, 3.0, 3.5, 4.0, 2.5], [4.0, 3.5, 2.5, 3.0, 2.0, 3.5], [3.0, 2.0, 4.0, 2.5, 3.5, 2.0])


def test_mixed_gateless_bias(device):
    Ks = [2.5, 3.5, 2.0, 4.0, 3.0, 2.5]
    run_mixed(device, 33, Ks, Ks, Ks, act = "relu2", gated = False, bias = True)


def test_mixed_expert_range(device):
    # Local tables cover global experts [4, 10); picks outside contribute zero
    Ks = [2.5, 3.5, 2.0, 4.0, 3.0, 2.5]
    run_mixed(device, 34, Ks, Ks, Ks, min_e = 4, max_e = 10)


def test_mixed_duplicates_and_batch(device):
    Ks = [2.5, 3.5, 4.0]
    run_mixed(device, 35, Ks, Ks, Ks, E = 3, topk = 3, bsz = 8)
    run_mixed(device, 36, [2.0, 3.0], [3.5, 2.5], [4.0, 2.0], E = 2, topk = 2, bsz = 1)


def test_mixed_shared(device):
    Ks = [2.5, 3.0, 3.5, 2.0, 4.0, 2.5]
    run_mixed(device, 37, Ks, Ks, Ks, shared = "gated", H = 200, H_out = 200, Hi = 256, Ho = 256)
    run_mixed(device, 38, Ks, Ks, Ks, shared = "plain")
