"""
Per-expert mixed bitrates in the fused MoE prefill kernel (ext.exl3_moe with k2 tables): each expert of a projection
carries its own EXL3 rate K in {2, 2.5, 3, 3.5, 4} (mul1). One launch per projection through the runtime-K instance,
checked against a per-expert dense reference (testlib.moe.expert_mlp_ref at the expert's own rates) at batch sizes
16 / 48 / 256. Uniform-K launches stay on the compiled instances (tests/kernels/moe/test_moe_half_rates.py).
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_scale, rand_trellis
from testlib.moe import contiguous_layout, exl3_linear_ref, moe_buffers, ptr_table

RATES = [2.0, 2.5, 3.0, 3.5, 4.0]
CAP = 256


def k2_of(K: float) -> int:
    return int(round(2 * K))               # 2*bits + half


def mixed_experts(E, H, I, Kg, Ku, Kd, gen, device):
    """Per-expert experts: {"g" | "u" | "d": [(trellis, suh, svh)] * E} with expert e at rate K[p][e]"""
    ex = {}
    for p, (k, n), Ks in (("g", (H, I), Kg), ("u", (H, I), Ku), ("d", (I, H), Kd)):
        ex[p] = [(rand_trellis(k, n, Ks[e], gen, device), rand_scale(k, gen, device), rand_scale(n, gen, device))
                 for e in range(E)]
    return ex


def expert_ref(ex, e, x, Ks):
    """Expert e on x; Ks = {"g" | "u" | "d": scalar rate of that expert's projection}"""
    g = exl3_linear_ref(x, *ex["g"][e], Ks["g"], "mul1")
    u = exl3_linear_ref(x, *ex["u"][e], Ks["u"], "mul1")
    ext.silu_mul(g, u, u, 0.0)
    return exl3_linear_ref(u, *ex["d"][e], Ks["d"], "mul1").float()


def ptr_tabs(ex, device):
    return {p: tuple(ptr_table([t[i] for t in ex[p]], device) for i in range(3)) for p in ex}


@pytest.mark.parametrize("bsz", [16, 48, 256])
def test_fused_prefill_mixed_rates(device, bsz):
    H, I, E = 256, 256, 8
    gen = torch.Generator().manual_seed(1000 + bsz)
    Ks = {p: [RATES[int(torch.randint(len(RATES), (1,), generator = gen))] for _ in range(E)] for p in ("g", "u", "d")}
    ex = mixed_experts(E, H, I, Ks["g"], Ks["u"], Ks["d"], gen, device)
    tabs = ptr_tabs(ex, device)
    k2 = {p: torch.tensor([k2_of(K) for K in Ks[p]], dtype = torch.uint8, device = device) for p in Ks}

    # One assignment per token, round-robin over experts: uneven counts stay small so every expert is exercised
    counts = [bsz // E + (1 if e < bsz % E else 0) for e in range(E)]
    A = sum(counts)
    y = (torch.randn((A, H), generator = gen) * 0.05).half().to(device)
    weight_sorted = (torch.rand((A,), generator = gen) * 0.5 + 0.5).half().to(device)
    token_sorted, expert_count, starts = contiguous_layout(counts, device)
    bufs = moe_buffers(H, I, CAP, device)

    out = torch.zeros((A, H), dtype = torch.float, device = device)
    # K scalars are unused when the tables are given (runtime-K instance); the first expert's rate is a placeholder
    ext.exl3_moe(y, out, expert_count, token_sorted, weight_sorted, *bufs, 0, 4, 4, 4,
                 *tabs["g"], *tabs["u"], *tabs["d"], False, True, False, True, False, True, 0.0, -1,
                 None, None, 1, CAP, 16, k2["g"], k2["u"], k2["d"])
    torch.cuda.synchronize()

    ref = torch.empty((A, H), dtype = torch.float, device = device)
    for e in range(E):
        a = int(starts[e])
        rows = slice(a, a + counts[e])
        ref[rows] = expert_ref(ex, e, y[rows], {p: Ks[p][e] for p in Ks}) * weight_sorted[rows].float().unsqueeze(1)
    scale = ref.abs().max().item()
    err = (out - ref).abs().max().item()
    assert err <= 3e-3 * scale, f"bsz {bsz}: err {err:.3e} vs scale {scale:.3e}"
