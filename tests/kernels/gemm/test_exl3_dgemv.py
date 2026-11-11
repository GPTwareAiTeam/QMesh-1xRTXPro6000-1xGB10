"""
Dense decode GEMV (quant/exl3_dgemv.cu, EXL3_DGEMV): the ordinary-launch split-K GEMV that exl3_gemm and
exl3_mgemm take ahead of every other path when the knob is on. mul1 codebook, integer K in {4, 5, 6, 8},
1..48 rows, k and n multiples of 128.

- every (K, rows, c_fp32) the runtime can ask for has an instance: a missing one fails here, never
  silently falls back to the vendor kernel
- against the fp64 trellis reference at rows 1 / 4 / 6 / 24 / 48 and every K, through exl3_gemm with the
  path forced (mode 2), half and fp32 output
- split-K and single-split shapes, repeat launches bit-identical (deterministic reduction order)
- a fan (exl3_mgemm with per-matrix widths, and with one input per matrix) equals the single launches
  bit for bit
- on by default (mode 1, the measured envelope): unset, an in-envelope call equals the forced kernel and an
  out-of-envelope one the vendor path, bit for bit
"""

import os

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref
from testlib.env import is_rocm
from testlib.exl3 import generator, rand_scale

pytestmark = pytest.mark.skipif(is_rocm(), reason = "CUDA-only kernel")

KS = [4, 5, 6, 8]
ROWS = [1, 4, 6, 24, 48]
SHAPES = [(1024, 512), (512, 1024), (2048, 384)]   # (k, n): split-K, wide-n, 3-column-block


def _weights(k, n, K, seed):
    g = generator(seed)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(seed)).reshape(k // 16, n // 16, -1)
    return p, rand_scale(k, g), rand_scale(n, g), g


def _gemm(A, tr, suh, svh, n, fp32, mode):
    ext.exl3_dgemv_set_enabled(mode)
    try:
        C = torch.empty(A.shape[0], n, dtype = torch.float if fp32 else torch.half, device = A.device)
        ext.exl3_gemm(A, tr, C, suh, torch.empty_like(A), svh, -1, False, True, 0)
        return C
    finally:
        ext.exl3_dgemv_set_enabled(-1)


def _assert_close(C, ref):
    e = C.cpu().double() - ref
    rel_rms = (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    rel_max = (e.abs().max() / ref.abs().max()).item()
    assert rel_rms < 2.5e-3 and rel_max < 4e-3, f"relative RMS {rel_rms:.2e}, max {rel_max:.2e}"


@pytest.mark.parametrize("K", KS)
def test_instances_exist(K):
    for m in range(1, 49):
        for fp32 in (False, True):
            assert ext.exl3_dgemv_has(K, 2, m, fp32), f"missing exl3_dgemv instance K={K} rows={m} fp32={fp32}"
    assert not ext.exl3_dgemv_has(K, 1, 4, False), "mcg must not be claimed"
    assert not ext.exl3_dgemv_has(K, 2, 49, False), "rows > 48 must not be claimed"


@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("k, n", SHAPES)
@torch.inference_mode()
def test_matches_reference(device, K, m, k, n):
    p, suh, svh, g = _weights(k, n, K, K * 100000 + m * 1000 + k + n)
    A = (torch.randn(m, k, generator = g) * 0.5).half()
    ref = tref.linear(A, p, suh, svh, K, "mul1")
    tr = torch.from_numpy(p.copy()).to(device)
    Ad, sd, vd = A.to(device), suh.to(device), svh.to(device)
    for fp32 in (False, True):
        C = _gemm(Ad, tr, sd, vd, n, fp32, 2)
        _assert_close(C, ref)
        C2 = _gemm(Ad, tr, sd, vd, n, fp32, 2)
        assert torch.equal(C, C2), "repeat launch differs"


@pytest.mark.skipif(bool(os.environ.get("EXL3_DGEMV")), reason = "EXL3_DGEMV set in the environment")
@torch.inference_mode()
def test_default_is_envelope(device):
    """Unset, the knob is mode 1: a 4-row 1024 x 512 call (inside the envelope) takes the new kernel,
    bit-identical to forcing it (mode 2); a 49-row call is outside the hard limit and stays vendor"""
    k, n, K = 1024, 512, 6
    p, suh, svh, g = _weights(k, n, K, 7)
    tr = torch.from_numpy(p.copy()).to(device)
    sd, vd = suh.to(device), svh.to(device)
    assert ext.exl3_dgemv_enabled()
    A = (torch.randn(4, k, generator = g) * 0.5).half().to(device)
    assert torch.equal(_gemm(A, tr, sd, vd, n, False, -1), _gemm(A, tr, sd, vd, n, False, 2))
    A49 = (torch.randn(49, k, generator = g) * 0.5).half().to(device)
    assert torch.equal(_gemm(A49, tr, sd, vd, n, False, -1), _gemm(A49, tr, sd, vd, n, False, 0))


def _ptrs(ts, device):
    return torch.tensor([t.data_ptr() for t in ts], dtype = torch.long, device = device)


@pytest.mark.parametrize("K", [5, 6])
@pytest.mark.parametrize("m", [1, 6, 24])
@torch.inference_mode()
def test_fans_equal_single_launches(device, K, m):
    k = 1024
    ns = [1024, 512, 256]
    mats = [_weights(k, n, K, 31 * K + i) for i, n in enumerate(ns)]
    tr = [torch.from_numpy(p.copy()).to(device) for p, _, _, _ in mats]
    su = [s.to(device) for _, s, _, _ in mats]
    sv = [v.to(device) for _, _, v, _ in mats]
    g = generator(m)
    x = (torch.randn(1, m, k, generator = g) * 0.5).half().to(device)

    # shared input, per-matrix widths (the x-side projection fan)
    outs = [torch.empty(m, n, dtype = torch.half, device = device) for n in ns]
    ext.exl3_dgemv_set_enabled(2)
    try:
        ext.exl3_mgemm(
            x, _ptrs(tr, device), outs[0].view(1, m, -1), _ptrs(su, device), torch.empty(3, m, k, dtype = torch.half, device = device),
            _ptrs(sv, device), torch.arange(3, dtype = torch.long, device = device).unsqueeze(0), None, K, -1, False, True,
            -1, -1, 0, 1, torch.tensor(ns, dtype = torch.int32, device = device), _ptrs(outs, device))
        for i, n in enumerate(ns):
            C = _gemm(x[0], tr[i], su[i], sv[i], n, False, 2)
            assert torch.equal(C, outs[i]), f"fan output {i} differs from its single launch"

        # one input per matrix, uniform width (wo_a's grouped slices)
        trs = tr[:1] * 3
        xs = (torch.randn(3, m, k, generator = g) * 0.5).half().to(device)
        C3 = torch.empty(3, m, ns[0], dtype = torch.half, device = device)
        ext.exl3_dgemv_set_enabled(2)    # _gemm above restored the env mode
        ext.exl3_mgemm(
            xs, _ptrs(trs, device), C3, _ptrs(su[:1] * 3, device), torch.empty(3, m, k, dtype = torch.half, device = device),
            _ptrs(sv[:1] * 3, device), None, None, K, -1, False, True, -1, -1, 0, 1, None, None)
        for j in range(3):
            C = _gemm(xs[j], tr[0], su[0], sv[0], ns[0], False, 2)
            assert torch.equal(C, C3[j]), f"per-input slot {j} differs from its single launch"
    finally:
        ext.exl3_dgemv_set_enabled(-1)


@pytest.mark.parametrize("m", [4, 24])
@torch.inference_mode()
def test_graph_replay_matches_eager(device, m):
    """The workspace pointer is baked into a captured graph and the split counters self-reset: replays
    must reproduce the eager result bit for bit, repeatedly"""
    k, n, K = 2048, 512, 6
    p, suh, svh, g = _weights(k, n, K, 99 + m)
    A = (torch.randn(m, k, generator = g) * 0.5).half().to(device)
    tr = torch.from_numpy(p.copy()).to(device)
    sd, vd = suh.to(device), svh.to(device)
    ref = _gemm(A, tr, sd, vd, n, False, 2)
    C = torch.empty(m, n, dtype = torch.half, device = device)
    Ah = torch.empty_like(A)
    ext.exl3_dgemv_set_enabled(2)
    try:
        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            ext.exl3_gemm(A, tr, C, sd, Ah, vd, -1, False, True, 0)
            torch.cuda.synchronize(device)
            with torch.cuda.graph(gr, stream = s):
                ext.exl3_gemm(A, tr, C, sd, Ah, vd, -1, False, True, 0)
        for _ in range(3):
            C.zero_()
            gr.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(C, ref)
    finally:
        ext.exl3_dgemv_set_enabled(-1)
