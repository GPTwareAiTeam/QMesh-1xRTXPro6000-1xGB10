import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext


def reference(x, idx):
    group = 32 if idx else 16
    z = x.float().reshape(x.shape[0], -1, group)
    a = z.abs().amax(-1)
    if idx:
        v = a.clamp_min(6 * 2**-126) * (1 / 6)
        bits = v.contiguous().view(torch.int32)
        exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).int()
        scales = exponent.to(torch.uint8)
        s = (exponent << 23).contiguous().view(torch.float32)
    else:
        # CUDA E4M3 conversion uses satfinite, unlike torch's overflow-to-NaN cast.
        s8 = (a.clamp_min(6 * 2**-9) / 6).clamp_max(448).to(torch.float8_e4m3fn)
        scales = s8.view(torch.uint8)
        s = s8.float()
    y = z / s.unsqueeze(-1)
    mag = y.abs()
    code = torch.zeros_like(mag, dtype=torch.uint8)
    for i, midpoint in enumerate((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)):
        code += ((mag > midpoint) | ((mag == midpoint) & bool(i & 1))).to(torch.uint8)
    code |= torch.signbit(y).to(torch.uint8) << 3
    code = code.reshape(x.shape)
    packed = code[:, ::2] | (code[:, 1::2] << 4)
    vals = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    value = vals[(code & 7).long()] * torch.where((code & 8) != 0, -1., 1.)
    decoded = (value.reshape_as(z) * s.unsqueeze(-1)).reshape_as(x).half()
    return packed, scales, decoded


@pytest.mark.parametrize('idx', [False, True])
def test_exact(idx):
    width = 128 if idx else 512
    torch.manual_seed(12)
    x = (torch.randn(64, width, device='cuda') * 8).half()
    x[0] = 0
    x[1] = torch.finfo(torch.float16).tiny / 1024
    x[2] = 60000
    x[3, ::2] *= -1
    # Groups with a scale of one exercise every midpoint and signed zero.
    x[4] = torch.tensor([6, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, -0., -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5], device='cuda').repeat(width // 16)
    p, s = ext.dsv4_fp4_quant(x, idx)
    rp, rs, rd = reference(x, idx)
    assert torch.equal(p, rp), [(int(i), int(j), int(p[i,j]), int(rp[i,j]), x[i,2*j:2*j+2].tolist(), int(s[i,j//8]), int(rs[i,j//8])) for i,j in (p != rp).nonzero()[:10].tolist()]
    assert torch.equal(s, rs)
    assert torch.equal(ext.dsv4_fp4_dequant(p, s, idx), rd)


@pytest.mark.parametrize('idx', [False, True])
def test_empty(idx):
    width = 128 if idx else 512
    x = torch.empty((0, width), device='cuda', dtype=torch.float16)
    p, s = ext.dsv4_fp4_quant(x, idx)
    assert ext.dsv4_fp4_dequant(p, s, idx).shape == x.shape


@pytest.mark.parametrize('idx', [False, True])
def test_invalid(idx):
    width = 128 if idx else 512
    x = torch.zeros((2, width), device='cuda', dtype=torch.float16)
    with pytest.raises(RuntimeError): ext.dsv4_fp4_quant(x.float(), idx)
    with pytest.raises(RuntimeError): ext.dsv4_fp4_quant(x[:, :-1], idx)
    p, s = ext.dsv4_fp4_quant(x, idx)
    with pytest.raises(RuntimeError): ext.dsv4_fp4_dequant(p, s.float(), idx)
    with pytest.raises(RuntimeError): ext.dsv4_fp4_dequant(p, s[:, :-1], idx)


@pytest.mark.parametrize('idx', [False, True])
def test_paged_scatter(idx):
    width, groups = (128, 4) if idx else (512, 32)
    stage = torch.randn((2, 3, width), device='cuda', dtype=torch.float16)
    packed = torch.zeros((12, width // 2), device='cuda', dtype=torch.uint8)
    scales = torch.zeros((12, groups), device='cuda', dtype=torch.uint8)
    bt = torch.tensor([[2, 0, 1], [0, 2, 1]], device='cuda', dtype=torch.int32)
    positions = torch.tensor([3, 7], device='cuda', dtype=torch.int32)
    ext.dsv4_fp4_scatter(stage, packed, scales, bt, 0, positions, 4, 9, 4, idx)
    for job, pos in enumerate((3, 7)):
        n = (pos + 9) // 4 - pos // 4
        for w in range(n):
            entry = pos // 4 + w
            row = int(bt[job, entry // 4]) * 4 + entry % 4
            p, s = ext.dsv4_fp4_quant(stage[job, w:w+1], idx)
            assert torch.equal(packed[row], p[0])
            assert torch.equal(scales[row], s[0])


@pytest.mark.parametrize('idx', [False, True])
def test_captured_scatter_matches_eager(idx):
    width, groups = (128, 4) if idx else (512, 32)
    stage = torch.randn((2, 3, width), device='cuda', dtype=torch.float16)
    bt = torch.tensor([[2, 0, 1], [3, 5, 4]], device='cuda', dtype=torch.int32)
    positions = torch.tensor([3, 7], device='cuda', dtype=torch.int32)
    packed = torch.zeros((24, width // 2), device='cuda', dtype=torch.uint8)
    scales = torch.zeros((24, groups), device='cuda', dtype=torch.uint8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ext.dsv4_fp4_scatter(stage, packed, scales, bt, 0, positions, 4, 9, 4, idx)
    for pos in ([3, 7], [8, 1]):
        positions.copy_(torch.tensor(pos, dtype=torch.int32, device='cuda'))
        stage.normal_()
        packed.zero_()
        scales.zero_()
        expected_p = torch.zeros_like(packed)
        expected_s = torch.zeros_like(scales)
        ext.dsv4_fp4_scatter(stage, expected_p, expected_s, bt, 0, positions, 4, 9, 4, idx)
        graph.replay()
        assert torch.equal(packed, expected_p)
        assert torch.equal(scales, expected_s)
