import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import generator, rand_linear


@pytest.mark.parametrize('batch,seq', [(1, 1), (1, 6), (2, 1), (2, 6)])
def test_index_projection_capture_matches_eager(batch, seq):
    m, max_seq, rd = 4, 16, 64
    rows = max_seq // m + 1
    w = rand_linear(512, 128, 3, generator(42), 'cuda')
    wk = ext.BC_LinearEXL3(w['trellis'], w['suh'], w['svh'], 3, None, False, False,
                           torch.empty((batch * rows, 512), device='cuda', dtype=torch.half))
    source = torch.randn((batch, rows, 512), device='cuda', dtype=torch.half)
    stage = torch.empty_like(source)
    keys = torch.empty((batch, rows, 128), device='cuda', dtype=torch.half)
    scratch = torch.empty((batch * rows, 512), device='cuda', dtype=torch.half)
    positions = torch.empty((batch, rows), device='cuda', dtype=torch.int32)
    pos = torch.tensor([3, 7][:batch], device='cuda', dtype=torch.int32)
    norm = torch.randn(128, device='cuda', dtype=torch.half)
    inv = torch.rand(rd // 2, device='cuda')
    bt = torch.arange(batch * 8, device='cuda', dtype=torch.int32).view(batch, 8)
    packed = torch.zeros((batch * 32, 256), device='cuda', dtype=torch.uint8)
    scales = torch.zeros((batch * 32, 32), device='cuda', dtype=torch.uint8)
    ipacked = torch.zeros((batch * 32, 64), device='cuda', dtype=torch.uint8)
    iscales = torch.zeros((batch * 32, 4), device='cuda', dtype=torch.uint8)

    def run():
        stage.copy_(source)
        ext.dsv4_fp4_index_keys(stage, keys, positions, scratch, pos, m, rd, wk, norm, 1e-6, inv)
        ext.dsv4_fp4_scatter(stage, packed, scales, bt, 0, pos, m, seq, 4, False)
        ext.dsv4_fp4_scatter(keys, ipacked, iscales, bt, 0, pos, m, seq, 4, True)

    run()
    ref_stage = source.clone()
    ref_keys = torch.empty_like(keys)
    ext.exl3_gemm(ref_stage.view(-1, 512), w['trellis'], ref_keys.view(-1, 128), w['suh'],
                  torch.empty_like(ref_stage).view(-1, 512), w['svh'], 0, False, False, 0)
    ext.rms_norm(ref_keys.view(-1, 128), norm, ref_keys.view(-1, 128), 1e-6, 0.0, 1.0, False, False, 1)
    ref_pos = (pos[:, None] // m + torch.arange(rows, device='cuda', dtype=torch.int32)) * m
    kr = ref_keys.view(batch, rows, 1, 128)[..., -rd:]
    lr = ref_stage.view(batch, rows, 1, 512)[..., -rd:]
    ext.rope(kr, kr, lr, lr, inv, 0, None, ref_pos, 1, 1.0, None, None, 1e-6, 0.0, 0.0, 0, 1, 0)
    torch.testing.assert_close(stage, ref_stage, rtol=0, atol=0)
    torch.testing.assert_close(keys, ref_keys, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for values in ([3, 7][:batch], [8, 1][:batch]):
        pos.copy_(torch.tensor(values, device='cuda', dtype=torch.int32))
        packed.zero_(); scales.zero_(); ipacked.zero_(); iscales.zero_()
        run()
        expected = [t.clone() for t in (stage, keys, packed, scales, ipacked, iscales)]
        packed.zero_(); scales.zero_(); ipacked.zero_(); iscales.zero_()
        graph.replay()
        for actual, ref in zip((stage, keys, packed, scales, ipacked, iscales), expected):
            assert torch.equal(actual, ref)


@pytest.mark.parametrize('batch,seq', [(1, 1), (1, 6), (2, 1), (2, 6)])
def test_grouped_index_writer_matches_single_rows(batch, seq):
    from types import SimpleNamespace
    from exllamav3.modules.dsv4 import DSV4Attention

    m, rd = 4, 64
    rows = seq // m + 1
    w = rand_linear(512, 128, 3, generator(42), 'cuda')
    norm = torch.randn(128, device='cuda', dtype=torch.half)

    class Projection:
        def forward(self, x, params):
            out = torch.empty((*x.shape[:-1], 128), device=x.device, dtype=x.dtype)
            ext.exl3_gemm(x.view(-1, 512), w['trellis'], out.view(-1, 128), w['suh'],
                          torch.empty_like(x).view(-1, 512), w['svh'], 0, False, False, 0)
            return out

    class Norm:
        def forward(self, x, params, out_dtype):
            y = torch.empty_like(x)
            ext.rms_norm(x.view(-1, 128), norm, y.view(-1, 128), 1e-6, 0., 1., False, False, 1)
            return y

    model = SimpleNamespace(idx_wk=Projection(), idx_k_norm=Norm(), compress_rate=m,
                            rope_head_dim=rd, inv_freq_compress=torch.rand(rd // 2, device='cuda'))
    source = torch.randn((batch, rows, 512), device='cuda', dtype=torch.half)
    pos = torch.tensor([3, 7][:batch], device='cuda', dtype=torch.int32)
    bt = torch.arange(batch * 8, device='cuda', dtype=torch.int32).view(batch, 8)
    grouped, single = source.clone(), source.clone()
    entries = (pos[:, None] // m + torch.arange(rows, device='cuda', dtype=torch.int32)).reshape(-1)
    grouped_keys = DSV4Attention._indexer_keys(model, grouped.view(-1, 512), {}, entries).view(batch, rows, 128)
    single_keys = torch.empty_like(grouped_keys)
    for b in range(batch):
        for j in range(rows):
            e = pos[b:b + 1] // m + j
            single_keys[b, j:j + 1] = DSV4Attention._indexer_keys(model, single[b, j:j + 1], {}, e)
    torch.testing.assert_close(grouped, single, atol=0, rtol=0)
    torch.testing.assert_close(grouped_keys, single_keys, atol=0, rtol=0)
    for indexer, a, b, width, groups in ((False, grouped, single, 256, 32),
                                         (True, grouped_keys, single_keys, 64, 4)):
        pa = torch.zeros((batch * 32, width), device='cuda', dtype=torch.uint8)
        sa = torch.zeros((batch * 32, groups), device='cuda', dtype=torch.uint8)
        pb, sb = torch.zeros_like(pa), torch.zeros_like(sa)
        ext.dsv4_fp4_scatter(a, pa, sa, bt, 0, pos, m, seq, 4, indexer)
        for job in range(batch):
            ext.dsv4_fp4_scatter(b[job], pb, sb, bt[job:job + 1], int(pos[job]), None, m, seq, 4, indexer)
        assert torch.equal(pa, pb)
        assert torch.equal(sa, sb)
