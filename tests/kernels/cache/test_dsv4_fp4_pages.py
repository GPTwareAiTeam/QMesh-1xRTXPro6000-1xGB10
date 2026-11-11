from types import SimpleNamespace
import torch
from exllamav3.cache.dsa import CacheLayer_dsa
from exllamav3.constants import PAGE_SIZE


def test_fp4_page_copy_and_mirror():
    attention = SimpleNamespace(compress_rate=4, head_dim=512, rope_head_dim=64,
                                index_head_dim=128, indexer_mode='full')
    a = CacheLayer_dsa(None, attention, 1, PAGE_SIZE * 2, kv_format='fp4')
    b = CacheLayer_dsa(None, attention, 2, PAGE_SIZE * 2, kv_format='fp4')
    a.alloc(torch.device('cuda'))
    b.alloc(torch.device('cuda'))
    for t in a.get_tensors():
        t.random_(0, 256)
    b.copy_page(a, 0, 1, PAGE_SIZE)
    for x, y in zip(a.get_tensors(), b.get_tensors()):
        assert torch.equal(x[0], y[1])
    assert a.pool_fp4.numel() + a.pool_fp4_scale.numel() == a.capacity * 288
    assert a.pool_idx_fp4.numel() + a.pool_idx_scale.numel() == a.capacity * 68
    assert a.storage_size() == a.capacity * (288 + 68)
    a.alloc_mirror(torch.device('cuda'))
    assert a.mirrors[torch.device('cuda')].kv_format == 'fp4'
    a.free()
    assert not a.get_tensors()


def test_fp4_partial_page_copy_preserves_bytes():
    attention = SimpleNamespace(compress_rate=4, head_dim=512, rope_head_dim=64,
                                index_head_dim=128, indexer_mode='full')
    a = CacheLayer_dsa(None, attention, 1, PAGE_SIZE * 2, kv_format='fp4')
    b = CacheLayer_dsa(None, attention, 2, PAGE_SIZE * 2, kv_format='fp4')
    a.alloc(torch.device('cuda'))
    b.alloc(torch.device('cuda'))
    for t in a.get_tensors():
        t.random_(0, 256)
    for t in b.get_tensors():
        t.fill_(173)
    b.copy_page(a, 0, 1, 43)
    for src, dst in zip(a.get_tensors(), b.get_tensors()):
        assert torch.equal(src[0, :10], dst[1, :10])
        assert torch.all(dst[1, 10:] == 173)
        assert torch.all(dst[0] == 173)


def test_fp4_page_rotation_preserves_bytes():
    from exllamav3.ext import exllamav3_ext as ext
    attention = SimpleNamespace(compress_rate=4, head_dim=512, rope_head_dim=64,
                                index_head_dim=128, indexer_mode='full')
    layer = CacheLayer_dsa(None, attention, 1, PAGE_SIZE * 3, kv_format='fp4')
    layer.alloc(torch.device('cuda'))
    order = torch.tensor([-1, 0, 0, 1, 1, 2, 2, -1], dtype=torch.int32, device='cuda')
    for t in layer.get_tensors():
        t.random_(0, 256)
        original = t.clone()
        ext.cache_rotate(t, order, torch.empty_like(t[0]))
        assert torch.equal(t[0], original[1])
        assert torch.equal(t[1], original[2])
        assert torch.equal(t[2], original[0])
