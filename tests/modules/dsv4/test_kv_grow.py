"""Growable DSA pools (EXL3_KV_GROW) on the tiny DSV4 model: the cached forward over a VMM-backed cache that grows page
by page gives the same logits as the fixed cache, page backing follows use (pages [0, n) only), trim releases the tail.
"""

import pytest
import torch

from exllamav3.cache.cache import Cache
from exllamav3.constants import PAGE_SIZE

from testlib.dsv4 import build_tiny_dsv4, fwd_cached
from testlib.tiny_models import DSV4_TINY


@pytest.fixture(scope = "module")
def models(tmp_path_factory, device):
    import os
    out = {}
    for grow in ("0", "1"):
        os.environ["EXL3_KV_GROW"] = grow
        config, model = build_tiny_dsv4(tmp_path_factory.mktemp(f"kvgrow{grow}"), seed = 13)
        cache = Cache(model, max_num_tokens = 262144, max_batch_size = 2)
        model.load(str(device))
        out[grow] = (model, cache)
    os.environ["EXL3_KV_GROW"] = "0"
    yield out
    for model, _ in out.values():
        model.unload()


def _pool_layers(cache):
    return [l for l in cache.layers.values() if hasattr(l, "ensure_pages") and getattr(l, "ranges", None)]


def test_grow_cache_backs_nothing_up_front(models):
    _, cache = models["1"]
    assert cache.growable()
    assert cache.backed_pages() == 0
    # one granule per pool tensor stays mapped (the range's floor), nothing per token
    assert all(l.backed_bytes() == sum(r.granule for r in l.ranges.values()) for l in _pool_layers(cache))


def test_grown_cache_matches_fixed_cache(models):
    torch.manual_seed(5)
    ids = torch.randint(0, DSV4_TINY["vocab_size"], (1, 900), dtype = torch.long)
    chunks = [256, 100, 300, 244]
    m0, c0 = models["0"]
    m1, c1 = models["1"]
    # Direct forward uses the slot-partitioned identity table: slot 0 owns pages [0, num_pages // 2)
    c1.ensure_pages(-(-ids.shape[1] // PAGE_SIZE))
    with torch.inference_mode():
        s0, s1 = c0.get_new_state(), c1.get_new_state()
    try:
        a = fwd_cached(m0, ids, s0, chunks)
        b = fwd_cached(m1, ids, s1, chunks)
    finally:
        s0.free(); s1.free()
    assert torch.equal(a, b), f"max diff {(a - b).abs().max().item()}"


def test_ensure_and_trim_follow_pages(models):
    _, cache = models["1"]
    layers = _pool_layers(cache)
    # Enough pages that every pool tensor spans several 2 MiB granules (a tiny-model page is a few KiB)
    n = cache.max_num_tokens // PAGE_SIZE
    cache.ensure_pages(n)
    assert cache.backed_pages() == n
    got = sum(l.backed_bytes() for l in layers)
    assert got >= n * cache.page_bytes()
    torch.cuda.synchronize()
    cache.trim_pages(2)
    assert cache.backed_pages() == 2
    assert sum(l.backed_bytes() for l in layers) < got
