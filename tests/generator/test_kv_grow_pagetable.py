"""PageTable on a growable cache (EXL3_KV_GROW): pages are backed lowest-first on demand, within the free budget before
any cached sequence is evicted, past it only after the cached sequences and by borrowing ColdSteal slots; unbacked
pages beyond both are never handed out; shrink gives the slots back."""

import pytest

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.pagetable import PageTable

PB = 1 << 20           # fake: one page costs 1 MiB of KV


class FakeCache:
    def __init__(self, max_pages):
        self.max_num_tokens = max_pages * PAGE_SIZE
        self._backed = 0
        self.layers = {}

    def growable(self):
        return True

    def backed_pages(self):
        return self._backed

    def page_bytes(self):
        return PB

    def ensure_pages(self, n):
        self._backed = max(self._backed, n)

    def trim_pages(self, n):
        self._backed = min(self._backed, n)


class FakeLender:
    def __init__(self, slots, slot_bytes):
        self.S, self.active, self.per = slots, slots, slot_bytes

    def lendable_bytes(self):
        return self.active * self.per

    def lent_bytes(self):
        return (self.S - self.active) * self.per

    def lend(self, nbytes):
        k = min(self.active, -(-nbytes // self.per))
        self.active -= k
        return k

    def give_back(self, nbytes):
        k = min(self.S - self.active, nbytes // self.per)
        self.active += max(0, k)
        return max(0, k)


def _pt(max_pages, free_pages, lender = None):
    cache = FakeCache(max_pages)
    pt = PageTable.__new__(PageTable)
    gen = type("G", (), {"enable_defrag": True})()
    PageTable.__init__(pt, gen, cache)
    pt.kv_backed_bytes = lambda: cache._backed * PB
    pt.grow_free_bytes = free_pages * PB
    pt.slot_lender = lender
    return pt, cache


def _alloc(pt, n_unique, hashes = ()):
    pages, *_ = pt.allocate_pages(list(hashes), n_unique, None)
    return pages


def test_backs_lowest_pages_first_within_free_budget():
    pt, cache = _pt(64, free_pages = 8)
    pages = _alloc(pt, 5)
    assert sorted(p.page_index for p in pages) == [0, 1, 2, 3, 4]
    assert cache.backed_pages() == 5


def test_never_hands_out_past_budget_without_lender():
    pt, cache = _pt(64, free_pages = 4)
    with pytest.raises(IndexError):        # eviction order exhausted: allocation refused, nothing unbacked used
        _alloc(pt, 5)
    assert cache.backed_pages() <= 4


def test_cached_sequence_evicted_before_slots_borrowed():
    lender = FakeLender(slots = 8, slot_bytes = 4 * PB)
    pt, cache = _pt(64, free_pages = 4, lender = lender)
    seq = _alloc(pt, 4)
    for p in seq:
        p.kv_position = PAGE_SIZE           # complete pages: a cached prompt once released
    pt.deallocate_pages(seq)
    again = _alloc(pt, 4)                   # free budget used up: reuse the cached pages, borrow nothing
    assert lender.active == 8 and cache.backed_pages() == 4
    assert sorted(p.page_index for p in again) == [0, 1, 2, 3]


def test_borrows_slots_for_live_context_then_gives_back():
    lender = FakeLender(slots = 8, slot_bytes = 4 * PB)
    pt, cache = _pt(64, free_pages = 4, lender = lender)
    live = _alloc(pt, 10)                   # 4 free + 6 borrowed -> 2 slots (8 MiB) lent
    assert cache.backed_pages() == 10
    assert lender.active == 6
    pt.deallocate_pages(live)
    pt.shrink()
    assert lender.active == 8               # all slots back
    assert cache.backed_pages() <= 4
