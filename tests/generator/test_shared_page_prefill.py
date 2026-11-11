"""
Identical prompts in one batch share CachePage objects from allocation on. When one sequence's prefill chunk ends
on a page boundary, the page one past the chunk may already have been completed by a sibling that is further
ahead; prefill must leave that page complete (its content is the same by construction) instead of resetting its
fill level, or the sibling's next recurrent checkpoint finds a half-empty page (seen with 8 identical prompts and
the adaptive prefill limiter cutting chunks). CPU only: the model call and the pages are stubs.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.job import Job

pytestmark = pytest.mark.nogpu


class ModelStub:
    caps = {}

    def __init__(self):
        self.calls = []

    def prefill(self, *, input_ids, params):
        self.calls.append((params["cache_seqlens"].item(), input_ids.shape[-1]))


def make_page(kv_position):
    return SimpleNamespace(kv_position = kv_position, prev_hash = None, phash = b"p%d" % kv_position,
                           sequence = torch.zeros((1, PAGE_SIZE), dtype = torch.long), can_revert = True)


def make_job(prompt_pages, kv_position, pages, chunk = PAGE_SIZE):
    job = Job(input_ids = torch.arange(prompt_pages * PAGE_SIZE + 1).unsqueeze(0))
    seq = job.sequences[0]
    seq.kv_position = kv_position
    seq.allocated_pages = pages
    seq.page_hashes = []
    seq.block_index_tensor = torch.tensor([[0]], dtype = torch.int32)
    job.recurrent_state = None
    job.generator = SimpleNamespace(model = ModelStub(), max_chunk_size = chunk, recurrent_cache = object(),
                                    cache = None, draft_model = None, mtp_draft = False, dflash_draft = False)
    return job


def test_chunk_end_keeps_a_page_completed_by_a_sibling():
    # Sibling already filled pages 0..2; this sequence is at page 1 and prefills one page (page 1).
    pages = [make_page(PAGE_SIZE), make_page(0), make_page(PAGE_SIZE), make_page(0)]
    job = make_job(4, PAGE_SIZE, pages)
    job.prefill([])
    assert job.generator.model.calls == [(PAGE_SIZE, PAGE_SIZE)]
    assert job.sequences[0].kv_position == 2 * PAGE_SIZE
    assert pages[1].kv_position == PAGE_SIZE
    assert pages[2].kv_position == PAGE_SIZE   # was reset to 0 before the fix
    assert pages[3].kv_position == 0


def test_chunk_end_still_marks_the_next_unfilled_page_empty():
    pages = [make_page(PAGE_SIZE), make_page(0), make_page(5), make_page(0)]
    job = make_job(4, PAGE_SIZE, pages)
    job.prefill([])
    assert pages[1].kv_position == PAGE_SIZE
    assert pages[2].kv_position == 0
