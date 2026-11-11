"""Generator.row_limited: the per-load draft window cap (draft_wide_jobs / draft_wide_window) and the row ceiling
(draft_max_rows) combine as the smaller of the two, apply only to sequences past prefill, and leave the window alone
below the job threshold."""

from types import SimpleNamespace

import pytest

from exllamav3.generator.generator import Generator


class _Job:
    def __init__(self, done, seqs = 1):
        self._done = done
        self.sequences = [None] * seqs

    def is_prefill_done(self):
        return self._done


def _gen(jobs, max_rows = 0, wide_jobs = 0, wide_window = 2, draft = 5):
    g = Generator.__new__(Generator)
    g.active_jobs = jobs
    g.draft_max_rows = max_rows
    g.draft_wide_jobs = wide_jobs
    g.draft_wide_window = wide_window
    g.num_draft_tokens = draft
    g._draft_window_last = None
    g.cache = SimpleNamespace(pad_slot = None)
    return g


@pytest.mark.parametrize("n_done, expect", [(1, 5), (2, 5), (3, 5), (4, 2), (8, 2)])
def test_wide_cap_applies_from_threshold(n_done, expect):
    g = _gen([_Job(True) for _ in range(n_done)], wide_jobs = 4, wide_window = 2)
    assert g.row_limited(5) == expect


def test_prefilling_jobs_do_not_count():
    g = _gen([_Job(True) for _ in range(3)] + [_Job(False) for _ in range(5)], wide_jobs = 4, wide_window = 2)
    assert g.row_limited(5) == 5


def test_tighter_of_both_caps_wins():
    jobs = [_Job(True) for _ in range(8)]
    assert _gen(jobs, max_rows = 16, wide_jobs = 4, wide_window = 3).row_limited(5) == 1    # 16 // 8 - 1
    assert _gen(jobs, max_rows = 48, wide_jobs = 4, wide_window = 3).row_limited(5) == 3    # wide cap tighter
    assert _gen(jobs, max_rows = 48).row_limited(5) == 5                                    # 48 // 8 - 1 = 5


def test_off_by_default_and_zero_window_untouched():
    jobs = [_Job(True) for _ in range(8)]
    assert _gen(jobs).row_limited(5) == 5
    assert _gen(jobs, wide_jobs = 4).row_limited(0) == 0
