"""Pad-to-bucket decode capture (EXL3_PW_BUCKETS).

Every distinct decode input shape captures its own piecewise graph Segments (transformer.py
_forward_piecewise is keyed by the input shape), its own C++ BC batch-attention slot and its own
drafter segments. With several users the batch row count drifts step to step (users join, finish), so
a B=8 server captured shapes 8, 7, 6, ... x every verify width, and ran out of P7 memory (RM2-memw,
ledger section 4: 480 segments at the OOM).

With EXL3_PW_BUCKETS="2,4,8" (any ascending list) a decode/verify step of B > 1 rows runs at the
smallest bucket >= B: the generator appends B_b - B pad rows, the model runs the bucket's graphs, and
the outputs are sliced back to the B real rows. B = 1 is never padded (the single-user path is the
release path, bit for bit). B above the largest bucket runs at its own shape.

A pad row never touches real state:
  * its block-table row points every page at the cache's trash page (one extra page per
    CacheLayer_dsa, past the page table's range), so pool / indexer writes land nowhere;
  * its recurrent state is the generator's pad state, on a slot reserved for it (one extra cache
    slot), reset to position 0 before every step, so ring / compressor writes land nowhere and no
    real state advances;
  * its router picks are replaced by the sentinel expert id (num_experts) at weight 0 before the
    expert split (block_sparse_mlp.py), so they never reach the GX seam as picks and never count
    toward an expert group; modules without an expert slice keep the ids and take weight 0;
  * its logits, tap states and drafter confidence are cut off before anything reads them.

The verify width is not padded: a pad column would be a real row's position past its allocated pages.
"""
from __future__ import annotations

import os

import torch


def _parse(raw: str) -> tuple[int, ...]:
    raw = raw.strip()
    if not raw or raw == "0":
        return ()
    b = tuple(int(v) for v in raw.split(",") if v.strip())
    if any(v < 1 for v in b) or list(b) != sorted(set(b)):
        raise ValueError(f"EXL3_PW_BUCKETS={raw!r}: expected an ascending list of distinct row counts >= 1")
    return b


BUCKETS = _parse(os.environ.get("EXL3_PW_BUCKETS", ""))
# Token id fed to pad rows (diagnostic knob: a run with another id must give bit-identical real rows, which is
# the direct test that nothing of a pad row reaches a real one)
PAD_TOKEN = int(os.environ.get("EXL3_PW_PAD_TOKEN", "0"))


def enabled() -> bool:
    return bool(BUCKETS)


def bucket_rows(rows: int) -> int:
    """Rows the step runs at: the smallest bucket >= rows, rows itself for rows == 1, when no bucket
    holds it, or with buckets off."""
    if rows <= 1 or not BUCKETS:
        return rows
    for b in BUCKETS:
        if b >= rows:
            return b
    return rows


def max_rows(rows: int) -> int:
    """Largest row count a step of at most `rows` real rows can run at (sizes the pinned buffers)."""
    return max(bucket_rows(r) for r in range(1, rows + 1)) if rows >= 1 else rows


# Real row count of the current bucketed step, as one persistent int32 device scalar per device: the
# captured graphs read it by pointer, so it is refreshed before every bucketed forward (set_real_rows)
_real_host = 0
_real_dev: dict = {}


def set_real_rows(rows: int):
    """Refresh the device scalar(s) before a bucketed forward. Stream-ordered: a replay queued after this
    reads the new value. The scalar for the current device is created here, eagerly, so it never comes
    from a graph capture's private pool"""
    global _real_host
    _real_host = rows
    if not _real_dev and torch.cuda.is_available():
        real_rows_tensor(torch.device("cuda", torch.cuda.current_device()))
    for t in _real_dev.values():
        t.fill_(rows)

def real_rows_tensor(device) -> torch.Tensor:
    device = torch.device(device)
    t = _real_dev.get(device)
    if t is None:
        t = _real_dev[device] = torch.full((1,), _real_host, dtype = torch.int32, device = device)
    return t

def pad_mask(batch: int, seq: int, device) -> torch.Tensor:
    """(batch * seq,) bool, True on the flattened rows of pad batch rows (row-major (b, s)). Device-only and
    sync-free, so it is safe inside a captured graph"""
    b = torch.arange(batch, device = device, dtype = torch.int32)
    return (b >= real_rows_tensor(device)).unsqueeze(1).expand(batch, seq).reshape(-1)
