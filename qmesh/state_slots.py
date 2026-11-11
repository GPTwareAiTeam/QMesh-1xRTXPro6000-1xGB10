"""Recurrent-state slot count for the pair's caches.

Every DSV4 / DSpark recurrent layer allocates one SWA ring, compressor buffer and overlap snapshot set per
slot (exllamav3/cache/dsa.py DSV4LayerState / DSparkLayerState), and `Cache(max_batch_size=...)` defaults
to 16 slots. The pair serves BATCH users at most, and the generator clamps its batch to the slot count
anyway (generator.py:449), so the slots past BATCH are dead memory: 0.50 GiB of target state at 16 slots
on pack N, i.e. ~31 MiB per slot (RM2-memw, ledger section 4).

QMESH_STATE_SLOTS overrides the count explicitly (a diagnosis that wants more concurrent states than
the batch, e.g. prefix-cache stashes restored next to live jobs); otherwise it follows BATCH.
"""
from __future__ import annotations

import os


def state_slots(batch: int) -> int:
    raw = os.environ.get("QMESH_STATE_SLOTS", "").strip()
    n = int(raw) if raw else batch
    if n < max(batch, 1):
        raise ValueError(f"QMESH_STATE_SLOTS={n} is below BATCH={batch}: the generator would clamp the batch to it")
    return n
