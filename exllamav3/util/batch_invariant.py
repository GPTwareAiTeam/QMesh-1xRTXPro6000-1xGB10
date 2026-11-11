"""
Batch invariance at decode row counts: a row's result must not depend on how many rows share the forward.

Kernels whose launch shape (tile, k split, cuBLAS algorithm, chunk count) is chosen from the row count produce
different fp32 summation orders at bsz 1 and bsz > 1, and with them different low bits (notes/BatchFix.md 6.7-6.9).
For row counts up to EXL3_BATCH_INVARIANT_ROWS the sites below run one launch shape for every row count. The default
0 keeps the row-count-dependent shapes, which are fixed per decode width on a single-user server and measured more
drafted tokens per round there (2026-11-07, d1 A/B/B/A); the batched server sets 8 (the fused MoE tier's MAX_BSZN),
together with EXL3_MOE_COOP_INVARIANT=1 for the coop kernel (qmesh/serve_pair.py, BATCH > 1):

  - Linear modules flagged with caps["batch_invariant_pad"] (the DSV4.1 Engram wkv and the LM head) zero-pad their
    input to exactly that many rows, so cuBLAS / the EXL3 GEMM sees one M (pad_rows)
  - hyperconnection mixes take the bsz 1 partials chunk count (chunk_rows)
  - the fused MoE kernels use one tile and no k split at every bsz (exl3_moe_coop.cu)

0 disables the padding and the chunk pin.
"""
import os

import torch

BI_ROWS = int(os.environ.get("EXL3_BATCH_INVARIANT_ROWS", "0"))


def chunk_rows(rows: int) -> int:
    """Row count to size a row-count-dependent partition with: every count up to BI_ROWS gets the bsz 1 shape"""
    return 1 if 0 < rows <= BI_ROWS else rows


def pad_rows(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(x as (BI_ROWS, k) with zero rows appended, original row count) for 1..BI_ROWS - 1 rows; otherwise
    (x as (rows, k), rows). Zero rows are inert: each output row depends only on its own input row"""
    rows = x.numel() // x.shape[-1]
    x2 = x.view(rows, x.shape[-1])
    if not 0 < rows < BI_ROWS:
        return x2, rows
    xp = x.new_zeros((BI_ROWS, x.shape[-1]))
    xp[:rows].copy_(x2)
    return xp, rows
