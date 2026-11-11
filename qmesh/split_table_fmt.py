"""Split-policy table file format (device layout of exl3_coldsteal.cuh CS_POL_*).

cells  int8 [LOAD_L=3][ROW_B=12][N+1=64][H+1=9][V=3]   (P7 hot load level, rows bucket, nd, h) -> (s, c, z)
thr    uint8 [ROW_B][2]                                  per rows bucket: level 0 if distinct hot <= thr[0], 1 if <= thr[1]
File: one flat int8 .npy of cells then thr (CS_POL_TABLE_BYTES bytes).
"""
import numpy as np

LOAD_L, ROW_B, POL_N, POL_H, POL_V = 3, 12, 63, 8, 3
CELLS_SHAPE = (LOAD_L, ROW_B, POL_N + 1, POL_H + 1, POL_V)
N_CELLS = int(np.prod(CELLS_SHAPE))
NBYTES = N_CELLS + ROW_B * 2
# Rows buckets of the device table (cs_row_bucket): rows 1..6, 7-8, 9-12, 13-16, 17-24, 25-32, 33+
ROW_EDGES = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 8), (9, 12), (13, 16), (17, 24), (25, 32), (33, 999)]


def row_bucket(rows):
    rows = np.asarray(rows)
    out = np.full(rows.shape, ROW_B - 1)
    for b, (lo, hi) in reversed(list(enumerate(ROW_EDGES))):
        out = np.where((rows >= lo) & (rows <= hi), b, out)
    return out


def empty():
    return np.zeros(CELLS_SHAPE, np.int8), np.full((ROW_B, 2), 255, np.uint8)


def pack(cells, thr):
    assert cells.shape == CELLS_SHAPE and thr.shape == (ROW_B, 2)
    return np.concatenate([cells.astype(np.int8).reshape(-1), thr.astype(np.uint8).view(np.int8).reshape(-1)])


def unpack(flat):
    flat = np.asarray(flat, np.int8).reshape(-1)
    assert flat.size == NBYTES, (flat.size, NBYTES)
    return flat[:N_CELLS].reshape(CELLS_SHAPE), flat[N_CELLS:].view(np.uint8).reshape(ROW_B, 2)


def save(path, cells, thr):
    np.save(path, pack(cells, thr))


def load(path):
    return unpack(np.load(path))
