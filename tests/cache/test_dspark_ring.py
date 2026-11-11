"""
DSpark drafter window rings (cache/dsa.py DSparkLayerState / DSparkState). The drafter reads only the sliding_window
main-kv rows below the block it drafts, so each sequence keeps a ring of ring_rows rows (window plus two pages)
instead of a context-sized paged plane. Checked here against a full-length reference plane holding every row ever
written:

- writes through a state's block table (dspark_write_rows) land at ring row position % ring_rows of that slot, so
  the window below the state's position reads back exactly the reference rows, after any number of wraps;
- a batched write's rows for a sequence that writes nothing (block table on the trash page) touch no slot;
- rollback_capacity never lets a rewind expose rows a later write overwrote; every rewind it allows reads back the
  reference window;
- stash / unstash round trip into another slot reproduces the window, and the restored state's rollback stays
  within the rows the checkpoint carried.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.cache.dsa import DSparkLayerState, DSparkState
from exllamav3.constants import PAGE_SIZE
from exllamav3.ext import exllamav3_ext as ext

WINDOW = 128
WIDTH = 64
SLOTS = 3


class FakeCache:
    """The two Cache methods DSparkState uses, over real layer states"""

    def __init__(self, layers):
        self.layers = layers
        self.released = []

    def get_all_recurrent_layers(self):
        return self.layers

    def release_state(self, state):
        self.released.append(state.slot)


def make(device, n_layers = 2):
    module = SimpleNamespace(dspark_ring_rows = -(-(WINDOW + 2 * PAGE_SIZE) // PAGE_SIZE) * PAGE_SIZE,
                             head_dim = WIDTH, sliding_window = WINDOW)
    layers = {}
    for i in range(n_layers):
        l = DSparkLayerState(module, SLOTS, 0, 0)
        l.alloc(device)
        layers[(i, 0)] = l
    return FakeCache(layers)


def write(cache, state, ref, rows, width_pages = 64):
    """rows (s, WIDTH) at the state's position into every layer (layer i gets rows + i) and into ref[i]"""
    s = rows.shape[0]
    pos = state.position
    bt = state.block_table(width_pages).to(rows.device)
    sl = torch.tensor([pos], dtype = torch.int32, device = rows.device)
    for i, l in enumerate(cache.layers.values()):
        ext.dspark_write_rows((rows + i).unsqueeze(0).contiguous(), l.ring, bt, sl)
        ref[i][pos : pos + s] = rows + i
    state.wrote(pos, s, s)


def window_rows(layer, state, position):
    """The rows the drafter reads below `position` (min(window, position) of them), through the ring addressing"""
    lo = max(0, position - WINDOW)
    flat = layer.ring.view(-1, WIDTH)
    idx = state.slot * layer.ring_rows + torch.arange(lo, position, device = flat.device) % layer.ring_rows
    return flat[idx]


def check_window(cache, state, ref):
    p = state.position
    lo = max(0, p - WINDOW)
    for i, l in enumerate(cache.layers.values()):
        assert torch.equal(window_rows(l, state, p), ref[i][lo:p]), f"layer {i}, position {p}"


@torch.inference_mode()
def test_writes_wrap_and_window_reads_back(device):
    cache = make(device)
    st = DSparkState(cache, 1, 0)
    ref = [torch.zeros((8192, WIDTH), dtype = torch.half, device = device) for _ in cache.layers]
    g = torch.Generator(device = "cpu").manual_seed(0)
    # Uneven chunks: page-crossing, ring-crossing, single rows (decode), several full wraps
    for n in [5, 300, 1, 1, 6, 250, 511, 3, 700, 6, 6, 1, 400, 129]:
        n = min(n, st.ring_rows)
        write(cache, st, ref, torch.randn((n, WIDTH), generator = g).half().to(device))
        check_window(cache, st, ref)
    assert st.position > 3 * st.ring_rows


@torch.inference_mode()
def test_other_slots_and_trash_page_are_untouched(device):
    cache = make(device)
    a, b = DSparkState(cache, 0, 0), DSparkState(cache, 2, 0)
    ref = [torch.zeros((4096, WIDTH), dtype = torch.half, device = device) for _ in cache.layers]
    g = torch.Generator(device = "cpu").manual_seed(1)
    write(cache, a, ref, torch.randn((600, WIDTH), generator = g).half().to(device))
    before = [l.ring[: l.trash_page].clone() for l in cache.layers.values()]
    # A batched write where row 1 writes nothing: its table points at the trash page
    rows = torch.randn((2, 6, WIDTH), generator = g).half().to(device)
    bt = torch.cat([b.block_table(16), b.trash_table(16)]).to(device)
    sl = torch.tensor([0, 600], dtype = torch.int32, device = device)
    for l in cache.layers.values():
        ext.dspark_write_rows(rows, l.ring, bt, sl)
    for l, prev in zip(cache.layers.values(), before):
        pps = l.pages_per_slot
        # slot 2 got its 6 rows; slots 0 and 1 are as before
        assert torch.equal(l.ring[: 2 * pps], prev[: 2 * pps])
        assert torch.equal(l.ring[2 * pps].view(-1, WIDTH)[:6], rows[0])
    check_window(cache, a, ref)


@torch.inference_mode()
def test_rollback_never_exposes_overwritten_rows(device):
    cache = make(device)
    st = DSparkState(cache, 0, 0)
    ref = [torch.zeros((8192, WIDTH), dtype = torch.half, device = device) for _ in cache.layers]
    g = torch.Generator(device = "cpu").manual_seed(2)
    # A speculative round: 6 rows written, 0..6 accepted, the rest rewound (write() accepts all, then rewind)
    for step in range(400):
        acc = int(torch.randint(0, 7, (1,), generator = g))
        rows = torch.randn((6, WIDTH), generator = g).half().to(device)
        write(cache, st, ref, rows)
        st.rewind(6 - acc)
        check_window(cache, st, ref)
        # Every rewind the state allows must still read back the reference window
        cap = st.rollback_capacity()
        assert cap >= 0
        if cap:
            p = st.position
            st.rewind(cap)
            check_window(cache, st, ref)
            st.position = p   # undo the probe rewind (rows above are still intact)
    assert st.position > st.ring_rows


@torch.inference_mode()
def test_rollback_capacity_bounds(device):
    cache = make(device)
    st = DSparkState(cache, 0, 0)
    ref = [torch.zeros((4096, WIDTH), dtype = torch.half, device = device) for _ in cache.layers]
    g = torch.Generator(device = "cpu").manual_seed(3)
    write(cache, st, ref, torch.randn((200, WIDTH), generator = g).half().to(device))
    assert st.rollback_capacity() == 200        # nothing overwritten yet: back to 0
    write(cache, st, ref, torch.randn((st.ring_rows, WIDTH), generator = g).half().to(device))
    # The last ring_rows rows are intact; the window must stay inside them
    assert st.rollback_capacity() == st.ring_rows - WINDOW
    with pytest.raises(AssertionError):
        st.rewind(st.ring_rows - WINDOW + 1)


@torch.inference_mode()
def test_stash_unstash_round_trip_into_another_slot(device):
    cache = make(device)
    a = DSparkState(cache, 0, 0)
    ref = [torch.zeros((8192, WIDTH), dtype = torch.half, device = device) for _ in cache.layers]
    g = torch.Generator(device = "cpu").manual_seed(4)
    for n in [700, 700, 33]:
        write(cache, a, ref, torch.randn((n, WIDTH), generator = g).half().to(device))
    stashed = a.stash()
    for k, l in cache.layers.items():
        assert stashed[k].shape == (a.ring_rows, WIDTH)
        assert stashed[k].device.type == "cpu"
    b = DSparkState(cache, 2, a.position, clear = False, stashed = stashed)
    check_window(cache, b, ref)
    # The restored state carries ring_rows rows: it can rewind within them, not below
    assert b.rollback_capacity() == a.ring_rows - WINDOW
    # Writes continue from the restored position
    write(cache, b, ref, torch.randn((300, WIDTH), generator = g).half().to(device))
    check_window(cache, b, ref)


@torch.inference_mode()
def test_stash_early_position_carries_only_written_rows(device):
    cache = make(device)
    a = DSparkState(cache, 1, 0)
    ref = [torch.zeros((1024, WIDTH), dtype = torch.half, device = device) for _ in cache.layers]
    g = torch.Generator(device = "cpu").manual_seed(5)
    write(cache, a, ref, torch.randn((256, WIDTH), generator = g).half().to(device))
    stashed = a.stash()
    assert all(stashed[k].shape[0] == 256 for k in cache.layers)
    b = DSparkState(cache, 0, 256, clear = False, stashed = stashed)
    check_window(cache, b, ref)
    assert b.rollback_capacity() == 256          # restored from position 0: nothing below is lost
