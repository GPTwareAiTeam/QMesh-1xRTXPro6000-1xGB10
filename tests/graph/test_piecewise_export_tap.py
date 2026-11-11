"""The piecewise decode graph path must leave the draft-model export tap where the eager path leaves it.

A replay runs no Python, so anything the step reads back on the host has to be produced after the replay: a tapped
block (params["export_state_layers"], DSpark/DFlash) appends `_tap_to_half(x.mean(dim = 2))` after graph B, in both
the plain piecewise body and the captured attention-body body (`_forward_piecewise_b1`, which is the decode path
whenever the attention has `b1_prep`, i.e. always on DSv4). The block's graph B must therefore never be merged into
the next block's graph A when the block is tapped: `_pw_defer` refuses export layers, and the tap never runs for a
pending block. If either guarantee is lost, params["export_states"] comes back empty and the drafter dies in
deepseek_v4_mtp.update_kv_from_target (`T = target_hidden[0].shape[1]`) on the first decode step — the exact
failure that EXL3_WHOLE_GRAPH=1 had before v1.2's tap was ported.

CPU test: capture is replaced by an eager passthrough (the Segment stand-in below), the block's HC sites, attention
and MLP are stubs with the same interface, and the same block body runs down the eager and the piecewise paths.
"""

import pytest
import torch

from exllamav3.modules import transformer
from exllamav3.modules.transformer import TransformerBlock

pytestmark = [pytest.mark.nogpu]

D = 8
H = 4


class _SegmentStub:
    """exllamav3.util.segment.Segment with capture removed: call the body and return its outputs"""

    def __init__(self, fn, carry = ()):
        self.fn = fn
        self.carry = carry
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        out = self.fn(*args)
        return tuple(out) if isinstance(out, (tuple, list)) else (out,)


class _HCStub:
    """HyperConnection with the torch-fallback update (hyperconnections.HyperConnection.apply_)"""

    def __init__(self, seed):
        g = torch.Generator().manual_seed(seed)
        self.hc_mult = H
        self.post_w = torch.randn(H, generator = g)
        self.comb_w = torch.randn(H, H, generator = g)

    def mix_norm(self, streams, params, norm):
        b, s, _, _ = streams.shape
        pre = torch.sigmoid(self.post_w) + 1e-6
        post = self.post_w.softmax(0).expand(b, s, H).contiguous()
        comb = self.comb_w.unsqueeze(0).unsqueeze(0).expand(b, s, H, H).contiguous()
        collapsed = (pre.view(1, 1, H, 1) * streams).sum(dim = 2)
        return post, comb, collapsed, False

    def apply_(self, x, y, post, comb, params):
        return post.unsqueeze(-1) * y.float().unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), x)


class _AttnStub:
    """Eager attention core. Without b1_prep the block takes _forward_piecewise"""

    tp_owner = None
    tp_reduce = False

    def forward(self, y, params):
        assert y.dtype == torch.half
        return (y.float() * 0.5).half()


class _AttnB1Stub(_AttnStub):
    """Attention with the device-driven body: the block takes _forward_piecewise_b1 (the production decode path)"""

    def b1_prep(self, params, seq = 1):
        # (ctx, arr, bt_st); ctx is (rsl, kl, m, w, kp, cand_regime, nnew) and keys the capture
        self.prep_seqs = getattr(self, "prep_seqs", []) + [seq]
        return (None, None, 0, 0, None, 0, 0), None, None

    def _batch_body(self, y, arr, rows, seq, total, rsl, kl, bt_st, m, w, kp, params, collect,
                    cand_regime = None, nnew = 0):
        assert y.dtype == torch.half
        return (y.float() * 0.5).half()


class _MLPStub:
    def forward(self, y, params):
        assert y.dtype == torch.half
        return (y.float().relu() * 0.25).half()


def _block(layer_idx = 0, b1 = False):
    # The block body only reads these attributes; Module.__init__ would need a config/tensor collection
    b = object.__new__(TransformerBlock)
    b.layer_idx = layer_idx
    b.out_dtype = None
    b.layer_scalar_f = None
    b.attn_resid_scalar = None
    b.mlp_resid_scalar = None
    b.attn_norm = None
    b.mlp_norm = None
    b.attn = _AttnB1Stub() if b1 else _AttnStub()
    b.attn_hc = _HCStub(1)
    b.mlp_hc = _HCStub(2)
    b.mlp = _MLPStub()
    b.device = torch.device("cpu")
    return b


def _streams(seed = 11, seq = 1):
    """The mHC residual: (b, s, H, D) fp32, seq inside the piecewise gate"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, seq, H, D, generator = g) * 0.5


def _run(monkeypatch, piecewise, layer_idx, params, b1 = False, seq = 1):
    # The module flags are read from the environment at import (EXL3_WHOLE_GRAPH_MAXSEQ, EXL3_DSV4_B1_MULTIROW);
    # pin the gate this test is written against so a runner's env cannot change which body runs
    monkeypatch.setattr(transformer, "Segment", _SegmentStub)
    monkeypatch.setattr(transformer, "_PIECEWISE", piecewise)
    monkeypatch.setattr(transformer, "_PIECEWISE_MAXSEQ", max(8, seq))
    monkeypatch.setattr(transformer, "_B1_MULTIROW", True)
    block = _block(layer_idx, b1)
    x = block.forward(_streams(seq = seq), params, None)
    return block, x


def _expected(x):
    """v1.2's export value: the collapsed stream mean, cast to fp16 and clamped"""
    return x.mean(dim = 2).half().clamp_(-65504.0, 65504.0)


def test_piecewise_tap_matches_eager(monkeypatch):
    layer = 3
    eager_params = {"export_state_layers": {layer}}
    graph_params = {"export_state_layers": {layer}}

    eager_block, eager_x = _run(monkeypatch, False, layer, eager_params)
    graph_block, graph_x = _run(monkeypatch, True, layer, graph_params)

    assert list(eager_params["export_states"]) and list(graph_params["export_states"])
    eager_tap, = eager_params["export_states"]
    graph_tap, = graph_params["export_states"]

    # The piecewise run really went through the segments (otherwise this proves nothing about the graph path)
    segs = list(graph_block._segments.values())
    assert len(segs) == 1 and all(seg.calls >= 1 for seg in segs[0])

    assert graph_tap.dtype == torch.half
    assert graph_tap.shape == (1, 1, D)
    assert torch.equal(graph_tap, eager_tap)
    assert torch.equal(graph_x, eager_x)
    assert torch.equal(graph_tap, _expected(eager_x))


def test_b1_path_taps_multirow(monkeypatch):
    """The captured attention body takes up to EXL3_WHOLE_GRAPH_MAXSEQ rows (the verify window) and taps every one"""
    layer = 5
    for seq in (1, 4):
        eager_params = {"export_state_layers": {layer}}
        graph_params = {"export_state_layers": {layer}}
        eager_block, eager_x = _run(monkeypatch, False, layer, eager_params, seq = seq)
        graph_block, graph_x = _run(monkeypatch, True, layer, graph_params, b1 = True, seq = seq)

        assert graph_block.attn.prep_seqs == [seq]
        assert len(graph_params["export_states"]) == 1
        graph_tap, = graph_params["export_states"]

        assert graph_tap.shape == (1, seq, D)
        assert torch.equal(graph_tap, _expected(eager_x))
        assert torch.equal(eager_x, graph_x)


def test_untapped_blocks_export_nothing(monkeypatch):
    for piecewise, b1 in ((False, False), (True, False), (True, True)):
        params = {"export_state_layers": {4}}
        _run(monkeypatch, piecewise, 3, params, b1 = b1)
        assert "export_states" not in params, (piecewise, b1)

    # No drafter attached at all
    params = {}
    _run(monkeypatch, True, 3, params)
    assert "export_states" not in params

    # Layer split across devices: instance 0 owns the export
    params = {"export_state_layers": {3}, "layer_instance": 1}
    _run(monkeypatch, True, 3, params)
    assert "export_states" not in params


def test_export_layer_never_defers(monkeypatch):
    """_pw_defer hands a block's MLP half to the next block's graph A; a tapped block must never be handed over or
    its tap is never computed (a replay runs no Python)"""
    monkeypatch.setattr(transformer, "_PW_MERGE", True)
    block = _block(3)
    nxt = _block(4)

    tapped = {"export_state_layers": {3}, "pw_next": nxt}
    assert block._pw_defer(_streams(), tapped, None) is False
    assert block._pw_defer(_streams(), {"pw_next": nxt}, None) is True

    # ... and the same through the whole block: the tapped block keeps its graph B and exports
    params = {"export_state_layers": {3}, "pw_next": nxt}
    _, x = _run(monkeypatch, True, 3, params)
    assert "pw_pending" not in params
    assert torch.equal(params["export_states"][0], _expected(x))


def test_tap_scale_knob_reaches_both_paths(monkeypatch):
    """EXL3_DSPARK_TAP_SCALE (power of two) is applied before the cast, identically on both paths"""
    monkeypatch.setattr(transformer, "_DSPARK_TAP_SCALE", 2.0)
    eager_params = {"export_state_layers": {1}}
    graph_params = {"export_state_layers": {1}}
    _, eager_x = _run(monkeypatch, False, 1, eager_params)
    _, graph_x = _run(monkeypatch, True, 1, graph_params)
    ref = (eager_x.mean(dim = 2).float() * 2.0).half().clamp_(-65504.0, 65504.0)
    assert torch.equal(eager_params["export_states"][0], ref)
    assert torch.equal(graph_params["export_states"][0], ref)


def test_drafter_consumer_keeps_its_piece_b_call():
    """The drafter's own segmented body calls block._piece_b directly: it must still return a 1-tuple"""
    import inspect

    import exllamav3.architecture.deepseek_v4_mtp as mtp

    assert hasattr(mtp.DeepseekV4MTPModel, "update_kv_from_target")
    src = inspect.getsource(mtp)
    assert "_piece_b(" in src
    # Direct call with the stub graph pieces: same arity the drafter uses (x, post, comb, y_attn, params)
    block = _block(0)
    x = _streams()
    post, comb, y = block._piece_a(x, {})
    res = block._piece_b(x, post, comb, (y.float() * 0.5).half(), {})
    assert isinstance(res, tuple) and len(res) == 1 and res[0].shape == x.shape
