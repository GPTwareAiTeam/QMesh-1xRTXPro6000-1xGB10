from __future__ import annotations
from typing_extensions import override
import torch
from ..util.tensor import to2
from ..model.config import Config
from . import Module, RMSNorm, LayerNorm, Attention, GatedDeltaNet, GatedMLP, MLP, BlockSparseMLP, Linear
from .hyperconnections import HyperConnection, hc_flush
from ..util import profile_opt
from ..util.segment import Segment
import math
import os

# Piecewise decode graphs (EXL3_WHOLE_GRAPH=1): each block runs graph A (attn HC mix + norm), the eager
# attention core, then graph B (attn residual + MLP + HC apply). Each block's residual is final at its exit.
_PIECEWISE = os.environ.get("EXL3_WHOLE_GRAPH", "1") != "0"
_PIECEWISE_MAXSEQ = int(os.environ.get("EXL3_WHOLE_GRAPH_MAXSEQ", "8"))
# Temporary A/B switch for the merged block-boundary graph (EXL3_PW_MERGE=0 keeps graph B separate). Remove at cutover
_PW_MERGE = os.environ.get("EXL3_PW_MERGE", "1") != "0"
_B1_EAGER_DIAG = os.environ.get("EXL3_DSV4_B1_EAGER_DIAG", "0") == "1"
_B1_MULTIROW = os.environ.get("EXL3_DSV4_B1_MULTIROW", "1") == "1"

# DSpark tap export. EXL3_DSPARK_TAP_SCALE: power-of-two scale applied before the fp16 cast (idea: philbert440,
# 1Cat-vLLM PR #1243). The drafter's only
# consumer is main_proj (no bias) -> main_norm (RMSNorm, scale-invariant up to eps), so the scale is absorbed.
# EXL3_TAP_PROBE=N logs the first N exports per layer (pre-cast magnitudes, overflow counts).
_DSPARK_TAP_SCALE = float(os.environ.get("EXL3_DSPARK_TAP_SCALE", "1.0"))
assert math.frexp(_DSPARK_TAP_SCALE)[0] == 0.5, "EXL3_DSPARK_TAP_SCALE must be a power of two"
_TAP_PROBE = int(os.environ.get("EXL3_TAP_PROBE", "0"))
_tap_probe_calls: dict = {}


def _tap_probe(layer_idx: int, x: torch.Tensor):
    n = _tap_probe_calls.get(layer_idx, 0)
    if n >= _TAP_PROBE:
        return
    _tap_probe_calls[layer_idx] = n + 1
    xf = x.float()
    fin = torch.isfinite(xf)
    mag = xf.abs().masked_fill(~fin, 0.0)
    tok = mag.reshape(-1, mag.shape[-1]).amax(dim = -1)
    nz = mag[mag > 0]
    hot = (tok > 4096.0).nonzero().flatten()[:8].tolist()
    print(f"[TAPPROBE] layer={layer_idx} dtype={x.dtype} shape={tuple(x.shape)} amax={float(mag.max()):.1f} "
          f"nonfinite={int((~fin).sum())} ge65504={int((xf.abs() >= 65504.0).sum())} "
          f"gt4096={int((mag > 4096.0).sum())} subfp16_min={int((nz < 6.1e-5).sum())}/{nz.numel()} "
          f"hot_tokens={[(i, round(float(tok[i]), 1)) for i in hot]} n_tokens={tok.numel()}", flush = True)


def _tap_to_half(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Export cast for DSpark taps: fp32 scale (power of two), fp16 cast, clamp kept as a safety net."""
    if _DSPARK_TAP_SCALE != 1.0:
        x = x.float() * _DSPARK_TAP_SCALE
    if _TAP_PROBE:
        _tap_probe(layer_idx, x)
    return x.half().clamp_(-65504.0, 65504.0)

# Opt-in per-phase decode marks (EXL3_ROUND_PROF=1): the attention core and the MLP core of each
# block, so an eager decode run attributes the verify forward between attention+dense and the MoE
# (which also holds the seam wait, measured separately by RemoteSeamHost's EXL3_SEAM_PROF). Only
# the eager path can host Python marks: with piecewise graphs the MLP is replayed from a graph.
_rp_on = os.environ.get("EXL3_ROUND_PROF", "0") == "1"
_rp_mark_fn = None

def _rpmark(name):
    global _rp_mark_fn
    if _rp_mark_fn is None:
        try:
            from ..generator.roundprof import mark as _m
        except Exception:
            _m = False
        _rp_mark_fn = _m
    if _rp_mark_fn:
        _rp_mark_fn(name)


class TransformerBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        attn_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | GatedDeltaNet | None = None,
        attn_post_norm: RMSNorm | LayerNorm | None = None,
        mlp_norm: RMSNorm | LayerNorm | None = None,
        mlp: MLP | GatedMLP | BlockSparseMLP | None = None,
        mlp_post_norm: RMSNorm | LayerNorm | None = None,
        attn_hc: HyperConnection | None = None,
        mlp_hc: HyperConnection | None = None,
        key_layer_scalar: str | None = None,
        key_attn_resid_scalar: str | None = None,
        key_mlp_resid_scalar: str | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.attn_norm = attn_norm
        self.attn = attn
        self.attn_post_norm = attn_post_norm
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.mlp_post_norm = mlp_post_norm
        self.attn_hc = attn_hc
        self.mlp_hc = mlp_hc
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.key_layer_scalar = key_layer_scalar
        self.key_attn_resid_scalar = key_attn_resid_scalar
        self.key_mlp_resid_scalar = key_mlp_resid_scalar
        self.layer_scalar_t = None
        self.layer_scalar_f = None
        self.attn_resid_scalar = None
        self.mlp_resid_scalar = None

        # Hyperconnection sites (mHC): the block's residual is (bsz, seq, hc_mult, hidden)
        # fp32 streams, mixed at each sublayer site instead of the plain residual add
        if attn_hc is not None or mlp_hc is not None:
            assert attn_hc is not None and mlp_hc is not None, \
                "hyperconnections require both attn_hc and mlp_hc"
            assert all(v is None for v in (
                attn_post_norm, mlp_post_norm,
                key_layer_scalar, key_attn_resid_scalar, key_mlp_resid_scalar,
            )), \
                "hyperconnections cannot combine with residual scalars/post-norms"

        self.register_submodule(self.attn_hc)
        self.register_submodule(self.attn_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.attn_post_norm)
        self.register_submodule(self.mlp_hc)
        self.register_submodule(self.mlp_norm)
        self.register_submodule(self.mlp)
        self.register_submodule(self.mlp_post_norm)

        self.num_slices = mlp.num_slices if mlp else 1


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]

    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.key_layer_scalar:
            self.layer_scalar_t = self.config.stc.get_tensor(
                self.key + "." + self.key_layer_scalar,
                None,
                allow_bf16 = True,
                no_defer = True,
            )
            assert self.layer_scalar_t.numel() == 1
            self.layer_scalar_f = self.layer_scalar_t.float().item()

        # TODO: Residual scalar tensors could be baked into preceding modules for models that use them
        #       (currently only Step3.7 vision tower)
        if self.key_attn_resid_scalar:
            self.attn_resid_scalar = self.config.stc.get_tensor(
                self.key + "." + self.key_attn_resid_scalar,
                device,
                allow_bf16 = True,
                no_defer = True,
            )
        if self.key_mlp_resid_scalar:
            self.mlp_resid_scalar = self.config.stc.get_tensor(
                self.key + "." + self.key_mlp_resid_scalar,
                device,
                allow_bf16 = True,
                no_defer = True,
            )

    def unload(self):
        super().unload()
        self.layer_scalar_t = None
        self.attn_resid_scalar = None
        self.mlp_resid_scalar = None

    def get_tensors(self):
        t = {}
        if self.key_layer_scalar is not None:
            t[self.key + "." + self.key_layer_scalar] = self.layer_scalar_t.data.contiguous()
        if self.key_attn_resid_scalar is not None:
            t[self.key + "." + self.key_attn_resid_scalar] = self.attn_resid_scalar.data.contiguous()
        if self.key_mlp_resid_scalar is not None:
            t[self.key + "." + self.key_mlp_resid_scalar] = self.mlp_resid_scalar.data.contiguous()
        return t

    def weights_numel(self):
        return (
            super().weights_numel() +
            (1 if self.key_layer_scalar is not None else 0) +
            (self.attn_resid_scalar.numel() if self.attn_resid_scalar is not None else 0) +
            (self.mlp_resid_scalar.numel() if self.mlp_resid_scalar is not None else 0)
        )

    @override
    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        # A deferred mHC apply_ (hyperconnections.hc_flush) lands before its streams move to another device
        if "hc_pending" in params and x.device != self.device:
            hc_flush(params)
        return super().prepare_for_device(x, params)

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        pending = params.pop("pw_pending", None)
        if (_PIECEWISE and 1 <= x.shape[1] <= _PIECEWISE_MAXSEQ and self.attn_hc is not None and
            self.mlp_hc is not None and not getattr(params.get("backend"), "disable_piecewise_capture", False)):
            return self._forward_piecewise(x, params, out_dtype, pending)
        assert pending is None, "deferred piecewise MLP half reached a non-piecewise block"

        export_state = params.get("export_state_layers")
        export_state = export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0

        y_resid = None  # pending attn output whose residual add is folded into the MLP input norm

        # Previous block's MLP output whose residual add is folded into this block's input norm (forward_ls sets
        # resid_defer when the next module is a TransformerBlock on the same device)
        pending = params.pop("resid_pending", None)
        fuse_in = (
            pending is not None and self.attn is not None and not self.attn_hc and
            isinstance(self.attn_norm, RMSNorm) and self.attn_norm.can_fuse_residual(x, pending)
        )
        if pending is not None and not fuse_in:
            x += pending

        if self.attn:
            if self.attn_hc:
                hc_post, hc_comb, y, normed = self.attn_hc.mix_norm(x, params, self.attn_norm)
                y = y.half()
                if self.attn_norm and not normed:
                    y = self.attn_norm.forward(y, params, out_dtype = torch.half)
            elif fuse_in:
                y = self.attn_norm.forward(pending, params, out_dtype = torch.half, residual_in = x)
            elif self.attn_norm:
                y = self.attn_norm.forward(x, params, out_dtype = torch.half)
            else:
                y = x.half()
            y = self.attn.forward(y, params)
            if _rp_on and y.shape[1] == 1:
                _rpmark("attn_end")
            if params.get("prefill") and not export_state:
                return x
            if self.attn_resid_scalar is not None:
                y *= self.attn_resid_scalar
            if self.attn_hc:
                x = self.attn_hc.apply_(x, y, hc_post, hc_comb, params)
            elif self.attn_post_norm:
                self.attn_post_norm.forward(y, params, residual = x)
            elif self.mlp is not None and self.mlp_norm is not None and self.mlp_norm.can_fuse_residual(x, y):
                y_resid = y
            else:
                x += y

        if self.mlp:
            if self.mlp_hc:
                hc_post, hc_comb, y, normed = self.mlp_hc.mix_norm(x, params, self.mlp_norm)
                y = y.half()
                if self.mlp_norm and not normed:
                    y = self.mlp_norm.forward(y, params, out_dtype = torch.half)
            else:
                params["residual"] = x
                if y_resid is not None:
                    y = self.mlp_norm.forward(y_resid, params, out_dtype = torch.half, residual_in = x)
                elif self.mlp_norm:
                    y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
                else:
                    y = x.half()
            y = self.mlp.forward(y, params)
            if _rp_on and y.shape[1] == 1:
                _rpmark("mlp_end")
            if self.mlp_resid_scalar is not None:
                y *= self.mlp_resid_scalar
            if self.mlp_hc:
                x = self.mlp_hc.apply_(x, y, hc_post, hc_comb, params)
            elif self.mlp_post_norm:
                self.mlp_post_norm.forward(y, params, residual = x)
            elif (
                params.get("resid_defer") and not export_state and self.layer_scalar_f is None and
                (out_dtype or self.out_dtype) in (None, x.dtype)
            ):
                params["resid_pending"] = y
                return x
            else:
                x += y

        if export_state:
            s = params.get("export_states")
            if not s:
                s = params["export_states"] = []
            # With hyperconnections the residual is a stream stack; export the stream mean as
            # the collapsed hidden state (streams start as broadcast copies of the embedding)
            x_ = x.mean(dim = 2) if self.attn_hc else x
            s.append(_tap_to_half(x_, self.layer_idx))

        if self.layer_scalar_f is not None:
            x *= self.layer_scalar_f

        return to2(x, out_dtype, self.out_dtype)

    def _pw_defer(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None) -> bool:
        """True when this block's MLP half may stay pending and run inside the next block's graph A (one replay per
        block boundary instead of two). The driver names the next module in params["pw_next"]; the next block must
        take the piecewise path too (same shape gate as forward)"""
        nxt = params.get("pw_next")
        if not _PW_MERGE or type(nxt) is not TransformerBlock or nxt.attn_hc is None or nxt.mlp_hc is None:
            return False
        if not (1 <= x.shape[1] <= _PIECEWISE_MAXSEQ) or params.get("prefill"):
            return False
        if out_dtype is not None or self.out_dtype is not None:
            return False
        export_state = params.get("export_state_layers")
        if export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0:
            return False
        return nxt.device == self.device

    def _forward_piecewise(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None, pending = None):
        """Decode body split around the attention core. Graph A: attn HC mix + norm, outputs (post, comb, y).
        Eager: attn.forward. Graph B: attn HC apply, MLP HC mix + norm, MLP, MLP HC apply. A block whose graph B is
        deferred (pending = the previous block's (block, x, post, comb, y_attn)) runs that graph B inside its own
        graph A. Residual x is threaded through the segments as a static buffer; HC post/comb are static outputs."""
        if (x.shape[1] == 1 or (_B1_MULTIROW and x.shape[0] == 1 and x.shape[1] <= 8)) and \
           self.attn is not None and hasattr(self.attn, "b1_prep"):
            b1 = self.attn.b1_prep(params, x.shape[1])
            if b1 is not None:
                return self._forward_piecewise_b1(x, params, out_dtype, pending, b1)
        if not hasattr(self, "_segments"):
            self._segments = {}
        key = (tuple(x.shape), x.dtype, out_dtype, None if pending is None else id(pending[0]))
        segs = self._segments.get(key)
        if segs is None:
            if pending is None:
                segs = (Segment(self._piece_a, carry = ("hc_pre",)), Segment(self._piece_b, carry = ("hc_pre",)))
            else:
                prev = pending[0]
                segs = (Segment(lambda *a: self._piece_ba(prev, *a), carry = ("hc_pre",)),
                        Segment(self._piece_b, carry = ("hc_pre",)))
            self._segments[key] = segs
        seg_a, seg_b = segs
        if pending is None:
            post, comb, y = seg_a(x, params)
        else:
            _, x_b, post_b, comb_b, y_b = pending
            x, post, comb, y = seg_a(x_b, post_b, comb_b, y_b, params)
        y_attn = self.attn.forward(y, params)
        if self._pw_defer(x, params, out_dtype):
            params["pw_pending"] = (self, x, post, comb, y_attn)
            return x
        x, = seg_b(x, post, comb, y_attn, params)
        export_state = params.get("export_state_layers")
        if export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0:
            states = params.get("export_states")
            if not states:
                states = params["export_states"] = []
            states.append(_tap_to_half(x.mean(dim = 2), self.layer_idx))
        return to2(x, out_dtype, self.out_dtype)

    def _piece_ba(self, prev, x_b, post_b, comb_b, y_b, params):
        """Previous block's graph B (its deferred MLP half), then this block's graph A"""
        x, = prev._piece_b(x_b, post_b, comb_b, y_b, params)
        post, comb, y = self._piece_a(x, params)
        return x, post, comb, y

    def _forward_piecewise_b1(self, x, params, out_dtype, pending, b1):
        """Decode body with the device-driven attention core inside the segment: graph A, the attention body
        (attn._batch_body) and graph B of the previous block (when pending) are one captured replay; graph B
        defers exactly as in _forward_piecewise.
        b1 is the attention's host prepass (ctx, arr, bt_st) for this step."""
        ctx, arr, bt_st = b1
        prev = None if pending is None else pending[0]
        # ctx[5] is the indexer selection regime and ctx[6] the step's new-entry run length: a step that
        # changes either must capture a body of its own (the run's base entry stays device-driven)
        key = ("b1", tuple(x.shape), x.dtype, out_dtype, id(ctx[0]), id(ctx[1]), ctx[5], ctx[6],
               None if prev is None else id(prev))
        if not hasattr(self, "_segments"):
            self._segments = {}
        segs = self._segments.get(key)
        if segs is None:
            segs = (Segment(lambda *a: self._piece_b1(ctx, prev, *a), carry = ("hc_pre",)),
                    Segment(self._piece_b, carry = ("hc_pre",)))
            self._segments[key] = segs
        seg_a, seg_b = segs
        if pending is None:
            x, post, comb, y = (self._piece_b1(ctx, prev, x, arr, bt_st, params)
                                if _B1_EAGER_DIAG else seg_a(x, arr, bt_st, params))
        else:
            _, x_b, post_b, comb_b, y_b = pending
            x, post, comb, y = (self._piece_b1(ctx, prev, x_b, post_b, comb_b, y_b, arr, bt_st, params)
                                if _B1_EAGER_DIAG else seg_a(x_b, post_b, comb_b, y_b, arr, bt_st, params))
        if self._pw_defer(x, params, out_dtype):
            params["pw_pending"] = (self, x, post, comb, y)
            return x
        x, = seg_b(x, post, comb, y, params)
        export_state = params.get("export_state_layers")
        if export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0:
            states = params.get("export_states")
            if not states:
                states = params["export_states"] = []
            states.append(_tap_to_half(x.mean(dim = 2), self.layer_idx))
        return to2(x, out_dtype, self.out_dtype)

    def _piece_b1(self, ctx, prev, *a):
        """Previous block's graph B, then graph A, then the attention body."""
        if prev is None:
            x, arr, bt_st, params = a
        else:
            x_b, post_b, comb_b, y_b, arr, bt_st, params = a
            x, = prev._piece_b(x_b, post_b, comb_b, y_b, params)
        post, comb, y = self._piece_a(x, params)
        rsl, kl, m, w, kp, cand_regime, nnew = ctx
        y_attn = self.attn._batch_body(y, arr, y.shape[0], y.shape[1], y.shape[0] * y.shape[1], rsl, kl, bt_st, m, w, kp,
                                       params, None, cand_regime = cand_regime, nnew = nnew)
        return x, post, comb, y_attn

    def _piece_a(self, x, params):
        post, comb, y, normed = self.attn_hc.mix_norm(x, params, self.attn_norm)
        if self.attn_norm and not normed:
            y = self.attn_norm.forward(y.half(), params, out_dtype = torch.half)
        return post, comb, y.half()

    def _piece_b(self, x, post, comb, y_attn, params):
        if self.attn_resid_scalar is not None:
            y_attn = y_attn * self.attn_resid_scalar
        x = self.attn_hc.apply_(x, y_attn, post, comb, params)
        hc_post, hc_comb, y, normed = self.mlp_hc.mix_norm(x, params, self.mlp_norm)
        if self.mlp_norm and not normed:
            y = self.mlp_norm.forward(y.half(), params, out_dtype = torch.half)
        y = self.mlp.forward(y.half(), params)
        if self.mlp_resid_scalar is not None:
            y = y * self.mlp_resid_scalar
        # Under HC fold, apply_ defers into params["hc_pending"]; flush it here so the update is captured in
        # this graph (a pending apply would not run on replay and would leak across the segment boundary)
        x = self.mlp_hc.apply_(x, y, hc_post, hc_comb, params)
        hc_flush(params)
        return x,


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": TransformerBlock,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "out_dtype": self.out_dtype,
                "key_layer_scalar": self.key_layer_scalar,
                "key_attn_resid_scalar": self.key_attn_resid_scalar,
                "key_mlp_resid_scalar": self.key_mlp_resid_scalar,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "attn_hc",
                "attn_norm",
                "attn",
                "attn_post_norm",
                "mlp_hc",
                "mlp_norm",
                "mlp",
                "mlp_post_norm",
            )},
            # Per-layer scalars load from the tensor collection, which TP children don't have
            "layer_scalar_f": self.layer_scalar_f,
            "attn_resid_scalar": producer.send(self.attn_resid_scalar),
            "mlp_resid_scalar": producer.send(self.mlp_resid_scalar),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        module = TransformerBlock(
            config = None,
            **exported["kwargs"],
            attn_hc = _import("attn_hc"),
            attn_norm = _import("attn_norm"),
            attn = _import("attn"),
            attn_post_norm = _import("attn_post_norm"),
            mlp_hc = _import("mlp_hc"),
            mlp_norm = _import("mlp_norm"),
            mlp = _import("mlp"),
            mlp_post_norm = _import("mlp_post_norm"),
        )

        module.layer_scalar_f = exported.get("layer_scalar_f")
        module.attn_resid_scalar = consumer.recv(exported.get("attn_resid_scalar"), cuda = True)
        module.mlp_resid_scalar = consumer.recv(exported.get("mlp_resid_scalar"), cuda = True)
        module.device = device
        return module


class ParallelDecoderBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        input_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | None = None,
        mlp: MLP | GatedMLP | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.input_norm = input_norm
        self.attn = attn
        self.mlp = mlp
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.register_submodule(self.input_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.mlp)

        self.num_slices = mlp.num_slices if mlp else 1

        self.tp_reduce = False


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        y = self.input_norm.forward(x, params, out_dtype = torch.half)
        y1 = self.attn.forward(y, params)
        if not params.get("prefill"):
            y2 = self.mlp.forward(y, params)
            y1 += y2

            if self.tp_reduce:
                self.tp_collect(params["backend"], y1)

            x += y1

        return to2(x, out_dtype, self.out_dtype)


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": ParallelDecoderBlock,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "out_dtype": self.out_dtype,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "input_norm",
                "attn",
                "mlp",
            )},
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        device = local_context["device"]

        def _import(name, **kwargs):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan, **kwargs) \
                if exported.get(name) else None

        module = ParallelDecoderBlock(
            config = None,
            **exported["kwargs"],
            input_norm = _import("input_norm"),
            attn = _import("attn", skip_reduction = True),
            mlp = _import("mlp", skip_reduction = True),
        )
        module.device = device

        # Use single reduction for sum of mlp and attn
        module.tp_reduce = True
        module.tp_owner = module.tp_single_owner(
            local_context, exported["attn"]["kwargs"]["key"], exported["mlp"]["kwargs"]["key"])
        return module
