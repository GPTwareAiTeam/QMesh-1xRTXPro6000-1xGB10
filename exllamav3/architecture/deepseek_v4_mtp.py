from __future__ import annotations
from typing_extensions import override
import torch

import weakref

from ..cache import Cache
from ..cache.dsa import DSparkState
from ..constants import PAGE_SIZE
from ..model.model import Model
from ..modules import RMSNorm, Linear, Embedding, GatedMLP, BlockSparseMLP, TransformerBlock, \
    HyperConnection, HyperHead
from ..modules.arch_specific.dspark import DSparkAttention, DSparkInputLayer, to_dev
from ..modules.attn import prepare_for_attn
from ..util.device_copy import to_device
from ..util.tensor import get_for_device
from ..util.segment import Segment
from ..modules.hyperconnections import hc_flush
from ..util import pad_bucket
import os as _os

# Opt-in per-round phase marks (EXL3_ROUND_PROF=1); lazy import to avoid a package cycle, no-op when off
_rp_on = _os.environ.get("EXL3_ROUND_PROF", "0") == "1"
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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .deepseek_v4 import DeepseekV4Config

# Capture the drafter's sampling chain (trunk head + sequential Markov loop + confidence
# head and its length gate) as one graph replay.
_MARKOV_GRAPH = _os.environ.get("EXL3_DSPARK_MARKOV_GRAPH", "1") == "1"
_BLOCK_GRAPH = _os.environ.get("EXL3_DSPARK_BLOCK_GRAPH", "1") == "1"

# EXL3_DSPARK_GATE: how the drafter decides each round's verified width from its per-position confidences.
#   "thr" (default): verify up to the first position whose confidence is below EXL3_DSPARK_CONF.
#   "ev": choose the width k that maximises expected emitted tokens per unit round cost,
#         E[k] / cost(k + 1), E[k] = 1 + sum_{i<=k} prod_{j<=i} min(1, s * c_j ** g),
#         with the measured relative round cost per verify-row count from EXL3_DSPARK_EV_COST
#         ("rows:cost,...", any unit; interpolated). g = EXL3_DSPARK_EV_GAMMA, s = EXL3_DSPARK_EV_SCALE,
#         k in [EXL3_DSPARK_EV_KMIN, block]. Computed on device inside the captured sampling graph.
_GATE = _os.environ.get("EXL3_DSPARK_GATE", "auto")
_EV_GAMMA = float(_os.environ.get("EXL3_DSPARK_EV_GAMMA", "1.0"))
_EV_SCALE = float(_os.environ.get("EXL3_DSPARK_EV_SCALE", "1.0"))
_EV_KMIN = int(_os.environ.get("EXL3_DSPARK_EV_KMIN", "1"))
_EV_COST_RAW = _os.environ.get("EXL3_DSPARK_EV_COST", "1:16.35,2:18.9,3:20.7,4:22.2,5:23.6,6:24.9,7:26.1,8:27.3")
#   "auto": the threshold gate with a content-adaptive threshold. An exponential average of accepted drafts per
#         round (half-life EXL3_DSPARK_AUTO_HL rounds) selects EXL3_DSPARK_AUTO_LO while it is below
#         EXL3_DSPARK_AUTO_SWITCH (low-acceptance text: prose, multilingual), else EXL3_DSPARK_CONF. The threshold
#         lives in a device tensor the captured sampling graph reads, so switching never re-captures.
# Defaults retuned for the 2.499-bpw TP=2 checkpoint (7 prompts, 256 tokens, ABBA: 143.3/144.7 -> 153.4/154.0 tok/s
# vs 0.35 / 1.5; 5 of 7 prompts faster, two short ones -2..-7%). It changes the verify width, and a different verify
# width is a different numeric path (batch-width non-invariance), so greedy output can differ on near-tied tokens
_AUTO_LO = float(_os.environ.get("EXL3_DSPARK_AUTO_LO", "0.25"))
_AUTO_SWITCH = float(_os.environ.get("EXL3_DSPARK_AUTO_SWITCH", "2.0"))
_AUTO_HL = float(_os.environ.get("EXL3_DSPARK_AUTO_HL", "32"))


def _ev_cost_vec(max_rows):
    """Relative round cost for 1..max_rows verify rows from EXL3_DSPARK_EV_COST, linearly interpolated and
    extrapolated with the end slopes."""
    pts = sorted((int(a), float(b)) for a, b in (p.split(":") for p in _EV_COST_RAW.split(",") if p.strip()))
    out = []
    for r in range(1, max_rows + 1):
        lo = max((p for p in pts if p[0] <= r), default = pts[0])
        hi = min((p for p in pts if p[0] >= r), default = pts[-1])
        if lo[0] == hi[0]:
            if r < pts[0][0]:
                (a0, c0), (a1, c1) = pts[0], pts[1]
                out.append(c0 - (c1 - c0) / (a1 - a0) * (a0 - r))
            elif r > pts[-1][0]:
                (a0, c0), (a1, c1) = pts[-2], pts[-1]
                out.append(c1 + (c1 - c0) / (a1 - a0) * (r - a1))
            else:
                out.append(lo[1])
        else:
            out.append(lo[1] + (hi[1] - lo[1]) * (r - lo[0]) / (hi[0] - lo[0]))
    return out

_EV_COST_CACHE = {}


def _ev_cost_dev(s, dtype, device):
    """Device cost vector for verify rows 2..s+1, built once per (s, dtype, device) so the captured sampling
    graph reads a persistent tensor instead of allocating one during capture (the eager warm-up calls fill it)."""
    key = (s, dtype, str(device))
    t = _EV_COST_CACHE.get(key)
    if t is None:
        t = torch.tensor(_ev_cost_vec(s + 1)[1:], dtype = dtype, device = device)
        _EV_COST_CACHE[key] = t
    return t



def _ev_lens(cs, threshold):
    """Per-row verified width from per-position confidences cs (b, s). 'thr': leading run above threshold.
    'ev': argmax over k of expected tokens / cost(k + 1); k = 0 (no speculation) is never chosen when
    EXL3_DSPARK_EV_KMIN >= 1."""
    if _GATE != "ev":
        # threshold: a float, or (gate "auto") a persistent (1,) device tensor the captured graph reads
        keep = cs >= threshold
        return torch.cumprod(keep.to(torch.int32), dim = 1).sum(dim = 1)
    b, s = cs.shape
    q = (cs.clamp_min(0) ** _EV_GAMMA * _EV_SCALE).clamp(max = 1.0)
    surv = torch.cumprod(q, dim = 1)                                    # P(first i drafts accepted)
    exp_tok = 1.0 + torch.cumsum(surv, dim = 1)                         # E[tokens] at k = 1..s
    cost = _ev_cost_dev(s, cs.dtype, cs.device)                         # rows 2..s+1
    rate = exp_tok / cost
    kmin = max(1, min(_EV_KMIN, s))
    if kmin > 1:
        rate = rate[:, kmin - 1:]
        return rate.argmax(dim = 1).to(torch.int32) + kmin
    return rate.argmax(dim = 1).to(torch.int32) + 1


class _AutoThreshold:
    """Content-adaptive gate threshold for EXL3_DSPARK_GATE=auto. The value the captured sampling graph compares
    against lives in a persistent device tensor (one per device), rewritten from the host only when the selected
    level changes, so the graph is captured once and replays read the current threshold. The selection uses an
    exponential average of accepted drafts per verify round; it is a host-side integer-derived value, identical on
    every TP rank because acceptance is rank-agreed."""

    def __init__(self, hi: float):
        self.hi = hi
        self.decay = 0.5 ** (1.0 / max(_AUTO_HL, 1e-6))
        self.avg = 2.0
        self.level = hi
        self.dev = {}

    def tensor(self, device):
        t = self.dev.get(str(device))
        if t is None:
            t = torch.full((1,), self.level, dtype = torch.float32, device = device)
            self.dev[str(device)] = t
        return t

    def reset(self):
        self.avg = 2.0
        self._set(self.hi)

    def observe(self, accepted: int):
        self.avg = self.avg * self.decay + accepted * (1.0 - self.decay)
        self._set(_AUTO_LO if self.avg < _AUTO_SWITCH else self.hi)

    def _set(self, level: float):
        if level != self.level:
            self.level = level
            for t in self.dev.values():
                t.fill_(level)

# DeepSeek-V4 MTP component = the DSPARK block drafter, stored under the mtp.* namespace:
# n_mtp_layers full transformer blocks (compressor-less DSA attention, 256-expert noaux MoE
# with shared expert, mHC residual streams), entered through main_proj/main_norm on block 0
# (projecting the concatenation of the trunk's stream-mean taps at dspark_target_layer_ids)
# and exited through block N-1's own hc_head collapse + final norm into the SHARED trunk
# head. The last block also carries a factorized-bigram markov head (per-token logit bias
# during the sequential sampling loop) and a confidence head (per-position score for
# dynamic draft length). Reference implementation ships in the checkpoint:
# hf/inference/model.py (DSparkBlock / DSparkAttention / forward_spec).


def _trace_window(env):
    """Diagnostic fixed verifier width for eager batch traces; None keeps the adaptive serving gate."""
    width = int(env.get("DSPARK_ROWTRACE_WIDTH", "0") or 0)
    return width if width > 1 and env.get("EXL3_WHOLE_GRAPH") == "0" else None

class DeepseekV4MTPModel(Model):

    def __init__(
        self,
        config: DeepseekV4Config,
        num_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        carry_pre: bool = False,
        q_head_norm: bool = True,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        h = config.hidden_size
        n_taps = max(len(config.dspark_target_layer_ids), 1)

        # Entry: main_proj / main_norm (consumed by update_kv_from_target) + the noise-block
        # embedder for the draft forward
        self.input_layer = DSparkInputLayer(
            config = config,
            key = "mtp.input",
            hidden_size = h,
            n_taps = n_taps,
            hc_mult = config.hc_mult,
            noise_token_id = config.dspark_noise_token_id,
            block_size = config.dspark_block_size,
            rms_norm_eps = config.rms_norm_eps,
        )
        self.modules = [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(config.num_mtp_layers):
            layer_type = config.mtp_layer_types[idx]
            assert layer_type == "sliding", \
                f"DeepseekV4 MTP: expected compressor-less (sliding) blocks, got {layer_type}"
            key = f"mtp.{idx}"
            attn = DSparkAttention(
                config = config,
                key = f"{key}.attn",
                layer_idx = idx,
                layer_type = layer_type,
                hidden_size = h,
                num_q_heads = config.num_q_heads,
                head_dim = config.head_dim,
                rope_head_dim = config.qk_rope_head_dim,
                q_lora_rank = config.q_lora_rank,
                o_groups = config.o_groups,
                o_lora_rank = config.o_lora_rank,
                sliding_window = config.sliding_window,
                compress_rate = None,
                index_n_heads = config.index_n_heads,
                index_head_dim = config.index_head_dim,
                index_topk = config.index_topk,
                rope_theta = config.rope_theta,
                compress_rope_theta = config.compress_rope_theta,
                rope_scaling = config.rope_scaling,
                rms_norm_eps = config.rms_norm_eps,
                qmap = "block.attn",
                out_dtype = torch.float,
                qbits_key = "mtp_bits",
                select_hq_bits = 2,
                q_head_norm = q_head_norm,
            )
            self.attn_modules.append(attn)
            mlp = BlockSparseMLP(
                config = config,
                key = f"{key}.ffn",
                hidden_size = h,
                intermediate_size = config.moe_intermediate_size,
                num_experts = num_experts or config.num_experts,
                num_experts_per_tok = num_experts_per_tok or config.num_experts_per_tok,
                key_up = "experts.{expert_idx}.w3",
                key_gate = "experts.{expert_idx}.w1",
                key_down = "experts.{expert_idx}.w2",
                key_routing_gate = "gate",
                key_e_score_bias = "gate.bias",
                qmap = "block.mlp",
                interm_dtype = torch.half,
                out_dtype = torch.float,
                activation_fn = "silu",
                act_limit = config.swiglu_limit,
                router_type = "sqrtsp",
                routed_scaling_factor = config.routed_scaling_factor,
                qbits_key = "mtp_bits",
                shared_experts = GatedMLP(
                    config = config,
                    key = f"{key}.ffn.shared_experts",
                    hidden_size = h,
                    intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                    key_up = "w3",
                    key_gate = "w1",
                    key_down = "w2",
                    qmap = "block.mlp",
                    out_dtype = torch.float,
                    activation_fn = "silu",
                    act_limit = config.swiglu_limit,
                    qbits_key = "mtp_bits",
                    select_hq_bits = 2,
                ),
            )
            def _hc(tag: str):
                return HyperConnection(
                    config = config,
                    key = f"{key}.hc_{tag}",
                    hc_mult = config.hc_mult,
                    hidden_size = h,
                    sinkhorn_iters = config.hc_sinkhorn_iters,
                    hc_eps = config.hc_eps,
                    rms_norm_eps = config.rms_norm_eps,
                    carry_pre = carry_pre,
                )
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(config, f"{key}.attn_norm", config.rms_norm_eps),
                    attn = attn,
                    attn_hc = _hc("attn"),
                    mlp_norm = RMSNorm(config, f"{key}.ffn_norm", config.rms_norm_eps),
                    mlp = mlp,
                    mlp_hc = _hc("ffn"),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1
        last = f"mtp.{config.num_mtp_layers - 1}"

        # Exit: the drafter's own stream collapse + final norm (logits via the trunk's
        # shared head at runtime). The draft forward runs modules[:fwd_end_idx]
        self.modules += [
            HyperHead(
                config = config,
                key = f"{last}.hc_head",
                hc_mult = config.hc_mult,
                rms_norm_eps = config.rms_norm_eps,
                hc_eps = config.hc_eps,
                carry_pre = carry_pre,
            ),
            RMSNorm(
                config = config,
                key = f"{last}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
        ]
        self.fwd_end_idx = len(self.modules)

        # Markov bigram head (per-token logit bias in the sampling loop) and confidence
        # head (per-position draft-length score), unquantized
        tp_size = getattr(config, "tp_size", 1)
        self.tp_size = tp_size
        self.markov_w1 = Embedding(
            config = config,
            key = f"{last}.markov_head.markov_w1",
            vocab_size = config.vocab_size // tp_size,
            hidden_size = config.dspark_markov_rank,
            allow_table = False,
        )
        # Device-resident (~63 MB): the sampling loop stays on-stream with no host syncs
        self.markov_w1.caps["prefer_cpu"] = False
        self.markov_w2 = Linear(
            config = config,
            key = f"{last}.markov_head.markov_w2",
            in_features = config.dspark_markov_rank,
            out_features = config.vocab_size // tp_size,
            qmap = None,
        )
        self.confidence = Linear(
            config = config,
            key = f"{last}.confidence_head.proj",
            in_features = h + config.dspark_markov_rank,
            out_features = 1,
            qmap = None,
            pad_to = 1,
        )
        self.modules += [self.markov_w1, self.markov_w2, self.confidence]


        # Draft length gate: keep the longest prefix with sigmoid(confidence) >= threshold
        import os
        self.draft_conf_threshold = float(os.environ.get("EXL3_DSPARK_CONF", "0.5"))
        self.auto_threshold = _AutoThreshold(self.draft_conf_threshold) if _GATE == "auto" else None
        # Diagnostic only: fixed verifier width for non-graphed eager batch traces; serving remains adaptive.
        self.diag_draft_window = _trace_window(os.environ)
        self._conf_stats = [] if os.environ.get("EXL3_DSPARK_CONF_STATS") else None

        self.logit_layer_idx = None
        self.caps.update({
            "supports_tp": False,
            "autosplit_load_fwd": False,
            "attach_target": True,
            "dflash_draft": True,
            "default_draft_size": config.dspark_block_size,
            # One forward per draft call covers exactly dspark_block_size positions; a longer window is
            # unproduceable (the Generator clamps num_draft_tokens to this with a startup warning)
            "max_draft_size": config.dspark_block_size,
            # Main-kv rows live in per-sequence window rings (DSparkLayerState), checkpointed with the target
            "recurrent_states": True,
        })
        self.recurrent_state_cls = DSparkState
        self.attached_model = None
        # Private trunk embedding/head instances, loaded lazily by attach_to when the
        # target is tensor-parallel (its own embedding and head then live in the rank
        # worker processes and cannot be borrowed)
        self.own_embed = None
        self.own_head = None
        self.draft_verifier_params.update({
            "export_state_layers": set(config.dspark_target_layer_ids),
        })

        # Activate all experts during any calibrated capture pass
        self.calibration_all_experts = True


    def attach_to(self, target):
        self.attached_model = weakref.ref(target)
        self.input_layer.attached_model = weakref.ref(target)
        if getattr(target, "loaded_tp", False):
            self._load_own_embed_head()


    def _load_own_embed_head(self):
        """Load private copies of the trunk's embedding (system RAM, prefer_cpu) and lm
        head (drafter's device). Only needed when the target runs TP; keeping the local
        head also keeps the sequential markov/argmax draft loop free of collectives."""
        if self.own_head is not None:
            return
        cfg = self.config
        embed = Embedding(
            config = cfg,
            key = "embed",
            vocab_size = cfg.vocab_size,
            hidden_size = cfg.hidden_size,
        )
        embed.load(torch.device("cpu"))
        head_alt_key = None
        if cfg.tie_word_embeddings and not cfg.stc.has_tensor("head"):
            head_alt_key = "embed"
        head = Linear(
            config = cfg,
            key = "head",
            qbits_key = "head_bits",
            alt_key = head_alt_key,
            in_features = cfg.hidden_size,
            out_features = cfg.vocab_size,
            caps = {"logits_output": True},
        )
        head.load(self.modules[self.fwd_end_idx - 1].device)
        self.own_embed = embed
        self.own_head = head
        self.input_layer.own_embed = embed


    @override
    @torch.inference_mode
    def forward(self, input_ids: torch.Tensor, params: dict, **kwargs) -> torch.Tensor:
        """Draft-block forward: seed token per row -> [seed, noise x (block-1)] -> input
        layer (embed + stream expand) -> blocks -> hc_head collapse -> norm. Returns the
        normed state (bsz, block, hidden) fp16 for sample_from_state."""
        x = self.prepare_inputs(input_ids, params)
        params["dspark_seed_ids"] = input_ids
        for i, m in enumerate(self.modules[:self.fwd_end_idx]):
            x = m.prepare_for_device(x, params)
            if i == self.fwd_end_idx - 1:
                params["dspark_prenorm"] = x.clone()    # confidence input is PRE-norm
            if (_BLOCK_GRAPH and x.shape[0] == 1 and isinstance(m, TransformerBlock)
                    and getattr(params.get("backend"), "piecewise_p2p_capture", False)):
                # Whole block as one graph: positions and block table enter as static tensors so the
                # captured attention never reads host state
                positions = get_for_device(params, "cache_seqlens", m.device)
                block_table = get_for_device(params, "block_table", m.device)
                if not hasattr(m, "_dspark_full_segments"):
                    m._dspark_full_segments = {}
                key = (tuple(x.shape), x.dtype, tuple(block_table.shape), id(params["cache"]))
                seg = m._dspark_full_segments.get(key)
                if seg is None:
                    def block_body(x, positions, block_table, p, block = m):
                        p["dspark_graph_positions"] = positions
                        p["dspark_graph_block_table"] = block_table
                        post, comb, y = block._piece_a(x, p)
                        y = block.attn.forward(y, p)
                        p.pop("dspark_graph_positions")
                        p.pop("dspark_graph_block_table")
                        result, = block._piece_b(x, post, comb, y, p)
                        hc_flush(p)
                        return result
                    seg = m._dspark_full_segments[key] = Segment(block_body, carry = ("hc_pre",))
                x, = seg(x, positions, block_table, params)
                rp = getattr(seg, "replays", 0)
                if rp in (1, 100) and getattr(self.config, "tp_rank", 0) == 0:
                    print(f"DSPARK_BLOCK_GRAPH key={key} graph={'yes' if seg.graph is not None else 'no'} "
                          f"replays={rp}", flush = True)
            else:
                x = m.forward(x, params)
        return x


    def update_kv_from_target(
        self,
        target_hidden: list,
        cache: Cache,
        params: dict,
        lengths: list[int] = None,
    ):
        """Write main-kv rows for freshly accepted/prefilled target positions into the drafter's window
        rings: concat the trunk tap states, project through main_proj/main_norm once, then per block
        derive and store that block's kv rows.

        params carries cache_seqlens (the rows' first positions) and recurrent_states (one DSparkState
        per batch row, at that position); lengths[i] is how many of row i's rows are real (default: all).
        A row longer than the ring only writes its tail: earlier rows would be overwritten before any
        draft could read them. Rows past a row's length are written too (one batched write) but land
        above its position, where the next write overwrites them before the drafter reads them; a row
        of length 0 writes into the trash page."""
        rsg = params["recurrent_states"]
        seqlens = params["cache_seqlens"]
        seqlens = seqlens.tolist() if torch.is_tensor(seqlens) else list(seqlens)
        bsz = len(rsg)
        T = target_hidden[0].shape[1]
        if lengths is None:
            lengths = [T] * bsz
        n = max(lengths)
        if n == 0:
            return
        ring_rows = rsg[0].ring_rows
        window = rsg[0].window
        # Rows that the drafter can still read: the last ring_rows of the row. Only a single-row write (prefill)
        # can be longer than the ring; a batched write's padding rows (past a row's length) must stay clear of
        # that row's window, which they do while they fit in the ring's slack above it
        skip = max(0, n - ring_rows)
        assert skip == 0 or all(ln == n for ln in lengths), "DSpark ring: a long write must not be padded"
        assert all(ln == 0 or n - ln <= ring_rows - window for ln in lengths), \
            "DSpark ring: padding rows of a batched write would overwrite a row's window"
        width = -(-(max(seqlens) + n) // PAGE_SIZE) + 1
        tables = []
        pos0 = []
        for st, p, ln in zip(rsg, seqlens, lengths):
            if ln > 0:
                assert st.position == p, f"DSpark ring state at {st.position}, rows start at {p}"
                tables.append(st.block_table(width))
            else:
                tables.append(st.trash_table(width))
            pos0.append(p + skip)
        target_hidden = [t[:, skip:n] for t in target_hidden]
        device = self.input_layer.main_proj.device
        for i in range(len(target_hidden)):
            target_hidden[i] = to_device(target_hidden[i], device)
        th = torch.cat(target_hidden, dim = -1)
        p = {
            "cache": cache,
            "block_table": torch.cat(tables, dim = 0),
            "cache_seqlens": torch.tensor(pos0, dtype = torch.int32),
            "dspark_main_x": self.input_layer.project_taps(th, {}),
        }
        for attn in self.attn_modules:
            mx = get_for_device(p, "dspark_main_x", attn.device)
            attn.update_kv_rows(mx, p)
        for st, q, ln in zip(rsg, seqlens, lengths):
            if ln > 0:
                st.wrote(q, ln, n)


    def _markov_body(self, state: torch.Tensor, seed: torch.Tensor, prenorm: torch.Tensor, p: dict):
        """Device-only form of sample_from_state, captured as one graph: trunk head over the whole
        block, the sequential markov chain (7 tiny steps, one collective each), the confidence head
        and the length gate. Returns (ids, conf_logits, conf_len); nothing is read back on the host.

        Every per-round value enters through a static tensor (state, seed, prenorm) and every
        per-round result leaves as a static output, so the captured body never depends on host state
        that changes between replays. `p` is the capture-time params copy."""
        import torch.nn.functional as F
        if self.own_head is not None:
            lm = self.own_head
        else:
            am = self.attached_model()
            lm = am.modules[am.logit_layer_idx]
        logits = lm.forward(lm.prepare_for_device(state, p), p).float()
        b, s, V = logits.shape
        dev = self.markov_w2.device
        logits = to_dev(logits, dev)
        seed = to_dev(seed, dev)
        if getattr(self, "_markov_w1_dev", None) is None:
            self._markov_w1_dev = to_dev(self.markov_w1.embedding.weight.data, dev)
        w2 = self.markov_w2
        tp = self.tp_size
        out = torch.empty((b, s + 1), dtype = torch.long, device = dev)
        out[:, 0] = seed[:, -1]
        embs = []
        for i in range(s):
            if tp > 1:
                local_ids = out[:, i] - self.config.tp_rank * self.markov_w1.vocab_size
                mask = (local_ids >= 0) & (local_ids < self.markov_w1.vocab_size)
                emb = F.embedding(torch.where(mask, local_ids, torch.zeros_like(local_ids)), self._markov_w1_dev).float()
                emb = emb * mask.unsqueeze(-1).float()
                p["backend"].all_reduce(emb)
                emb = emb.half()
            else:
                emb = F.embedding(out[:, i], self._markov_w1_dev).half()
            embs.append(emb)
            bias = w2.forward(emb.unsqueeze(1), p).float()
            if tp > 1:
                bias = p["backend"].all_gather(bias, tp)
            logits[:, i] += bias[:, 0]
            out[:, i + 1] = torch.argmax(logits[:, i], dim = -1)

        # Confidence head and the length gate, all device side: column 0 is the (unused) seed
        # position, the generator crops it
        cdev = self.confidence.device
        xpre = to_dev(prenorm, cdev).half()
        me = to_dev(torch.stack(embs, dim = 1), cdev).half()
        logit = self.confidence.forward(torch.cat((xpre, me), dim = -1), p).float().squeeze(-1)
        cs = torch.sigmoid(logit)
        conf_out = torch.cat((torch.zeros_like(logit[:, :1]), logit), dim = 1)
        lens = self._real_lens(_ev_lens(cs, self._gate_threshold(cs.device)), p)
        return out, conf_out, lens

    @staticmethod
    def _real_lens(lens: torch.Tensor, p: dict) -> torch.Tensor:
        """Pad-to-bucket rows (util/pad_bucket.py) have no length: the batch's verify width is the max over
        the real rows only. Device-side, so the captured sampling graph serves every row count of its bucket"""
        if not p.get("pad_bucket"):
            return lens
        pad = pad_bucket.pad_mask(lens.shape[0], 1, lens.device)
        return lens.masked_fill(pad, 0)

    def _gate_threshold(self, device):
        """The gate's threshold: the fixed float, or under EXL3_DSPARK_GATE=auto the persistent device tensor."""
        if self.auto_threshold is None:
            return self.draft_conf_threshold
        return self.auto_threshold.tensor(device)

    def observe_accepted(self, accepted: int):
        """Per verify round: accepted drafts (rank-agreed). Feeds the content-adaptive threshold."""
        if self.auto_threshold is not None:
            self.auto_threshold.observe(accepted)

    def reset_adaptive(self):
        """Start of a new job: forget the previous job's content estimate."""
        if self.auto_threshold is not None:
            self.auto_threshold.reset()

    def gate_summary(self) -> dict:
        """The length gate in effect, for the run log: the verify width it picks is a different numeric
        path, so a run's numbers only compare against another run with the same gate and constants."""
        out = {
            "gate": _GATE,
            "conf": self.draft_conf_threshold,
            "markov_graph": _MARKOV_GRAPH,
            "block_graph": _BLOCK_GRAPH,
        }
        if self.auto_threshold is not None:
            out["auto_lo"] = _AUTO_LO
            out["auto_switch"] = _AUTO_SWITCH
            out["auto_hl"] = _AUTO_HL
            out["auto_avg"] = round(self.auto_threshold.avg, 3)
            out["auto_level"] = self.auto_threshold.level
        if _GATE == "ev":
            out["ev_cost"] = _EV_COST_RAW
            out["ev_kmin"] = _EV_KMIN
            out["ev_gamma"] = _EV_GAMMA
            out["ev_scale"] = _EV_SCALE
        return out




    def _sample_graphed(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """Captured sample_from_state: one graph replay for head + markov chain + confidence instead
        of ~50 host launches. The host values the generator needs (draft length, and the ids it
        stages for the verifier) are read once afterwards, as before."""
        dev = self.markov_w2.device
        seed = to_dev(params["dspark_seed_ids"], dev)
        prenorm = params["dspark_prenorm"]
        if not hasattr(self, "_sample_segments"):
            self._sample_segments = {}
        key = (tuple(state.shape), state.dtype, tuple(prenorm.shape), prenorm.dtype, self.draft_conf_threshold)
        seg = self._sample_segments.get(key)
        if seg is None:
            # The gemms inside the body keep their own recorded sub-graphs (exllamav3_ext/graph.cu):
            # one eager call only flips them to record-on-next-call, which the outer capture would
            # then trigger on a second stream. Two eager calls settle them before we capture.
            self._markov_body(state, seed, prenorm, params)
            self._markov_body(state, seed, prenorm, params)
            seg = self._sample_segments[key] = Segment(self._markov_body)
        out, conf, lens = seg(state, seed, prenorm, params)
        rp = getattr(seg, "replays", 0)
        if rp in (1, 100) and self.config.tp_rank == 0:
            print(f"MARKOV_GRAPH key={key} graph={'yes' if seg.graph is not None else 'no'} replays={rp}", flush = True)
        if params.get("export_draft_conf"):
            params["draft_conf"] = conf
        if params.get("defer_draft_confidence_len"):
            # Host-free path: the generator (or the TP2 runner) resolves the length on device
            params["draft_confidence_len_device"] = lens.max().reshape(1).to(device = out.device, dtype = torch.long)
        else:
            params["draft_confidence_len"] = int(lens.max().item())
        if self._conf_stats is not None:
            self._conf_stats.append(torch.sigmoid(conf[0, 1:].float()).tolist())
        backend = params.get("backend")
        if not params.get("defer_draft_confidence_len") and getattr(backend, "trace_draft_window", False):
            backend.trace_window(params["draft_confidence_len"])
        if _rp_on: _rpmark("ds_sample_end")
        return out


    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """Trunk head over all block positions at once, then the sequential greedy loop
        with the markov bigram bias. Returns (bsz, block + 1) ids [seed, drafts...]; the
        generator crops the seed."""
        if _MARKOV_GRAPH and "dspark_prenorm" in params:
            return self._sample_graphed(state, params)
        import torch.nn.functional as F
        if self.own_head is not None:
            lm = self.own_head
        else:
            am = self.attached_model()
            lm = am.modules[am.logit_layer_idx]
        logits = lm.forward(lm.prepare_for_device(state, params), params).float()
        if _rp_on: _rpmark("ds_fwd_end")
        b, s, V = logits.shape
        # Sequential in the sampled chain but fully on-device: embedding gather + bias
        # gemv + argmax per step, no host round trips (single sync when consumed)
        dev = self.markov_w2.device
        logits = to_dev(logits, dev)
        seed = to_dev(params["dspark_seed_ids"], dev)
        if getattr(self, "_markov_w1_dev", None) is None:
            self._markov_w1_dev = to_dev(self.markov_w1.embedding.weight.data, dev)
        w2 = self.markov_w2
        tp = self.tp_size
        out = torch.empty((b, s + 1), dtype = torch.long, device = dev)
        out[:, 0] = seed[:, -1]
        embs = []
        for i in range(s):
            if tp > 1:
                local_ids = out[:, i] - self.config.tp_rank * self.markov_w1.vocab_size
                mask = (local_ids >= 0) & (local_ids < self.markov_w1.vocab_size)
                emb = F.embedding(torch.where(mask, local_ids, torch.zeros_like(local_ids)), self._markov_w1_dev).float()
                emb = emb * mask.unsqueeze(-1).float()
                params["backend"].all_reduce(emb)
                emb = emb.half()
            else:
                emb = F.embedding(out[:, i], self._markov_w1_dev).half()
            embs.append(emb)
            bias = w2.forward(emb.unsqueeze(1), params).float()
            if tp > 1:
                bias = params["backend"].all_gather(bias, tp)
            logits[:, i] += bias[:, 0]
            out[:, i + 1] = torch.argmax(logits[:, i], dim = -1)
            sync_ids = getattr(params.get("backend"), "sync_draft_ids", None)
            if sync_ids is not None:
                sync_ids(out[:, i + 1])
        if _rp_on: _rpmark("ds_loop_end")

        # Confidence-capped draft length: proj(cat([pre-norm hidden, markov emb])) per
        # position; the generator clamps its window to the longest all-confident prefix
        # (batch max), 0 = skip drafting this round
        cdev = self.confidence.device
        xpre = to_dev(params["dspark_prenorm"], cdev).half()
        me = to_dev(torch.stack(embs, dim = 1), cdev).half()
        conf = self.confidence.forward(torch.cat((xpre, me), dim = -1), params)
        cs = torch.sigmoid(conf.float().squeeze(-1))
        if params.get("export_draft_conf"):
            # Raw confidence logits for the generator's calibrator (binned 1.0 wide, so logits, not sigmoid).
            # Column 0 is the seed position, which the generator crops (draft_conf[:, 1:]); layout (bsz, block + 1)
            logit = conf.float().squeeze(-1)
            params["draft_conf"] = torch.cat((torch.zeros_like(logit[:, :1]), logit), dim = 1)
        if self._conf_stats is not None:
            self._conf_stats.append(cs[0].tolist())
        lens = self._real_lens(_ev_lens(cs, self._gate_threshold(cs.device)), params)
        if params.get("defer_draft_confidence_len"):
            params["draft_confidence_len_device"] = lens.max().reshape(1).to(device = out.device, dtype = torch.long)
        else:
            params["draft_confidence_len"] = (self.diag_draft_window - 1 if self.diag_draft_window is not None
                                             else int(lens.max().item()))
        if _rp_on: _rpmark("ds_conf_end")
        backend = params.get("backend")
        if not params.get("defer_draft_confidence_len") and getattr(backend, "trace_draft_window", False):
            backend.trace_window(params["draft_confidence_len"])
        return out


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}
