#!/usr/bin/env python3
"""Per-owner VRAM budget for the Recipe A hybrid on the production pair (P7 GPU0 hot + GX1 cold).

Loads exactly as qmesh/e2e_prompts.py does at the shipped config, then reports, per device:

  MEM_OWNER                      resident CUDA tensors by owner (deduplicated by storage)
  MEMB                           engine accounting: weights / kv cache / recurrent / statics /
                                 generator / other, plus statics by tag (graph pools, workspaces)
  MEM_KV                         pool bytes per token, scaled to 64k / 131k / 1M tokens
  MEM_PREFILL / MEM_DECODE       peak allocated+reserved during a long-context prefill, then
                                 during the decode steps that follow it, with free at peak

Phases: after load, after the warm-up decode (graphs captured -> statics exist), after the long
prefill, and after decode.

Env:
  CHECKPOINT, DRAFT_CHECKPOINT (defaults to CHECKPOINT),
  PLACEMENT (default: the pack's PACK_MANIFEST.json, else qmesh/placement.final.json),
  KV_BITS (4|8|16, default 4), CACHE_TOKENS (default 1048576),
  DSPARK (1 loads the drafter), NUM_DRAFT (default 5), WARMUP_NEW (decode tokens of the warm-up
  prompt, default 24), LONG_PROMPT_FILE or CORPUS+LONG_TOKENS, LONG_NEW (decode tokens after the
  long prompt, default 16), MAX_CHUNK (default 2048), PREFILL_ADAPTIVE, PREFILL_ADAPTIVE_LOG,
  EXL3_WHOLE_GRAPH, PREFILL_ARMS ("depth:chunk,..." prefill arms in one load), SEAM_ROWS (largest
  seam bucket, only used to count a chunk's round trips), BYNAME_MIN_MIB (drafter MEM_BYNAME floor),
  DEPTH_PROBES / DEPTH_MAX_S, LOAD_ONLY,
  BATCH (served batch, default 1: sizes the generator and the recurrent-state slots, as serve_pair.py does),
  QMESH_STATE_SLOTS (explicit slot count, >= BATCH; qmesh/state_slots.py)
 """
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
os.environ.setdefault("EXL3_WHOLE_GRAPH", "1")

import torch

from exllamav3 import Cache, CacheLayer_quant, Config, Generator, GreedySampler, Job, Model, Tokenizer
from exllamav3.util.memory import vram_accounting
from qmesh.prefill_limit import PAGE_SIZE, PREFILL_ADAPTIVE, PrefillChunkLimiter
from qmesh import placement as seam_placement
from qmesh.state_slots import state_slots

CHECKPOINT = os.environ["CHECKPOINT"]
DRAFT_CHECKPOINT = os.environ.get("DRAFT_CHECKPOINT", CHECKPOINT)
# The pack's own table (qmesh/placement.py), not a repo default: this script walks the long prefill,
# which touches far more experts than the identity phase, and a wrong cold count here is what made
# the server refuse 2 of 384 ids on 2026-11-01 18:03.
PLACEMENT = seam_placement.resolve(CHECKPOINT)
KV_BITS = int(os.environ.get("KV_BITS", "4"))
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", "1048576"))
DSPARK = os.environ.get("DSPARK") == "1"
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "5"))
BATCH = int(os.environ.get("BATCH", "1"))
STATE_SLOTS = state_slots(BATCH)
WARMUP_NEW = int(os.environ.get("WARMUP_NEW", "24"))
LONG_PROMPT_FILE = os.environ.get("LONG_PROMPT_FILE", "")
LONG_TOKENS = int(os.environ.get("LONG_TOKENS", "0"))
LONG_NEW = int(os.environ.get("LONG_NEW", "16"))
# Prompt format for LONG_PROMPT_FILE. The v1.2 needle harness that scored 5/5 on the 128k fixture
# (artifacts/2x-rtx-tp2/fp4kv/needle.sh, PROMPT_IDS=8) does not feed the file as raw text: it builds
# "PROMPTS[i] + '\n\n' + transcript" and wraps that in the checkpoint's own chat template with
# add_bos=False. A raw add_bos=True read makes the base model continue the prose, which is what the
# first 131k arm did. LONG_CHAT=1 reproduces the reference format; LONG_LEAD is the instruction that
# precedes the transcript (PROMPTS[8] in run_tp2.py).
LONG_CHAT = os.environ.get("LONG_CHAT") == "1"
LONG_LEAD = os.environ.get("LONG_LEAD", "Name three planets in our solar system.")
ENC_DIR = os.environ["ENC_DIR"]
CORPUS_FILES = [p for p in os.environ.get("CORPUS", "").split(",") if p.strip()]
MAX_CHUNK = int(os.environ.get("MAX_CHUNK", "2048"))
# Deep-context probes for the statics (dsv4_attn_out and friends): tokens from a tiled corpus, and
# the wall time a probe may be predicted to take before it is skipped (the previous probe scales it).
DEPTH_MAX_S = float(os.environ.get("DEPTH_MAX_S", "420"))
# "depth:chunk,depth:chunk,..." prefills a fresh prompt of `depth` tokens at each chunk size in one load
# (see the PREFILL_ARM phase below). SEAM_ROWS is the largest seam row bucket the client asks for (the
# client splits a prefill chunk into pieces of at most this many rows), used only for the round-trip
# count in the arm line; it defaults to the largest bucket in the request so the two cannot disagree.
PREFILL_ARMS = [tuple(int(v) for v in a.split(":"))
                for a in os.environ.get("PREFILL_ARMS", "").split(",") if a.strip()]
SEAM_BUCKETS = [int(b) for b in os.environ.get("SEAM_BUCKETS", "1,6,64").split(",") if b.strip()]
SEAM_ROWS = int(os.environ.get("SEAM_ROWS") or max(SEAM_BUCKETS))
# Wall-clock discipline for the arms: ARM_MAX_S caps one arm (the job is cancelled after the chunk
# in flight and the arm is reported with aborted=1), ARM_TOTAL_S caps the whole arms phase, so deep
# arms are skipped when the measured rate says they cannot finish.
ARM_MAX_S = float(os.environ.get("ARM_MAX_S", "0"))
ARM_TOTAL_S = float(os.environ.get("ARM_TOTAL_S", "0"))
LOAD_ONLY = os.environ.get("LOAD_ONLY") == "1"
# PREFILL_MEM_TRACE=<path prefix> records the CUDA allocator's history for the first arm's prefill and
# dumps `<prefix>.snapshot.pickle` when that prefill completes. The live-tensor walk above sees what
# survives a chunk, not the transients that peak in the middle of one, and the chunk's peak is exactly
# what the limiter has to make room for. Parsing lives in
# the snapshot_top.py capture tool.
PREFILL_MEM_TRACE = os.environ.get("PREFILL_MEM_TRACE", "")
DEVICE = torch.device("cuda:0")
G = 2 ** 30

assert KV_BITS in (4, 8, 16), "KV_BITS must be 4, 8 or 16"

KV_CACHE_KWARGS = ({} if KV_BITS == 16 else
                   {"kv_format": "fp4"} if KV_BITS == 4 else
                   {"layer_type": CacheLayer_quant, "k_bits": 8, "v_bits": 8})


def _io_counters() -> dict:
    """This process's byte counters (/proc/self/io), for separating disk reads from CPU work."""
    out = {}
    try:
        with open("/proc/self/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                try:
                    out[k.strip()] = int(v)
                except ValueError:
                    pass
    except OSError:
        pass
    return out

def gib(n) -> float:
    return n / G


def walk_tensors(root, seen: set, out: dict):
    """CUDA tensors reachable from root through attributes/containers, deduplicated by storage.
    `seen` is shared between calls, so nothing is counted twice across owners."""
    stack = [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.is_cuda and o.device == DEVICE:
                key = o.untyped_storage().data_ptr()
                if key not in out:
                    out[key] = o.numel() * o.element_size()
            continue
        if isinstance(o, (str, bytes, int, float, bool, type(None))):
            continue
        if isinstance(o, (list, tuple, set)):
            stack.extend(o)
            continue
        if isinstance(o, dict):
            stack.extend(o.values())
            continue
        try:
            stack.extend(list(vars(o).values()))
        except TypeError:
            pass


class Owner:
    def __init__(self):
        self.seen: set = set()
        self.accounted: set = set()
        self.rows: list = []
        self.count: dict = {}

    def take(self, label, root, count = 0):
        out: dict = {}
        walk_tensors(root, self.seen, out)
        new = sum(v for k, v in out.items() if k not in self.accounted)
        self.accounted.update(out)
        self.rows.append((label, gib(new)))
        if count:
            self.count[label] = count
        return new

    def totals(self) -> dict:
        t: dict = {}
        for label, gb in self.rows:
            t[label.split()[0]] = t.get(label.split()[0], 0.0) + gb
        return t


def take_mlp(own: Owner, mlp, tag: str, expert_label: str):
    """Split a routed MoE: router, shared experts (+latent projections, norms), hot experts, rest."""
    for name in ("routing_gate", "gate"):
        m = getattr(mlp, name, None)
        if isinstance(m, torch.nn.Module):
            own.take(f"router {tag}", m)
            break
    shared = []
    for name in ("shared_gate", "shared_experts", "latent_in", "latent_out",
                 "shared_experts_post_norm", "router_pre_norm", "routed_pre_norm", "routed_post_norm"):
        m = getattr(mlp, name, None)
        if isinstance(m, torch.nn.Module):
            shared.append(m)
    own.take(f"shared {tag}", shared)
    experts, n_hot = [], 0
    for list_name in ("gates", "ups", "downs"):
        for lin in getattr(mlp, list_name, []) or []:
            experts.append(lin)
            probe: dict = {}
            walk_tensors(lin, set(), probe)
            if probe:
                n_hot += 1
    own.take(f"{expert_label} {tag}", experts, count = n_hot)
    own.take(f"moe_other {tag}", mlp)


def take_block(own: Owner, m, tag: str):
    attn = getattr(m, "attn", None)
    if isinstance(attn, torch.nn.Module):
        own.take(f"attn {tag}", attn)
    mlp = getattr(m, "mlp", None)
    if isinstance(mlp, torch.nn.Module):
        if "BlockSparse" in type(mlp).__name__:
            take_mlp(own, mlp, tag, "hot_experts")
        else:
            own.take(f"dense_mlp {tag}", mlp)
    own.take(f"block_other {tag}", m)


def report(own: Owner, title: str):
    for label, gb in own.rows:
        if gb >= 0.02:
            extra = f" n={own.count[label]}" if label in own.count else ""
            print(f"MEM_OWNER {title} {gb:.3f} {label}{extra}", flush = True)
    for cat, gb in sorted(own.totals().items(), key = lambda kv: -kv[1]):
        if gb >= 0.02:
            print(f"MEM_GROUP {title} {gb:.3f} {cat}", flush = True)


ARENA_BLOCK_BYTES = 128 << 20


def _view_key(t: torch.Tensor):
    """Key a tensor the way vram_accounting does: pooled weights live as views into 128 MiB arena
    blocks (counted per view), everything else is counted once per storage."""
    s = t.untyped_storage()
    if s.nbytes() == ARENA_BLOCK_BYTES:
        return ("view", t.data_ptr(), t.numel() * t.element_size())
    return ("storage", s.data_ptr(), s.nbytes())


def walk_named(root, skip_ids = frozenset()):
    """Every CUDA tensor reachable from root, with the attribute path that first reached its key."""
    out: dict = {}
    ids: set = set()
    stack = [(root, "")]
    while stack:
        o, path = stack.pop()
        if id(o) in ids or id(o) in skip_ids:
            continue
        ids.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.is_cuda and o.device == DEVICE:
                out.setdefault(_view_key(o), (path, o))
            continue
        if isinstance(o, (str, bytes, int, float, bool, type(None))):
            continue
        if isinstance(o, (list, tuple, set)):
            stack.extend((v, f"{path}[{i}]") for i, v in enumerate(o))
            continue
        if isinstance(o, dict):
            stack.extend((v, f"{path}[{k}]") for k, v in o.items())
            continue
        try:
            stack.extend((v, f"{path}.{k}") for k, v in vars(o).items())
        except TypeError:
            pass
    return out


def _norm_path(path: str) -> str:
    return path.replace("[", ".").replace("]", "")



def classify_target(path: str) -> str:
    p = _norm_path(path).lower()
    leaf = p.rsplit(".", 1)[-1]
    if ".attn." in p and ("compressor" in p or "indexer" in p):
        return "attn." + ("compressor" if "compressor" in p else "indexer")
    if ".experts." in p or ".experts" in p:
        if leaf == "trellis":
            return "hot_experts.trellis"
        if leaf in ("suh", "svh", "mul1", "scale"):
            return "hot_experts.aux_" + leaf
        return "hot_experts.other"
    if "shared_experts" in p:
        return "shared_experts.trellis" if leaf == "trellis" else "shared_experts.aux"
    if ".ffn.gate" in p or "routing_gate" in p:
        return "router"
    if ".attn." in p:
        for comp in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "sink", "norm"):
            if comp in p:
                return f"attn.{comp}"
        return "attn.other"
    if ".engram" in p:
        return "engram"
    if p.startswith("head.") or ".head." in p:
        return "head"
    if "embed" in p:
        return "embed"
    if "norm" in leaf or "hc_" in p or "expandstream" in p:
        return "norms_hc"
    if p.startswith(("aligner", "image_", "vision")):
        return "vision_aligner"
    return "other:" + p.rsplit(".", 2)[-2] + "." + leaf


def itemize(root, title: str, classify, skip_ids = frozenset(), engine_gib = None):
    """Print MEM_WITEM: CUDA weight bytes of root grouped by classify(path), deduplicated the way
    vram_accounting deduplicates (arena views once each, other tensors once per storage)."""
    rows = walk_named(root, skip_ids)
    by: dict = {}
    ex: dict = {}
    for key, (path, _t) in rows.items():
        cat = classify(path)
        by[cat] = by.get(cat, 0) + key[-1]
        ex.setdefault(cat, []).append(path)
    for cat, nb in sorted(by.items(), key = lambda kv: -kv[1]):
        if gib(nb) < 0.01:
            continue
        sample = ", ".join(ex[cat][:2])
        print(f"MEM_WITEM {title} {gib(nb):.3f} {cat} [{sample}]", flush = True)
    total = sum(by.values())
    tail = (f" engine={engine_gib:.3f} arena_tail={engine_gib - gib(total):.3f}"
            if engine_gib is not None else "")
    print(f"MEM_WITEM {title} {gib(total):.3f} TOTAL n_views={len(rows)}{tail}", flush = True)
    return by


def dump_byname(root, title: str, skip_ids = frozenset(), min_mib: float = 0.0, top: int = 400):
    """Print MEM_BYNAME: every CUDA tensor reachable from root, deduplicated by storage pointer, with
    dtype and shape, plus a rollup by module suffix. Answers 'which tensors make up 6.703 GiB' when the
    checkpoint file on disk only accounts for 5.4 of it (stacked/padded copies, fp32 heads, private
    embed/head, arena tail)."""
    rows = walk_named(root, skip_ids)
    items = []
    for _key, (path, t) in rows.items():
        n = t.numel() * t.element_size()
        if gib(n) < min_mib:
            continue
        items.append((n, path, str(t.dtype).replace("torch.", ""), tuple(t.shape)))
    items.sort(reverse = True)
    tot = sum(n for n, *_ in items)
    print(f"MEM_BYNAME {title} n={len(rows)} listed={len(items)} shown={min(top, len(items))} "
          f"total_gib={gib(tot):.3f} min_mib={min_mib}", flush = True)
    roll: dict = {}
    for n, path, dt, shape in items:
        leaf = path.rsplit(".", 1)[-1]
        roll[(leaf, dt)] = roll.get((leaf, dt), 0) + n
    for (leaf, dt), n in sorted(roll.items(), key = lambda kv: -kv[1])[:40]:
        print(f"MEM_BYNAME_ROLL {title} {gib(n):8.3f} {dt:10s} {leaf}", flush = True)
    for n, path, dt, shape in items[:top]:
        print(f"MEM_BYNAME_ITEM {title} {gib(n):8.3f} {dt:10s} {str(shape):28s} {path}", flush = True)


def classify_generator(path: str) -> str:
    p = _norm_path(path)
    # first attribute under the walk root is the generator's own slot; keep two levels of detail
    parts = [s for s in p.split(".") if s]
    return ".".join(parts[:2]) if parts else "?"


def classify_drafter(path: str) -> str:
    p = _norm_path(path).lower()
    leaf = p.rsplit(".", 1)[-1]
    if ".experts" in p:
        return "drafter_experts.trellis" if leaf == "trellis" else "drafter_experts.aux"
    if "shared_experts" in p:
        return "drafter_shared.trellis" if leaf == "trellis" else "drafter_shared.aux"
    if ".attn." in p:
        return "drafter_attn"
    if "markov" in p or "conf" in p:
        return "drafter_head." + leaf
    if "norm" in leaf or "hc_" in p:
        return "drafter_norms_hc"
    if p.startswith("mtp.2") or p.startswith("mtp"):
        return "drafter_block_other." + leaf
    return "drafter_other." + leaf


def engine_report(model, cache, generator, title: str):
    r = vram_accounting(model, cache, generator, [DEVICE])[0]
    print(f"MEMB {title} weights={gib(r.weights):.3f} kv_cache={gib(r.cache):.3f} "
          f"recurrent={gib(r.recurrent):.3f} statics={gib(r.statics):.3f} "
          f"generator={gib(r.generator):.3f} other={gib(r.other):.3f}", flush = True)
    for tag, b in sorted(r.statics_by_tag.items(), key = lambda kv: -kv[1]):
        print(f"MEMB statics_tag {title} {tag}={gib(b):.3f}", flush = True)
    free, total = torch.cuda.mem_get_info(DEVICE)
    print(f"MEMB device {title} total={gib(total):.3f} used={(total - free) / G:.3f} "
          f"torch_reserved={gib(torch.cuda.memory_reserved(DEVICE)):.3f} "
          f"torch_allocated={gib(torch.cuda.memory_allocated(DEVICE)):.3f} "
          f"non_torch={(total - free - torch.cuda.memory_reserved(DEVICE)) / G:.3f} "
          f"free={gib(free):.3f}", flush = True)
    # The allocator's own view, once per report: `active` is memory torch believes is in use (a per-shape
    # cache shows up here as growth that never comes back), `inactive_split` is freed-and-split memory
    # that only exists without expandable segments. Read together with the chunk limiter's room_MiB these
    # say whether a shrunken grant is caused by live growth or by the pool's shape history.
    st = torch.cuda.memory_stats(DEVICE)
    print(f"MEMB_ALLOC {title} expandable={os.environ.get('EXL3_EXPANDABLE_SEGMENTS_ACTIVE', 'env' if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') else 'off')} "
          f"active={gib(st.get('active_bytes.all.current', 0)):.3f} "
          f"inactive_split={(st.get('inactive_split_bytes.all.current', 0)) / G:.3f} "
          f"segments={int(st.get('segment.all.current', 0))} "
          f"reserved={gib(torch.cuda.memory_reserved(DEVICE)):.3f}", flush = True)


def peaks(title: str, base_alloc: int):
    free, total = torch.cuda.mem_get_info(DEVICE)
    peak_a = torch.cuda.max_memory_allocated(DEVICE)
    peak_r = torch.cuda.max_memory_reserved(DEVICE)
    print(f"MEM_PEAK {title} peak_alloc={gib(peak_a):.3f} peak_reserved={gib(peak_r):.3f} "
          f"rise_alloc={gib(peak_a - base_alloc):.3f} free_at_peak={gib(total - peak_r):.3f} "
          f"free_now={gib(free):.3f} used_now={(total - free) / G:.3f}", flush = True)


def main():
    placement, placement_path, placement_sha = seam_placement.load(CHECKPOINT)
    print(f"CLIENT_PLACEMENT path={placement_path} sha={placement_sha[:16]} ckpt={CHECKPOINT}",
          flush = True)
    hot = seam_placement.hot(placement)
    config = Config.from_directory(CHECKPOINT)
    config.infer_params.moe_cpu_split = 1
    config.infer_params.moe_remote_hot = hot
    config.infer_params.moe_remote_warm = seam_placement.warm(placement)
    config.infer_params.remote_expert_host = {
        "host": os.environ["SEAM_PEER"],
        "port": int(os.environ.get("SEAM_PORT", "47311")),
    }
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_history=5, max_batch_size=STATE_SLOTS,
                  **KV_CACHE_KWARGS)
    t0 = time.time()
    io0 = _io_counters()
    model.load(device="cuda:0", progressbar=False)
    load_s = time.time() - t0
    free, total = torch.cuda.mem_get_info(DEVICE)
    print(f"P7_LOAD_OK {load_s:.1f}s vram={gib(torch.cuda.memory_allocated(DEVICE)):.2f}GiB "
          f"free={gib(free):.2f}GiB total={gib(total):.2f}GiB kv_bits={KV_BITS} "
          f"cache_tokens={CACHE_TOKENS} dspark={int(DSPARK)} batch={BATCH} state_slots={STATE_SLOTS}", flush = True)
    io1 = _io_counters()
    print(f"LOADIO seconds={load_s:.1f} read_gib={gib(io1['read_bytes'] - io0['read_bytes']):.2f} "
          f"rchar_gib={gib(io1['rchar'] - io0['rchar']):.2f} "
          f"mib_per_s={gib(io1['read_bytes'] - io0['read_bytes']) * 1024 / max(load_s, 1e-9):.1f} "
          f"cpu_s={time.process_time():.1f}", flush = True)
    if LOAD_ONLY:
        return 0

    tokenizer = Tokenizer.from_config(config)
    draft_model = None
    draft_cache = None
    if DSPARK:
        draft_config = Config.from_directory(DRAFT_CHECKPOINT)
        draft_model = Model.from_config(draft_config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=CACHE_TOKENS, max_history=5,
                            max_batch_size=STATE_SLOTS, **KV_CACHE_KWARGS)
        t1 = time.time()
        draft_model.load(device="cuda:0", progressbar=False)
        print(f"DSPARK_LOADED {time.time() - t1:.1f}s drafter={DRAFT_CHECKPOINT}", flush = True)
    generator = Generator(model, cache, tokenizer, max_batch_size=BATCH, max_chunk_size=MAX_CHUNK,
                          draft_model=draft_model, draft_cache=draft_cache,
                          num_draft_tokens=NUM_DRAFT if DSPARK else None)
    engine_report(model, cache, generator, "after_load")
    if draft_model is not None:
        engine_report(draft_model, draft_cache, None, "drafter_after_load")

    # KV/state cost, scaled from the allocation the shipped CACHE_TOKENS size actually makes
    r0 = vram_accounting(model, cache, generator, [DEVICE])[0]
    print(f"MEM_KV pool_bytes={gib(r0.cache):.3f} cache_tokens={CACHE_TOKENS} "
          f"bytes_per_token={r0.cache / max(CACHE_TOKENS, 1):.1f} "
          f"at_64k={gib(r0.cache * 65536 / max(CACHE_TOKENS, 1)):.3f} "
          f"at_131k={gib(r0.cache * 131072 / max(CACHE_TOKENS, 1)):.3f} "
          f"recurrent_bytes={gib(r0.recurrent):.3f}", flush = True)
    if draft_cache is not None:
        r1 = vram_accounting(draft_model, draft_cache, None, [DEVICE])[0]
        print(f"MEM_KV drafter_pool_bytes={gib(r1.cache):.3f} "
              f"drafter_recurrent_bytes={gib(r1.recurrent):.3f}", flush = True)


    # --- itemised weight / generator split (owner question: every piece by module type) -------
    iskip = {id(model.config), id(getattr(model.config, "stc", None)), id(cache), id(generator),
             id(tokenizer)}
    if draft_model is not None:
        iskip |= {id(draft_model), id(draft_cache), id(draft_model.config),
                  id(getattr(draft_model.config, "stc", None))}
    iskip.discard(None)
    itemize(model, "target", classify_target, iskip, engine_gib = gib(r0.weights))
    if draft_model is not None:
        itemize(draft_model, "drafter", classify_drafter,
                {id(draft_model.config), id(getattr(draft_model.config, "stc", None)),
                 id(draft_cache), id(generator), id(model), id(cache)})
        # Every CUDA tensor the drafter owns, by name, deduplicated by storage pointer: the resident total
        # (6.7 GiB) is well above the checkpoint's tensor bytes (5.4 GiB), which is either a stacked copy,
        # an upcast head or a private embed/head, and the rollup names which.
        dump_byname(draft_model, "drafter",
                    {id(draft_model.config), id(getattr(draft_model.config, "stc", None)),
                     id(draft_cache), id(generator), id(model), id(cache)},
                    min_mib = float(os.environ.get("BYNAME_MIN_MIB", "0.5")))
    itemize(generator, "generator", classify_generator,
            {id(model), id(cache), id(draft_model), id(draft_cache), id(tokenizer),
             id(model.config), id(getattr(model.config, "stc", None))})
    # --- after load ------------------------------------------------------------------------
    skip = {id(draft_model), id(draft_cache), id(generator), id(model), id(model.config),
            id(getattr(model.config, "stc", None))}
    if draft_model is not None:
        skip |= {id(draft_model), id(draft_model.config), id(draft_cache)}
    skip.discard(None)
    own = Owner()
    own.seen |= skip
    for m in model.modules:
        name, key = type(m).__name__, getattr(m, "key", "") or ""
        if name == "TransformerBlock":
            take_block(own, m, key)
        elif name == "EngramLayer":
            own.take(f"engram {key}", m)
        elif name == "Embedding" or key in ("embed",):
            own.take("embed", m)
        elif name in ("HyperHead",) or key in ("head",):
            own.take("head", m)
        elif key in ("norm", "hc_head", "hc_expand") or "Norm" in name or name == "ExpandStreams":
            own.take(f"norms_hc {key or name}", m)
        else:
            own.take(f"other {key or name}", m)
    report(own, "target")
    if draft_model is not None:
        down = Owner()
        down.seen |= {id(draft_model), id(draft_model.config), id(draft_cache), id(generator)}
        for m in draft_model.modules:
            name, key = type(m).__name__, getattr(m, "key", "") or ""
            if hasattr(m, "attn") and hasattr(m, "mlp"):
                take_block(down, m, key)
            elif "BlockSparse" in name:
                take_mlp(down, m, key, "drafter_experts")
            elif "head" in key.lower() or "Head" in name:
                down.take(f"drafter_head {key or name}", m)
            else:
                down.take(f"drafter_other {key or name}", m)
        report(down, "drafter")
    # --- warm-up: capture graphs, then account for statics --------------------------------
    def run(ids, max_new, chunk_label, max_s = None):
        limiter = None
        if PREFILL_ADAPTIVE:
            generator.prefill_chunk_limit = limiter = PrefillChunkLimiter(DEVICE)
        job = Job(input_ids = ids, max_new_tokens = max_new, sampler = GreedySampler())
        # The trace has to start before the chunk allocates, so it goes on ahead of enqueue and comes off
        # as soon as this arm's prefill has produced its first result: after that the only allocations are
        # decode steps, which would bury the prefill peak in 24 rounds of noise.
        trace_path = f"{PREFILL_MEM_TRACE}.{chunk_label}.snapshot.pickle"
        trace_here = bool(PREFILL_MEM_TRACE) and chunk_label.startswith("arm_") \
                     and not os.path.exists(trace_path)
        if trace_here:
            torch.cuda.memory._record_memory_history(enabled = "all", max_entries = 400000)
            print(f"MEM_TRACE on path={trace_path} "
                  f"base_alloc_mib={torch.cuda.memory_allocated(DEVICE) / 2 ** 20:.0f}", flush = True)
        generator.enqueue(job)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        base_alloc = torch.cuda.memory_allocated(DEVICE)
        t = time.time()
        first_t = None
        prefill_t = None
        prefill_peak = None
        out = None
        gen = None  # first result that carries the generated token ids: an arm is often stopped
        n = 0  # results yielded (one per token, plus the prefill and eos markers)
        aborted = False
        while generator.num_remaining_jobs():
            # Per-arm wall limit: a prefill arm that runs past it is cancelled after the chunk in
            # flight (generator.cancel releases the pages, so the loop then drains and returns).
            if max_s and not aborted and time.time() - t > max_s:
                generator.cancel(job)
                aborted = True
                print(f"MEM_ABORT {chunk_label} elapsed_s={time.time() - t:.1f} max_s={max_s}",
                      flush = True)
                break
            for r in generator.iterate():
                if r.get("stage") == "error":
                    print(f"MEM_ERROR {chunk_label} error={r.get('error')!r}", flush = True)
                    err = r.get("error")
                    raise err if isinstance(err, BaseException) else RuntimeError(f"job error: {err!r}")
                if first_t is None:
                    first_t = time.time() - t
                if prefill_t is None and (r.get("text") or r.get("stage") == "streaming"
                                          or job.is_prefill_done()):
                    prefill_t = time.time() - t
                    prefill_peak = torch.cuda.max_memory_allocated(DEVICE)
                    torch.cuda.reset_peak_memory_stats(DEVICE)
                    if trace_here:
                        torch.cuda.synchronize()
                        torch.cuda.memory._dump_snapshot(trace_path)
                        torch.cuda.memory._record_memory_history(enabled = None)
                        trace_here = False
                        print(f"MEM_TRACE dump path={trace_path} "
                              f"prefill_peak_mib={(prefill_peak - base_alloc) / 2 ** 20:.0f}",
                              flush = True)
                if gen is None and r.get("token_ids") is not None:
                    gen = r  # before eos, which a one-token or time-limited arm may never reach
                if r.get("eos"):
                    out = r
                n += 1
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(DEVICE)
        prefill_peak = peak if prefill_peak is None else prefill_peak
        out = out or gen or {}
        return {"wall_s": time.time() - t, "ttft_s": first_t, "steps": n,
                "chunks": limiter.chunks if limiter else None,
                "shrunk": limiter.shrunk if limiter else None,
                "prefill_s": prefill_t or 0.0,
                "aborted": aborted,
                "peak_rise_mib": (prefill_peak - base_alloc) / 2 ** 20,
                "decode_peak_rise_mib": (peak - base_alloc) / 2 ** 20,
                "decide_us_median": limiter.summary()["decide_us_median"] if limiter else None,
                # The tokens this arm produced, so two arms can be compared for identity and not
                # only for rate. `out` is the eos chunk's result; an aborted or time-limited arm
                # never reaches eos, so None here means "no identity" and must never be a match.
                "gen_ids": None if not out.get("token_ids") else out["token_ids"].tolist()[0][:4],
                "gen_text": out.get("text")}

    warm_ids = tokenizer.encode("Write a paragraph about the history of sailing ships.",
                                add_bos = True, encode_special_tokens = True)
    torch.cuda.synchronize()
    w = run(warm_ids, WARMUP_NEW, "warm")
    print(f"MEM_WARM {json.dumps(w)}", flush = True)
    engine_report(model, cache, generator, "after_warm")

    # --- prefill chunk arms (chunk size x context depth, one load) -------------------------
    # PREFILL_ARMS="depth:chunk,depth:chunk,..." walks a fresh prompt of `depth` tokens through prefill at
    # each chunk size and prints one PREFILL_ARM line per arm (host-timed prefill, engine tok/s, chunk count,
    # allocator peak rise) plus the limiter's own PREFILL_CHUNK_PEAK/PREFILL_CHUNK_ADAPT lines for the refit
    # of PREFILL_ROW_KIB / PREFILL_DEPTH_BASE_MIB / PREFILL_DEPTH_BYTES. Each arm's prompt is distinct (the
    # corpus is rolled by the running total), so no arm is served out of the prompt cache.
    if PREFILL_ARMS:
        sources = ([open(LONG_PROMPT_FILE).read()] if LONG_PROMPT_FILE else []) + \
                  [open(p).read() for p in CORPUS_FILES]
        src = torch.cat([tokenizer.encode(t, add_bos = False, encode_special_tokens = True)[0]
                         for t in sources])
        bos = tokenizer.encode("a", add_bos = True, encode_special_tokens = True)[0, :1]
        print(f"PREFILL_ARMS source_tokens={int(src.shape[-1])} arms={PREFILL_ARMS}", flush = True)
        # Every layer with warm (GX1) experts issues seam calls, so a prefill piece costs one round
        # trip per such layer; the count comes from the same placement the model is loaded with.
        COLD_LAYERS = sum(1 for v in placement.get("layers", {}).values() if v.get("warm"))
        print(f"PREFILL_ARMS seam_layers={COLD_LAYERS} seam_rows={SEAM_ROWS}", flush = True)
        consumed = 0
        tok_s_seen = []  # (depth, chunk, tok/s) of arms that completed prefill, for the estimate below
        arms_t0 = time.time()
        for depth, chunk in PREFILL_ARMS:
            # The page pool holds CACHE_TOKENS / PAGE_SIZE pages and enqueue() asserts on overflow: a job of
            # `depth` prompt tokens plus one generated token needs ceil((depth + 1) / PAGE_SIZE) pages
            if (depth + 1 + PAGE_SIZE - 1) // PAGE_SIZE > CACHE_TOKENS // PAGE_SIZE:
                print(f"PREFILL_ARM depth={depth} chunk={chunk} skipped=page_limit "
                      f"cache_tokens={CACHE_TOKENS}", flush = True)
                continue
            # Budget: an arm is skipped when the fastest rate seen so far predicts it cannot finish
            # inside what is left of ARM_TOTAL_S (the deepest arm is last, so this only prunes it).
            left_s = ARM_TOTAL_S - (time.time() - arms_t0) if ARM_TOTAL_S else None
            rate_hint = max((t for _, _, t in tok_s_seen), default = None)
            if left_s is not None and rate_hint:
                need_s = depth / rate_hint * 1.25
                if need_s > left_s:
                    print(f"PREFILL_ARM depth={depth} chunk={chunk} skipped=budget "
                          f"left_s={left_s:.0f} need_s={need_s:.0f} rate={rate_hint:.1f}", flush = True)
                    continue
            reps = depth // int(src.shape[-1]) + 1
            ids_a = torch.cat([bos, src.roll(consumed).repeat(reps)[: depth - 1]]).unsqueeze(0)
            consumed = (consumed + 7919) % int(src.shape[-1])
            generator.max_chunk_size = chunk
            max_s = min(ARM_MAX_S, left_s) if left_s is not None else ARM_MAX_S
            r = run(ids_a, 1, f"arm_{depth}_{chunk}", max_s = max_s or None)
            tps = None if r["aborted"] or not r["prefill_s"] else depth / r["prefill_s"]
            if tps:
                tok_s_seen.append((depth, chunk, tps))
            print(f"PREFILL_ARM depth={depth} chunk={chunk} tokens={int(ids_a.shape[-1])} "
                  f"prefill_s={r['prefill_s']:.2f} "
                  f"tok_per_s={'none' if not tps else f'{tps:.1f}'} "
                  f"chunks={r['chunks']} shrunk={r['shrunk']} aborted={int(r['aborted'])} "
                  f"wall_s={r['wall_s']:.2f} peak_rise_mib={r['peak_rise_mib']:.0f} "
                  f"pieces={None if r['chunks'] is None else -(-chunk // SEAM_ROWS)} "
                  f"round_trips={None if r['chunks'] is None else -(-chunk // SEAM_ROWS) * COLD_LAYERS} "
                  f"round_trips_total={None if r['chunks'] is None else -(-chunk // SEAM_ROWS) * COLD_LAYERS * r['chunks']} "
                  f"decide_us_median={r['decide_us_median']} "
                  f"gen_ids={r['gen_ids']} gen_text={r['gen_text']!r}", flush = True)
            free_now, _total_now = torch.cuda.mem_get_info(DEVICE)
            print(f"PREFILL_ARM_PEAK depth={depth} chunk={chunk} "
                  f"peak_rise_mib={r['peak_rise_mib']:.0f} "
                  f"decode_peak_rise_mib={r['decode_peak_rise_mib']:.0f} "
                  f"free_now_gib={free_now / G:.3f}", flush = True)
            engine_report(model, cache, generator, f"after_arm_{depth}_{chunk}")

        generator.max_chunk_size = MAX_CHUNK
        print(f"PREFILL_ARMS_DONE arms_s={time.time() - arms_t0:.0f} "
              f"rates={tok_s_seen}", flush = True)

    # --- long-context prefill + decode peaks ----------------------------------------------
    if LONG_PROMPT_FILE:
        raw = open(LONG_PROMPT_FILE).read()
        if LONG_CHAT:
            # Same path as the reference harness: instruction, blank line, transcript, chat template,
            # no BOS. The template puts the assistant header after the transcript, which is what makes
            # the model answer the request block instead of continuing the story.
            import sys as _sys
            if ENC_DIR not in _sys.path:
                _sys.path.insert(0, ENC_DIR)
            from encoding import encode_messages
            body = f"{LONG_LEAD}\n\n{raw}" if LONG_LEAD else raw
            text = encode_messages([{"role": "user", "content": body}], thinking_mode = "chat")
            ids = tokenizer.encode(text, add_bos = False, encode_special_tokens = True)
            print(f"LONG_CHAT lead={LONG_LEAD!r} enc={ENC_DIR} chars={len(text)}", flush = True)
            if LONG_TOKENS:
                print("LONG_TOKENS ignored with LONG_CHAT=1 (truncating ids would cut the template "
                      "footer); cut the fixture file instead", flush = True)
        else:
            ids = tokenizer.encode(raw, add_bos = True, encode_special_tokens = True)
            if LONG_TOKENS:
                ids = ids[:, :LONG_TOKENS]
        print(f"LONG_PROMPT tokens={int(ids.shape[-1])} file={LONG_PROMPT_FILE}", flush = True)
        print(f"LONG_PROMPT_TAIL {raw[-160:]!r}", flush = True)
    elif LONG_TOKENS:
        pieces = [tokenizer.encode(open(p).read(), add_bos = False, encode_special_tokens = True)[0]
                  for p in CORPUS_FILES]
        flat = torch.cat(pieces)
        bos = tokenizer.encode("a", add_bos = True, encode_special_tokens = True)[0, :1]
        reps = LONG_TOKENS // int(flat.shape[-1]) + 1
        ids = torch.cat([bos, flat.repeat(reps)[: LONG_TOKENS - 1]]).unsqueeze(0)
        print(f"LONG_PROMPT tokens={int(ids.shape[-1])} corpus={len(CORPUS_FILES)}", flush = True)
    else:
        ids = None

    if ids is not None:
        assert (int(ids.shape[-1]) + LONG_NEW + PAGE_SIZE - 1) // PAGE_SIZE <= CACHE_TOKENS // PAGE_SIZE, \
            f"long prompt {int(ids.shape[-1])} + {LONG_NEW} needs " \
            f"{(int(ids.shape[-1]) + LONG_NEW + PAGE_SIZE - 1) // PAGE_SIZE} pages > {CACHE_TOKENS // PAGE_SIZE}"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        base_alloc = torch.cuda.memory_allocated(DEVICE)
        limiter = None
        if PREFILL_ADAPTIVE:
            generator.prefill_chunk_limit = limiter = PrefillChunkLimiter(DEVICE)
        # IGNORE_EOS=1 mirrors tp2_bench.run_prompt: force LONG_NEW tokens so a short answer cannot end
        # the job before it prints (the reference needle arm ran MAX_NEW=200 for the same reason).
        ignore_eos = os.environ.get("IGNORE_EOS") == "1"
        job = Job(input_ids = ids, max_new_tokens = LONG_NEW,
                  min_new_tokens = LONG_NEW if ignore_eos else 0, sampler = GreedySampler(),
                  stop_conditions = [] if ignore_eos else [tokenizer.eos_token_id])
        generator.enqueue(job)
        t = time.time()
        st = None
        prefill_done = False
        prefill_wall = 0.0
        rows = 0
        # Generated text of the long job, so a needle prompt can be scored (NEEDLE_TEXT=1): the answers
        # are the only evidence that the 131k prefill kept the needle, and the events are otherwise
        # counted and thrown away.
        gen_text = []
        gen_ids = []
        # The first yielded event is stage "started", long before any prompt token is ingested, and a job
        # that dies inside prefill() reports zeros in its own counters, so the split below is timed on the
        # host as well: prefill ends when the job reports prefill done (or streams its first token), and a
        # "stage": "error" result (generator.reap_failed_job) is re-raised instead of being read as an eos.
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("stage") == "error":
                    print(f"MEM_ERROR long_prompt error={r.get('error')!r}", flush = True)
                    err = r.get("error")
                    raise err if isinstance(err, BaseException) else RuntimeError(f"job error: {err!r}")
                if r.get("eos"):
                    st = r
                if r.get("text"):
                    gen_text.append(str(r["text"]))
                # The generator reports generated ids under "token_ids" (tp2_bench reads that key);
                # "ids" stays as a fallback for older result shapes. Reading only "ids" is why the
                # first 131k arm printed LONG_TEXT_IDS [] although it produced 80 tokens.
                if r.get("token_ids") is not None:
                    gen_ids.append(r["token_ids"])
                elif r.get("ids") is not None:
                    gen_ids.append(r["ids"])
                if not prefill_done and (r.get("text") or r.get("stage") == "streaming"
                                         or job.is_prefill_done()):
                    prefill_done = True
                    prefill_wall = time.time() - t
                    peaks("prefill", base_alloc)
                    torch.cuda.reset_peak_memory_stats(DEVICE)
                    base_alloc = torch.cuda.memory_allocated(DEVICE)
                rows += 1
        torch.cuda.synchronize()
        wall = time.time() - t
        st = st or {}
        pf = float(st.get("time_enqueued", 0.0)) + float(st.get("time_prefill", 0.0))
        pf_wall = prefill_wall if prefill_wall else pf
        gen = float(st.get("time_generate", 0.0))
        print(f"MEM_PREFILL tokens={int(ids.shape[-1])} prefill_s={pf:.2f} "
              f"prefill_wall_s={pf_wall:.2f} "
              f"tok_per_s={int(ids.shape[-1]) / max(pf_wall, 1e-9):.1f} "
              f"chunks={limiter.chunks if limiter else None} "
              f"shrunk={limiter.shrunk if limiter else None}", flush = True)

        print(f"MEM_DECODE steps={int(st.get('new_tokens', 0))} wall_s={gen:.2f} "
              f"events={rows} total_wall_s={wall:.2f}", flush = True)
        if gen_text or gen_ids:
            text = "".join(gen_text)
            # A result's ids arrive as a tensor ([ids]) or a python list, and depending on the build the
            # outer dim is present: the first short run crashed on int() of a list. Flatten to scalars.
            def _flat(seq):
                for t in seq:
                    if hasattr(t, "tolist"):
                        t = t.tolist()
                    if isinstance(t, (list, tuple)):
                        yield from _flat(t)
                    else:
                        yield int(t)
            flat = list(_flat(gen_ids))
            print(f"LONG_TEXT chars={len(text)} tokens={len(flat)} "
                  f"text={text!r}", flush = True)
            print(f"LONG_TEXT_IDS {flat}", flush = True)
        peaks("decode", base_alloc)
        engine_report(model, cache, generator, "after_long")

    # --- depth probes: statics (dsv4_attn_out et al.) at deeper context ---------------------
    probes = [int(v) for v in os.environ.get("DEPTH_PROBES", "524288,1048570").split(",") if v.strip()]
    if probes:
        texts = ([open(LONG_PROMPT_FILE).read()] if LONG_PROMPT_FILE else []) + \
                [open(p).read() for p in CORPUS_FILES]
        if not texts:
            print("MEM_DEPTH skipped=no-prompt-source", flush = True)
        else:
            base = torch.cat([tokenizer.encode(t, add_bos = False, encode_special_tokens = True)[0]
                              for t in texts])
            bos = tokenizer.encode("a", add_bos = True, encode_special_tokens = True)[0, :1]
            print(f"MEM_DEPTH source_tokens={int(base.shape[-1])} probes={probes}", flush = True)
            prev_want, prev_s = 0, 0.0
            for want in probes:
                # enqueue() asserts when a job needs more pages than the pool holds; one generated token
                # needs its own page
                if (want + 1 + PAGE_SIZE - 1) // PAGE_SIZE > CACHE_TOKENS // PAGE_SIZE:
                    print(f"MEM_DEPTH tokens={want} skipped=page_limit cache_tokens={CACHE_TOKENS} "
                          f"pages={(want + 1 + PAGE_SIZE - 1) // PAGE_SIZE}/{CACHE_TOKENS // PAGE_SIZE}",
                          flush = True)
                    continue
                if prev_want and prev_s:
                    pred = prev_s * want / prev_want
                    if pred > DEPTH_MAX_S:
                        print(f"MEM_DEPTH tokens={want} skipped=predicted_s={pred:.0f} "
                              f"limit={DEPTH_MAX_S}", flush = True)
                        continue
                reps = want // int(base.shape[-1]) + 1
                # Distinct content per probe, so page hashes from the previous probe cannot turn the
                # prefill into a cache hit (which would time nothing and measure nothing)
                pids = torch.cat([bos, base.roll(prev_want).repeat(reps)[: want - 1]]).unsqueeze(0)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(DEVICE)
                pbase = torch.cuda.memory_allocated(DEVICE)
                limiter = None
                if PREFILL_ADAPTIVE:
                    generator.prefill_chunk_limit = limiter = PrefillChunkLimiter(DEVICE)
                pst = None
                generator.enqueue(Job(input_ids = pids, max_new_tokens = 1, sampler = GreedySampler()))
                t = time.time()
                while generator.num_remaining_jobs():
                    for r in generator.iterate():
                        if r.get("stage") == "error":
                            print(f"MEM_ERROR depth_{want} error={r.get('error')!r}", flush = True)
                            err = r.get("error")
                            raise err if isinstance(err, BaseException) else \
                                RuntimeError(f"job error: {err!r}")
                        if r.get("eos"):
                            pst = r
                torch.cuda.synchronize()
                pst = pst or {}
                ps = float(pst.get("time_enqueued", 0.0)) + float(pst.get("time_prefill", 0.0))
                wall = time.time() - t
                ps = ps if ps else wall
                prev_want, prev_s = want, ps
                print(f"MEM_DEPTH tokens={int(pids.shape[-1])} prefill_s={ps:.2f} "
                      f"tok_per_s={int(pids.shape[-1]) / max(ps, 1e-9):.1f} "
                      f"chunks={limiter.chunks if limiter else None} "
                      f"shrunk={limiter.shrunk if limiter else None} "
                      f"total_wall_s={wall:.2f}", flush = True)


if __name__ == "__main__":
    main()
