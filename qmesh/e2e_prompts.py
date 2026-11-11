#!/usr/bin/env python3
"""Multi-prompt P7-hot + GX1-cold end-to-end driver over the native RDMA seam.

Generalizes qmesh/e2e_seam.py: reads a prompt list (JSON array of strings, or a text file with
one prompt per record separated by a blank line), runs greedy decoding, and prints one JSON
line per prompt plus a summary. Timing comes from the engine's own job accounting
(time_enqueued / time_prefill / time_generate / new_tokens) and from wall clock.

  CHECKPOINT   model dir (required)
  DRAFT_CHECKPOINT  drafter (mtp) model dir (default: CHECKPOINT), e.g. the HC1 overlay
  PLACEMENT    placement json (default: the pack's PACK_MANIFEST.json, else qmesh/placement.final.json)
  PROMPTS      path to prompt list (JSON array)  [default: built-in 3 prompts]
  PROMPTS_TXT  path to a text file; each non-empty line is one prompt (overrides PROMPTS)
  CORPUS       comma-separated corpus txt files; each file is truncated to CORPUS_TOKENS tokens
  CORPUS_TOKENS default 24000
  LONG_TOKENS  one long-context prompt of this many tokens, tiled from CORPUS
  LONG_PROMPT_FILE  one long prompt read from a file (overrides CORPUS/LONG_TOKENS)
  MAX_NEW      new tokens per prompt (default 256)
  WARMUP_NEW   new tokens for warm-up prompts (default MAX_NEW)
  MAX_CHUNK    prefill chunk rows (default 2048); 'auto' sizes it once, after the loads, to the largest
               chunk whose predicted transient still fits the VRAM the card has left
  DSPARK       1 to enable the DSpark draft model
  NUM_DRAFT    draft width K (default 5)
  ARMS         comma-separated arms measured in ONE process after a single load
               (default: 'dspark1' when DSPARK=1 else 'dspark0'). Arm names: 'dspark1' (draft at
               NUM_DRAFT) and 'dspark0' (drafter resident but idle: num_draft_tokens = 0, so the
               target-only arm costs no second load). Each arm runs the identical prompt list and
               prints its own rows and `SUMMARY` line, tagged "arm". The drafter is loaded once if
               any arm drafts.
  CACHE_TOKENS KV pool tokens, allocated before model.load (default 4096; >= prompt + MAX_NEW)
  OUT          jsonl output path (also echoed to stdout)
  MAX_PROMPTS  cap the prompt list (default 0 = all)
  QMESH_B1_CACHE_SLOTS  1 opts into one target/draft recurrent slot for this B1 driver only

The round-profiled arm (`EXL3_ROUND_PROF=1`) needs its own process: roundprof resolves its
function bodies at import and the per-layer marks only fire in eager mode, so that run also sets
EXL3_WHOLE_GRAPH=0 and is not part of ARMS.
"""
from __future__ import annotations

import json
import hashlib
import os
import time
import sys

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")

import torch

from exllamav3 import Cache, CacheLayer_quant, Config, Generator, GreedySampler, Job, Model, Tokenizer
from qmesh.prefill_limit import PREFILL_ADAPTIVE, PrefillChunkLimiter, resolve_max_chunk
from qmesh import prefix_cache
from qmesh import placement as seam_placement

CHECKPOINT = os.environ["CHECKPOINT"]
DRAFT_CHECKPOINT = os.environ.get("DRAFT_CHECKPOINT", CHECKPOINT)
# qmesh/placement.py owns the choice (explicit PLACEMENT, else the pack's manifest, else the repo
# default) so this script, mem_budget.py and the loaders cannot each pick a different table: the
# 18:03 prefill died because it was the one client that still read the 2.8535 table under the 4620
# server. The resolved sha is also what the seam handshake checks against the server's Hello.
PLACEMENT = seam_placement.resolve(CHECKPOINT)

MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
MAX_CHUNK_RAW = os.environ.get("MAX_CHUNK", "2048")   # int rows, or 'auto': resolved after the loads
DSPARK = os.environ.get("DSPARK") == "1"
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "5"))
WARMUP = int(os.environ.get("WARMUP", "0"))
WARMUP_NEW = int(os.environ.get("WARMUP_NEW") or MAX_NEW)
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", "4096"))
OUT = os.environ.get("OUT", "")
_b1_slots = os.environ.get('QMESH_B1_CACHE_SLOTS', '0')
if _b1_slots not in ('0', '1'):
    raise ValueError('QMESH_B1_CACHE_SLOTS must be 0 or 1')
CACHE_SLOT_KWARGS = {'max_batch_size': 1} if _b1_slots == '1' else {}
MEM_BREAKDOWN = os.environ.get("MEM_BREAKDOWN") == "1"
CS_MEM_DIAG = os.environ.get('QMESH_CS_MEM_DIAG') == '1'


def cs_memory(phase):
    if CS_MEM_DIAG:
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(0)
        print(f'CS_MEMORY phase={phase} allocated={torch.cuda.memory_allocated(0)} '
              f'reserved={torch.cuda.memory_reserved(0)} free={free} total={total}', flush=True)

# MARGIN_PROBE=1 prints, per prompt, the top-2 logits that produced the first generated token and
# their gap. Two arms can differ in text while agreeing on 128 tokens out of 128 until one near-tied
# step flips; the gap is what says whether a flip is a reordering artefact (gap far below the kernels'
# measured 2e-3 relative difference) or a real disagreement. Read once per prompt, after the sampler
# already consumed those logits, so it cannot change the token it measures.
MARGIN_PROBE = os.environ.get("MARGIN_PROBE") == "1"
DEBUG_RESULTS = os.environ.get("DEBUG_RESULTS") == "1"
CORPUS_TOKENS = int(os.environ.get("CORPUS_TOKENS", "24000"))
CORPUS_FILES = [p for p in os.environ.get("CORPUS", "").split(",") if p.strip()]
LONG_TOKENS = int(os.environ.get("LONG_TOKENS", "0"))
LONG_PROMPT_FILE = os.environ.get("LONG_PROMPT_FILE", "")
ARMS = [a.strip() for a in os.environ.get("ARMS", "").split(",") if a.strip()]
# KV_BITS=4 (default) is DeepSeek's trained FP4 layout for the compressed pools: whole 512-wide
# latent rows plus indexer keys, as the model was trained; 8 packs the nope part with CacheLayer_quant;
# 16 is the fp16 pool. The DSA pools take the same kwargs on target and draft caches
KV_BITS = int(os.environ.get("KV_BITS", "4"))
assert KV_BITS in (4, 8, 16), "KV_BITS must be 4, 8 or 16"
KV_CACHE_KWARGS = ({} if KV_BITS == 16 else
                   {"kv_format": "fp4"} if KV_BITS == 4 else
                   {"layer_type": CacheLayer_quant, "k_bits": 8, "v_bits": 8})
DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three primary colors and explain why they matter in painting.",
]
ARM_NAMES = ("dspark0", "dspark1")


def load_prompts():
    """Returns a list of (label, prompt) pairs."""
    txt = os.environ.get("PROMPTS_TXT") or ""
    corpus = os.environ.get("CORPUS") or ""
    if corpus:
        out = []
        for path in corpus.split(","):
            path = path.strip()
            if not path:
                continue
            text = open(path, encoding="utf-8", errors="replace").read()
            out.append((os.path.basename(path), text[: CORPUS_TOKENS * 5]))
        return out
    if txt:
        text = open(txt, encoding="utf-8", errors="replace").read()
        recs = [r.strip() for r in text.split("\n\n") if r.strip()]
        return [(f"line{i}", r) for i, r in enumerate(recs)]
    path = os.environ.get("PROMPTS", "")
    if path:
        data = json.load(open(path))
        if isinstance(data, dict):
            data = [v for k, v in sorted(data.items())]
        return [(f"p{i}", p if isinstance(p, str) else p.get("prompt", ""))
                for i, p in enumerate(data)]
    return [(f"p{i}", p) for i, p in enumerate(DEFAULT_PROMPTS)]


def run_arm(spec, generator, draft_model, tokenizer, prompts, long_ids, fh, max_chunk=None):
    """Runs the whole prompt list for one arm and returns its SUMMARY dict. Both arms run in the
    same process after one load: 'dspark0' idles the drafter by zeroing the draft window, which
    the generator reads per iteration, instead of unloading it."""
    use_draft = spec != "dspark0"
    generator.num_draft_tokens = NUM_DRAFT if use_draft else 0
    rows = []
    for idx, (label, prompt) in enumerate(prompts):
        ids = (long_ids if prompt is None else
               tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True))
        mn = WARMUP_NEW if label.startswith("warm") else MAX_NEW
        assert int(ids.shape[-1]) + mn <= CACHE_TOKENS, \
            f"prompt {label}: {int(ids.shape[-1])} tokens + {mn} new exceed CACHE_TOKENS={CACHE_TOKENS}"
        limiter = None
        if PREFILL_ADAPTIVE:
            generator.prefill_chunk_limit = limiter = PrefillChunkLimiter(torch.device("cuda:0"))
        job = Job(input_ids=ids, max_new_tokens=mn, sampler=GreedySampler())
        generator.enqueue(job)
        cs_memory('before_prompt')
        prefill_logged = False
        decode_logged = False
        t0 = time.time()
        text = ""
        chunks = 0
        eos_result = None
        probed = False
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if result.get('stage') == 'prefill' and not prefill_logged:
                    cs_memory('prefill')
                    prefill_logged = True
                if result.get('stage') == 'streaming' and not decode_logged:
                    cs_memory('first_decode')
                    decode_logged = True
                if DEBUG_RESULTS:
                    print("DBG " + json.dumps(
                        {k: (list(v.shape) if isinstance(v, torch.Tensor) else v)
                         for k, v in result.items() if k != "text"}, default=str), flush=True)
                if MARGIN_PROBE and not probed and result.get("logits") is not None:
                    probed = True
                    top = torch.topk(result["logits"][0, 0].float(), 2)
                    chosen = result.get("token_ids")
                    print(f"MARGIN spec={spec} prompt={label} "
                          f"top_ids={[int(t) for t in top.indices]} "
                          f"top_logits={[round(float(v), 5) for v in top.values]} "
                          f"gap={float(top.values[0] - top.values[1]):.5f} "
                          f"chosen={int(chosen[0, 0]) if chosen is not None else -1}", flush=True)
                chunk = result.get("text", "")
                if chunk:
                    text += chunk
                    chunks += 1
                if result.get("eos"):
                    eos_result = result
                if result.get("stage") == "error":
                    # Generator.reap_failed_job contains a forward failure as an "error" result with
                    # eos=True. Without this the arm reports 0 tokens with rc=0 and the exception
                    # that killed the job never reaches the log (measured 2026-11-01: a seam arm
                    # looked "empty" while the real failure was one layer-0 request).
                    print(f"ARM_JOB_ERROR {spec} prompt={label} error={result.get('error')!r}",
                          file = sys.stderr, flush = True)
        wall = time.time() - t0
        st = eos_result or {}
        new_tokens = st.get("new_tokens") or chunks
        prefill_s = (st.get("time_enqueued", 0.0) + st.get("time_prefill", 0.0)) or float("nan")
        gen_s = st.get("time_generate", float("nan"))
        decode_ms = 1000.0 * gen_s / max(new_tokens - 1, 1)
        acc = st.get("accepted_draft_tokens", 0)
        rej = st.get("rejected_draft_tokens", 0)
        rounds = getattr(job, "verify_rounds", 0)
        toks_per_round = (new_tokens / rounds) if rounds else None
        free, total = torch.cuda.mem_get_info(0)
        row = {
            "i": idx, "label": label, "arm": spec,
            "prompt_tokens": int(ids.shape[-1]), "new_tokens": new_tokens, "max_chunk": max_chunk,
            # Tokens of this prompt that came out of the page/recurrent cache instead of being prefilled.
            # The prefix-cache arm reads this: a repeat of a prefix must show it near prompt_tokens and
            # drop ttft_ms by the prefill time it saved (generator.py reports it in complete pages).
            "cached_tokens": st.get("cached_tokens"),
            "ttft_ms": round(1000.0 * prefill_s, 1),
            "decode_ms_per_tok": round(decode_ms, 2),
            "wall_s": round(wall, 3),
            "accepted": acc, "rejected": rej, "verify_rounds": rounds,
            "accept_rate": round(acc / max(acc + rej, 1), 4) if (acc or rej) else None,
            "tokens_per_round": round(toks_per_round, 3) if toks_per_round else None,
            "verify_widths": {str(k): v for k, v in sorted(getattr(job, "verify_widths", {}).items())},
            "gpu0_free_gib": round(free / 2**30, 2),
            "prefill_chunks": limiter.chunks if limiter else None,
            "prefill_shrunk": limiter.shrunk if limiter else None,
            "text": text[:400],
        }
        # Recurrent checkpoints a later prompt in this run can resume from (RecurrentCache is an
        # OrderedDict; current_size is its byte accounting). An arm that matches no pages shows the pool
        # growing with nothing restored, which is what separates "cache on" from "cache paid off".
        rc = getattr(generator, "recurrent_cache", None)
        if rc is not None:
            print(f"PREFIX_CACHE_STATE arm={spec} prompt={label} checkpoints={len(rc)} "
                  f"mib={rc.current_size // 2**20} evictions={rc.metrics.get('stash_evictions', 0)}",
                  flush=True)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
        if MEM_BREAKDOWN and prompt is None:
            # Peak VRAM of the long-context prefill, reported the moment it finishes rather than
            # folded into the end-of-run peak (which later short prompts cannot exceed anyway).
            free_l, total_l = torch.cuda.mem_get_info(0)
            g = 2 ** 30
            print(f"MEMB after_long arm={spec} tokens={int(ids.shape[-1])} "
                  f"used={(total_l - free_l) / g:.3f} free={free_l / g:.3f} "
                  f"peak_allocated={torch.cuda.max_memory_allocated(0) / g:.3f} "
                  f"peak_reserved={torch.cuda.max_memory_reserved(0) / g:.3f} "
                  f"prefill_chunks={limiter.chunks if limiter else 0}", flush=True)
        if idx >= WARMUP:
            rows.append(row)
        del job

    def med(key):
        vals = sorted(r[key] for r in rows if r.get(key) is not None)
        return vals[len(vals) // 2] if vals else None

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    hist = {}
    for r in rows:
        for k, v in r["verify_widths"].items():
            hist[int(k)] = hist.get(int(k), 0) + v
    n_rounds = sum(hist.values())
    return {
        "arm": spec,
        "n": len(rows), "dspark": use_draft, "num_draft": NUM_DRAFT if use_draft else None,
        "max_new": MAX_NEW, "max_chunk": max_chunk,
        "decode_ms_per_tok_mean": mean("decode_ms_per_tok"),
        "decode_ms_per_tok_median": med("decode_ms_per_tok"),
        "ttft_ms_mean": mean("ttft_ms"),
        "tokens_per_round_mean": mean("tokens_per_round"),
        "accept_rate_mean": mean("accept_rate"),
        "verify_width_hist": {str(k): v for k, v in sorted(hist.items())},
        "verify_width_mean": (round(sum(k * v for k, v in hist.items()) / n_rounds, 3)
                              if n_rounds else None),
        "gate": (draft_model.gate_summary() if use_draft and draft_model is not None
                 and hasattr(draft_model, "gate_summary") else None),
        "gpu0_free_gib_min": min((r["gpu0_free_gib"] for r in rows), default=None),
        "tokens": sum(r["new_tokens"] for r in rows),
        # Prompt tokens served from the prefix cache across the arm, and the prompts that used it: the
        # pair of numbers an A/B of EXL3_PREFIX_CACHE=1 vs 0 is read from
        "cached_tokens": sum(r.get("cached_tokens") or 0 for r in rows),
        "prompts_with_cache_hits": sum(1 for r in rows if (r.get("cached_tokens") or 0) > 0),
        "wall_s": round(sum(r["wall_s"] for r in rows), 2),
    }


def main():
    placement, placement_path, placement_sha = seam_placement.load(CHECKPOINT)
    # Printed before any model work: the seam handshake refuses a server whose Hello advertises
    # another table, so this number either matches the server's or names the mismatch here.
    print(f"CLIENT_PLACEMENT path={placement_path} sha={placement_sha[:16]} ckpt={CHECKPOINT}",
          flush=True)
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
    # Recipe A sizes its hybrid expert pools during load, so the cache has to exist first
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_history=5, **CACHE_SLOT_KWARGS, **KV_CACHE_KWARGS)
    if _b1_slots == '1':
        print(f'B1_CACHE_SLOTS target={cache.num_slots} cache_tokens={CACHE_TOKENS}', flush=True)
    t0 = time.time()
    model.load(device="cuda:0", progressbar=False)
    free, total = torch.cuda.mem_get_info(0)
    print(f"P7_LOAD_OK {time.time() - t0:.1f}s "
          f"vram={torch.cuda.memory_allocated(0) / 2**30:.2f}GiB "
          f"free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB", flush=True)

    tokenizer = Tokenizer.from_config(config)
    arms = ARMS or [("dspark1" if DSPARK else "dspark0")]
    for spec in arms:
        assert spec in ARM_NAMES, f"unknown arm '{spec}' (known: {', '.join(ARM_NAMES)})"
    # One drafter load serves every arm that drafts; a dspark0-only run still loads it only if a
    # later arm needs it, so the process never pays two drafter loads
    want_draft = any(spec != "dspark0" for spec in arms)
    draft_model = None
    draft_cache = None
    if want_draft:
        draft_config = Config.from_directory(DRAFT_CHECKPOINT)
        draft_model = Model.from_config(draft_config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=CACHE_TOKENS, max_history=5, **CACHE_SLOT_KWARGS, **KV_CACHE_KWARGS)
        if _b1_slots == '1':
            print(f'B1_CACHE_SLOTS draft={draft_cache.num_slots} cache_tokens={CACHE_TOKENS}', flush=True)
        t1 = time.time()
        draft_model.load(device="cuda:0", progressbar=False)
        cs_memory('after_draft_load')
    # MAX_CHUNK is resolved here, after every load and immediately before the Generator, because 'auto' reads
    # what the card actually has left once the weights, the KV pool and the drafter are resident. A numeric
    # MAX_CHUNK comes back unchanged.
    max_chunk = resolve_max_chunk(torch.device("cuda:0"), MAX_CHUNK_RAW)
    print(f"MAX_CHUNK_EFFECTIVE rows={max_chunk} raw={MAX_CHUNK_RAW}", flush=True)
    # The recurrent-state pool is sized from EXL3_PREFIX_CACHE* (default on, which is what this driver has
    # always run); generator_kwargs() states no checkpoint interval unless asked, so an arm with the knob
    # unset keeps the model's own 2048/32768 schedule and stays comparable with earlier numbers.
    generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=max_chunk,
                          draft_model=draft_model, draft_cache=draft_cache,
                          num_draft_tokens=NUM_DRAFT if want_draft else None,
                          **prefix_cache.generator_kwargs())
    prefix_cache.apply(generator)
    print(prefix_cache.summary(), flush=True)
    if MEM_BREAKDOWN:
        # Resident VRAM by owner after load (target + drafter + caches + generator statics),
        # before any job; then the peak since this reset is reported at the end
        from exllamav3.util.memory import vram_accounting
        g = 2 ** 30
        device = torch.device("cuda:0")
        for name, m, c in (("target", model, cache), ("drafter", draft_model, draft_cache)):
            if m is None:
                continue
            r = vram_accounting(m, c, generator if name == "target" else None, [device])[0]
            print(f"MEMB {name} weights={r.weights / g:.3f} kv_cache={r.cache / g:.3f} "
                  f"recurrent={r.recurrent / g:.3f} statics={r.statics / g:.3f} "
                  f"generator={r.generator / g:.3f} other={r.other / g:.3f} kv_bits={KV_BITS}", flush=True)
            if r.statics_by_tag:
                for tag, b in sorted(r.statics_by_tag.items(), key=lambda kv: -kv[1]):
                    print(f"MEMB statics_tag {name} {tag}={b / g:.3f}", flush=True)
        free, total = torch.cuda.mem_get_info(device)
        print(f"MEMB device total={total / g:.3f} used={(total - free) / g:.3f} "
              f"torch_reserved={torch.cuda.memory_reserved(device) / g:.3f} "
              f"torch_allocated={torch.cuda.memory_allocated(device) / g:.3f} "
              f"non_torch={(total - free - torch.cuda.memory_reserved(device)) / g:.3f}", flush=True)
        torch.cuda.reset_peak_memory_stats(device)

    prompts = load_prompts()[: int(os.environ.get("MAX_PROMPTS", "0")) or None]
    # Warm-up prompts are synthetic, excluded from the summary, and never touch the prompt list
    warm = ["Write a paragraph about the history of sailing ships and their navigation instruments."] * WARMUP
    prompts = [(f"warm{i}", p) for i, p in enumerate(warm)] + prompts
    # LONG_TOKENS: a single long-context prompt tiled from the corpus files, for peak-VRAM and
    # prefill-rate measurement at a real context width
    long_ids = None
    if LONG_PROMPT_FILE:
        long_ids = tokenizer.encode(open(LONG_PROMPT_FILE).read(), add_bos=True,
                                    encode_special_tokens=True)
        print(f"LONG_PROMPT tokens={int(long_ids.shape[-1])} file={LONG_PROMPT_FILE}", flush=True)
        prompts = [("long", None)] + prompts[:WARMUP]
    elif LONG_TOKENS:
        cat = []
        for path in (CORPUS_FILES or []):
            cat.append(tokenizer.encode(open(path).read(), add_bos=False,
                                        encode_special_tokens=True)[0])
        assert cat, "LONG_TOKENS needs CORPUS=<file[,file...]>"
        flat = torch.cat(cat)
        bos = tokenizer.encode("a", add_bos=True, encode_special_tokens=True)[0, :1]
        reps = LONG_TOKENS // int(flat.shape[-1]) + 1
        long_ids = torch.cat([bos, flat.repeat(reps)[: LONG_TOKENS - 1]]).unsqueeze(0)
        print(f"LONG_PROMPT tokens={int(long_ids.shape[-1])} files={len(cat)}", flush=True)
        prompts = [("long", None)] + prompts[:WARMUP]
    fh = open(OUT, "a", encoding="utf-8") if OUT else None
    for spec in arms:
        print(f"ARM_START {spec} prompts={len(prompts)} max_new={MAX_NEW} "
              f"num_draft={NUM_DRAFT if spec != 'dspark0' else 0}", flush=True)
        summary = run_arm(spec, generator, draft_model, tokenizer, prompts, long_ids, fh,
                          max_chunk=max_chunk)
        print("SUMMARY " + json.dumps(summary), flush=True)
        if fh:
            fh.write("SUMMARY " + json.dumps(summary) + "\n")
            fh.flush()
        print(f"ARM_END {spec}", flush=True)
    if MEM_BREAKDOWN:
        g = 2 ** 30
        device = torch.device("cuda:0")
        free_now, total = torch.cuda.mem_get_info(device)
        print(f"MEMB peak_since_load allocated={torch.cuda.max_memory_allocated(device) / g:.3f} "
              f"reserved={torch.cuda.max_memory_reserved(device) / g:.3f} "
              f"device_used_now={(total - free_now) / g:.3f} free_now={free_now / g:.3f}", flush=True)
    if fh:
        fh.close()
    if os.environ.get("QMESH_COLDSTEAL") == "1":
        # Release pinned pool allocations while the CUDA context and model are still alive.
        from qmesh import coldsteal
        coldsteal.manager().close()


if __name__ == "__main__":
    main()
