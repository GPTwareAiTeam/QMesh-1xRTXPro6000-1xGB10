#!/usr/bin/env python3
"""Teacher-forced perplexity of the 4460 (official-FP8-quantized) EXL3 build as it serves over the seam.

One process = one client half of a pair (P7 GPU0/1 hot experts + its own GX cold experts), scoring the
production path: `Cache` + `Generator` prefill, i.e. exactly the kernels, bucket geometry and remote
expert picks that serving uses. Nothing here is TP=2: the 2xRTX harness (2x-rtx-tp2/chunk-src/qmesh/
tp2/panel_ppl.py) is the origin of the protocols and is adapted in three ways -- the model comes from
`Config.from_directory($CHECKPOINT)` plus the pack's placement instead of a rank overlay, there is no
distributed rank, and logits are captured single-process.

Protocols (unchanged, so the numbers compare with the published ones):
  WIKI2  raw wikitext-2 test split, n_ctx tokens per chunk, chunk[0] <- BOS, score positions
         [n_ctx/2, n_ctx - 1) predicting tokens t+1, 140 chunks (143220 positions) for the full split.
  PANEL  sudoingX/dgx-spark-deepseek-v41 conversations rendered by the checkpoint's own encoder,
         assistant-only mask, position t predicts token t+1, truncated to --panel-max tokens.

Set CHECKPOINT, MODEL_DIR, SEAM_PEER and the required dataset paths before running.
KV mode is fixed per process (it is a Cache construction argument): KV_BITS=4 (shipping FP4), 8 or 16.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

# Scoring runs the same graph-free path as the shipping arm: the generator's decode graph and the
# fused expert split are not involved in a prefill-only job, and the fused split changes the pick
# geometry the seam server was validated against
os.environ.setdefault("EXL3_WHOLE_GRAPH", "0")
os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import panel_encode_4460 as PE  # noqa: E402  (pure-stdlib module: safe without CUDA)

CHECKPOINT = os.environ.get("CHECKPOINT")
SEAM_PEER = os.environ.get("SEAM_PEER")
SEAM_PORT = int(os.environ.get("SEAM_PORT", "47311"))
KV_BITS = int(os.environ.get("KV_BITS", "4"))
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", "4096"))
PANEL_FILE = os.environ.get("PANEL_FILE")
SCORE_CHUNK = 64  # logits rows scored per logsumexp block (memory)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kv_cache_kwargs(bits: int) -> dict:
    """Same mapping as the shipping client (qmesh/e2e_prompts.py): 4 = trained FP4 pools."""
    from exllamav3 import CacheLayer_quant
    assert bits in (4, 8, 16), "KV_BITS must be 4, 8 or 16"
    if bits == 16:
        return {}
    if bits == 4:
        return {"kv_format": "fp4"}
    return {"layer_type": CacheLayer_quant, "k_bits": 8, "v_bits": 8}


# --- model -----------------------------------------------------------------------------------------

def build_model(kv_bits: int, ctx: int):
    """The shipping seam client, minus drafter: hot experts on this GPU, cold served by the peer."""
    import torch
    from exllamav3 import Cache, Config, Model, Tokenizer
    from qmesh import placement as seam_placement

    placement, placement_path, placement_sha = seam_placement.load(CHECKPOINT)
    # Printed before any model work: the seam handshake refuses a server whose Hello advertises
    # another table, so this number either matches the server's or names the mismatch here.
    log(f"CLIENT_PLACEMENT path={placement_path} sha={placement_sha[:16]} ckpt={CHECKPOINT}")
    hot = seam_placement.hot(placement)
    config = Config.from_directory(CHECKPOINT)
    config.infer_params.moe_cpu_split = 1
    config.infer_params.moe_remote_hot = hot
    config.infer_params.remote_expert_host = {"host": SEAM_PEER, "port": SEAM_PORT}
    model = Model.from_config(config)
    # The cache has to exist before load(): loading is what allocates its layer tensors
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_history=2, **kv_cache_kwargs(kv_bits))
    t0 = time.time()
    model.load(device="cuda:0", progressbar=False)
    free, total = torch.cuda.mem_get_info(0)
    log(f"P7_LOAD_OK {time.time() - t0:.1f}s vram={torch.cuda.memory_allocated(0) / 2**30:.2f}GiB "
        f"free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB kv_bits={kv_bits} ctx={ctx}")
    tokenizer = Tokenizer.from_config(config)
    return model, torch.device("cuda", 0), tokenizer, cache, placement_path, placement_sha


# --- cached (production) path ----------------------------------------------------------------------
# `flash_attn_nc` is a stateless single-pass path and is NOT prefix-stable on these builds: the same
# token at the same absolute position changes NLL when only the trailing chunk length changes, so a
# score taken with it describes a model nobody serves. Everything below scores through the cached
# path the generator serves with.

CACHED_GEN = None
CACHED_CAP: list | None = None   # per-prefill-chunk logits captured from the full-stack forward


def _install_prefill_with_logits(model):
    """Replace model.prefill with the same cached forward plus the head over every prompt row.

    Model.prefill stops at the last cache-writing module, so it never runs the head (job.prefill
    ignores its return value). The full forward is the same kernels plus the head over every row --
    the call shape job.py already uses for MTP drafting, minus `last_tokens_only`.
    """
    def prefill(input_ids, params = None):
        p = dict(params or {})
        p.pop("last_tokens_only", None)
        y = model.forward(input_ids, p)
        lg = y.get("logits") if isinstance(y, dict) else y
        if lg is not None and CACHED_CAP is not None:
            CACHED_CAP.append(lg)
        return None
    model.prefill = prefill


def build_cached_scorer(model, cache, tokenizer, chunk: int):
    """Install the prefill hook that returns logits for every prompt row, and the Generator driving it."""
    from exllamav3 import Generator
    # recurrent_cache_size=0: the release documents -rcs 0 as "checkpoints larger than the whole
    # budget are not stored at all ... a later restore misses and falls back to replay", which is
    # exactly right here -- every sequence is prefilled once from position 0 and never resumed
    gen = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=chunk,
                    recurrent_cache_size=0)
    _install_prefill_with_logits(model)
    return gen


def cached_logits(model, device, ids: list[int]):
    """[1, len(ids), vocab] logits of one sequence from one cached prefill (production path)."""
    import torch
    from exllamav3 import GreedySampler, Job
    global CACHED_CAP
    gen = CACHED_GEN
    CACHED_CAP = []
    # Drop every cached page: a panel conversation's prompt is often a prefix of the previous one, and
    # a prefix hit would skip exactly the prefill rows this harness needs logits for
    gen.pagetable.reset_page_table()
    gen.enqueue(Job(input_ids = torch.tensor([ids], dtype = torch.long), max_new_tokens = 1,
                    min_new_tokens = 1, sampler = GreedySampler(), stop_conditions = [],
                    return_logits = True))
    # Prefill covers every token but the last; the generator feeds the last one as the first decode
    # step and returns that row's pre-sampling logits in the result (return_logits)
    last = []
    while gen.num_remaining_jobs():
        for result in gen.iterate():
            if "error" in result:
                raise result["error"]
            if "logits" in result:
                last.append(result["logits"])
    cap, CACHED_CAP = CACHED_CAP, None
    if not cap or not last:
        raise RuntimeError("cached prefill captured no logits")
    tail = last[0].to(device = cap[0].device).reshape(1, -1, last[0].shape[-1])[:, :1, :]
    w = min(tail.shape[-1], *(c.shape[-1] for c in cap))   # scoring reads only the first `vocab` columns
    lg = torch.cat([c[..., :w] for c in cap] + [tail[..., :w].to(cap[0].dtype)], dim = 1)
    assert lg.shape[1] == len(ids), f"captured {lg.shape[1]} logits rows for {len(ids)} tokens"
    return lg


def nll_at(lg, rows, targets, vocab: int):
    """NLL (nats) of targets given logits rows, over the first `vocab` columns, in row blocks."""
    import torch
    out = torch.empty(rows.numel(), dtype = torch.float64, device = lg.device)
    for a in range(0, rows.numel(), SCORE_CHUNK):
        b = min(a + SCORE_CHUNK, rows.numel())
        sel = lg.index_select(0, rows[a:b])[:, :vocab].float()
        lse = torch.logsumexp(sel, -1)
        picked = sel.gather(-1, targets[a:b].unsqueeze(-1)).squeeze(-1)
        out[a:b] = (lse - picked).double()
    return out


# --- protocols --------------------------------------------------------------------------------------

def run_wikitext(model, device, raw_tok, args, vocab: int, bos: int | None) -> dict:
    import torch
    text = PE.load_wikitext2_raw()
    tokens = raw_tok.encode(text, add_special_tokens = False).ids
    n_ctx = args.ctx
    if len(tokens) < 2 * n_ctx:
        raise ValueError(f"wikitext-2 needs >= {2 * n_ctx} tokens, got {len(tokens)}")
    n_chunks_total = len(tokens) // n_ctx
    n_chunks = n_chunks_total if not args.chunks else min(args.chunks, n_chunks_total)
    first = n_ctx // 2
    log(f"WIKI2 tokens={len(tokens)} chunks={n_chunks_total} (scoring {n_chunks}) n_ctx={n_ctx} "
        f"positions=[{first},{n_ctx - 2}] zip={PE.WIKITEXT_ZIP}")

    prefix_chunks = sorted({int(x) for x in args.prefix_chunks.split(",") if x.strip()})
    quick: dict[str, dict] = {}
    total_nll = total_nll2 = 0.0
    total_count = 0
    per_chunk = []
    t0 = time.time()
    for i in range(n_chunks):
        start = i * n_ctx
        chunk = list(tokens[start:start + n_ctx])
        if bos is not None:
            chunk[0] = bos
        lg = cached_logits(model, device, chunk)[0]
        rows = torch.arange(first, n_ctx - 1, device = device, dtype = torch.long)
        targets = torch.tensor(tokens[start + first + 1:start + n_ctx], dtype = torch.long, device = device)
        nll = nll_at(lg, rows, targets, vocab)
        total_nll += nll.sum().item()
        total_nll2 += (nll * nll).sum().item()
        total_count += nll.numel()
        per_chunk.append(round(math.exp(total_nll / total_count), 4))
        # The published comparisons used only the first 8 chunks, so report that prefix from the same
        # pass: identical rows, no extra forward
        if (i + 1) in prefix_chunks:
            quick[str(i + 1)] = dict(positions = total_count,
                                     **_ppl_block(total_nll, total_nll2, total_count))
            log(f"WIKI2 QUICK{i + 1} ppl={quick[str(i + 1)]['ppl']:.4f} +/- "
                f"{quick[str(i + 1)]['ppl_stderr']:.5f} positions={total_count}")
        log(f"WIKI2 chunk {i + 1}/{n_chunks} running_ppl={per_chunk[-1]:.4f} ({time.time() - t0:.1f}s)")
        del lg, nll
        torch.cuda.empty_cache()
    res = {"chunks_scored": n_chunks, "chunks_total": n_chunks_total, "n_ctx": n_ctx,
           "positions": total_count, "elapsed_s": round(time.time() - t0, 1)}
    if total_count:
        mean_nll = total_nll / total_count
        mean_nll2 = total_nll2 / total_count
        var = max(0.0, mean_nll2 - mean_nll * mean_nll)
        stderr_nll = math.sqrt(var / (total_count - 1)) if total_count > 1 else 0.0
        ppl = math.exp(mean_nll)
        res.update({"nll_mean": mean_nll, "ppl": ppl, "ppl_stderr": stderr_nll * ppl,
                    "per_chunk_running_ppl": per_chunk})
        if quick:
            res["quick"] = quick
        log(f"WIKI2 RESULT ppl={ppl:.4f} +/- {stderr_nll * ppl:.5f} nll={mean_nll:.6f} "
            f"positions={total_count}")
    return res


def _ppl_block(nll_sum: float, nll2_sum: float, count: int) -> dict:
    mean = nll_sum / count
    var = max(0.0, nll2_sum / count - mean * mean)
    stderr = math.sqrt(var / (count - 1)) if count > 1 else 0.0
    ppl = math.exp(mean)
    return {"nll_mean": mean, "ppl": ppl, "ppl_stderr": stderr * ppl}


def run_panel(model, device, enc, args, vocab: int) -> dict:
    """Score every --panel-file; returns per-file results plus the aggregate over all of them."""
    import torch
    tot_nll = tot_nll2 = 0.0
    tot_count = 0
    by_dom: dict[str, list] = {}
    out_files: dict[str, dict] = {}
    prefixes = sorted({int(x) for x in args.prefix.split(",") if x.strip()})
    t_all = time.time()
    for pf in args.panel_files:
        name = os.path.basename(pf)
        rows = [json.loads(line) for line in open(pf, encoding = "utf-8")]
        rows = [r for r in rows if r.get("split", "panel") == "panel"]
        if args.convs:
            rows = rows[:args.convs]
        log(f"PANEL file={pf} conversations={len(rows)} panel_max={args.panel_max}")
        f_nll = f_nll2 = 0.0
        f_count = 0
        f_dom: dict[str, list] = {}
        per_conv = []
        prefix_out: dict[str, dict] = {}
        t0 = time.time()
        for i, r in enumerate(rows):
            ids, score = enc(r)
            ids, score = ids[:args.panel_max], score[:args.panel_max]
            assert 2 <= len(ids) <= args.ctx, f"{r['id']}: rendered length {len(ids)} outside 2..{args.ctx}"
            lg = cached_logits(model, device, ids)[0]
            sc = torch.tensor(score, dtype = torch.bool, device = device)
            pos = sc[1:].nonzero().squeeze(-1)      # position t predicts token t+1
            n_sum = n2_sum = 0.0
            n_count = 0
            if pos.numel():
                targets = torch.tensor(ids[1:], dtype = torch.long, device = device)[pos]
                nll = nll_at(lg, pos, targets, vocab)
                n_sum = nll.sum().item()
                n2_sum = (nll * nll).sum().item()
                n_count = int(nll.numel())
                f_nll += n_sum
                f_nll2 += n2_sum
                f_count += n_count
                dom = str(r.get("domain", "?")).split("_")[0]
                f_dom.setdefault(dom, []).append((n_sum, n_count))
                per_conv.append({"id": r["id"], "domain": r.get("domain"), "length": len(ids),
                                 "scored": n_count, "nll_sum": n_sum})
            # Cumulative prefixes: the earlier quick panels scored only the first 8 conversations, so a
            # full-panel run can also answer those, in the same order the references used
            if not args.convs and (i + 1) in prefixes:
                prefix_out[str(i + 1)] = dict(positions = f_count,
                                              **_ppl_block(f_nll, f_nll2, f_count))
                log(f"PANEL {name} prefix{i + 1} ppl={prefix_out[str(i + 1)]['ppl']:.4f} "
                    f"positions={f_count}")
            log_every = 10 if len(rows) > 20 else 1
            if (i + 1) % log_every == 0 or i + 1 == len(rows):
                running = math.exp(f_nll / f_count) if f_count else float("nan")
                log(f"PANEL {name} {i + 1}/{len(rows)} running_ppl={running:.4f} positions={f_count} "
                    f"({time.time() - t0:.1f}s)")
            del lg
            torch.cuda.empty_cache()
        res = {"file": pf, "conversations": len(rows), "panel_max": args.panel_max,
               "positions": f_count, "elapsed_s": round(time.time() - t0, 1)}
        if f_count:
            res.update(_ppl_block(f_nll, f_nll2, f_count))
            res["by_domain"] = {k: {"nll": sum(v[0] for v in x) / sum(v[1] for v in x),
                                    "positions": sum(v[1] for v in x)}
                                for k, x in sorted(f_dom.items())}
            res["per_conversation"] = per_conv
            res["prefix"] = prefix_out
            log(f"PANEL {name} ppl={res['ppl']:.4f} nll={res['nll_mean']:.6f} positions={f_count} "
                f"({res['elapsed_s']}s)")
            tot_nll += f_nll
            tot_nll2 += f_nll2
            tot_count += f_count
            for k, v in f_dom.items():
                by_dom.setdefault(k, []).extend(v)
        out_files[name] = res
    agg = {"files": list(out_files), "conversations": sum(v["conversations"] for v in out_files.values()),
           "panel_max": args.panel_max, "positions": tot_count, "elapsed_s": round(time.time() - t_all, 1),
           "by_file": out_files}
    if tot_count:
        agg.update(_ppl_block(tot_nll, tot_nll2, tot_count))
        agg["by_domain"] = {k: {"nll": sum(v[0] for v in x) / sum(v[1] for v in x),
                                "positions": sum(v[1] for v in x)}
                            for k, x in sorted(by_dom.items())}
        log(f"PANEL RESULT ppl={agg['ppl']:.4f} nll={agg['nll_mean']:.6f} positions={tot_count} "
            f"files={len(out_files)}")
    return agg

SANITY_TEXTS = {
    "facts": "The capital of France is Paris. The capital of Germany is Berlin. "
             "The capital of Italy is Rome. The capital of Spain is Madrid. "
             "The capital of Portugal is Lisbon. The capital of Austria is Vienna.",
    "repeat": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
              "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
              "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
              "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
    "code": "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n\n"
            "x = add(2, 3)\ny = mul(x, 4)\nprint(x, y)\n",
}


def run_sanity(model, device, tokenizer, vocab: int) -> dict:
    """Absolute forward check before the long protocols: boilerplate the model nearly predicts.

    ppl far above these values means the forward is wrong (cold experts not picked, a bad bucket,
    a corrupt pack), not that the data is hard -- the quick panels are only a few hundred positions.
    """
    import torch
    out = {}
    for name, text in SANITY_TEXTS.items():
        ids = tokenizer.encode(text, add_bos = True)[0].tolist()
        lg = cached_logits(model, device, ids)[0]
        pos = torch.arange(len(ids) - 1, device = device)
        targets = torch.tensor(ids[1:], dtype = torch.long, device = device)
        nll = nll_at(lg, pos, targets, vocab)
        top1 = lg.index_select(0, pos)[:, :vocab].float().argmax(-1)
        mean = nll.mean().item()
        out[name] = {"tokens": len(ids), "nll_mean": mean, "ppl": math.exp(mean),
                     "top1_acc": (top1 == targets).float().mean().item()}
        log(f"SANITY {name} tokens={len(ids)} nll={mean:.4f} ppl={out[name]['ppl']:.3f} "
            f"top1={out[name]['top1_acc']:.3f}")
        del lg
    return out



# --- dry run (no CUDA, no model) --------------------------------------------------------------------

def resolve_placement(ckpt: str):
    """qmesh.placement's order, without importing the extension: explicit, pack manifest, repo default."""
    explicit = os.environ.get("PLACEMENT")
    if explicit:
        return explicit, "env PLACEMENT"
    mp = os.path.join(ckpt, "PACK_MANIFEST.json")
    try:
        p = json.load(open(mp)).get("placement")
    except (OSError, ValueError):
        p = None
    if isinstance(p, str) and p:
        for cand in (p, os.path.join(ckpt, p)):
            if os.path.exists(cand):
                return os.path.abspath(cand), f"pack manifest {mp}"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "qmesh", "placement.final.json"), "repo default"


def dry_run(args) -> int:
    """Exercise everything that can be checked without a GPU: placement, packs, encoders, protocols."""
    ok = True
    path, how = resolve_placement(CHECKPOINT)
    if not os.path.exists(path):
        log(f"DRY_FAIL placement_missing {path} ({how})")
        return 1
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    log(f"DRY placement path={path} sha={sha[:16]} via={how}")
    try:
        man = json.load(open(os.path.join(CHECKPOINT, "PACK_MANIFEST.json")))
        want = man.get("placement_sha256")
        log(f"DRY pack={CHECKPOINT} tensors={man.get('tensors')} mode={man.get('mode')} "
            f"manifest_placement_sha={str(want)[:16]}")
        if want and want != sha:
            log(f"DRY_FAIL placement_sha_mismatch manifest={want[:16]} resolved={sha[:16]}")
            ok = False
    except (OSError, ValueError) as e:
        log(f"DRY_FAIL pack_manifest {e!r}")
        ok = False
    log(f"DRY env kv_bits={KV_BITS} cache_tokens={CACHE_TOKENS} peer={SEAM_PEER}:{SEAM_PORT} "
        f"ckpt={CHECKPOINT} model_dir={PE.MODEL_DIR} wikitext_zip={PE.WIKITEXT_ZIP}")
    for k in ("REMOTE_TRANSPORT", "SEAM_DEV", "SEAM_GID", "SEAM_BUCKETS", "SEAM_BUILD_ID"):
        log(f"DRY env {k}={os.environ.get(k, '<unset>')}")

    try:
        raw_tok, enc = PE.make_encoder()
    except Exception as e:  # a missing tokenizer.json / encoding module is the usual cause
        log(f"DRY_FAIL encoder {e!r}")
        return 1
    if args.mode in ("both", "wiki2"):
        try:
            text = PE.load_wikitext2_raw()
            tokens = raw_tok.encode(text, add_special_tokens = False).ids
            n_ctx = args.ctx
            total = len(tokens) // n_ctx
            n = min(args.chunks, total) if args.chunks else total
            # The model's own BOS id comes from the exllamav3 tokenizer at scoring time; the raw
            # tokenizer here is only used for the text -> ids count
            log(f"DRY WIKI2 tokens={len(tokens)} chunks={total} (scoring {n}) positions_per_chunk="
                f"{n_ctx - 1 - n_ctx // 2} positions={n * (n_ctx - 1 - n_ctx // 2)}")
        except Exception as e:
            log(f"DRY_FAIL wikitext {e!r}")
            ok = False
    if args.mode in ("both", "panel"):
        refs = list(args.reference or [])
        for i, pf in enumerate(args.panel_files):
            try:
                rows = [json.loads(line) for line in open(pf, encoding = "utf-8")]
            except OSError as e:
                log(f"DRY_FAIL panel_open {pf} {e!r}")
                ok = False
                continue
            rows = [r for r in rows if r.get("split", "panel") == "panel"]
            if args.convs:
                rows = rows[:args.convs]
            n_pos = 0
            trunc = 0
            t0 = time.time()
            for r in rows:
                ids, score = enc(r)
                if len(ids) > args.panel_max:
                    trunc += 1
                ids, score = ids[:args.panel_max], score[:args.panel_max]
                assert 2 <= len(ids) <= args.ctx, f"{r['id']}: rendered length {len(ids)} outside 2..{args.ctx}"
                n_pos += sum(score[1:])
            msg = (f"DRY PANEL file={pf} conversations={len(rows)} panel_max={args.panel_max} "
                   f"positions={n_pos} truncated={trunc} ({time.time() - t0:.1f}s)")
            ref = None
            if i < len(refs):
                try:
                    ref = json.load(open(refs[i], encoding = "utf-8"))
                except (OSError, ValueError) as e:
                    log(f"DRY_FAIL reference {refs[i]} {e!r}")
                    ok = False
            if ref:
                rpos = ref.get("positions")
                match = "" if rpos is None else (" MATCH" if int(rpos) == n_pos else
                                                 f" MISMATCH(want {rpos})")
                if match.startswith(" MISMATCH"):
                    ok = False
                msg += f" reference_positions={rpos}{match} reference_nll_native={ref.get('nll_native')}"
            log(msg)
    log("DRY_OK" if ok else "DRY_FAILED")
    return 0 if ok else 1


def main():
    global CACHED_GEN
    ap = argparse.ArgumentParser(allow_abbrev = False)
    ap.add_argument("--mode", default = "both", choices = ["both", "wiki2", "panel"])
    ap.add_argument("--ctx", type = int, default = 2048, help = "n_ctx for wiki2 and the max sequence length")
    ap.add_argument("--chunks", type = int, default = 0, help = "limit the number of wiki2 chunks (0 = all)")
    ap.add_argument("--convs", type = int, default = 0, help = "limit the number of panel conversations")
    ap.add_argument("--panel-max", type = int, default = 2048)
    ap.add_argument("--panel-file", action = "append", default = None,
                    help = "panel jsonl to score; repeat to score several in one model load")
    ap.add_argument("--reference", action = "append", default = None,
                    help = "upstream kld json per panel file (same order); reports its published nll/ppl")
    ap.add_argument("--prefix", default = "8",
                    help = "comma list of conversation counts to also report as running prefixes")
    ap.add_argument("--prefix-chunks", default = "8",
                    help = "comma list of wiki2 chunk counts to also report as running prefixes")
    ap.add_argument("--kv-bits", type = int, default = KV_BITS,
                    help = "KV cache mode for this process (must match the server's expectation)")
    ap.add_argument("--out", default = "")
    ap.add_argument("--sanity", action = "store_true", help = "score boilerplate texts first (forward check)")
    ap.add_argument("--dry-run", action = "store_true", help = "encode + resolve only: no CUDA, no model")
    ap.add_argument("--prefill-chunk", type = int, default = 0,
                    help = "cached path prefill chunk size (0 = one chunk per sequence)")
    args = ap.parse_args()
    args.panel_files = args.panel_file or ([PANEL_FILE] if PANEL_FILE else [])
    if not CHECKPOINT:
        ap.error("set CHECKPOINT to the hot expert pack")
    if not SEAM_PEER and not args.dry_run:
        ap.error("set SEAM_PEER to the cold expert server address")
    if args.mode in ("both", "panel") and not args.panel_files:
        ap.error("provide --panel-file or set PANEL_FILE")
    if not (2 * 256 <= args.ctx <= 65536):
        raise ValueError("--ctx out of range")
    if not (2 <= args.panel_max <= args.ctx):
        raise ValueError("--panel-max must be within [2, ctx]")
    if args.kv_bits not in (4, 8, 16):
        raise ValueError("--kv-bits must be 4, 8 or 16")
    if args.dry_run:
        sys.exit(dry_run(args))

    model, device, tokenizer, cache, placement_path, placement_sha = build_model(args.kv_bits, args.ctx)
    import torch
    vocab = tokenizer.actual_vocab_size
    prefill_chunk = args.prefill_chunk or args.ctx
    CACHED_GEN = build_cached_scorer(model, cache, tokenizer, prefill_chunk)
    log(f"ATTN cached prefill_chunk={prefill_chunk} vocab={vocab} "
        f"(production path: full-stack forward + head)")
    out = {"model_dir": PE.MODEL_DIR, "checkpoint": CHECKPOINT, "placement": placement_path,
           "placement_sha": placement_sha, "kv_bits": args.kv_bits, "ctx": args.ctx,
           "prefill_chunk": prefill_chunk, "cache_tokens": CACHE_TOKENS,
           "seam_peer": f"{SEAM_PEER}:{SEAM_PORT}", "build_id": os.environ.get("SEAM_BUILD_ID", ""),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.sanity:
        out["sanity"] = run_sanity(model, device, tokenizer, vocab)
    if args.reference:
        refs = {}
        for i, rf in enumerate(args.reference):
            name = os.path.basename(args.panel_files[i]) if i < len(args.panel_files) else f"ref{i}"
            try:
                ref = json.load(open(rf, encoding = "utf-8"))
                r = {k: ref.get(k) for k in ("positions", "nll_native", "ppl_native", "nll_ours")}
                if ref.get("nll_native") is not None:
                    r["ppl_native"] = math.exp(ref["nll_native"])
                if ref.get("nll_ours") is not None:
                    r["ppl_ours"] = math.exp(ref["nll_ours"])
                refs[name] = r
            except Exception as e:  # a missing/foreign reference file must not fail the run
                log(f"reference {rf}: {e!r}")
        out["reference"] = refs

    if args.mode in ("both", "wiki2"):
        raw_tok = PE.make_encoder()[0]
        out["wiki2"] = run_wikitext(model, device, raw_tok, args, vocab, tokenizer.bos_token_id)
    if args.mode in ("both", "panel"):
        enc = PE.make_encoder()[1]
        out["panel"] = run_panel(model, device, enc, args, vocab)

    def _trim(v):   # keep the RESULT log line readable: drop the 199-row per-conversation tables
        if isinstance(v, dict):
            return {k: _trim(x) for k, x in v.items() if k != "per_conversation"}
        return v
    log("RESULT " + json.dumps(_trim(out)))
    if args.out:
        with open(args.out, "w", encoding = "utf-8") as f:
            json.dump(out, f, indent = 1)
        log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
