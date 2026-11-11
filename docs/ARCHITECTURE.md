# Architecture

DeepSeek-V4.1-Flash-1017 has 40 MoE layers with 384 routed experts each (top-8 per token), plus attention, shared
experts, Engram tables and a DSpark drafter. At 2.935 bpw the routed experts alone are about 120 GB: too much for one
96 GB card, and the GB10's 128 GB of unified memory is shared with its OS and has a fraction of the RTX's bandwidth.
R5 splits the model where each half is strongest.

```
                         RTX PRO 6000 (96 GB)                                    GB10 (128 GB unified)
  request ─▶ HTTP ─▶ scheduler ─▶ per layer:                                     per layer:
                         attention (batched), dense, shared expert
                         router ─┬─ hot experts (5,660): coop kernel  ┐
                                 ├─ ColdSteal: cached cold experts    │  bf16   cold experts (9,700):
                                 └─ cold picks ──────── RoCE ──────────┼───────▶ one fused launch per
                         combine ◀──────────────────── RoCE ───────────┘◀─────── layer, fixed-order sum
                         DSpark drafter + EV gate ─▶ verify
                         KV cache (FP4, growable to 5M tokens)
```

## Expert placement

Experts are ranked by measured routing frequency on real traffic. The 5,660 most-routed (900 at 4 bpw, 4,760 at 3 bpw)
live on the RTX; the other 9,700 (7,800 at 3 bpw, 1,900 at 2 bpw) on the GB10. A placement file binds the two packs:
it defines which global expert IDs each half owns, and both packs' manifests carry its hash, so mismatched halves refuse
to start. The measured ranking beats a balanced placement by 13.35 vs 14.15 ms/token.

## The seam

Each MoE layer the RTX sends the hidden states of tokens routed to cold experts, plus their expert IDs and weights, to
the GB10 and waits for the weighted sum back. Transport is RDMA over RoCE v2 through pinned host buffers (not
GPUDirect): a control connection negotiates row buckets, data connections carry bf16 requests and bf16 replies. One
loaded GB10 worker serves many client sessions in turn, so restarting the HTTP server does not reload 120 GB of
experts. A dead handle fails its session, not the worker.

On the GB10 a native C path serves every layer: one plan plus one gate/up launch and one down launch per decode piece
across all bit-width groups, and a fixed-order reduction, so the same input always produces the same output (racing
atomics made drafts get rejected spuriously).

## ColdSteal

While the RTX waits for the GB10, it is idle. ColdSteal fills that time: each layer keeps up to 8 recently used cold
experts in RTX memory, and when a token routes to one of them the RTX computes it itself, inside the same cooperative
launch as the hot experts, before the remaining cold picks are sent. Fewer rows cross the link and the GB10 finishes
sooner. The slots are refilled least-recently-used, and eviction is fenced so a slot is never overwritten while a
launch reads it. Stolen picks are not bit-identical to the GB10's arithmetic (well within quantisation noise; see the
quality tables).

## Growable KV cache

The cache pools are reserved as CUDA virtual address ranges for the full 5,000,192 tokens and backed with physical
memory only as pages are used (`cuMemCreate` / `cuMemMap`, 2 MiB granules). Base addresses never move, so captured
CUDA graphs stay valid as the cache grows. 890 bytes per token (FP4), so 5M tokens is 4.1 GiB.

The cache grows into a 512 MiB free budget first. Past that, it borrows ColdSteal slots: the proxy stops admitting
experts to the highest slot of every layer, evicts it, signals the slot clean, and its memory is unmapped and handed to
the cache, 480 MiB per slot across 40 layers. With 4 × 1M-token requests, 7 of 8 slots were lent. When the queue
drains, the cache releases pages from the top, only as far as returning every slot needs, keeping the prompt cache
below, and the slots are re-mapped and re-admitted. A single request is limited to 1,048,576 tokens and refused above.

## Decoding

DSpark drafts up to 5 tokens per round with the model's own draft heads. The EV gate picks how many to verify (3 or
5) from the drafter's per-position confidences and a round-cost table measured on this pair: the width that maximises
expected tokens per unit of round time. From 4 concurrent sequences the verify window is capped at 2 drafts, because
each extra verify row costs more GB10 time when the GB10 is serving everyone. Up to 8 requests decode as one batch;
attention for the whole batch runs in one batched pass. Each transformer block replays as a captured CUDA graph.

## Prefill

Prompts are processed in 16,384-token chunks; an adaptive limiter shrinks a chunk when free memory is short. The prefix
cache (4 GiB) keeps KV and recurrent state for repeated prompt prefixes. Prefill is bounded by the cold experts on the
GB10: 1.1–1.5k tok/s.
