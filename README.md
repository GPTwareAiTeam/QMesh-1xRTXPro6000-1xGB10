# DeepSeek-V4.1-Flash-1017 EXL3 on one RTX PRO 6000 + one GB10

A disaggregated inference runtime for DeepSeek-V4.1-Flash-1017 across two very different machines: one NVIDIA RTX PRO
6000 Blackwell Max-Q (96 GB) and one NVIDIA GB10 system (ASUS Ascent GX10 / DGX Spark class, 128 GB unified memory),
joined by a 100 GbE RoCE link. The RTX runs attention, the dense layers, the most-routed experts, the KV cache and the
DSpark drafter; the GB10 runs the remaining routed experts. Built on [ExLlamaV3](https://github.com/turboderp-org/exllamav3).

Release **R5**.

| | Peak | Average | Conditions |
|---|---|---|---|
| **8 users** | **214.9 tok/s** | 207.7 tok/s | each user with a 2k-token prompt in context; all 8 decoding together |
| 8 users, long context | 200.6 tok/s | 190.3 tok/s | each user with a 24.6k-token prompt in context |
| **1 user** | **128.1 tok/s** | 106.9 tok/s | peak: structured text, 24k-token prompt in context; average: all content types and prompt lengths |
| **KV cache** | **5M tokens** | | grows on demand; up to 1M tokens per request |
| **Quality** | **ppl 1.5861** | | 1017 panel; lower than every published quantised build, on both published panels |

- [Architecture](docs/ARCHITECTURE.md): the expert split, the RDMA seam, ColdSteal and the growable cache.
- [Running](docs/RUNNING.md): prerequisites, configuration, launch and the shipped settings.
- [Results](docs/RESULTS.md): every number below with its method, settings and raw data.
- [Credits](CREDITS.md): what this builds on, and from whom.

## Model

Official [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) weights,
quantised to EXL3 with the routed experts split by measured routing frequency:

**RTXPro6000 900 @ 4bpw + 4760 @ 3bpw, GB10 7800 @ 3bpw + 1900 @ 2bpw**

15,360 routed experts, 2.935 bpw average. Dense linears at 4–5 bpw, a 6-bit LM head, a 3-bit drafter. Every routed
expert and dense linear was encoded from the official checkpoint; the Engram tables are the official tensors.

## Performance

Measured on the release build, shipped settings, greedy, 256 generated tokens, prefix cache off, no warm cache.
Decode throughput excludes prefill. Details and raw files: [Results](docs/RESULTS.md).

**Single user, by content** (each 24k row has a 24k-token prompt in context; every request generates exactly 256 tokens; two runs each):

| Content | tok/s | Tokens per verify step |
|---|---|---|
| Structured (JSON, tables), 24k | 119.0 / 128.1 | 3.85 |
| Code, 24k | 103.7 / 106.3 | 3.14 |
| Math, 24k | 97.3 / 95.7 | 3.14 |
| Prose, 24k | 78.9 / 79.8 | 2.44 |
| 7 short prompts | 96.7 / 104.1 | 3.15 |
| 40 held-out prompts | 107.4 / 111.0 | 3.20 |
| All (sum of tokens / sum of decode time) | **106.9** | |

**Concurrent users.** N users send requests at the same moment. Each request carries a cold prompt of about 2,075
("2k") or 24,603 ("24k") tokens, which is prefilled into the cache first, so decoding runs with that much context
already present per user; then it generates exactly 256 tokens (`ignore_eos`, as in vLLM's benchmarks). Users get
structured, math, code and prose prompts in rotation, so every user count sees the same content mix. Four
repetitions per cell, 0 errors. Average (peak run in brackets):

| Users | 2k: steady-state tok/s | 2k: total tok/s | 24k: steady-state tok/s | 24k: total tok/s | Per user tok/s, 2k / 24k | Last user's first token, 2k / 24k |
|---|---|---|---|---|---|---|
| 1 | 101.9 (127.6) | 101.9 (127.6) | 98.8 (127.1) | 98.8 (127.1) | 101.9 / 98.8 | 2.0 s / 20.2 s |
| 2 | 137.0 (158.8) | 127.7 (146.9) | 128.5 (149.1) | 115.0 (131.3) | 70.8 / 65.3 | 3.6 s / 39.2 s |
| 4 | 162.4 (165.5) | 148.9 (154.1) | 154.2 (159.7) | 136.0 (143.8) | 41.6 / 39.3 | 7.7 s / 79.4 s |
| 8 | **207.7 (214.9)** | 195.1 (206.0) | **190.3 (200.6)** | 148.6 (173.5) | 26.9 / 26.1 | 14.2 s / 162.6 s |

- **Steady-state** counts only the interval in which all N users are decoding at once: what the server sustains at
  batch size N.
- **Total** is the conventional aggregate (as vLLM's `benchmark_serving` reports it): all generated tokens over the time
  from the first user's first token to the last user's last token. It is lower because users finish at different
  times and the last ones decode alone; at 24k the first tokens are also staggered by prefill.
- **Per user** is each user's own decode rate, averaged.

**Per user, by content type, under load** (same run as the table above; each user's own decode rate, averaged
over the users of that type; 2k / 24k-token prompt in context). The 1-user cells are a single request each.

| Content | 1 user | 2 users | 4 users | 8 users |
|---|---|---|---|---|
| Structured | 127.6 / 127.1 | 76.7 / 79.8 | 44.1 / 42.9 | 28.8 / 28.1 |
| Math | 94.2 / 100.6 | 80.0 / 67.6 | 41.7 / 38.9 | 27.1 / 26.7 |
| Code | 101.8 / 101.8 | 68.5 / 63.4 | 42.1 / 40.1 | 26.1 / 26.4 |
| Prose | 83.9 / 65.8 | 57.8 / 50.4 | 38.6 / 35.2 | 25.4 / 23.3 |

The spread between types shrinks as users are added: at 8 users each batch step serves every user at once, so a
request's speed depends less on how well the drafter guesses its own text.

**Prefill and context** (cold, single request):

| Prompt | First token | Prefill tok/s |
|---|---|---|
| 24,011 tokens | 16.5 s | 1,452 |
| 130,995 tokens | 92.7 s | 1,414 |
| 999,983 tokens | 849.9 s | 1,177 |

| Context test | Result |
|---|---|
| Needles, 3 facts at 10/50/90 % depth | **3/3 at 8k, 32k, 64k and 128k** |
| 1 user, 1.01M-token prompt | answered |
| 4 users × 1.01M tokens at once (4.04M in the cache) | all four on topic; ColdSteal gave 7 of 8 slots per layer to the cache, then took them all back |
| Request over 1M tokens | refused with HTTP 400 |

## Quality

Both published DeepSeek-V4.1-Flash-1017 panels (199 conversations each; 237,457 and 254,846 scored assistant
positions, the same positions the published packs were scored on).

**Perplexity**, through this runtime's production prefill path:

| Build | Expert bpw | 1017 panel (native 1017 replies) | sudoingX panel |
|---|---|---|---|
| DeepSeek-V4.1-Flash-1017, original weights | — | 1.5720 | 1.6005 |
| **This build (R5)** | **2.935** | **1.5861** | **1.6067** |
| sudoingX v2.0 | 2.80 | 1.5869 | 1.6117 |
| HeNryous | 2.75 | 1.5871 | 1.6119 |
| object-oriented-architects | 2.7 | 1.5931 | 1.6146 |
| sudoingX v1.2 | 2.70 | 1.5946 | 1.6159 |
| LibertAIDAI | 2.9 | 1.6040 | 1.6155 |

**Fidelity**, KL(native ‖ quantised) at every scored position, with every pack scored by DeepSeek's own
`inference/model.py` with its tensors overlaid (sudoingX's harness and method, run here on R5; other rows as published
by sudoingX):

| Build | Expert bpw | sudoingX panel: KLD / p99 / top-1 | 1017 panel: KLD / p99 / top-1 | 8 domains equally weighted |
|---|---|---|---|---|
| **This build (R5)** | **2.935** | 0.0214 / 0.282 / 95.68 % | 0.0228 / 0.318 / 95.66 % | 0.0333 / 0.0419 |
| sudoingX v2.0 | 2.80 | 0.0198 / 0.276 / 95.94 % | 0.0213 / 0.317 / 95.97 % | 0.0331 / 0.0413 |
| HeNryous | 2.75 | 0.0201 / 0.285 / 95.96 % | 0.0216 / 0.330 / 95.97 % | 0.0337 / 0.0426 |
| object-oriented-architects | 2.7 | 0.0232 / 0.325 / 95.65 % | 0.0249 / 0.381 / 95.65 % | 0.0390 / 0.0500 |
| sudoingX v1.2 | 2.70 | 0.0243 / 0.339 / 95.47 % | 0.0262 / 0.386 / 95.49 % | 0.0405 / 0.0517 |
| LibertAIDAI | 2.9 | 0.0304 / 0.383 / 94.76 % | 0.0322 / 0.433 / 94.70 % | 0.0454 / 0.0571 |

R5 has the lowest perplexity of the published builds on both panels. On KLD it is third over all positions, behind
sudoingX v2.0 and HeNryous, level with v2.0 with the 8 domains weighted equally, and the closest of all on
multilingual (0.0443 vs 0.0585 for v2.0). Its gap is on code, agentic and chat; code is 65 % of the scored positions.
The perplexity rows of the other builds were scored by their authors on DGX Spark, ours through this runtime.
Speculative decoding does not affect any quality number: the target model decides every token.

## Features

| Feature | Effect |
|---|---|
| Expert split across RTX and GB10 by measured routing | 13.35 vs 14.15 ms/token against a balanced placement |
| RDMA expert seam (RoCE, multi-session, bf16 replies) | one GB10 worker serves many client sessions |
| ColdSteal: the RTX computes cold experts it has cached while it waits for the GB10, up to 8 per layer | −4.8 % ms/token single user; +2.2 to +6.6 % tok/s at 1–8 users |
| Growable KV cache with slot lending (CUDA VMM) | 5M-token pool, 1M per request; ColdSteal slots lent to the cache under load and returned when idle |
| Batched attention at 2+ users | −6 to −7 % ms/token at 2–8 users |
| DSpark drafting with an adaptive K (EV gate, K ∈ {3,5}) | ~2.8 tokens per round; −4.4 % ms/token vs always 5 |
| Draft width cap from 4 concurrent users | −12.5 % ms/token at 8 users |
| Piecewise CUDA graphs, HC fold, dense GEMV, cooperative MoE kernel | 1,894 → 682 launches per round; −7.0 %, −4.4 % ms/token |
| GB10 cold experts in native C instead of Python | 113.98 → 77.19 ms/token when it landed (single user, an earlier build) |
| GB10 fused stage: each bit-width group's gate/up and down in one launch | +4.9 % decode, perplexity unchanged |
| GB10 one launch set per layer across all bit-width groups | −6.2 % ms per round, −2.4 % ms/token; host launch overhead now ~5 µs of a 0.3–2.5 ms call |
| GB10 fixed-order expert combine | deterministic sums; −3.6 % ms/token |
| Prefix cache, FP4 KV, 16K prefill chunks with an adaptive limiter | repeated 10.8K prompt: 9,848 → 289 ms to first token |

What each of these builds on is in [Built on, with thanks](#built-on-with-thanks) below.

## Built on, with thanks

| Who | What this project uses | What we did with it |
|---|---|---|
| **[turboderp — ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (MIT) | The engine underneath: the EXL3 format, trellis quantiser and decode kernels, the cooperative MoE kernel, the fused HC-mix kernel, DeepSeek-V4 attention, cache and the DSpark component, graph capture, the batched generator, loaders and converter. | Forked at v1.6.5; everything else in this README is built on it. Upstream's copyright and licence are kept. |
| **[DeepSeek — DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (MIT) | The model and its official weights, the DSpark drafter and its confidence head, the Engram tables, the reference `inference/model.py`. | Quantised to the mixed EXL3 build above; the confidence head drives our EV gate; the reference model scores our KLD. |
| **[Zeuss5 — cuda-exl3](https://github.com/Zeuss5/cuda-exl3)** (MIT) | The EXL3 MoE GEMM the GB10 runs (vendored in `vendor/cuda-exl3/`, licence and notice kept). | Added fractional-bitrate tiles, an fp32 caller accumulator and a partial-map fix; our C cold-expert path, fused stage, one-launch layer plan and fixed-order combine run on it. |
| **[sudoingX — dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41)** (Apache-2.0) | The two published panels and panel protocol, the KLD harness `p4_kld.py` and its Engram reader, the published comparison rows. | Reproduced the protocol to score R5 on the same positions; ran `p4_kld.py` unmodified on R5. No code redistributed. |
| **HeNryous, object-oriented-architects, LibertAIDAI** | Their published scores in the quality tables. | Quoted with attribution; their weights were not used. |
| **[KTransformers](https://github.com/kvcache-ai/ktransformers)**, **[Niko1221 — Strata](https://github.com/Niko1221/Strata)** (MIT), **Si et al. / WiSP** | Ideas: compute an expert where its weights live; an adaptive GPU expert cache in front of a slower tier; LRU over frequency for expert residency. | Combined into ColdSteal across an RTX + GB10 split, with steals folded into the cooperative launch. |
| **[diffbot](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)** (MIT) | Ideas: a confidence gate on DSpark's verify width; an 8-bit peer-push prefill all-reduce. | Our EV gate prices each width by this pair's measured round cost and is on by default; `prefill_ar.cu` is our own int8/fp32 implementation of the all-reduce (used by the two-RTX build). No diffbot code or weights used. |
| **philbert440 — 1Cat-vLLM PR #1243** | Idea: scale the drafter's input taps by a power of two before the fp16 cast. | Kept as `EXL3_DSPARK_TAP_SCALE` (default 1.0, measured neutral). |
| **[FLA](https://github.com/fla-org/flash-linear-attention)**, **rocm_wmma_gemm** (Adel Johar), **CarouselAether's ROCm fork**, **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (all MIT) | Inherited with upstream: linear-attention kernels, the ROCm GEMM and RDNA WMMA code, the DRY sampler semantics and the WikiText-2 perplexity protocol. | Kept with their licences; the WikiText-2 protocol is our quality harness's reference. |

Full detail, including reference implementations and tools: [CREDITS.md](CREDITS.md).

## Hardware and software

- One NVIDIA RTX PRO 6000 Blackwell (96 GB, `sm_120`) and one NVIDIA GB10 system (128 GB unified memory, `sm_121`),
  joined by 100 GbE with RoCE. The release was measured on the Max-Q (power-capped) RTX PRO 6000.
- Python ≥ 3.10, CUDA-enabled PyTorch on both hosts, the CUDA toolkit, Ninja and a C++/CUDA compiler for the native
  extensions, Docker with NVIDIA GPU access, `rdma-core`.

## Running it

See [Running](docs/RUNNING.md). In short: build the hot (RTX) and cold (GB10) packs from the official weights, describe
the pair in a `pairs.conf`, and start it with `qmesh/release/pair_server.sh <pair> start`. The shipped settings in
`qmesh/release/serving.env` load automatically. The API is OpenAI-compatible on loopback port 8000.

## Known limitations

- Cold prefill runs at 1.1–1.5k tok/s because cold experts cross the link to the GB10: a 1M-token prompt takes about
  14 minutes to its first token, and 8 users at 24k wait up to 166 s.
- The growable 5M cache decodes 1.7–5.5 % slower than ColdSteal with a small fixed 64k cache; it is still 8–13 % faster
  than running without ColdSteal at the same 5M capacity. The cause of the residual is open.
- Multi-user batching (`BATCH=8`) and ColdSteal change numerics slightly: a request's greedy output can differ on
  near-tied tokens depending on what else is running. Quality is unaffected (see Quality).
- Drafting is slower on prose, where the drafter's guesses are accepted less often (2.4 tokens per verify step, against 3.8 on
  structured text).

## License

MIT, as ExLlamaV3. Model weights are distributed by DeepSeek under their own license.
