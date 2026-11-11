# Measured results, release R5

Every number in the README, with how it was measured and its raw file. Release R5 is the runtime in this repository with
the shipped settings in `qmesh/release/serving.env` and `qmesh/release/shipping.env`. Measured 2026-11-10/11.

## Test system

| | |
|---|---|
| Hot side | NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB, power-capped |
| Cold side | NVIDIA GB10 (ASUS Ascent GX10), 128 GB unified memory |
| Link | 100 GbE, RoCE v2, host-staged RDMA |
| Model | official DeepSeek-V4.1-Flash-1017, EXL3, routed experts RTXPro6000 900 @ 4bpw + 4760 @ 3bpw, GB10 7800 @ 3bpw + 1900 @ 2bpw (2.935 bpw average over 15,360 experts) |
| Serving settings | `BATCH=8`, ColdSteal 8 slots × 8 steals per layer, batched attention, draft cap 2 from 4 jobs, growable KV 5,000,192 tokens / 1,048,576 per request, FP4 KV, DSpark K ≤ 5 with the EV gate, `MAX_CHUNK=16384`, CUDA graphs |

Two identical pairs were used: speed and quality on one (RTX GPU1 + GB10 #2), needles on the other (RTX GPU0 + GB10 #1, the production pair). No measurement overlapped another GPU job or a compile.
Power-capped cards drift by a few percent between loads; compare rows within a table, not across tables.

## 1. Single user, by content

`tools/r5-bench/content.py`. HTTP, streaming, greedy, prefix cache off. Each 24k prompt is a corpus text of one content
type (structured, code, math, prose) followed by a task; plus 7 short prompts and 40 held-out prompts. Every request
generates exactly 256 tokens (`ignore_eos`). Two repetitions, the second in reverse order. Decode tok/s = completion
tokens over the time from the first to the last streamed token. 102 requests, 0 errors.
Raw: [`data/speed/content.jsonl`](data/speed/content.jsonl).

| Content | tok/s, run 1 / run 2 | Tokens per verify step | First token |
|---|---|---|---|
| Structured 24k | 119.0 / 128.1 | 3.82 / 3.88 | 18.1–18.5 s |
| Code 24k | 103.7 / 106.3 | 3.12 / 3.16 | 19.5–19.6 s |
| Math 24k | 97.3 / 95.7 | 3.08 / 3.20 | 18.8 s |
| Prose 24k | 78.9 / 79.8 | 2.49 / 2.39 | 19.5–19.7 s |
| 7 short prompts | 96.7 / 104.1 | 3.02 / 3.27 | 0.25 s |
| 40 held-out prompts | 107.4 / 111.0 | 3.18 / 3.21 | 0.44–0.46 s |
| All | 106.9 | | |

Content speed follows drafter acceptance (tokens per verify step); a verify step costs about the same on every type.

## 2. Concurrent users

`tools/r5-bench/concurrency.py`. N users send requests at the same moment. User i gets content type (i + repetition)
mod 4 from structured, math, code and prose, each prompt a distinct corpus text of the target size with a unique first
line, so with four repetitions every user count sees each type equally. Prompts are cold (prefix cache off): about
2,075 ("2k") or 24,603 ("24k") tokens, prefilled before decoding starts. Greedy; every request generates exactly 256
tokens (`ignore_eos`), so all users decode the same amount. Streamed tokens are timestamped individually.
Four repetitions per cell, alternating cell order, 0 errors. Dev pair, release runtime.
Raw: [`data/speed/concurrency.jsonl`](data/speed/concurrency.jsonl).

Three measures, each reported as the average over the four repetitions with the best repetition in brackets:

- **Steady-state**: tokens over the interval in which all N users are decoding at once (latest first token to earliest
  last token). What the server sustains at batch size N.
- **Total**: all generated tokens over the time from the first user's first token to the last user's last token, the
  conventional client-side aggregate (as vLLM's `benchmark_serving` reports it). Lower than steady state because users
  finish at different times and the last ones decode alone; at 24k the first tokens are also staggered by prefill.
- **Per user**: each user's own decode rate (tokens over its first-to-last-token time), averaged over users.

| Users | 2k: steady-state | 2k: total | 24k: steady-state | 24k: total | Per user, 2k / 24k | Last user's first token, 2k / 24k | Peak RTX memory |
|---|---|---|---|---|---|---|---|
| 1 | 101.9 (127.6) | 101.9 (127.6) | 98.8 (127.1) | 98.8 (127.1) | 101.9 / 98.8 | 2.0 s / 20.2 s | 93.4 / 94.1 GiB |
| 2 | 137.0 (158.8) | 127.7 (146.9) | 128.5 (149.1) | 115.0 (131.3) | 70.8 / 65.3 | 3.6 s / 39.2 s | 93.6 / 94.1 GiB |
| 4 | 162.4 (165.5) | 148.9 (154.1) | 154.2 (159.7) | 136.0 (143.8) | 41.6 / 39.3 | 7.7 s / 79.4 s | 93.6 / 94.3 GiB |
| 8 | 207.7 (214.9) | 195.1 (206.0) | 190.3 (200.6) | 148.6 (173.5) | 26.9 / 26.1 | 14.2 s / 162.6 s | 93.4 / 94.3 GiB |

At 1 user the rows are the mean of one run per content type: structured and math run at ~127 tok/s, code and prose
lower (table 1).

## 3. Prefill

`tools/r5-bench/prefill.py`. One cold request, distinct prompt, 16 output tokens. Prefill tok/s = prompt tokens over
the time to the first token. Raw: [`data/speed/prefill.jsonl`](data/speed/prefill.jsonl).

| Prompt tokens | First token | tok/s |
|---|---|---|
| 24,011 | 16.5 s | 1,452 |
| 130,995 | 92.7 s | 1,414 |
| 999,983 | 849.9 s | 1,177 |

## 4. Long context

**Needles** (`tools/r5-bench/needles.py`): three invented facts planted at 10 %, 50 % and 90 % depth of a haystack
mixing structured, math, code, prose and multilingual text, three questions after it, greedy. A fact counts when its unique answer string appears in
the reply. Raw (with replies): [`data/speed/needles.jsonl`](data/speed/needles.jsonl).

| Context | Facts | First token | Prefill tok/s | Decode after the prompt |
|---|---|---|---|---|
| 8,159 | 3/3 | 6.9 s | 1,179 | 107 tok/s |
| 32,836 | 3/3 | 29.2 s | 1,125 | 124 tok/s |
| 65,433 | 3/3 | 55.6 s | 1,177 | 107 tok/s |
| 130,930 | 3/3 | 116.0 s | 1,129 | 126 tok/s |

**Million-token requests.** One request of 1,010,525 tokens: answered. Four concurrent requests of 1,010,525 tokens each
(4.04M tokens in the cache): all four answered on topic in 3,403 s. As the cache grew to 3.44 GiB, ColdSteal lent 7 of
its 8 slots per layer to it; when the requests finished, all 7 came back and 15,180 pages were released, keeping
506 MiB of prompt cache. A request of 1,075,000 tokens was refused with HTTP 400. No errors.

## 5. Quality: perplexity

`tools/r5-bench/panel_ppl_4460.py`: teacher-forced, through the production cached prefill path (FP4 KV, 2048-token
chunks), the panels rendered with the checkpoint's own encoder following sudoingX's panel protocol, assistant positions
only, each conversation cut at 2,048 tokens. Raw: [`data/quality/`](data/quality/).

| Panel | Conversations | Scored positions | R5 ppl |
|---|---|---|---|
| 1017 panel (native 1017 replies) | 199 | 237,457 | 1.5861 ± 0.0035 |
| sudoingX panel | 199 | 254,846 | 1.6067 ± 0.0034 |
| WikiText-2 raw test, full | 140 windows | 143,220 | 2.4017 ± 0.0121 |

Published rows for the other builds (README) are their authors', scored on DGX Spark with the same panels and positions.

## 6. Quality: fidelity (KLD)

sudoingX's `p4_kld.py` from [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41),
unmodified: DeepSeek's own `inference/model.py` streamed layer by layer, run once with the official weights (top-128
log-probabilities saved at every scored position) and once with every EXL3 tensor R5 ships overlaid (all 15,360 routed
experts, attention, shared experts, LM head). KL(native ‖ R5) uses the native top-128 support plus one tail bucket.
Run on the RTX PRO 6000; the only addition is that script's Engram disk reader, built from its own source. Raw:
[`data/kld/`](data/kld/).

| | sudoingX panel | 1017 panel |
|---|---|---|
| Scored positions | 254,846 | 237,457 |
| KLD, mean | 0.0214 | 0.0228 |
| KLD, p99 | 0.282 | 0.318 |
| KLD, 8 domains weighted equally | 0.0333 | 0.0419 |
| Top-1 agreement | 95.68 % | 95.66 % |
| Perplexity, native → R5 (reference implementation) | 1.6018 → 1.6084 | 1.5728 → 1.5873 |

By domain, KLD (sudoingX panel / 1017 panel), against the published builds:

| Domain | R5 | sudoingX v2.0 | HeNryous 2.75 | OOA 2.7 | sudoingX v1.2 | LibertAIDAI 2.9 |
|---|---|---|---|---|---|---|
| code | 0.0141 / 0.0140 | 0.0122 / 0.0124 | 0.0125 / 0.0126 | 0.0140 / 0.0142 | 0.0150 / 0.0153 | 0.0211 / 0.0210 |
| agentic | 0.0241 / 0.0273 | 0.0175 / 0.0241 | 0.0183 / 0.0242 | 0.0208 / 0.0269 | 0.0221 / 0.0280 | 0.0329 / 0.0393 |
| math | 0.0172 / 0.0168 | 0.0159 / 0.0147 | 0.0152 / 0.0144 | 0.0188 / 0.0179 | 0.0202 / 0.0194 | 0.0252 / 0.0253 |
| long | 0.0266 / 0.0289 | 0.0256 / 0.0266 | 0.0258 / 0.0249 | 0.0285 / 0.0302 | 0.0304 / 0.0316 | 0.0346 / 0.0385 |
| knowledge | 0.0408 / 0.0499 | 0.0396 / 0.0495 | 0.0382 / 0.0502 | 0.0474 / 0.0592 | 0.0479 / 0.0628 | 0.0550 / 0.0706 |
| writing | 0.0482 / 0.0770 | 0.0480 / 0.0748 | 0.0514 / 0.0801 | 0.0561 / 0.0980 | 0.0614 / 0.1008 | 0.0655 / 0.1104 |
| multilingual | **0.0443 / 0.0536** | 0.0585 / 0.0693 | 0.0601 / 0.0734 | 0.0698 / 0.0790 | 0.0697 / 0.0814 | 0.0571 / 0.0637 |
| chat | 0.0511 / 0.0673 | 0.0475 / 0.0594 | 0.0482 / 0.0608 | 0.0568 / 0.0747 | 0.0575 / 0.0739 | 0.0716 / 0.0881 |

The IFEval-reply panel was not scored: its native replies are not published.

## 7. Tests

`pytest tests` on this tree in the release container: **6,425 passed, 0 failed**, 895 skipped (tests whose models are
not present on the test machine, and tests marked slow), 45 min.

## 8. How each optimisation was measured

Speed claims in [CREDITS.md](../CREDITS.md) and the README feature table come from back-to-back A/B runs on one pair
with the arms interleaved (A A B B or A B B A) and per-request paired confidence intervals; each claim names its
setting. The growable-cache comparison (release config against no ColdSteal at the same 5M capacity, and against a
fixed 64k cache), six mirrored server starts, 44 paired requests per cell:

| Users | ColdSteal + growable vs no ColdSteal | ColdSteal + growable vs ColdSteal + fixed 64k |
|---|---|---|
| 1 | −7.9 % ms/token (−10.2..−5.2) | +1.7 % (+0.8..+2.7) |
| 2 | −12.0 % (−13.7..−10.1) | +2.7 % (+1.5..+3.9) |
| 4 | −13.3 % (−14.6..−11.9) | +5.5 % (+2.8..+9.6) |
| 8 | −10.7 % (−12.3..−9.0) | +4.2 % (+2.8..+5.7) |
