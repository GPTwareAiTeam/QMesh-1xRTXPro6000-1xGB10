# R5 benchmark drivers

These are the R5 measurement drivers, not benchmark results. They require the same R5 source, packs, hardware, server configuration, tokenizer and inputs as the measured run for a comparable result. Preserve the raw outputs and record their input hashes, source/pack identifiers, hardware and environment. Do not substitute datasets and call the resulting numbers the published measurements. No external datasets or model weights are redistributed here; obtain them under their applicable terms. The exact original held-out prompts, corpus and panel files are required to reproduce those workloads; availability and redistribution rights have not been established here.

## HTTP workloads

Run against an already configured R5 OpenAI-compatible server; use an empty output directory and the identical server settings used for the measurement. The model name in the requests is `deepseek-v4.1-flash-pair`. Paths are local to the client:

```sh
mkdir -p results
python tools/r5-bench/content.py http://SERVER:PORT results/content.jsonl \
  --corpus-dir /path/to/corpus --held-prompts /path/to/prompts_held.json
R5_CORPUS=/path/to/corpus NV_GPU=0 python tools/r5-bench/concurrency.py http://SERVER:PORT results/concurrency.jsonl 4
R5_CORPUS=/path/to/corpus python tools/r5-bench/needles.py http://SERVER:PORT results/needles.jsonl
python tools/r5-bench/prefill.py http://SERVER:PORT results/prefill.jsonl
python tools/r5-bench/summarize.py results
```

Content requires `structured_24k.txt`, `math_24k.txt`, `code_24k.txt`, and `prose_24k.txt` in the corpus directory, and a JSON array of at least 40 held-out prompt strings. Each corpus text is preceded by the first short prompt. Seven built-in short prompts and the first 40 held prompts are also measured. Two repetitions reverse both set and within-set order. Requests stream greedily at temperature 0 and generate exactly 256 tokens (`ignore_eos`), with no retries. Speculative verify rounds are inferred as completion tokens minus accepted draft tokens, not directly measured.

Concurrency gives user i of a wave the corpus text of content type (i + repetition) mod 4 (structured, math, code, prose) at the 2k or 24k size, with a unique first line, for 1/2/4/8 users (`USERS=1,2` selects cells; the last argument is the repetition count), exactly 256 tokens per request (`ignore_eos`, which the server accepts as a request field), and reports steady-state throughput (all users decoding), total throughput and per-user decode rates. Needles plant three facts at 10/50/90 % depth of a haystack built from the five `_128k` corpus files and ask for them at 8k, 32k, 64k and 128k tokens. Prefill uses the same filler at nominal 24k, 131k and 1M targets and max 16 tokens. Actual prompt counts come from server usage, not the nominal target. TTFT is elapsed time to first nonempty content; decode is first to last nonempty content. Aggregate decode throughput sums completion tokens minus one over the interval from the earliest first token to the latest last token, not the interval after all users reach decode. Mean user throughput is the arithmetic mean of individual decode rates. Concurrency samples `nvidia-smi` memory.used every second on the client machine; `--gpu` selects the P7 device (default index 1). Missing samples are null. Ensure the client is on the relevant GPU host. Records include errors; summarization does not erase them.

## Teacher-forced quality

Quality runs directly on the hot GPU with a compatible cold expert seam server running, not via HTTP. Install this repository and its runtime dependencies, plus `tokenizers`; use the original model directory containing `tokenizer.json` and its `encoding` module. Supply the exact hot pack and its placement, original model tokenizer, WikiText-2 raw zip and panel JSONL externally. The panel encoder follows the `sudoingX/dgx-spark-deepseek-v41` panel protocol. The WikiText archive must contain `wikitext-2-raw/wiki.test.raw` (the original driver used the `ggml-org/ci` WikiText-2 raw archive). Panel reference JSON files are optional and do not replace the panel data.

```sh
export CHECKPOINT=/path/to/hot-pack MODEL_DIR=/path/to/original-model
export SEAM_PEER=SERVER SEAM_PORT=47311 PLACEMENT=/path/to/placement.json
export WIKITEXT_ZIP=/path/to/wikitext-2-raw-v1.zip
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/r5-bench/panel_ppl_4460.py --dry-run --mode both --panel-file /path/to/panel.jsonl
python tools/r5-bench/panel_ppl_4460.py --mode panel --panel-file /path/to/panel.jsonl --sanity --out results/panel.json
python tools/r5-bench/panel_ppl_4460.py --mode wiki2 --sanity --out results/wiki2.json
```

Set the seam transport, device, tuning and other server/client environment to match the measured deployment; `KV_BITS` defaults to 4, `CACHE_TOKENS` to 4096. The R5 run used a 5,000,192-token cache and additional R5 serving settings; reproducing it requires those settings and matching placement on both ends. Run separate panel files separately if matching that procedure. `--reference` may be repeated in panel-file order. Default context and panel truncation are 2048. WikiText scores the second half of each non-overlapping chunk, predicting the next token, with BOS replacing the first token where available; `--chunks 0` scores all chunks. Panel scoring filters to split `panel`, renders with the model encoder, truncates and scores assistant-only next-token positions. Both use the cached production prefill path. Prefixes of eight chunks/conversations are reported as well as full results; `--chunks` and `--convs` restrict the run and are not full-panel measurements.
