# Credits

This project is a disaggregated runtime for DeepSeek-V4.1-Flash-1017 across one RTX PRO 6000 and one GB10, built on
turboderp's ExLlamaV3. This file says what comes from where: the upstream engine, the model, other people's code
vendored here, and other people's ideas, methods and data we used or built on. [README.md](README.md) describes the
runtime; [docs/RESULTS.md](docs/RESULTS.md) has the measurements.

## Upstream engine

**turboderp — [ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (MIT, see [LICENSE](LICENSE)).
The engine this runtime builds on: the EXL3 quantisation format, the trellis quantiser and decode kernels, the
cooperative MoE kernel, the fused hyper-connection (HC-mix) kernel, the DeepSeek-V4 attention, DSA cache and DSpark
drafter component, CUDA graph capture, the paged cache, recurrent checkpoints, chunked prefill, the coop autotuner,
the batched generator, sampler, loaders and converter. This tree is a fork of upstream at `cdc785db` (v1.6.5) and
keeps upstream's copyright notice as MIT requires; turboderp and the upstream contributors (including Async0x42,
oobabooga, Fabiano and others in upstream's history) wrote that code. This project's work starts after that point.

Inherited with upstream and kept under their own licences:
- **[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** (FLA) — Songlin Yang, Yu Zhang,
  Zhiyuan Li (MIT): a forward-only subset of v0.5.2, vendored in `exllamav3/vendor/fla/` with its licence and per-file
  headers.
- **rocm_wmma_gemm** — Adel Johar (MIT): vendored at `ea3aa74` in `exllamav3/exllamav3_ext/rocm/vendor/rocm_wmma_gemm/`
  with its licence; the ROCm build path only.
- **CarouselAether's ROCm fork** of ExLlamaV3: the RDNA WMMA fragment code in `exllamav3_ext/rocm/rdna_wmma.cuh` and
  an optimisation to the fused HC-mix kernel, which our HC fold extends.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT): the DRY sampler semantics (`exllamav3_ext/generator/dry.cu`,
  the DRY scheme designed by p-e-w), the `mmvq` K-split idea in the RDNA GEMV, and the WikiText-2 perplexity
  protocol (`perplexity.c`) used by `eval/ppl.py` and our quality harness.
- Reference implementations followed by upstream's architecture code: Hugging Face
  [transformers](https://github.com/huggingface/transformers) (DeepSeek-V4, GLM, Qwen2-VL image processing — the
  latter adapted), [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm),
  [guidance-ai/llguidance](https://github.com/guidance-ai/llguidance) and
  [formatron](https://github.com/Dan-wanna-M/formatron) for constrained generation, and the model vendors' own
  modelling code.

## Model

- **[DeepSeek — DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (MIT): the
  model and its official weights, the base of every quantised tensor in this build; the DSpark multi-token-prediction
  drafter design and its per-position confidence head, which our EV gate reads; the Engram tables, used verbatim; and
  the reference `inference/model.py`, used to score KLD.

## Other people's code used here

- **[Zeuss5 — cuda-exl3](https://github.com/Zeuss5/cuda-exl3)** (MIT), vendored at `6a1ffc3` in `vendor/cuda-exl3/`
  with its `LICENSE` and `NOTICE` unchanged. Its EXL3 MoE GEMM is the kernel our GB10 cold-expert path runs on
  (`SEAM_MOE_KERNEL=cudaexl3`). We added three changes, documented in `vendor/cuda-exl3/VENDORED`: fractional-bitrate
  (half k-group) tiles, which this checkpoint's 2- and 3-bit experts need; a caller-supplied fp32 accumulator, so one
  layer's bit-width groups sum into one output; and a fix for partial routing maps in `exl3_moe_combine`. Our C
  cold-expert path, fused per-group stage, one-launch-per-layer plan and fixed-order combine are built on top of it.
  cuda-exl3 itself credits ExLlamaV3, vLLM and the projects listed in its `NOTICE`.
- **[sudoingX — dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41)** (Apache-2.0):
  - **The two published panels** (199 conversations each; the sudoingX panel of native replies and the 1017 panel)
    and the **panel protocol**: rendering with the checkpoint's own encoder, assistant-only scoring, 2,048-token cut.
    Our perplexity harness (`tools/r5-bench/panel_ppl_4460.py`, `panel_encode_4460.py`) reproduces it, so our numbers
    are on the same 237,457 and 254,846 positions as the published packs.
  - **The KLD harness**, `scripts/quantize/p4_kld.py` with its `dsv41q` package, run unmodified for our fidelity
    table, including its Engram disk reader (`patches/engram_disk/engram_disk.cpp`, from its vLLM patch 0002), built
    from its own source. No sudoingX code is redistributed here.
  - **The comparison rows**: the published perplexity and KLD figures for sudoingX v2.0 and v1.2 and for the packs
    below, quoted with attribution.
- **Published DeepSeek-V4.1-Flash-1017 EXL3 packs** whose published scores we compare against:
  [HeNryous](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark),
  [object-oriented-architects](https://huggingface.co/object-oriented-architects/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-TensorFold),
  [LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw). Their weights were not used.

## Other people's ideas we built on

- **ColdSteal** — the RTX computing cold experts it has cached while it waits for the GB10 — combines three published
  ideas, applied here across an RTX + GB10 split with the steals folded into the cooperative launch:
  - [KTransformers](https://github.com/kvcache-ai/ktransformers) (kvcache-ai): compute an expert on the device where
    its weights already live, and overlap that work with the other device's.
  - [Strata](https://github.com/Niko1221/Strata) (Niko1221, MIT): an adaptive GPU-resident expert cache in front of a
    slower expert tier, with the split decided by measured cost.
  - Si et al. / WiSP: at hundreds of experts per layer, least-recently-used residency beats frequency-based residency,
    and prediction or prefetch adds little; ColdSteal's slots are LRU with no predictor.
- **Hot/cold expert placement.** Putting the most-routed experts on the fast device and the rest on a slower tier is
  a common offloading idea (KTransformers, Strata and others). Our ranking was measured here on this model's routing
  (1.63M routed positions across content types, context lengths, prefill and decode); Strata's profile was used for the
  sibling project's ranking, not this one.
- **[diffbot](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)** (MIT): a
  confidence-based gate for DSpark's verify width (off by default there), which preceded our EV gate; and the idea of an
  8-bit peer-push prefill all-reduce (their `fp8ar`), which `exllamav3_ext/parallel/prefill_ar.cu` implements with int8
  and exact fp32 modes (used by the two-RTX build, not by this pair). No diffbot code or weights are used here.
- **philbert440 — 1Cat-vLLM PR #1243**: the idea of scaling the drafter's input taps by a power of two before the fp16
  cast, kept here as `EXL3_DSPARK_TAP_SCALE` (default 1.0, measured neutral).
- **A "decode candidates" kit produced for this project**: the prefetch-ring hoist and programmatic dependent launch
  inside the cooperative MoE kernel, both bitwise identical to the previous arithmetic.
- **Marlin** (IST-DASLab): the BK=64 GEMM tile shape referenced in cuda-exl3's GEMM.

## From the sibling project

[DSV41-Flash-EXL3-2xRTXPRO6000](https://github.com/GPTwareAiTeam/DSV41-Flash-EXL3-2xRTXPRO6000) (same authors, MIT):
the piecewise CUDA graphs and DSpark tap integration, the FP4 KV cache, the prefix cache, pinned tuner caches and the
warm-up prefill were first written there and ported here; `serve_pair.py` keeps its server's scheduling structure and
wire protocol.

## Reference data only (no code taken)

DeepSeek-V4.1-Flash-1017 deployments on comparable hardware published by sudoingX, diffbot, HeNryous,
object-oriented-architects, LibertAIDAI, MiaAI-Lab, sybil-solutions, sfxnz, SSHdotCodes, acrogenesis and llamaperf,
used as reference points.

## Tools

NVIDIA **CUDA** (including the virtual memory management API, `cuMemCreate` / `cuMemMap`, that the growable cache is
built on), **rdma-core** (ibverbs, for the RoCE seam), **PyTorch** (runtime and CUDA graphs), **Triton**,
**tilelang** (the reference model's kernels, for KLD), **safetensors**, **tokenizers**, **Ninja** and **Docker**.

## Licence

MIT, inherited from ExLlamaV3; the text in [LICENSE](LICENSE) is unchanged. Vendored components keep their own
licences (`exllamav3/vendor/fla/LICENSE`, `exllamav3/exllamav3_ext/rocm/vendor/rocm_wmma_gemm/LICENSE`,
`vendor/cuda-exl3/LICENSE` and `NOTICE`). Model weights are not part of this repository; DeepSeek publishes them under
MIT.
