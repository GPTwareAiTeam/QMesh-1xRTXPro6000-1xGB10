#!/usr/bin/env python3
"""P7 hot + GX1 cold end-to-end driver over the native RDMA seam (REMOTE_TRANSPORT=seam).

Based on p7_smoke.py. Greedy decoding of fixed prompts; reports first-token and decode
ms/token, the generated text, and (with DSPARK=1) accepted draft tokens per verify round.
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")

import torch

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from qmesh.prefill_limit import PREFILL_ADAPTIVE, PrefillChunkLimiter
from qmesh import placement as seam_placement

CHECKPOINT = os.environ["CHECKPOINT"]
PLACEMENT = seam_placement.resolve(CHECKPOINT)   # the pack's table, else $PLACEMENT, else the repo default
MAX_NEW = int(os.environ.get("MAX_NEW", "64"))
PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three primary colors and explain why they matter in painting.",
]


def main():
    placement, _path, _sha = seam_placement.load(CHECKPOINT)
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
    cache = Cache(model, max_num_tokens=4096, max_history=5)
    t0 = time.time()
    model.load(device="cuda:0", progressbar=False)
    print(f"P7_LOAD_OK {time.time() - t0:.1f}s vram={torch.cuda.memory_allocated(0)/2**30:.2f}GiB",
          flush=True)

    tokenizer = Tokenizer.from_config(config)
    draft_model = None
    draft_cache = None
    dspark = os.environ.get("DSPARK") == "1"
    if dspark:
        draft_config = Config.from_directory(CHECKPOINT)
        draft_model = Model.from_config(draft_config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=4096, max_history=5)
        draft_model.load(device="cuda:0", progressbar=False)
        print("DSPARK_LOADED", flush=True)
    generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=2048,
                          draft_model=draft_model, draft_cache=draft_cache,
                          num_draft_tokens=5 if dspark else None)
    if PREFILL_ADAPTIVE:
        generator.prefill_chunk_limit = PrefillChunkLimiter(torch.device("cuda:0"))

    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
        job = Job(input_ids=ids, max_new_tokens=MAX_NEW, sampler=GreedySampler())
        generator.enqueue(job)
        t0 = time.time()
        marks = []
        text = ""
        eos_result = None
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                chunk = result.get("text", "")
                if chunk:
                    text += chunk
                    marks.append(time.time() - t0)
                if result.get("eos"):
                    eos_result = result
        total = time.time() - t0
        first = marks[0] if marks else float("nan")
        decode = (total - first) / max(len(marks) - 1, 1) if len(marks) > 1 else float("nan")
        line = (f"E2E prompt={prompt!r} total={total:.3f}s first_token={first:.3f}s "
                f"decode_ms_per_tok={decode * 1000:.1f} chunks={len(marks)}")
        if dspark and eos_result is not None:
            job_obj = eos_result["job"]
            acc = getattr(job_obj, "accepted_draft_tokens", 0)
            rej = getattr(job_obj, "rejected_draft_tokens", 0)
            line += f" accepted_draft_tokens={acc} rejected_draft_tokens={rej}"
        print(line, flush=True)
        print(f"TEXT {text!r}", flush=True)


if __name__ == "__main__":
    main()
