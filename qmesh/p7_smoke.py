#!/usr/bin/env python3
"""P7 hot-expert + GX1 cold-expert smoke test."""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")

import torch

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from qmesh import placement as seam_placement

CHECKPOINT = os.environ["CHECKPOINT"]
PLACEMENT = seam_placement.resolve(CHECKPOINT)   # the pack's table, else $PLACEMENT, else the repo default
REMOTE_HOST = os.environ["REMOTE_HOST"]
REMOTE_PORT = int(os.environ.get("REMOTE_PORT", "29517"))


def main():
    placement, _path, _sha = seam_placement.load(CHECKPOINT)
    hot = seam_placement.hot(placement)
    config = Config.from_directory(CHECKPOINT)
    config.infer_params.moe_cpu_split = 1
    config.infer_params.moe_remote_hot = hot
    config.infer_params.moe_remote_warm = seam_placement.warm(placement)
    config.infer_params.remote_expert_host = {"host": REMOTE_HOST, "port": REMOTE_PORT}
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_history=5)
    t0 = time.time()
    model.load(device="cuda:0", progressbar=False)
    print(f"P7_LOAD_OK {time.time() - t0:.1f}s vram={torch.cuda.memory_allocated(0)/2**30:.2f}GiB", flush=True)

    tokenizer = Tokenizer.from_config(config)
    draft_model = None
    draft_cache = None
    if os.environ.get("DSPARK") == "1":
        draft_config = Config.from_directory(CHECKPOINT)
        draft_model = Model.from_config(draft_config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=4096, max_history=5)
        draft_model.load(device="cuda:0", progressbar=False)
        print("DSPARK_LOADED", flush=True)
    generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=2048,
                          draft_model=draft_model, draft_cache=draft_cache,
                          num_draft_tokens=5 if draft_model is not None else None)
    ids = tokenizer.encode("Hi", add_bos=True, encode_special_tokens=True)
    job = Job(input_ids=ids, max_new_tokens=5, min_new_tokens=5, sampler=GreedySampler())
    generator.enqueue(job)
    t0 = time.time()
    marks = []
    text = ""
    while generator.num_remaining_jobs():
        results = generator.iterate()
        for result in results:
            chunk = result.get("text", "")
            if chunk:
                text += chunk
                marks.append(time.time() - t0)
    total = time.time() - t0
    first = marks[0] if marks else float("nan")
    decode = (total - first) / max(len(marks) - 1, 1) if len(marks) > 1 else float("nan")
    print(f"SMOKE_OK total={total:.3f}s first_token={first:.3f}s "
          f"decode_per_tok={decode:.3f}s chunks={len(marks)} text={text!r}", flush=True)


if __name__ == "__main__":
    main()
