"""Load the final P7 hot-only configuration with the null remote (same recipe as qmesh/p7_smoke.py)."""
import json, os, sys, time
os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
sys.path.insert(0, "/src/exllamav3")
import torch
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from qmesh.prof.null_remote import install
from qmesh import placement as seam_placement

CHECKPOINT = os.environ["CHECKPOINT"]
PLACEMENT = seam_placement.resolve(CHECKPOINT)   # the pack's table, else $PLACEMENT, else the repo default


def load(max_history=5):
    install()
    placement, _path, _sha = seam_placement.load(CHECKPOINT)
    hot = seam_placement.hot(placement)
    config = Config.from_directory(CHECKPOINT)
    config.infer_params.moe_cpu_split = 1
    config.infer_params.moe_remote_hot = hot
    config.infer_params.moe_remote_warm = seam_placement.warm(placement)
    config.infer_params.remote_expert_host = {"host": "null", "port": 0}
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_history=max_history)
    t0 = time.time()
    model.load(device="cuda:0", progressbar=False)
    print(f"LOAD {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated(0)/2**30:.2f}GiB", flush=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=2048)
    return config, model, cache, tokenizer, generator


def new_job(tokenizer, n, prompt="Hi"):
    ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
    return Job(input_ids=ids, max_new_tokens=n, min_new_tokens=n, sampler=GreedySampler())
