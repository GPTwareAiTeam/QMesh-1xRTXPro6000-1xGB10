"""
DSV4.1 (q_head_norm=False) multi-job decode: the batched attention body (EXL3_DSV41_BATCH) against the per-job loop on
the tiny random model. Two jobs prefilled to different lengths, then decode steps of verify width S through both paths
on identical states; logits must match to the chunk-shape noise floor at every step (the batched rows only change
fp16 reduction order). Covers the ring shift past the SWA window and the CSA/HCA pools.
"""

import pytest
import torch

from exllamav3.cache.cache import Cache
from exllamav3.modules import dsv4

from testlib.dsv4 import assert_logits_close, attention_layers, build_tiny_dsv4, fwd_modules, noise_floor
from testlib.tiny_models import DSV4_TINY


@pytest.fixture(scope = "module")
def model41(tmp_path_factory, device):
    config, model = build_tiny_dsv4(tmp_path_factory.mktemp("dsv41_batch"), seed = 17)
    cache = Cache(model, max_num_tokens = 8192, max_batch_size = 4)
    model.load(str(device))
    for a in attention_layers(model):
        a.q_head_norm = False       # DSV4.1 attention (architecture/deepseek_v41.py)
        a.x_fan_ready = False
        a.x_fan = None
    yield model, cache
    model.unload()


def _step(model, ids, states, batched):
    """One decode step of ids (B, S) for the states (one per row), through the batched body or the per-job loop"""
    dsv4.dsv41_batch = batched
    params = {"attn_mode": "flash_attn", "recurrent_states": states}
    return fwd_modules_b(model, ids, params)


def fwd_modules_b(model, ids, params):
    params["input_ids"] = ids
    x = ids
    with torch.inference_mode():
        for m in model.modules:
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
    return x.float().cpu()


def _advance(states, n):
    for s in states:
        s.position += n
        s.post_advance()


def _prefill(model, cache, prompts):
    states = []
    with torch.inference_mode():
        for p in prompts:
            s = cache.get_new_state()
            fwd_modules_b(model, p, {"attn_mode": "flash_attn", "recurrent_states": [s]})
            _advance([s], p.shape[1])
            states.append(s)
    return states


@pytest.mark.parametrize("S", [1, 4, 6])
def test_batched_matches_per_job(model41, S):
    model, cache = model41
    torch.manual_seed(31 + S)
    V = DSV4_TINY["vocab_size"]
    lens = (37, 141)
    prompts = [torch.randint(0, V, (1, n), dtype = torch.long) for n in lens]
    steps = [torch.randint(0, V, (2, S), dtype = torch.long) for _ in range(12)]
    # Noise floor of this model on a single-sequence nc forward (fp16 tiling perturbation)
    ref_ids = torch.randint(0, V, (1, 160), dtype = torch.long)
    ref = fwd_modules(model, ref_ids, {"attn_mode": "flash_attn_nc"})
    floor_kl, floor_am = noise_floor(model, ref_ids, ref)
    kl_tol = max(5e-4, 1.5 * floor_kl)
    arg_tol = min(0.99, floor_am - 0.05)

    a_states = _prefill(model, cache, prompts)
    b_states = _prefill(model, cache, prompts)
    try:
        for k, ids in enumerate(steps):
            got = _step(model, ids, a_states, True)
            want = _step(model, ids, b_states, False)
            assert got.shape == want.shape == (2, S, V)
            assert torch.isfinite(got).all()
            assert_logits_close(got.view(-1, V), want.view(-1, V), kl_tol, arg_tol, f"S={S} step {k}")
            _advance(a_states, S)
            _advance(b_states, S)
    finally:
        dsv4.dsv41_batch = False
        for s in a_states + b_states:
            s.free()
