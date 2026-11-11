"""CPU MoE worker serving an MXFP4 layer (moe_cpu_host register_layer(fp4=...), cpu/moe_handoff.cu fp4 dispatch):
through the real worker process, both handoffs equal exl3_moe_fp4_forward in-process bit for bit, with skipped picks:
- submit_issue/submit_collect (host-sequenced), rows 1 and 6;
- tier_issue/tier_collect (device-sequenced; on a side stream joined at the collect, and with QMESH_CPU_TIER_SIDE=0
  on the compute stream), eager and as a captured CUDA graph replayed with fresh inputs more times than there are slots (the slot, job descriptor, ring tail and flags all come from device state on every replay)."""
import json
import os
import struct
import tempfile
from types import SimpleNamespace

import pytest
import torch

from testlib.isolated import device_env, run_isolated

H, I, TOPK = 512, 256, 3
SLOTS = 4


def _write_pack(td):
    path = os.path.join(td, "cold-L00.safetensors")
    entries, payload, tensors = {}, [], {}
    offset = 0
    gen = torch.Generator().manual_seed(449)
    for slot in range(SLOTS):
        for proj, out, inp in (("gate", I, H), ("up", I, H), ("down", H, I)):
            for suffix, shape in (("weight", (out, inp // 2)), ("scale", (out, inp // 32))):
                t = torch.randint(118, 125, shape, dtype=torch.uint8, generator=gen) if suffix == "scale" else \
                    torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)
                tensors[(slot, proj, suffix)] = t
                raw = t.numpy().tobytes()
                key = f"slot.{slot}.{proj}.{suffix}"
                entries[key] = {"dtype": "U8", "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
                payload.append(raw)
                offset += len(raw)
    header = {"__metadata__": {"experts": json.dumps(list(range(SLOTS)))}, **entries}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        f.write(b"".join(payload))
    return path, tensors


def _case(device_sequenced):
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.model.moe_cpu_host import MoeCpuHost

    with tempfile.TemporaryDirectory() as td:
        path, tensors = _write_pack(td)
        config = SimpleNamespace(directory=td, infer_params=SimpleNamespace(moe_cpu_component="text", moe_cpu_threads=2,
                                                                                moe_cpu_pinned_arena=False))
        host = MoeCpuHost(config)
        dev = torch.device("cuda:0")
        gen = torch.Generator().manual_seed(17)

        def inputs(rows):
            x = (torch.randn(rows, H, generator=gen) * .1).half()
            sel = torch.randint(0, SLOTS, (rows, TOPK), dtype=torch.int64, generator=gen)
            sel[0, 0] = -1
            if rows > 1:
                sel[1, 1] = -1
            weights = torch.randn(rows, TOPK, generator=gen).half()
            return x, sel, weights

        def reference(x, sel, weights):
            expected = torch.empty(x.shape[0], H, dtype=torch.float32)
            ext.exl3_moe_fp4_forward(ref, x.contiguous(), sel.contiguous(), weights.contiguous(), expected, 2)
            return expected

        try:
            li = host.register_layer("fp4", [], [], [], 0, 10.0, H, H, TOPK,
                                     proj_dims=None, fp4={"path": path, "slots": SLOTS})
            host.ensure_started()
            ref = ext.exl3_moe_fp4_make_layer(
                [tensors[s, "gate", "weight"] for s in range(SLOTS)], [tensors[s, "gate", "scale"] for s in range(SLOTS)],
                [tensors[s, "up", "weight"] for s in range(SLOTS)], [tensors[s, "up", "scale"] for s in range(SLOTS)],
                [tensors[s, "down", "weight"] for s in range(SLOTS)], [tensors[s, "down", "scale"] for s in range(SLOTS)], 10.0)
            try:
                if not device_sequenced:
                    for rows in (1, 6):
                        x, sel, weights = inputs(rows)
                        result = host.submit_collect(host.submit_issue(li, x.to(dev), sel.to(dev), weights.to(dev)))
                        assert torch.equal(result.cpu(), reference(x, sel, weights))
                    return
                for rows in (1, 6):
                    # Eager: the partial is added into a non-zero base, so the in-place add is checked too
                    x, sel, weights = inputs(rows)
                    base = torch.randn(rows, H, generator=gen)
                    out = base.to(dev)
                    host.tier_collect(host.tier_issue(li, x.to(dev), sel.to(dev), weights.to(dev)), out)
                    assert torch.equal(out.cpu(), base + reference(x, sel, weights))
                    # Captured graph, static buffers refilled before every replay
                    sx, ssel, sw = (t.to(dev) for t in inputs(rows))
                    sout = torch.zeros(rows, H, device=dev)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(stream):
                        g.capture_begin()
                        sout.zero_()
                        host.tier_collect(host.tier_issue(li, sx, ssel, sw), sout)
                        g.capture_end()
                    torch.cuda.current_stream().wait_stream(stream)
                    for _ in range(2 * SLOTS + 1):
                        x, sel, weights = inputs(rows)
                        sx.copy_(x)
                        ssel.copy_(sel)
                        sw.copy_(weights)
                        g.replay()
                        assert torch.equal(sout.cpu(), reference(x, sel, weights))
                    # An all-skipped call adds nothing (the worker may skip it; the collect must not read the slot)
                    ssel.fill_(-1)
                    g.replay()
                    assert torch.count_nonzero(sout.cpu()) == 0
            finally:
                ext.exl3_moe_fp4_free_layer(ref)
        finally:
            host.shutdown()


def worker_case():
    _case(False)


def tier_case():
    _case(True)


def tier_main_stream_case():
    import os
    os.environ["QMESH_CPU_TIER_SIDE"] = "0"
    _case(True)


@pytest.mark.gpu
def test_fp4_worker(device):
    run_isolated(worker_case, env=device_env(device), timeout=1200)


@pytest.mark.gpu
def test_fp4_worker_device_sequenced_graph(device):
    run_isolated(tier_case, env=device_env(device), timeout=1200)


@pytest.mark.gpu
def test_fp4_worker_device_sequenced_graph_main_stream(device):
    run_isolated(tier_main_stream_case, env=device_env(device), timeout=1200)
