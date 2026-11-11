"""CPU-only check of CudaExl3Moe.from_pack: a synthetic shard, no GPU, no extension.

The vendor extension is stubbed out (`cuda_exl3.ops` is not imported), so this runs anywhere torch
is installed, including a container on a box whose GPUs are busy. What it pins down is the part the
GX box cannot tell us quickly: which pack keys go where in the stacked slabs, how experts are
grouped by (gate words, down words, codebook), and that the bytes land unmodified.

    PYTHONPATH=/src/exllamav3 python3 qmesh/tests/cx_direct_test.py

The device-side oracle - the same slabs built by restacking real modules, compared tensor by tensor
- is qmesh/tests/cx_direct_check.py and needs the loaded layer, i.e. a GX.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import types

import torch


def _stub_ext():
    """Make `from qmesh.exl3_moe_cuda import ...` work without the compiled extension."""
    os.environ["CUDA_EXL3_MOE_ACC_CAP"] = "1"   # skip the import-time op call
    pkg = types.ModuleType("cuda_exl3")
    pkg.__path__ = []
    ops = types.ModuleType("cuda_exl3.ops")
    pkg.ops = ops
    sys.modules.setdefault("cuda_exl3", pkg)
    sys.modules.setdefault("cuda_exl3.ops", ops)


def _write_shard(path, tensors):
    """tensors: list of (name, torch cpu tensor) -> a safetensors file, header then data."""
    hdr, data, off = {}, [], 0
    for name, t in tensors:
        t = t.contiguous()
        raw = t.numpy().tobytes()
        hdr[name] = {"dtype": {torch.int16: "I16", torch.half: "F16", torch.int32: "I32"}[t.dtype],
                     "shape": list(t.shape), "data_offsets": [off, off + len(raw)]}
        data.append(raw)
        off += len(raw)
    blob = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for raw in data:
            f.write(raw)


def _expert(layer, e, kt, nt, w13, w2, cb):
    """One expert's nine tensors: w1 gate, w3 up, w2 down, plus the codebook flag."""
    p = f"layers.{layer}.ffn.experts.{e}"
    out = [
        (p + ".w1.trellis", torch.randint(-3000, 3000, (kt, nt, w13), dtype=torch.int16)),
        (p + ".w1.suh", torch.randn(16 * kt).half()),
        (p + ".w1.svh", torch.randn(16 * nt).half()),
        (p + ".w3.trellis", torch.randint(-3000, 3000, (kt, nt, w13), dtype=torch.int16)),
        (p + ".w3.suh", torch.randn(16 * kt).half()),
        (p + ".w3.svh", torch.randn(16 * nt).half()),
        (p + ".w2.trellis", torch.randint(-3000, 3000, (nt, kt, w2), dtype=torch.int16)),
        (p + ".w2.suh", torch.randn(16 * nt).half()),
        (p + ".w2.svh", torch.randn(16 * kt).half()),
    ]
    if cb == 2:
        out.append((p + ".w1.mul1", torch.zeros((), dtype=torch.int32)))
        out.append((p + ".w3.mul1", torch.zeros((), dtype=torch.int32)))
        out.append((p + ".w2.mul1", torch.zeros((), dtype=torch.int32)))
    elif cb == 1:
        out.append((p + ".w1.mcg", torch.zeros((), dtype=torch.int32)))
        out.append((p + ".w3.mcg", torch.zeros((), dtype=torch.int32)))
        out.append((p + ".w2.mcg", torch.zeros((), dtype=torch.int32)))
    return out


def main():
    _stub_ext()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from qmesh.exl3_moe_cuda import CB_3INST, CB_MUL1, CudaExl3Moe

    layer, kt, nt, w13, w2 = 5, 4, 3, 64, 32
    # The cold set is a *subset* of the checkpoint's experts, and the pack keys carry the checkpoint's
    # global ids (verified against the real 4460 shard: `layers.5.ffn.experts.<global>.w1.trellis`).
    # The ids the server receives are local to this subset, so a layer whose cold list is not
    # 0,1,2,... is the normal case -- layer 5 of the real pack has 254 ids between 0 and 383.
    # Global ids here are deliberately non-contiguous, out of order, and interleaved with the
    # groups, so `source` (local) and the shard key (global) can not be confused silently:
    #   local 0 = global 11 (MUL1), local 1 = global 3 (MUL1),
    #   local 2 = global 40 (3INST), local 3 = global 17 (3INST).
    cold = [11, 3, 40, 17]
    shape_of = {11: (CB_MUL1, w13, w2), 3: (CB_MUL1, w13, w2),
                40: (CB_3INST, 84, 52), 17: (CB_3INST, 84, 52)}
    src = {}
    tensors = []
    for e in cold:
        cb, words, dw = shape_of[e]
        for name, t in _expert(layer, e, kt, nt, words, dw, cb):
            src[name] = t
            tensors.append((name, t))
    # A second layer's experts live in the same shard file: the reader must pick by layer number.
    for e in (0, 1):
        tensors.extend(_expert(layer + 1, e, kt, nt, w13, w2, CB_3INST))

    tmp = tempfile.mkdtemp(prefix = "cx_direct_")
    path = os.path.join(tmp, "cold-L05.safetensors")
    _write_shard(path, tensors)

    hit = CudaExl3Moe.from_pack(path, layer, cold, device = "cpu")
    assert hit.hidden == 16 * kt, hit.hidden
    assert hit.inter == 16 * nt, hit.inter
    assert hit.num_experts == 4
    assert len(hit.groups) == 2, hit.describe()
    m1, m3 = hit.groups
    assert (m1.bits, m1.frac, m1.cb, m1.n_experts) == (4, False, CB_MUL1, 2), hit.describe()
    assert (m3.bits, m3.frac, m3.cb, m3.n_experts) == (5, True, CB_3INST, 2), hit.describe()
    # `source` labels the experts of this group as the layer's forward sees them: positions in
    # `cold`, i.e. the same labels `__init__` gives the module's per-expert list. A global id here
    # (11 or 40 on a 4-expert layer) would index `remap` out of bounds on the device.
    assert m1.source == [0, 1] and m3.source == [2, 3], (m1.source, m3.source)
    assert m1.remap.tolist() == [0, 1, -1, -1], m1.remap.tolist()
    assert m3.remap.tolist() == [-1, -1, 0, 1], m3.remap.tolist()
    assert m1.remap.dtype == torch.int64 and m1.remap.numel() == len(cold)

    def check(grp, i, expert):  # i = position in the group, expert = global checkpoint id
        p = f"layers.{layer}.ffn.experts.{expert}"
        assert torch.equal(grp.trellis13[i, :, :nt], src[p + ".w1.trellis"]), "gate trellis"
        assert torch.equal(grp.trellis13[i, :, nt:], src[p + ".w3.trellis"]), "up trellis"
        assert torch.equal(grp.suh13[i, 0], src[p + ".w1.suh"]), "gate suh"
        assert torch.equal(grp.suh13[i, 1], src[p + ".w3.suh"]), "up suh"
        assert torch.equal(grp.svh13[i, :16 * nt], src[p + ".w1.svh"]), "gate svh"
        assert torch.equal(grp.svh13[i, 16 * nt:], src[p + ".w3.svh"]), "up svh"
        assert torch.equal(grp.trellis2[i], src[p + ".w2.trellis"]), "down trellis"
        assert torch.equal(grp.suh2[i, 0], src[p + ".w2.suh"]), "down suh"
        assert torch.equal(grp.svh2[i], src[p + ".w2.svh"]), "down svh"

    # Group order is sorted by key, member order is the order of `cold`: slot 0 of the MUL1 group is
    # global 11 (first in `cold`), slot 1 is global 3. Reading by `source` would swap them.
    check(m1, 0, 11)
    check(m1, 1, 3)
    check(m3, 0, 40)
    check(m3, 1, 17)
    assert hit.nbytes() == sum(g.nbytes() for g in hit.groups) and hit.nbytes() > 0
    # A missing expert is a load-time error, not a silently shorter layer.
    try:
        CudaExl3Moe.from_pack(path, layer, [11, 7], device = "cpu")
    except KeyError as e:
        assert "experts.7.w1.trellis" in str(e), e
    else:
        raise AssertionError("a missing expert id was accepted")
    print(f"CX_DIRECT_TEST ok groups={hit.describe()} cold={cold} source={m1.source}+{m3.source} "
          f"remap={m1.remap.tolist()}/{m3.remap.tolist()} slab_mib={hit.nbytes() / 2**20:.3f} "
          f"hidden={hit.hidden} inter={hit.inter}")

if __name__ == "__main__":
    main()
