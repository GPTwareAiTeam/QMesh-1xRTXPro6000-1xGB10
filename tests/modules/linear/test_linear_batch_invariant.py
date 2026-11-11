"""
Batch-invariant decode Linear (caps["batch_invariant_pad"], util/batch_invariant.py): a flagged module zero-pads
1..EXL3_BATCH_INVARIANT_ROWS - 1 input rows to that many, so row 0's output is bit-identical at every decode batch
size, for the FP16 path (cuBLAS chooses its algorithm by M) and the EXL3 path (tile and k split by M). Shapes are
the DSV4.1 Engram wkv (6144 -> 25600, FP16) and a head-like EXL3 8-bit projection. An unflagged module is the
control: it must still return the same values for row 0 up to rounding, and its shape handling is unchanged.
"""

import pytest
import torch

from exllamav3.modules import Linear
from exllamav3.util.batch_invariant import BI_ROWS

# The batched server's contract: these tests run under the env qmesh/serve_pair.py sets for BATCH > 1
pytestmark = pytest.mark.skipif(BI_ROWS != 8, reason = "run with EXL3_BATCH_INVARIANT_ROWS=8 (the batched-server env)")

from testlib.checkpoint import module_config
from testlib.exl3 import checkpoint_tensors, generator

KEY = "model.layers.0.mlp.up_proj"


def fp16_module(directory, device, k, n, flag):
    w = (torch.randn((n, k), generator = generator(7)) * 0.02).half()
    module = Linear(module_config({f"{KEY}.weight": w}, directory), KEY, k, n, out_dtype = torch.half,
                    caps = {"batch_invariant_pad": True} if flag else None)
    module.load(device)
    assert module.quant_type == "fp16"
    return module


def exl3_module(directory, device, k, n, flag):
    t = checkpoint_tensors(KEY, k, n, 8, generator(11), codebook = "mul1")
    module = Linear(module_config(t, directory), KEY, k, n, caps = {"batch_invariant_pad": True} if flag else None)
    module.load(device)
    assert module.quant_type == "exl3"
    return module


@pytest.mark.parametrize("kind,k,n", [("fp16", 6144, 25600), ("exl3", 5120, 16384)])
def test_row0_identical_across_batch(tmp_path, device, kind, k, n):
    assert BI_ROWS == 8
    build = fp16_module if kind == "fp16" else exl3_module
    flagged = build(tmp_path / "a", device, k, n, True)
    plain = build(tmp_path / "b", device, k, n, False)
    x = (torch.randn((BI_ROWS, 1, k), generator = generator(3)) * 0.5).half().to(device)
    ref = flagged.forward(x[:1], {})
    assert ref.shape == (1, 1, n)
    for rows in range(2, BI_ROWS + 1):
        y = flagged.forward(x[:rows], {})
        assert y.shape == (rows, 1, n)
        assert torch.equal(y[0], ref[0]), f"{kind}: row 0 differs at {rows} rows"
    # The control computes the same function
    p = plain.forward(x[:1], {})
    assert ((p.float() - ref.float()).abs().max() / ref.float().abs().max()).item() < 1e-2


def test_prefill_rows_unpadded(tmp_path, device):
    """Row counts above BI_ROWS go through unchanged (no padding, same result as the unflagged module)"""
    flagged = fp16_module(tmp_path / "a", device, 1024, 512, True)
    plain = fp16_module(tmp_path / "b", device, 1024, 512, False)
    x = (torch.randn((3, 40, 1024), generator = generator(5)) * 0.5).half().to(device)
    assert torch.equal(flagged.forward(x, {}), plain.forward(x, {}))
