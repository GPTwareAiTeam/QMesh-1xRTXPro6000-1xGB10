import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.moe_cpu import cpu_runtime   # noqa: F401 (fixture)

pytestmark = pytest.mark.nogpu


HIDDEN = 512
INTERMEDIATE = 256
EXPERTS = 5
TOPK = 6
LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
)


def make_layer(gen):
    gate_w, gate_s, up_w, up_s, down_w, down_s = [], [], [], [], [], []
    for _ in range(EXPERTS):
        gate_w.append(torch.randint(0, 256, (INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8, generator=gen))
        gate_s.append(torch.randint(118, 125, (INTERMEDIATE, HIDDEN // 32), dtype=torch.uint8, generator=gen))
        up_w.append(torch.randint(0, 256, (INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8, generator=gen))
        up_s.append(torch.randint(118, 125, (INTERMEDIATE, HIDDEN // 32), dtype=torch.uint8, generator=gen))
        down_w.append(torch.randint(0, 256, (HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8, generator=gen))
        down_s.append(torch.randint(118, 125, (HIDDEN, INTERMEDIATE // 32), dtype=torch.uint8, generator=gen))
    return ext.exl3_moe_fp4_make_layer(gate_w, gate_s, up_w, up_s, down_w, down_s, 0.0), (
        gate_w, gate_s, up_w, up_s, down_w, down_s
    )


def unpack(weight, scale):
    nibbles = torch.empty(weight.shape[0], weight.shape[1] * 2, dtype=torch.long)
    nibbles[:, 0::2] = (weight.to(torch.long) & 15)
    nibbles[:, 1::2] = (weight.to(torch.long) >> 4)
    factors = torch.pow(2.0, scale.to(torch.float64) - 127.0).repeat_interleave(32, dim=1)
    return LUT[nibbles] * factors


def reference(x, selected, weights, matrices, act_limit):
    gate_w, gate_s, up_w, up_s, down_w, down_s = matrices
    gate = [unpack(w, s) for w, s in zip(gate_w, gate_s)]
    up = [unpack(w, s) for w, s in zip(up_w, up_s)]
    down = [unpack(w, s) for w, s in zip(down_w, down_s)]
    x64 = x.to(torch.float64)
    result = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float64)
    for row in range(x.shape[0]):
        for pick in range(selected.shape[1]):
            expert = int(selected[row, pick])
            if expert < 0:
                continue
            g = x64[row] @ gate[expert].T
            u = x64[row] @ up[expert].T
            a = torch.nn.functional.silu(g)
            if act_limit != 0:
                a = a.clamp(max=act_limit)
                u = u.clamp(-act_limit, act_limit)
            result[row] += weights[row, pick].to(torch.float64) * (a * u @ down[expert].T)
    return result


def assert_matches(out, expected):
    error = torch.linalg.vector_norm(out.to(torch.float64) - expected) / torch.linalg.vector_norm(expected)
    assert torch.isfinite(out).all()
    assert error <= 2e-2, f"relative Frobenius error {error.item():.6g} exceeds 2e-2"


def run_case(rows, act_limit, selected=None, gen=None, num_threads=4):
    gen = gen or torch.Generator().manual_seed(211)
    gate_w, gate_s, up_w, up_s, down_w, down_s = [], [], [], [], [], []
    for _ in range(EXPERTS):
        gate_w.append(torch.randint(0, 256, (INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8, generator=gen))
        gate_s.append(torch.randint(118, 125, (INTERMEDIATE, HIDDEN // 32), dtype=torch.uint8, generator=gen))
        up_w.append(torch.randint(0, 256, (INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8, generator=gen))
        up_s.append(torch.randint(118, 125, (INTERMEDIATE, HIDDEN // 32), dtype=torch.uint8, generator=gen))
        down_w.append(torch.randint(0, 256, (HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8, generator=gen))
        down_s.append(torch.randint(118, 125, (HIDDEN, INTERMEDIATE // 32), dtype=torch.uint8, generator=gen))
    matrices = gate_w, gate_s, up_w, up_s, down_w, down_s
    handle = ext.exl3_moe_fp4_make_layer(*matrices, act_limit)
    try:
        x = (torch.randn(rows, HIDDEN, generator=gen) * 0.5).half()
        if selected is None:
            selected = torch.randint(-1, EXPERTS, (rows, TOPK), dtype=torch.long, generator=gen)
            selected[0, 0] = -1
        weights = (torch.rand(rows, TOPK, generator=gen) * 0.30 + 0.05).half()
        out = torch.empty(rows, HIDDEN, dtype=torch.float32)
        ext.exl3_moe_fp4_forward(handle, x, selected, weights, out, num_threads)
        expected = reference(x, selected, weights, matrices, act_limit)
        return out, expected, x, selected, weights
    finally:
        ext.exl3_moe_fp4_free_layer(handle)


@pytest.mark.parametrize("rows", (1, 3, 9, 17))
@pytest.mark.parametrize("act_limit", (0.0, 10.0))
def test_moe_fp4_matches_reference(rows, act_limit, cpu_runtime):
    out, expected, _, _, _ = run_case(rows, act_limit)
    assert_matches(out, expected)


def test_moe_fp4_skipped_rows_are_zero(cpu_runtime):
    selected = torch.full((3, TOPK), -1, dtype=torch.long)
    selected[1, 0] = 2
    out, _, _, _, _ = run_case(3, 0.0, selected=selected)
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    selected.fill_(-1)
    out, _, _, _, _ = run_case(3, 0.0, selected=selected)
    assert torch.equal(out, torch.zeros_like(out))


def test_moe_fp4_repeated_experts(cpu_runtime):
    selected = torch.zeros(3, TOPK, dtype=torch.long)
    selected[0, 1] = selected[0, 2] = 3
    out, expected, _, _, _ = run_case(3, 10.0, selected=selected)
    assert_matches(out, expected)


def test_moe_fp4_thread_determinism(cpu_runtime):
    gen = torch.Generator().manual_seed(313)
    matrices = None
    gate_w, gate_s, up_w, up_s, down_w, down_s = [], [], [], [], [], []
    for _ in range(EXPERTS):
        gate_w.append(torch.randint(0, 256, (INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8, generator=gen))
        gate_s.append(torch.randint(118, 125, (INTERMEDIATE, HIDDEN // 32), dtype=torch.uint8, generator=gen))
        up_w.append(torch.randint(0, 256, (INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8, generator=gen))
        up_s.append(torch.randint(118, 125, (INTERMEDIATE, HIDDEN // 32), dtype=torch.uint8, generator=gen))
        down_w.append(torch.randint(0, 256, (HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8, generator=gen))
        down_s.append(torch.randint(118, 125, (HIDDEN, INTERMEDIATE // 32), dtype=torch.uint8, generator=gen))
    matrices = gate_w, gate_s, up_w, up_s, down_w, down_s
    handle = ext.exl3_moe_fp4_make_layer(*matrices, 0.0)
    try:
        x = (torch.randn(9, HIDDEN, generator=gen) * 0.5).half()
        selected = torch.randint(-1, EXPERTS, (9, TOPK), dtype=torch.long, generator=gen)
        weights = (torch.rand(9, TOPK, generator=gen) * 0.30 + 0.05).half()
        outputs = []
        for threads in (1, 4, 16):
            out = torch.empty(9, HIDDEN, dtype=torch.float32)
            ext.exl3_moe_fp4_forward(handle, x, selected, weights, out, threads)
            outputs.append(out)
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(outputs[0], outputs[2])
    finally:
        ext.exl3_moe_fp4_free_layer(handle)


def test_moe_fp4_expert_bytes(cpu_runtime):
    gen = torch.Generator().manual_seed(414)
    handle, _ = make_layer(gen)
    try:
        expected = 2 * (INTERMEDIATE * HIDDEN // 2 + INTERMEDIATE * HIDDEN // 32) + (
            HIDDEN * INTERMEDIATE // 2 + HIDDEN * INTERMEDIATE // 32
        )
        assert ext.exl3_moe_fp4_expert_bytes(handle) == expected
    finally:
        ext.exl3_moe_fp4_free_layer(handle)


def test_moe_fp4_invalid_inputs(cpu_runtime):
    gen = torch.Generator().manual_seed(515)
    handle, _ = make_layer(gen)
    try:
        x = torch.zeros(1, HIDDEN, dtype=torch.float16)
        weights = torch.ones(1, TOPK, dtype=torch.float16)
        out = torch.empty(1, HIDDEN, dtype=torch.float32)
        selected = torch.zeros(1, TOPK, dtype=torch.long)
        selected[0, 0] = EXPERTS
        with pytest.raises(RuntimeError):
            ext.exl3_moe_fp4_forward(handle, x, selected, weights, out, 1)
        selected.zero_()
        with pytest.raises(RuntimeError):
            ext.exl3_moe_fp4_forward(handle, x.float(), selected, weights, out, 1)
        with pytest.raises(RuntimeError):
            ext.exl3_moe_fp4_forward(handle, x, selected, weights, torch.empty(1, HIDDEN - 1), 1)
    finally:
        ext.exl3_moe_fp4_free_layer(handle)
