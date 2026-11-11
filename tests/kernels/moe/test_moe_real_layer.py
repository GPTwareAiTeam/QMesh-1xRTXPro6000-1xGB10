"""
Real-weight check: one DeepSeek-V4.1-Flash-1017 EXL3 mixed-2.565 MoE layer (layer 3) built through BlockSparseMLP with
the architecture's own arguments (architecture/deepseek_v4.py). Routing is pinned via external_selected /
external_weights so the layer and the dense reference consume identical routing. The reference is a per-expert
dense computation (exl3_linear_ref, each expert at its own K, silu(gate)*up, down, weighted sum) over the layer's
loaded LinearEXL3 tensors. Checkpoint is read-only. Run inside exl3-worker on GPU1 only.
"""
import os

import pytest
import torch

from exllamav3 import Config
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from exllamav3.ext import exllamav3_ext as ext
from testlib.moe import exl3_linear_ref

CKPT = os.environ.get("EXL3_TEST_MOE_CHECKPOINT")
LAYER = 3


@pytest.fixture(scope = "module")
def layer(device):
    if not CKPT or not os.path.isdir(CKPT):
        pytest.skip("set EXL3_TEST_MOE_CHECKPOINT to a mounted checkpoint")
    cfg = Config.from_directory(CKPT)
    mlp = BlockSparseMLP(
        config = cfg,
        key = f"layers.{LAYER}.ffn",
        hidden_size = cfg.hidden_size,
        intermediate_size = cfg.moe_intermediate_size,
        num_experts = cfg.num_experts,
        num_experts_per_tok = cfg.num_experts_per_tok,
        key_up = "experts.{expert_idx}.w3",
        key_gate = "experts.{expert_idx}.w1",
        key_down = "experts.{expert_idx}.w2",
        key_routing_gate = "gate",
        key_e_score_bias = "gate.bias",
        qmap = "block.mlp",
        interm_dtype = torch.half,
        out_dtype = torch.float,
        activation_fn = "silu",
        act_limit = cfg.swiglu_limit,
        router_type = "sqrtsp",
    )
    mlp.load(device)
    return mlp


def dense_reference(layer, x, sel, w):
    """Per-expert dense oracle: sum_k w[t,k] * expert_{sel[t,k]}(x[t]). Each expert at its own K."""
    out = torch.zeros((x.shape[0], layer.hidden_size), dtype = torch.float, device = x.device)
    for t in range(x.shape[0]):
        for k in range(sel.shape[1]):
            e = int(sel[t, k])
            g = layer.gates[e].inner
            u = layer.ups[e].inner
            d = layer.downs[e].inner
            xt = x[t:t + 1]
            gg = exl3_linear_ref(xt, g.trellis, g.suh, g.svh, g.K, "mul1")
            uu = exl3_linear_ref(xt, u.trellis, u.suh, u.svh, u.K, "mul1")
            ext.silu_mul(gg, uu, uu, layer.act_limit)
            dd = exl3_linear_ref(uu, d.trellis, d.suh, d.svh, d.K, "mul1").float()
            out[t] += w[t, k].float() * dd[0]
    return out


@pytest.mark.parametrize("bsz", [1, 6, 48])
def test_real_layer_mixed_fused_vs_reference(device, layer, bsz):
    assert layer.mixed_k, "layer 3 should carry per-expert mixed rates"
    gen = torch.Generator(device = "cpu").manual_seed(bsz)
    E, k = layer.num_experts, layer.num_experts_per_tok
    x = (torch.randn((bsz, layer.hidden_size), generator = gen) * 0.05).half().to(device)
    sel = torch.stack([torch.randperm(E, generator = gen)[:k] for _ in range(bsz)]).to(device)
    w = (torch.rand((bsz, k), generator = gen) * 0.5 + 0.1).half().to(device)

    params = {"external_selected": sel, "external_weights": w}
    got = layer.forward(x.view(1, bsz, layer.hidden_size), params).view(bsz, -1).float()
    ref = dense_reference(layer, x, sel, w)
    scale = ref.abs().max().item()
    err = (got - ref).abs().max().item()
    print(f"bsz {bsz}: err {err:.3e} scale {scale:.3e} rel {err / max(scale, 1e-12):.3e}")
    assert err <= 3e-3 * scale, f"bsz {bsz}: err {err:.3e} vs scale {scale:.3e}"
