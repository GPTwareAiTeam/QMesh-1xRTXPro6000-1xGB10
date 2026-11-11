"""
The split placement permutation moves the router's expert columns into hot-to-cold order right after load, and
every per-expert routing tensor has to follow it or selection runs in the wrong space. The vision selection bias
(DeepSeek-V4 gate.bias_vl, used for image rows) was missed: the router gate was permuted and the bias was not, so
image rows selected top-k on a bias whose entries no longer belonged to the columns they were scored against
(IdSpace, 2026-11-04). Checked on the real class without weights, on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.modules.block_sparse_mlp import BlockSparseMLP


pytestmark = pytest.mark.nogpu


def _module(perm, num_experts = 8):
    m = BlockSparseMLP.__new__(BlockSparseMLP)
    m.modules = []
    m.num_experts = num_experts
    m.key = "model.layers.0.mlp"
    m.device = torch.device("cpu")
    m.config = SimpleNamespace(infer_params = SimpleNamespace())
    E = num_experts
    m.cpu_split_first = num_experts - 2
    m._split_perm = perm
    m._split_dynamic = False            # the dynamic-placement block needs a live worker; not under test
    inner = SimpleNamespace(weight = torch.arange(E * 3, dtype = torch.float).reshape(3, E), bias = None)
    m.routing_gate = SimpleNamespace(inner = inner)
    m.e_score_correction_bias = torch.arange(E, dtype = torch.float)
    m.per_expert_scale = None
    m.e_score_bias_vl = torch.arange(E, dtype = torch.float) + 100.0
    return m


def test_vision_bias_follows_the_split_permutation():
    perm = [5, 2, 7, 0, 3, 6, 1, 4]
    m = _module(perm)
    before = m.e_score_bias_vl.clone()
    m.cpu_post_load()
    assert m.e_score_bias_vl.tolist() == before[perm].tolist()
    # the gate columns and the two per-expert biases must all be in the same order
    assert m.routing_gate.inner.weight[0].tolist() == [float(p) for p in perm]
    assert m.e_score_correction_bias.tolist() == [float(p) for p in perm]


def test_vision_bias_permutation_leaves_an_unpermuted_split_alone():
    m = _module(None)                   # identity split, no placement profile: nothing to move
    before = m.e_score_bias_vl.clone()
    m.cpu_post_load()
    assert torch.equal(m.e_score_bias_vl, before)


def test_routing_config_built_before_the_permutation_is_corrected():
    """A reload that runs load_routing before cpu_post_load would keep the checkpoint-order bias and a centered
    fp16 copy of it; the hook must repoint the config and drop the copy"""
    perm = [3, 0, 2, 1, 7, 5, 6, 4]
    m = _module(perm)
    stale = m.e_score_bias_vl.clone()
    cfg = SimpleNamespace(e_score_bias_vl = stale, e_score_bias_vl_h = stale.half())
    m.routing_cfg = cfg
    m.cpu_post_load()
    assert cfg.e_score_bias_vl is m.e_score_bias_vl
    assert cfg.e_score_bias_vl_h is None
