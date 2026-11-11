"""CPU-only coverage of real split-loader registration and profile isolation."""
import ast
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


# Compile the real loader method in isolation: importing exllamav3 would load its CUDA extension.
_source = Path(__file__).resolve().parents[2] / "exllamav3/modules/block_sparse_mlp_cpu.py"
_tree = ast.parse(_source.read_text())
_cls = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and any(
    isinstance(m, ast.FunctionDef) and m.name == "load_cpu_split" for m in n.body))
_method = next(m for m in _cls.body if isinstance(m, ast.FunctionDef) and m.name == "load_cpu_split")
_method.returns = None
for _arg in _method.args.args:
    _arg.annotation = None
_ns = {"os": os, "torch": SimpleNamespace(device=lambda d: d, float=float),
       "trellis_rate": lambda _: 3}
exec(compile(ast.fix_missing_locations(ast.Module(body=[_method], type_ignores=[])), str(_source), "exec"), _ns)
load_cpu_split = _ns["load_cpu_split"]


class Expert:
    def __init__(self, name):
        self.key = name


class STC:
    tensor_file_map = {}

    def has_tensor(self, key):
        return True

    def get_tensor(self, key, device, optional=False, **kwargs):
        return None if key.endswith(".bias") else 1

    def list_tensors(self, key):
        return {key + ".trellis": {"shape": (1, 1, 48)}}


def module(remote=True):
    hot, warm = [2, 0], [3, 1, 4]
    ip = SimpleNamespace(moe_remote_hot={"layer": hot} if remote else None,
                         moe_remote_warm={"layer": warm}, remote_expert_host="gx" if remote else None)
    m = SimpleNamespace(key="layer", num_experts=5, gated=False, tid2eid_key=None,
                        config=SimpleNamespace(stc=STC(), infer_params=ip),
                        ups=[Expert(f"up.{e}") for e in range(5)],
                        downs=[Expert(f"down.{e}") for e in range(5)], gates=[], modules=[],
                        activation_fn="silu", act_limit=None, num_experts_per_tok=1,
                        interm_dtype=None, num_local_experts=5, routing_first=0, routing_last=5)
    return m, hot, warm


def test_remote_saved_profile_cannot_change_request_mapping(monkeypatch):
    m, hot, warm = module()
    registered = []
    host = SimpleNamespace(register_layer=lambda key, gates, ups, downs, *a, **kw: registered.extend(ups) or 0)
    fake = ModuleType("qmesh.remote_host")
    fake.RemoteExpertHost = SimpleNamespace(for_config=lambda ip: host)
    monkeypatch.setitem(sys.modules, "qmesh.remote_host", fake)
    monkeypatch.delenv("EXL3_MOE_CPU_SPLIT_STATS", raising=False)
    monkeypatch.setenv("REMOTE_TRANSPORT", "tcp")
    # Nonidentity stats would have overwritten the placement, even if persisted earlier.
    monkeypatch.setitem(_ns, "_expert_stats_load", lambda *a: {"layer": [1, 5, 2, 4, 3]})
    assert load_cpu_split(m, "cpu", 3)
    assert m._split_perm == hot + warm
    for request_id, global_id in enumerate(warm):
        assert m._split_perm[m.cpu_split_first + request_id] == global_id
        assert registered[request_id] == f"up.{global_id}"


def test_explicit_stats_and_remote_placement_refused_before_mutation(monkeypatch, tmp_path):
    m, _, _ = module()
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"layer": [1, 5, 2, 4, 3]}))
    monkeypatch.setenv("EXL3_MOE_CPU_SPLIT_STATS", str(stats))
    original = list(m.ups)
    with pytest.raises(RuntimeError, match="EXL3_MOE_CPU_SPLIT_STATS cannot be used with remote placement"):
        load_cpu_split(m, "cpu", 3)
    assert m.ups == original
