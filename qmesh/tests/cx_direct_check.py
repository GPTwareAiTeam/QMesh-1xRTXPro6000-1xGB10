"""Device oracle for CudaExl3Moe.from_pack: build one layer both ways and compare every byte.

Run it on a GX before trusting a server log full of `SEAM_MOE_DIRECT`. The module path (`build_layer`
+ `CudaExl3Moe(ffn, drop_source=False)`) is the layout the kernel has been serving; the direct path
reads the same tensors out of the pack shard into the same six slabs. If the grouping and the slice
placement in `from_pack` are right, the two objects are equal tensor for tensor, so a later
difference in output can only come from the pack or the reader - not from a mis-stacked slab that
would still "work" and quietly compute the wrong experts.

    PYTHONPATH=/src/exllamav3 python3 qmesh/tests/cx_direct_check.py --layers 0,20,39

Needs the loaded cold pack (--model-dir, default /cold) and the placement json that says which
experts of each layer are cold. Holds two copies of ONE layer (about 5.5 GiB), so run it on a box
that is not serving: this is a bring-up check, not a measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _avail_mib():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def check_layer(model, placement, model_dir, layer, device):
    from qmesh.exl3_moe_cuda import CudaExl3Moe
    from qmesh.worker_seam_server import build_layer
    pack = os.path.join(model_dir, f"cold-L{layer:02d}.safetensors")
    cold = placement["layers"][str(layer)]["warm"]
    if not cold:
        print(f"CXD_CHECK layer={layer} skip=no_cold_experts")
        return True
    avail0 = _avail_mib()
    base = next(m for m in model.modules if m.key == f"layers.{layer}").mlp
    ffn, n = build_layer(base, cold, device)
    del base
    assert n == len(cold), (n, len(cold))
    t0 = time.perf_counter()
    # drop_source=False: the module path must keep every expert so the comparison below has a
    # reference for the *sources* too, not just the stacks.
    ref = CudaExl3Moe(ffn, device, drop_source = False)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = CudaExl3Moe.from_pack(pack, layer, cold, device)
    t_direct = time.perf_counter() - t0
    ok = True
    if (got.hidden, got.inter, got.num_experts) != (ref.hidden, ref.inter, ref.num_experts):
        print(f"CXD_MISMATCH layer={layer} geometry ref={(ref.hidden, ref.inter, ref.num_experts)} "
              f"direct={(got.hidden, got.inter, got.num_experts)}")
        ok = False
    # `source` must be positions in the layer's cold list (what the forward's ids are), not the pack's
    # global expert ids: `remap` is indexed by source, so a global id is an out-of-bounds index_put on
    # the device, i.e. a context-killing assert in a fail-stop server. Cheap to verify before the
    # group keys are compared, which would otherwise report this as a mysterious "grouping" mismatch.
    for g in got.groups:
        if g.source and max(g.source) >= got.num_experts:
            print(f"CXD_MISMATCH layer={layer} source_out_of_range n={got.num_experts} "
                  f"source={g.source} (global ids leaked into source)")
            ok = False
    key = lambda g: (g.bits, g.frac, g.cb, tuple(g.source))
    ref_groups = {key(g): g for g in ref.groups}
    got_groups = {key(g): g for g in got.groups}
    if sorted(ref_groups) != sorted(got_groups):
        print(f"CXD_MISMATCH layer={layer} grouping ref={sorted(ref_groups)} "
              f"direct={sorted(got_groups)} cold={cold[:8]}{'...' if len(cold) > 8 else ''}")
        return False
    for k in sorted(ref_groups):
        a, b = ref_groups[k], got_groups[k]
        for name in ("trellis13", "suh13", "svh13", "trellis2", "suh2", "svh2"):
            ta, tb = getattr(a, name), getattr(b, name)
            if ta.shape != tb.shape or ta.dtype != tb.dtype:
                print(f"CXD_MISMATCH layer={layer} group={k} tensor={name} "
                      f"ref={tuple(ta.shape)}/{ta.dtype} direct={tuple(tb.shape)}/{tb.dtype}")
                ok = False
                continue
            if not torch.equal(ta, tb):
                bad = int((ta != tb).sum())
                print(f"CXD_MISMATCH layer={layer} group={k} tensor={name} "
                      f"differing={bad} of {ta.numel()}")
                ok = False
        if not torch.equal(a.remap, b.remap):
            print(f"CXD_MISMATCH layer={layer} group={k} remap")
            ok = False
    if ok:
        print(f"CXD_OK layer={layer} n={len(cold)} groups={got.describe()} "
              f"source={[len(g.source) for g in got.groups]} "
              f"slab_mib={got.nbytes() / 2**20:.0f} module_build_ref_ms={1e3 * t_ref:.0f} "
              f"direct_ms={1e3 * t_direct:.0f} avail_mib={avail0}->{_avail_mib()} "
              f"peak_mib={ref.nbytes() / 2**20 + got.nbytes() / 2**20:.0f}", flush=True)
    # Release both copies before the next layer: the oracle deliberately holds two.
    del ref, got, ffn
    torch.cuda.empty_cache()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default = os.environ.get("SEAM_MODEL", "/cold"))
    ap.add_argument("--placement", default = os.environ.get("SEAM_PLACEMENT",
                                                          "/cold/placement.exl3.json"))
    ap.add_argument("--layers", default = "0,20,39")
    ap.add_argument("--device", default = "cuda:0")   # exl3_moe_max_concurrency needs an index
    ap.add_argument("--margin-mib", type = int, default = 7000,
                    help = "refuse to start below this MemAvailable: the check holds two layers")
    a = ap.parse_args()
    avail = _avail_mib()
    if avail >= 0 and avail < a.margin_mib and os.environ.get("SEAM_FORCE_LOAD", "0") == "0":
        print(f"CXD_NO_MEMORY avail_mib={avail} need_mib={a.margin_mib} - free the box or set "
              f"SEAM_FORCE_LOAD=1")
        return 3
    placement = a.placement
    if not placement or not os.path.exists(placement):
        print("CXD_NO_PLACEMENT: pass --placement (json with layers.<N>.warm = cold expert ids)")
        return 3
    pl = json.load(open(placement))
    from exllamav3 import Config, Model
    device = torch.device(a.device)
    if device.index is None:          # bare "cuda" reaches exl3_moe_max_concurrency(None)
        device = torch.device(device.type, 0)
    config = Config.from_directory(a.model_dir)
    model = Model.from_config(config)
    rc = 0
    for layer in [int(x) for x in a.layers.split(",") if x.strip()]:
        if not check_layer(model, pl, a.model_dir, layer, device):
            rc = 1
    try:
        config.stc.close()
    except Exception:  # noqa: BLE001 - the verdict is already printed
        pass
    print(f"CXD_DONE layers={a.layers} rc={rc} avail_mib={_avail_mib()}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
