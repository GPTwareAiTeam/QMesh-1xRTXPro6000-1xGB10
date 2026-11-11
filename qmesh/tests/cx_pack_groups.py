"""Per-layer bitrate groups of a cold pack, read from the safetensors headers alone.

`safetensors` keeps every shape in the JSON header that precedes the data, so the grouping
`CudaExl3Moe.from_pack` derives (gate words, down words, codebook) can be listed without loading a
byte of the pack, without torch and without a GPU. That makes it usable on a GX host before a load:

    python3 qmesh/tests/cx_pack_groups.py --model-dir /cold \
        --placement /cold/placement.exl3.json --multi 2

What it answers: which layers have more than one bitrate group (a mixed layer exercises the group
split, a single-group layer does not) and whether the placement's cold ids are the checkpoint's
global ids -- the ids `from_pack` must index the shard with, while the kernel-facing `source` stays
local to the cold list.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def codebook(hdr, prefix):
    if prefix + ".mul1" in hdr:
        return "mul1"
    if prefix + ".mcg" in hdr:
        return "mcg"
    return "3inst"


def layer_report(model_dir, placement, layer):
    path = os.path.join(model_dir, f"cold-L{layer:02d}.safetensors")
    if not os.path.exists(path):
        return None
    hdr = header(path)
    cold = [int(e) for e in placement["layers"][str(layer)]["warm"]] if placement else None
    ids = cold if cold is not None else sorted(
        {int(k.split(".")[4]) for k in hdr if k.startswith(f"layers.{layer}.ffn.experts.")})
    groups, missing = {}, []
    for e in ids:
        p = f"layers.{layer}.ffn.experts.{e}"
        try:
            w13 = hdr[p + ".w1.trellis"]["shape"][2]
            w2 = hdr[p + ".w2.trellis"]["shape"][2]
        except KeyError:
            missing.append(e)
            continue
        groups.setdefault((int(w13), int(w2), codebook(hdr, p + ".w1")), 0)
        groups[(int(w13), int(w2), codebook(hdr, p + ".w1"))] += 1
    desc = ",".join(f"{k[0]}w/{k[1]}w/{k[2]}:{v}" for k, v in sorted(groups.items()))
    return {"layer": layer, "cold": len(ids), "global_ids": bool(cold is not None),
            "groups": len(groups), "desc": desc, "missing": missing,
            "first_ids": ids[:6]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default = os.environ.get("SEAM_MODEL", "/cold"))
    ap.add_argument("--placement", default = os.environ.get("SEAM_PLACEMENT",
                                                          "/cold/placement.exl3.json"))
    ap.add_argument("--layers", default = "")
    ap.add_argument("--multi", type = int, default = None,
                    help = "print only the first layer with at least this many groups")
    a = ap.parse_args()
    placement = json.load(open(a.placement)) if a.placement and os.path.exists(a.placement) else None
    if placement is None:
        print("PACK_GROUPS no placement: cold id sets read from the shards", file = sys.stderr)
    layers = [int(x) for x in a.layers.split(",") if x.strip()] if a.layers else (
        sorted(int(k) for k in placement["layers"]) if placement else range(64))
    seen = []
    for layer in layers:
        r = layer_report(a.model_dir, placement, layer)
        if r is None:
            continue
        seen.append((r["layer"], r["groups"]))
        print(f"PACK_GROUPS layer={r['layer']} cold={r['cold']} groups={r['groups']} "
              f"ids={'global' if r['global_ids'] else 'shard'} first={r['first_ids']} "
              f"desc={r['desc']}"
              + (f" MISSING={r['missing'][:8]}" if r["missing"] else ""))
    if a.multi is not None:
        print("PACK_GROUPS multi=" + ",".join(str(l) for l, g in seen if g >= a.multi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
