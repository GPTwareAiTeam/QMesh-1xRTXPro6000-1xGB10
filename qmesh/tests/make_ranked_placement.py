"""Write the ranked placement next to the pack's own table: same sets, deterministic order.

The pack's placement fixes *which* experts are hot, which cold ones are 4-bit and which are 2-bit,
and in which order the seam's cold-local ids are numbered. Only that last part is order by
accident: `warm` is ascending, so every layer's cold list interleaves 2-bit and 4-bit experts, and
the seam's group boundary (`bits4:n`) sits wherever the interleave put it. That costs the client a
sort it cannot do without a GPU pass, and it makes the boundary a property of the numbers rather
than of the file.

This writes `placement.<...>.ranked.json` with the identical sets and bit widths:

  hot   : the same experts, lowest global rank first (expert_ranking_2.25.csv);
  warm  : the same cold experts, all 4-bit first, then all 2-bit, each lowest rank first;
  warm_bpw: permuted with warm, so `warm_bpw[i]` is still the bitrate of `warm[i]`;
  warm_4bpw_n: the group boundary, i.e. `warm[:warm_4bpw_n]` are exactly the 4-bit experts.

Set equality against the source file is asserted per layer before anything is written, and the
output is re-read and re-checked afterwards. CPU only, no pack access: run it anywhere.

    python3 qmesh/tests/make_ranked_placement.py \
        --src /path/to/placement.json --ranking /path/to/ranking.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_ranks(path):
    """{(layer, expert): rank} from the ranking csv. Ranks are global, so they order a subset."""
    ranks = {}
    with open(path, newline = "") as fh:
        for row in csv.DictReader(fh):
            ranks[(int(row["layer"]), int(row["expert_id"]))] = int(row["rank"])
    if not ranks:
        raise SystemExit(f"no rows in {path}")
    return ranks


def ranked_layer(layer, src, ranks):
    hot, warm, bpw = list(src["hot"]), list(src["warm"]), list(src["warm_bpw"])
    if len(warm) != len(bpw):
        raise SystemExit(f"layer {layer}: warm {len(warm)} != warm_bpw {len(bpw)}")
    n_experts = len(hot) + len(warm)
    if sorted(hot + warm) != list(range(n_experts)):
        raise SystemExit(f"layer {layer}: source hot+warm is not a permutation of 0..{n_experts - 1}")

    def rank(e):
        try:
            return ranks[(layer, e)]
        except KeyError:
            raise SystemExit(f"layer {layer}: expert {e} has no rank in the ranking csv")

    four = [e for e, b in zip(warm, bpw) if b == 4]
    two = [e for e, b in zip(warm, bpw) if b == 2]
    if len(four) + len(two) != len(warm):
        raise SystemExit(f"layer {layer}: warm_bpw has widths other than 4 and 2")
    out_hot = sorted(hot, key = rank)
    out_four = sorted(four, key = rank)
    out_two = sorted(two, key = rank)
    out = dict(src)
    out["hot"] = out_hot
    out["warm"] = out_four + out_two
    out["warm_bpw"] = [4] * len(out_four) + [2] * len(out_two)
    out["warm_4bpw_n"] = len(out_four)
    return out


def check(layer, src, out):
    """The contract the client and the server both rely on, checked against the source layer."""
    assert set(out["hot"]) == set(src["hot"]), f"layer {layer}: hot set changed"
    assert set(out["warm"]) == set(src["warm"]), f"layer {layer}: warm set changed"
    assert len(out["hot"]) == len(src["hot"]) and len(out["warm"]) == len(src["warm"])
    assert out["hot_bpw"] == src["hot_bpw"], f"layer {layer}: hot_bpw changed"
    assert sorted(out["warm_bpw"]) == sorted(src["warm_bpw"]), f"layer {layer}: widths changed"
    # warm_bpw is still the width of the expert in the same position.
    for e, b in zip(out["warm"], out["warm_bpw"]):
        i = src["warm"].index(e)
        assert src["warm_bpw"][i] == b, f"layer {layer}: expert {e} changed width"
    n4 = out["warm_4bpw_n"]
    assert out["warm_bpw"][:n4] == [4] * n4, f"layer {layer}: 4-bit experts are not first"
    assert all(b == 2 for b in out["warm_bpw"][n4:]), f"layer {layer}: 2-bit experts are not last"
    assert n4 == sum(1 for b in src["warm_bpw"] if b == 4), f"layer {layer}: boundary != 4-bit count"
    assert sorted(out["hot"] + out["warm"]) == list(range(len(out["hot"]) + len(out["warm"])))


def main(argv = None):
    ap = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    ap.add_argument("--src", required = True)
    ap.add_argument("--ranking", required = True)
    ap.add_argument("--out", default = None,
                    help = "default: the source path with `.ranked.json` appended")
    args = ap.parse_args(argv)

    out_path = args.out or os.path.splitext(args.src)[0] + ".ranked.json"
    if os.path.abspath(out_path) == os.path.abspath(args.src):
        raise SystemExit("refusing to overwrite the source placement")
    ranks = load_ranks(args.ranking)
    src_sha = sha256(args.src)
    with open(args.src) as fh:
        src = json.load(fh)
    if src.get("ranked"):
        raise SystemExit(f"{args.src} is already a ranked placement")
    out = {k: v for k, v in src.items() if k != "layers"}
    out["ranked"] = True
    out["source_placement"] = os.path.abspath(args.src)
    out["source_sha256"] = src_sha
    out["ranking_sha256"] = sha256(args.ranking)
    out["layers"] = {}
    total4 = 0
    for layer_s, layer_src in sorted(src["layers"].items(), key = lambda kv: int(kv[0])):
        layer = int(layer_s)
        layer_out = ranked_layer(layer, layer_src, ranks)
        check(layer, layer_src, layer_out)
        out["layers"][layer_s] = layer_out
        total4 += layer_out["warm_4bpw_n"]
    if total4 != src.get("warm_4bpw_experts", total4):
        raise SystemExit(f"4-bit cold experts: {total4} vs header {src.get('warm_4bpw_experts')}")
    out["warm_4bpw_experts"] = total4

    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent = 1, sort_keys = True)
        fh.write("\n")
    os.replace(tmp, out_path)

    again = json.load(open(out_path))
    for layer_s, layer_src in src["layers"].items():
        check(int(layer_s), layer_src, again["layers"][layer_s])
    print(f"RANKED source={args.src} sha256={src_sha}")
    print(f"RANKED out={out_path} sha256={sha256(out_path)}")
    print(f"RANKED layers={len(out['layers'])} hot={out['hot_experts']} warm={out['warm_experts']} "
          f"warm_4bpw={total4} warm_2bpw={out['warm_experts'] - total4} "
          f"sets_equal=1 widths_equal=1 permutation=1")
    layer0 = out["layers"]["0"]
    print(f"RANKED layer0 hot[:8]={layer0['hot'][:8]} warm[:4]={layer0['warm'][:4]} "
          f"warm_4bpw_n={layer0['warm_4bpw_n']} warm_bpw[:4]={layer0['warm_bpw'][:4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
