#!/usr/bin/env python3
"""Pack the checkpoint's routed MXFP4 experts into per-layer cold-slot safetensors."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import struct
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig", required=True)
    p.add_argument("--placement", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", default="0-39")
    p.add_argument("--jobs", type=int, default=8)
    return p.parse_args()


def layer_ids(spec):
    values = set()
    for part in spec.split(","):
        a, sep, b = part.partition("-")
        if sep:
            values.update(range(int(a), int(b) + 1))
        else:
            values.add(int(a))
    if any(x < 0 or x > 39 for x in values):
        raise ValueError("layers must be in the range 0-39")
    return sorted(values)


def open_index(orig):
    with open(os.path.join(orig, "model.safetensors.index.json"), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def load_header(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        raw = os.pread(fd, 8, 0)
        if len(raw) != 8:
            raise IOError(f"short safetensors prefix: {path}")
        n = struct.unpack("<Q", raw)[0]
        hdrbytes = os.pread(fd, n, 8)
        if len(hdrbytes) != n:
            raise IOError(f"short safetensors header: {path}")
        hdr = json.loads(hdrbytes)
        return fd, 8 + n, hdr
    except BaseException:
        os.close(fd)
        raise


def pack_layer(args):
    layer, warm, orig, weight_map, out_dir = args
    tensors = []
    for slot, expert in enumerate(warm):
        for part, projection in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
            for suffix in ("weight", "scale"):
                key = f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
                shard = weight_map[key]
                tensors.append((f"slot.{slot}.{part}.{suffix}", key, shard))

    # Open each referenced shard once; header offsets are relative to its data section.
    shards = {}
    try:
        for _, _, shard in tensors:
            if shard not in shards:
                shards[shard] = load_header(os.path.join(orig, shard))
        entries = []
        total = 0
        for dest, key, shard in tensors:
            _, _, header = shards[shard]
            info = header[key]
            if info["dtype"] not in ("I8", "F8_E8M0"):
                raise ValueError(f"unexpected dtype {info['dtype']} for {key}")
            start, end = info["data_offsets"]
            entries.append({"name": dest, "shape": info["shape"], "size": end - start,
                            "fd": shards[shard][0], "offset": shards[shard][1] + start})
            total += end - start

        meta = {"layer": str(layer), "experts": json.dumps(warm),
                "format": "mxfp4_e2m1_e8m0_32"}
        out_header = {e["name"]: {"dtype": "U8", "shape": e["shape"],
                                    "data_offsets": [sum(x["size"] for x in entries[:i]),
                                                     sum(x["size"] for x in entries[:i + 1])]}
                      for i, e in enumerate(entries)}
        out_header["__metadata__"] = meta
        encoded = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((-len(encoded)) % 8)

        os.makedirs(out_dir, exist_ok=True)
        final = os.path.join(out_dir, f"cold-L{layer:02d}.safetensors")
        tmp = final + ".tmp"
        digest = hashlib.sha256()
        with open(tmp, "wb", buffering=0) as f:
            prefix = struct.pack("<Q", len(encoded)) + encoded
            f.write(prefix)
            digest.update(prefix)
            for e in entries:
                remaining, offset = e["size"], e["offset"]
                while remaining:
                    chunk = os.pread(e["fd"], min(8 * 1024 * 1024, remaining), offset)
                    if not chunk:
                        raise IOError(f"short read for tensor {e['name']}")
                    f.write(chunk)
                    digest.update(chunk)
                    offset += len(chunk)
                    remaining -= len(chunk)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        return layer, len(prefix) + total, digest.hexdigest(), len(warm), os.path.basename(final)
    except BaseException:
        try:
            os.unlink(os.path.join(out_dir, f"cold-L{layer:02d}.safetensors.tmp"))
        except FileNotFoundError:
            pass
        raise
    finally:
        for fd, _, _ in shards.values():
            os.close(fd)


def main():
    global_args = parse_args()
    if global_args.jobs < 1:
        raise ValueError("--jobs must be positive")
    os.makedirs(global_args.out, exist_ok=True)
    if shutil.disk_usage(global_args.out).free < 200_000_000_000:
        raise OSError(f"insufficient free space in {global_args.out}; need 200e9 bytes")
    with open(global_args.placement, encoding="utf-8") as f:
        placement = json.load(f)
    weight_map = open_index(global_args.orig)
    layers = layer_ids(global_args.layers)
    jobs = [(L, placement["layers"][str(L)]["warm"], global_args.orig, weight_map, global_args.out) for L in layers]
    manifest = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=global_args.jobs) as pool:
        for L, size, sha, slots, filename in pool.map(pack_layer, jobs):
            manifest[str(L)] = {"n_slots": slots, "bytes": size, "sha256": sha}
    with open(os.path.join(global_args.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(global_args.out, "SHA256SUMS"), "w", encoding="utf-8") as f:
        for L in layers:
            filename = f"cold-L{L:02d}.safetensors"
            f.write(f"{manifest[str(L)]['sha256']}  {filename}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"pack_cold_fp4: {exc}", file=sys.stderr)
        sys.exit(1)
