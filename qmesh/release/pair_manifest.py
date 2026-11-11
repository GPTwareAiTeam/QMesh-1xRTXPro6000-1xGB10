#!/usr/bin/env python3
"""pair_manifest.py -- write the one file that says what a pair is supposed to be serving.

    pair_manifest.py write <pair> <sha> [--ref F] [--log F] [--by S]
    pair_manifest.py show <pair>

The manifest is written *last* by an update, after the release is cut, both halves are built from it,
the symlinks are flipped and the smoke row passed. pair_check.sh (the queue guard) reads it and
pair_sync_check.py compares it against what the two boxes and two containers really report; nothing
else is allowed to decide whether a pair is in sync.

Atomic: written to a temp file in the same directory and renamed, so a reader never sees half a file.
"""
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("PAIR_ART", os.path.expanduser("~/recipe-artifacts/gpu0-gx1"))
STATE = os.environ.get("PAIR_STATE", os.path.join(ART, "pairs"))
LIB = "libqmesh_expert_seam"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def die(msg):
    print(f"pair_manifest: {msg}", file=sys.stderr)
    sys.exit(2)


def load_conf(pair):
    conf = os.environ.get("PAIR_CONF", os.path.join(HERE, "pairs.conf"))
    keys = ["pair", "p7ct", "gpu", "seam_dev", "seam_gid", "gx_gid", "seam_peer", "gx_host", "gxct",
            "root", "port", "buckets", "p7_arch", "gx_arch"]
    for line in open(conf):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if f[0] == pair:
            d = dict(zip(keys, f))
            d["root"] = os.environ.get("PAIR_ROOT_OVERRIDE") or d["root"]
            return d
    die(f"pair {pair} not in {conf}")


def read_remote(host, path):
    if os.environ.get("PAIR_DRY") == "1" and os.environ.get("PAIR_DRY_SSH"):
        cmd = [os.environ["PAIR_DRY_SSH"]]
    else:
        cmd = ["ssh"]
    p = subprocess.run(cmd + SSH_OPTS + [host, f"cat {path} 2>/dev/null"],
                       capture_output=True, text=True)
    return p.stdout.strip()


def read_local(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def json_or_empty(text):
    try:
        return json.loads(text) if text else {}
    except ValueError:
        return {}


def stamp_json(text):
    """A .so.stamp / BUILD_STAMP.json: empty when absent or unreadable."""
    return json_or_empty(text)


def write(pair, sha, ref_file="", log="", by=""):
    cfg = load_conf(pair)
    root = cfg["root"]
    rel = os.path.join(root, "releases", sha)
    try:
        rj = json.load(open(os.path.join(rel, "RELEASE.json")))
    except Exception as exc:                                  # noqa: BLE001
        die(f"cannot read {rel}/RELEASE.json: {exc}")
    arch_p7 = "x86_64" if cfg["p7_arch"] == "sm_120" else "aarch64"
    arch_gx = "x86_64" if cfg["gx_arch"] == "sm_120" else "aarch64"

    lib_p7 = stamp_json(read_local(os.path.join(rel, "src", "qmesh", "native", "bin",
                                                f"{LIB}.{arch_p7}.so.stamp")))
    lib_gx = stamp_json(read_remote(cfg["gx_host"], f"{rel}/src/qmesh/native/bin/"
                                   f"{LIB}.{arch_gx}.so.stamp"))
    ext_p7 = stamp_json(read_local(os.path.join(rel, "ext", "BUILD_STAMP.json")))
    ext_gx = stamp_json(read_remote(cfg["gx_host"], f"{rel}/ext/BUILD_STAMP.json"))

    for name, want_arch, got in (("p7 lib", cfg["p7_arch"], lib_p7), ("gx lib", cfg["gx_arch"], lib_gx)):
        if not got:
            die(f"{name} has no build stamp beside it; refuse to write a manifest for {sha[:8]}")
        if got.get("source_sha") not in (sha, None):
            die(f"{name} was built from {str(got.get('source_sha'))[:8]}, not {sha[:8]}")
        if got.get("sm") not in (want_arch, None):
            die(f"{name} was built for {got.get('sm')}, want {want_arch}")
    for name, side, got in (("p7 ext", "p7", ext_p7), ("gx ext", "gx", ext_gx)):
        if not got:
            die(f"{name} has no BUILD_STAMP.json; refuse to write a manifest for {sha[:8]}")

    ref_ids = {}
    if ref_file and os.path.exists(ref_file):
        ref_ids = json_or_empty(read_local(ref_file)).get("rows", [])
    doc = {
        "format": 1,
        "pair": pair,
        "release_sha": sha,
        "build_id": rj.get("build_id"),
        "tree_digest": rj.get("digest"),
        "seam": rj.get("seam") or {},
        "branch": rj.get("branch"),
        "cut_at": rj.get("cut_at"),
        "lib": {arch_p7: {k: lib_p7.get(k) for k in ("sha256", "build_id", "source_sha", "sm")},
                arch_gx: {k: lib_gx.get(k) for k in ("sha256", "build_id", "source_sha", "sm")}},
        # "cx" on gx: cuda-exl3 is gitignored and built per checkout, so the manifest has to name the
        # kernel the release was built with -- a promotion compares releases by this, and without it a
        # release that never built its kernel would look like any other (pair_sync_check.py).
        "ext": {"p7": {k: ext_p7.get(k) for k in ("sha256", "build_id", "source_sha", "arch_list")},
                "gx": {k: ext_gx.get(k) for k in ("sha256", "build_id", "source_sha", "arch_list", "cx")}},
        "ref_ids": ref_ids,
        "p7": {"label": "P7", "container": cfg["p7ct"], "arch": arch_p7, "root": root,
               "sm": cfg["p7_arch"], "cuda_arch": "12.0" if cfg["p7_arch"] == "sm_120" else "12.1",
               "gpu": int(cfg["gpu"])},
        "gx": {"label": "GX", "container": cfg["gxct"], "arch": arch_gx, "root": root,
               "sm": cfg["gx_arch"], "cuda_arch": "12.0" if cfg["gx_arch"] == "sm_120" else "12.1",
               "host": cfg["gx_host"]},
        "seam_addr": {"peer": cfg["seam_peer"], "port": int(cfg["port"]),
                      "buckets": cfg["buckets"], "dev": cfg["seam_dev"],
                      "gid": int(cfg["seam_gid"]), "gx_gid": int(cfg["gx_gid"])},
        "updated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_by": by or os.environ.get("USER", "") + "@" + os.uname().nodename,
        "log": log,
    }
    os.makedirs(STATE, exist_ok=True)
    path = os.path.join(STATE, f"{pair}.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    print(f"MANIFEST {path} pair={pair} sha={sha[:12]} build={doc['build_id']}")
    return 0


def show(pair):
    path = os.path.join(STATE, f"{pair}.json")
    try:
        doc = json.load(open(path))
    except OSError:
        print(f"MANIFEST none {path}", file=sys.stderr)
        return 2
    print(json.dumps(doc, indent=2, sort_keys=True))
    return 0


def main():
    if len(sys.argv) < 3:
        die(__doc__)
    what = sys.argv[1]
    if what == "write":
        pair, sha = sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""
        args = sys.argv[3:]
        if not sha or sha.startswith("--"):
            die("write needs <pair> <sha>")
        def opt(name):
            return args[args.index(name) + 1] if name in args else ""
        return write(pair, sha, opt("--ref"), opt("--log"), opt("--by"))
    if what == "show":
        return show(sys.argv[2])
    die(f"unknown subcommand {what!r}")


if __name__ == "__main__":
    sys.exit(main())
