#!/usr/bin/env python3
"""Battery for pair_pack_disk.py -- expert packs live on the drive of the box that serves them.

Owner rule 2026-11-02: P7 hot packs on the P7's own drive, GX cold packs on the GX's own disk, never
across the network. The checker is what refuses a release whose pack does not satisfy it, so its two
refusals (a mount that names the 12 TB drive, and a path or symlink that leaves the allowed tree) are
pinned here: one unit case per rule with probe stubbed, then one end-to-end case over a real
directory that has a link pointing outside the allowed root.

No container, GPU, ssh or queue is touched.
"""
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PAIR_QUEUE_ROOT", "/tmp/packdiskq")
os.environ["PAIR_PACK_GX_HOME"] = "/home/test"
_s = importlib.util.spec_from_file_location("pd", os.path.join(HERE, "pair_pack_disk.py"))
pd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pd)

fails = 0


def check(name, cond, detail=""):
    global fails
    print(("OK   " if cond else "FAIL ") + name + (" | " + detail if detail else ""))
    if not cond:
        fails += 1


def probe_d(real="/models/dsv41/pack", fstype="ext4", links=(), missing=None, failed=None,
            dev="66306"):
    return {"real": real, "fstype": fstype, "dev": dev, "links": list(links), "missing": missing,
            "real_failed": failed, "rc": 0, "err": ""}


POOL_DEV = "66306"        # /dev/nvme1n1p1, the 4 TB pool, mounted at /models/dsv41
ROOT_DEV = "66305"        # /dev/nvme0n1p2, the 1.9 TB root disk that holds the /models directory


def p7_spec(mounts):
    return {"mounts": mounts}


def gx_spec(mounts):
    return {"mounts": mounts}


GOOD_P7 = p7_spec([["__PAIR_ROOT__", "/pair"], ["/models/dsv41", "/models/dsv41"],
                   ["/usr/local/cuda", "/opt/cuda"]])
GOOD_GX = gx_spec([["__PAIR_ROOT__", "/pair"], ["/home/test/cold-pack-a", "/cold"]])


def run(p7spec, p7probe, gxspec=None, gxprobe=None, placement=None, poolprobe=None, homeprobe=None,
        live="spec", hot_sha="", cold_sha=""):
    """-> (fails, printed) with spec, live mounts, manifests and probe stubbed.

    `live` is what the running GX container mounts at /cold: "spec" (same as the spec, the common case),
    a path string (a switched pack), or None (no container at all).
    """
    pd.spec_mounts = lambda cfg, side: [tuple(m) for m in
                                        (p7spec if side == "p7" else (gxspec or GOOD_GX))["mounts"]]
    pd.manifest_refs = lambda pack: (placement, None)
    pd.p7_placement_sha = lambda pack: hot_sha
    pd.cold_placement_sha = lambda host, pack: cold_sha
    spec_cold = next((s for s, d in pd.spec_mounts({"pair": "dev"}, "gx") if d == "/cold"), "")
    if live == "spec":
        live_cold = spec_cold
    else:
        live_cold = live or ""
    # pc.live() inspects the real box; the live /cold source is the one thing check_gx needs from it
    pd.pc.live = lambda cfg, side, _lc=live_cold: (
        None if _lc is None else {"running": True, "mounts": {"/cold": _lc} if _lc else {}})

    def fake_probe(host, path):
        if host is None:
            if path == pd.P7_ALLOW:
                return poolprobe or probe_d(real=pd.P7_ALLOW, dev=POOL_DEV)
            return p7probe
        if path == "/home/test":
            return homeprobe or probe_d(real="/home/test", dev=ROOT_DEV)
        return gxprobe or probe_d("/home/test/cold-pack-a", dev=ROOT_DEV)
    pd.probe = fake_probe
    out = []
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        pd.check_p7({"pair": "dev"}, out, True)
        pd.check_gx({"pair": "dev", "gx_host": "gx-dev.invalid"}, out, True)
    return out, buf.getvalue()


def reasons(fs):
    return " | ".join(f"{s}:{w}:{p}:{r}" for s, w, p, r in fs)


# 1. the spec must not hand the pair the 12 TB drive
fs, _ = run(p7_spec(GOOD_P7["mounts"] + [["/mnt/network/models", "/mnt/network/models"]]),
            probe_d(), placement="/models/dsv41/placement.json")
check("a P7 mount naming the 12 TB drive is refused",
      any(w == "mount" and p == "/mnt/network/models" and "12 TB" in r for s, w, p, r in fs), reasons(fs))

# 2. the spec must mount the pool itself. /models is a directory on the 1.9 TB root disk with the
#    4 TB NVMe mounted inside it, so a /models:/models bind exposes the pool only by accident and a
#    string test cannot say which device a pack landed on.
fs, _ = run(p7_spec([["__PAIR_ROOT__", "/pair"], ["/models", "/models"]]), probe_d())
check("a spec that mounts the /models directory instead of the pool is refused",
      any(w == "mount" and p == "/models/dsv41" for s, w, p, r in fs), reasons(fs))
fs, _ = run(p7_spec([["__PAIR_ROOT__", "/pair"], ["/mnt/network/models", "/models/dsv41"]]), probe_d())
check("a pool mounted from another disk is refused",
      any(w == "mount" and p == "/models/dsv41" and "not from the pool itself" in r for s, w, p, r in fs),
      reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), placement="/models/dsv41/placement.json")
check("the pool mounted from itself, a pack on its device, passes", not fs, reasons(fs))

# 3. st_dev, not a path prefix: a pack that is under /models/dsv41 but on the root disk is not on the pool
fs, _ = run(GOOD_P7, probe_d(real="/models/dsv41/pack", dev=ROOT_DEV))
check("a pack under the pool path but on another device is refused",
      any(w == "pack" and "is on device" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(dev=ROOT_DEV), placement="/models/dsv41/placement.json",
            poolprobe=probe_d(real=pd.P7_ALLOW, dev=ROOT_DEV))
check("a pool and a pack on the same device pass even when it is not the 4 TB one",
      not fs, reasons(fs))

# 4. the pack itself, and everything a link inside it resolves to, stays under the P7 root
fs, _ = run(GOOD_P7, probe_d(real="/mnt/network/pack"))
check("a pack that resolves outside the P7 drive is refused",
      any(w == "pack" and "/mnt/network/pack" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7,
            probe_d(links=[("/models/dsv41/pack/model-00041.safetensors",
                            "/mnt/network/models/model-00041.safetensors")]))
check("a pack's link into the 12 TB drive is refused",
      any(w == "link" and "/mnt/network" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(links=[("/models/dsv41/pack/x.safetensors", "/models/dsv41/serve/x.safetensors")]))
check("a link that stays on the P7 drive passes", not fs, reasons(fs))

# 5. a pack on a network filesystem is refused even when its path looks local
fs, _ = run(GOOD_P7, probe_d(fstype="nfs4"))
check("a pack on an NFS mount is refused", any("sits on nfs4" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), poolprobe=probe_d(real=pd.P7_ALLOW, fstype="nfs4", dev=POOL_DEV))
check("a pool on NFS is refused", any(w == "pool" and "own NVMe" in r for s, w, p, r in fs), reasons(fs))

# 6. the manifest's placement profile is read at serving time and is a pack path too
fs, _ = run(GOOD_P7, probe_d(real="/mnt/x"), placement="/mnt/network/placement.json")
check("a placement profile outside the P7 drive is refused",
      any(w == "placement" for s, w, p, r in fs), reasons(fs))

# 7. the GX half: /cold on the box's own disk, never /mnt (P7 over NFS) and never NFS at all
fs, _ = run(GOOD_P7, probe_d(),
            gx_spec(GOOD_GX["mounts"] + [["/mnt/network/models", "/mnt/network/models"]]))
check("a GX mount naming /mnt is refused", any(s == "gx" and w == "mount" for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(),
            gxprobe=probe_d(real="/mnt/network/cold-pack", dev=ROOT_DEV))
check("a GX pack under /mnt is refused",
      any(s == "gx" and w == "pack" and "under /mnt" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), gxprobe=probe_d(fstype="nfs"))
check("a GX pack on NFS is refused", any(s == "gx" and "sits on nfs" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), gxprobe=probe_d(missing="/home/x/pack"))
check("a GX pack that does not exist is refused", any(s == "gx" and w == "pack" for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d())
check("a local GX pack with no links passes", not fs, reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), gxprobe=probe_d(links=[("/home/x/pack/a", "/net/host/pack/a")]))
check("a GX link into /net is refused", any(s == "gx" and w == "link" for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(),
            gxprobe=probe_d("/home/test/cold-pack-a", dev=POOL_DEV))
check("a GX pack on another device than the box's own disk is refused",
      any(s == "gx" and w == "pack" and "root NVMe" in r for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), homeprobe=probe_d(missing="/home/test"))
check("a GX whose own disk cannot be probed is refused",
      any(s == "gx" and w == "home" for s, w, p, r in fs), reasons(fs))

# 6b. a pack switch is a labelled config change, never a silent one and never a refusal of a valid live
#     pack: the checker must judge what the container mounts now, not what the spec was written with.
fs, shown = run(GOOD_P7, probe_d(), live="/home/test/cold-pack-b",
                gxprobe=probe_d("/home/test/cold-pack-b", dev=ROOT_DEV))
check("a /cold that is not the spec's pack is reported as a labelled config change",
      "PACK_DISK_CONFIG gx /cold spec=/home/test/cold-pack-a "
      "live=/home/test/cold-pack-b" in shown, shown)
check("...and the live pack is the one judged, so the switched pack passes", not fs, reasons(fs))
fs, shown = run(GOOD_P7, probe_d(), live="/home/test/cold-pack-b",
                gxprobe=probe_d("/home/test/cold-pack-b", fstype="nfs"))
check("...so a switched pack served over NFS is still refused",
      any(s == "gx" and "sits on nfs" in r for s, w, p, r in fs), reasons(fs))
fs, shown = run(GOOD_P7, probe_d(), live=None)
check("no container with a /cold mount is labelled, and the spec pack is judged",
      "live=absent" in shown and not fs, shown + " / " + reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), hot_sha="a" * 64, cold_sha="b" * 64)
check("two halves that declare different placement_sha256 are refused",
      any(s == "gx" and w == "placement" for s, w, p, r in fs), reasons(fs))
fs, _ = run(GOOD_P7, probe_d(), hot_sha="a" * 64, cold_sha="a" * 64)
check("the same placement_sha256 on both halves passes", not fs, reasons(fs))

# 7. end to end over a real tree: the shipped hot pack carries links, and a link into the 12 TB drive
#    (which is what the 4460 pack did through its overlay before 2026-11-02) must be caught by the
#    filesystem walk, not only by a stubbed probe.
W = tempfile.mkdtemp(prefix="packdisk.")
try:
    old_allow, old_pack = pd.P7_ALLOW, pd.P7_PACK
    root = os.path.join(W, "models", "dsv41")
    pack = os.path.join(root, "pack")
    outside = os.path.join(W, "data12", "models", "exl3")
    for d in (pack, outside):
        os.makedirs(d)
    open(os.path.join(pack, "model-00001-of-00041.safetensors"), "w").write("bytes")
    open(os.path.join(outside, "model-00041-of-00041.safetensors"), "w").write("bytes")
    os.symlink(os.path.join(outside, "model-00041-of-00041.safetensors"),
               os.path.join(pack, "model-00041-of-00041.safetensors"))
    import json
    open(os.path.join(pack, "PACK_MANIFEST.json"), "w").write(json.dumps(
        {"placement": os.path.join(root, "packs", "placement.json"), "overlay": outside}))
    os.makedirs(os.path.join(root, "packs"))
    open(os.path.join(root, "packs", "placement.json"), "w").write("{}")
    pd.P7_ALLOW, pd.P7_PACK = root, pack
    pd.pc.spec = lambda cfg, side: GOOD_P7 if side == "p7" else GOOD_GX
    pd.spec_mounts = lambda cfg, side: ([["__PAIR_ROOT__", "/pair"], [root, root]]
                                        if side == "p7" else
                                        [["__PAIR_ROOT__", "/pair"],
                                         ["/home/test/cold-pack-a", "/cold"]])
    pd.probe = lambda host, path: probe_d("/home/test/cold-pack-a")
    real_probe = pd.probe
    # p7 paths are read locally by the checker itself, so restore the real probe for host None only.
    sys.path.insert(0, HERE)
    import pair_pack_disk as _real
    pd.probe = lambda host, path: _real.probe(None, path) if host is None else real_probe(None, path)
    fs = []
    with contextlib.redirect_stdout(io.StringIO()):
        pd.check_p7({"pair": "dev"}, fs, True)
    check("a real symlink out of the pack is refused by the walk",
          any(w == "link" and os.path.join(outside, "model-00041") in r for s, w, p, r in fs), reasons(fs))
    os.unlink(os.path.join(pack, "model-00041-of-00041.safetensors"))
    open(os.path.join(pack, "model-00041-of-00041.safetensors"), "w").write("bytes")
    fs = []
    with contextlib.redirect_stdout(io.StringIO()):
        pd.check_p7({"pair": "dev"}, fs, True)
    check("the same pack with its bytes local passes", not fs, reasons(fs))
    pd.P7_ALLOW, pd.P7_PACK = old_allow, old_pack
finally:
    shutil.rmtree(W, ignore_errors=True)

# 8. the hot half is not a constant of the checker: a pair whose cold half is switched (4460 -> 4780p8,
#    2026-11-03) must not be validated against the previous pack's placement refs by default.
_old_pack, _old_env = pd.P7_PACK, os.environ.pop("PAIR_PACK_P7", None)
pd.P7_PACK = ""
_err = io.StringIO()
_rc = 0
with contextlib.redirect_stderr(_err), contextlib.redirect_stdout(io.StringIO()):
    _rc = pd.main(["pair_pack_disk.py", "dev"])
check("an unnamed hot half is refused, not checked against the previous pack",
      _rc == 2 and "PACK_DISK_USAGE" in _err.getvalue() and "PAIR_PACK_P7" in _err.getvalue(),
      "rc=%s err=%s" % (_rc, _err.getvalue().strip()[:120]))
if _old_env is not None:
    os.environ["PAIR_PACK_P7"] = _old_env
pd.P7_PACK = _old_pack

print("\nRESULT %s fails=%d" % ("PASS" if not fails else "FAIL", fails))
sys.exit(1 if fails else 0)
