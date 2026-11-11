#!/usr/bin/env python3
"""pair_pack_disk.py -- expert packs live on the drive of the box that serves them.

    pair_pack_disk.py <pair> [--quiet]

Owner rule (2026-11-02): "gx10 weights go on the 4TB NVMe of the GX10s, gpu0/gpu1 weights go on the
4TB drive of the P7." A pack served over the network makes every decode step wait for a filesystem,
and on this hardware the GX boxes mount network storage from P7 over NFS, so a
pack under /mnt is P7's disk being read across the wire by the GX half.

Checked against the spec the pair actually runs (containers/<pair>-<side>.json), so this check can
not drift from what the containers mount:

  p7  no mount may name /mnt -- the 12 TB drive is not the pair's, and the containers stop mounting
      it (a mount in the spec that names it is refused here). The spec must mount the pool itself,
      /models/dsv41:/models/dsv41: /models is a directory on the 1.9 TB root disk and the 4 TB NVMe
      is a submount inside it, so a /models:/models bind exposes the pool only by accident and a
      string comparison cannot tell whether the bytes are on the pool. The check therefore compares
      st_dev of the hot pack with st_dev of /models/dsv41 -- the same device, not the same prefix.
      The pack, the placement profile its PACK_MANIFEST.json names, and every symlink inside those
      must also resolve under /models/dsv41, on a local filesystem. The manifest's `overlay`
      is provenance (the tree the pack was built from, not read by a server), so it is printed but
      not walked: a pack file that still points into it is caught by the link check, which is exact,
      where walking the overlay would flag files nothing loads.

  gx  /cold must be the cold pack on the GX's own root disk: same st_dev as the GX home directory,
      not a symlink that resolves into /mnt or /net, not a network filesystem.

Prints PACK_DISK lines, then PACK_DISK_OK, or PACK_DISK_FAIL side=.. what=.. path=.. reason=..
Exit 0 clean / 3 refused / 2 usage. PAIR_PACK_DISK=0 is for stub harnesses only (no packs to see).
"""
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pair_containers as pc  # config parser and spec loader: one source of truth

P7_ALLOW = os.environ.get("PAIR_PACK_P7_ROOT", "/models/dsv41")
# Which hot half is the client of the pack the GX box serves is a property of THE PAIR, not of this
# file: the cold half names it through its PACK_MANIFEST placement_sha256, pair_ckpt_xcheck compares
# that against /cold, and the caller passes the named half in. A hard-coded exl3-p7hot-4460-hc1 here
# silently validated the PREVIOUS pack's placement refs after the 4780p8 switch (2026-11-03), so an
# unnamed hot half is refused rather than checked against a stale one.
P7_PACK = os.environ.get("PAIR_PACK_P7", "")
# Filesystems that put the bytes somewhere other than the box's own disk. virtiofs is a VM's local
# disk handed through, so it is not in the list; an NFS mount of a local disk is.
REMOTE_FS = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "9p", "sshfs", "fuse.sshfs", "ceph", "glusterfs",
             "lustre", "gpfs", "afs", "ncpfs", "vboxsf"}

# One round trip per path: realpath, the filesystem and the device it lands on, and every symlink
# below it (a link is how a pack names a file that lives somewhere else -- if that somewhere is NFS,
# the pack is not local). find -type l does not follow links, so the walk stays inside the tree.
# st_dev is what tells "my pack is on the pool" apart from "the pool happens to be bind-mounted
# inside a directory I mounted": on the P7 /models is the root NVMe and the pool is a submount.
PROBE = r"""
p=$1
[ -e "$p" ] || { printf 'MISSING %s\n' "$p"; exit 0; }
r=$(readlink -f -- "$p") || { printf 'REALFAIL %s\n' "$p"; exit 0; }
printf 'REAL %s\n' "$r"
ft=$(findmnt -T "$r" -no FSTYPE 2>/dev/null | head -1)
[ -n "$ft" ] || ft=$(stat -f -c %T "$r" 2>/dev/null)
printf 'FSTYPE %s\n' "${ft:-unknown}"
printf 'DEV %s\n' "$(stat -c %d -- "$r" 2>/dev/null || echo unknown)"
if [ -d "$p" ]; then
  find "$p" -type l -exec sh -c 'printf "LINK %s\t%s\n" "$1" "$(readlink -f -- "$1" 2>/dev/null || echo MISSING)"' _ {} \; 2>/dev/null
fi
"""


def ssh_argv():
    if os.environ.get("PAIR_DRY") == "1" and os.environ.get("PAIR_DRY_SSH"):
        return [os.environ["PAIR_DRY_SSH"]]
    return shlex.split(os.environ.get("PAIR_SSH") or "ssh")


def probe(host, path):
    """-> {real, fstype, dev, links: [(name, target)]} or {missing|real_failed: path}."""
    if host is None:
        cmd = ["sh", "-c", PROBE, "_", path]
    else:
        cmd = ssh_argv() + pc.SSH_OPTS + [host, "sh -c " + shlex.quote(PROBE) + " _ " + shlex.quote(path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = {"real": None, "fstype": "unknown", "dev": None, "links": [], "missing": None,
           "real_failed": None, "rc": p.returncode, "err": (p.stderr or "").strip()[:300]}
    for line in p.stdout.splitlines():
        if line.startswith("REAL "):
            out["real"] = line[5:]
        elif line.startswith("FSTYPE "):
            out["fstype"] = line[7:]
        elif line.startswith("DEV "):
            out["dev"] = line[4:] if line[4:] not in ("", "unknown") else None
        elif line.startswith("LINK "):
            name, _, target = line[5:].partition("\t")
            out["links"].append((name, target))
        elif line.startswith("MISSING "):
            out["missing"] = line[8:]
        elif line.startswith("REALFAIL "):
            out["real_failed"] = line[9:]
    if out["rc"] != 0 and not p.stdout:
        out["real_failed"] = path
    return out


def remote_fs(fstype):
    f = (fstype or "").lower()
    return f in REMOTE_FS or f.startswith("nfs") or f.startswith("fuse.sshfs") or f.startswith("cifs")


def under(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


def manifest_refs(pack):
    """(placement, overlay) from the pack's manifest, both absolute paths or None.

    `placement` is read at serving time (the client resolves CLIENT_PLACEMENT through it), so it is a
    pack path like any other. `overlay` is provenance -- the tree the pack was built from -- and is
    not read by a server; a pack file that still points into it is caught by the link check instead,
    which is exact, where walking the whole overlay would flag files nothing loads.
    """
    man = os.path.join(pack, "PACK_MANIFEST.json")
    try:
        d = json.load(open(man))
    except (OSError, ValueError):
        return None, None
    placement = d.get("placement") if isinstance(d.get("placement"), str) else None
    # A relative placement is an in-pack name (placement-strict 4a4efacf writes it that way and
    # qmesh/placement.py resolves it against the pack dir first): resolve it the same way here.
    if placement and not os.path.isabs(placement):
        placement = os.path.join(pack, placement)
    overlay = d.get("overlay") if isinstance(d.get("overlay"), str) else None
    return (placement or None), (overlay or None)


def spec_mounts(cfg, side):
    """[(src, dst)] from the container spec, read raw.

    The spec loader in pair_containers.py refuses a /mnt mount by exiting 2, which is the right
    behaviour for plan/apply but would turn this checker's report into an abort: the smoke wants one
    PACK_DISK_FAIL line per bad mount and rc 3, so read the file here and let check_p7/check_gx name
    the mount. A missing or malformed spec is the loader's business, not this file's.
    """
    path = os.path.join(pc.SPEC_DIR, f"{cfg['pair']}-{side}.json")
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return []
    d = pc.resolve_mounts(d)
    return [(m[0], m[1]) for m in d.get("mounts") or []
            if isinstance(m, list) and len(m) >= 2]


def check_p7(cfg, fails, quiet):
    pool_src = None
    for src, dst in spec_mounts(cfg, "p7"):
        if src == "/mnt" or src.startswith("/mnt/") or dst == "/mnt" or dst.startswith("/mnt/"):
            fails.append(("p7", "mount", dst,
                          f"the pair must not mount the 12 TB drive ({src}); packs live on the "
                          f"serving box's own disk (owner rule 2026-11-02)"))
        if dst == P7_ALLOW:
            pool_src = src
    if pool_src is None:
        fails.append(("p7", "mount", P7_ALLOW, f"the spec mounts no {P7_ALLOW}: the hot pack is unreachable"))
    elif pool_src != P7_ALLOW:
        fails.append(("p7", "mount", P7_ALLOW,
                      f"mounted from {pool_src}, not from the pool itself (mount {P7_ALLOW}:{P7_ALLOW})"))

    poold = probe(None, P7_ALLOW)
    if poold["missing"] or poold["real_failed"] or not poold["real"]:
        fails.append(("p7", "pool", P7_ALLOW, f"cannot be resolved on P7 ({poold['err'] or 'missing'})"))
    elif remote_fs(poold["fstype"]):
        fails.append(("p7", "pool", P7_ALLOW, f"sits on {poold['fstype']}: the pool must be the P7's own NVMe"))
    pr = probe(None, P7_PACK)
    placement, overlay = manifest_refs(P7_PACK)
    roots = [("pack", P7_PACK)] + ([("placement", placement)] if placement else [])
    nlinks = 0
    for what, path in roots:
        r = pr if path == P7_PACK else probe(None, path)
        if r["missing"]:
            fails.append(("p7", what, path, "does not exist on P7"))
            continue
        if r["real_failed"] or not r["real"]:
            fails.append(("p7", what, path, f"cannot be resolved ({r['err'] or 'readlink failed'})"))
            continue
        if not under(r["real"], P7_ALLOW):
            fails.append(("p7", what, path,
                          f"resolves to {r['real']}, outside {P7_ALLOW} (packs live on the P7 4 TB drive)"))
        if poold["dev"] and r["dev"] and r["dev"] != poold["dev"]:
            fails.append(("p7", what, path,
                          f"is on device {r['dev']}, but {P7_ALLOW} is device {poold['dev']}: the path is "
                          f"under the pool only as a prefix, not on the 4 TB NVMe"))
        if remote_fs(r["fstype"]):
            fails.append(("p7", what, path, f"sits on {r['fstype']}: a pack must not be served over the network"))
        for name, target in r["links"]:
            nlinks += 1
            if target == "MISSING":
                fails.append(("p7", "link", name, "is a dangling link (its target does not exist)"))
            elif not under(target, P7_ALLOW):
                fails.append(("p7", "link", name,
                              f"points at {target}, outside {P7_ALLOW} (a pack's bytes stay on the drive)"))
    if not quiet:
        print(f"PACK_DISK p7 pack={P7_PACK} real={pr['real'] or '-'} fstype={pr['fstype']} "
              f"dev={pr['dev'] or '-'} pool={P7_ALLOW} pool_dev={poold['dev'] or '-'} "
              f"placement={placement or '-'} overlay={overlay or '-'} links={nlinks}")


def p7_placement_sha(pack):
    """> placement_sha256 the P7 client half declares ("" when absent); it is a local file."""
    try:
        d = json.load(open(os.path.join(pack, "PACK_MANIFEST.json")))
    except (OSError, ValueError):
        return ""
    v = d.get("placement_sha256")
    return v if isinstance(v, str) else ""


def cold_placement_sha(host, pack):
    """placement_sha256 the pack's own manifest declares ("" when it cannot be read).

    A pack switch is a config change, and the row that records it has to carry the sha the pack says it
    holds -- otherwise two arms at different packs are indistinguishable in the log.
    """
    cmd = ssh_argv() + pc.SSH_OPTS + [host, "python3 -c "
           + shlex.quote("import json,sys;print(json.load(open(sys.argv[1] + '/PACK_MANIFEST.json'))"
                         ".get('placement_sha256',''))") + " " + shlex.quote(pack)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def check_gx(cfg, fails, quiet):
    spec_src = None
    for s, dst in spec_mounts(cfg, "gx"):
        if dst == "/cold":
            spec_src = s
        elif s == "/mnt" or s.startswith("/mnt/") or dst == "/mnt" or dst.startswith("/mnt/"):
            fails.append(("gx", "mount", dst,
                          f"the GX container must not mount /mnt ({s}): that is P7's disk over NFS"))
    if spec_src is None:
        fails.append(("gx", "mount", "/cold", "the spec mounts no /cold: the cold pack is unreachable"))
        return
    # The pack to judge is the one behind /cold *now*, not the one the spec was written with: a run that
    # switched packs (k=4/5/6 variants, 2026-11-03) is serving the live mount, and a checker reading the
    # spec would either bless a pack nothing serves or refuse a pack that is fine. Both cases must show up
    # as a labelled config change here -- printed even under --quiet, because which pack is served is the
    # label every number measured under it has to carry, not chatter.
    lv = pc.live(cfg, "gx") or {}
    live_src = (lv.get("mounts") or {}).get("/cold", "")
    src = live_src or spec_src
    if live_src and spec_src and live_src != spec_src:
        print(f"PACK_DISK_CONFIG gx /cold spec={spec_src} live={live_src} (pack switch: judging live; "
              f"sync containers/{cfg['pair']}-gx.json so plan/apply do not recreate onto the old pack)")
    elif not live_src:
        print(f"PACK_DISK_CONFIG gx /cold spec={spec_src} live=absent (no container with a /cold mount; "
              f"judging the spec)")
    r = probe(cfg["gx_host"], src)
    if r["missing"]:
        fails.append(("gx", "pack", src, f"does not exist on {cfg['gx_host']}"))
        return
    if r["real_failed"] or not r["real"]:
        fails.append(("gx", "pack", src, f"cannot be resolved on {cfg['gx_host']} ({r['err'] or 'readlink failed'})"))
        return
    # Same device as the box's own disk, not merely a path that does not start with /mnt: a /cold
    # that is a symlink into the NFS mount passes a prefix test only until the link is resolved, and
    # `under(real, "/mnt")` above depends on the GX having /mnt mounted at all.
    home = os.environ.get("PAIR_PACK_GX_HOME", os.path.expanduser("~"))
    hd = probe(cfg["gx_host"], home)
    if hd["missing"] or hd["real_failed"] or not hd["real"]:
        fails.append(("gx", "home", home,
                      f"cannot be resolved on {cfg['gx_host']} ({hd['err'] or 'missing'}); cannot prove "
                      f"the pack sits on the box's own disk"))
    elif hd["dev"] and r["dev"] and r["dev"] != hd["dev"]:
        fails.append(("gx", "pack", src,
                      f"is on device {r['dev']}, but the box's own disk ({home}) is device {hd['dev']}: "
                      f"the cold pack must live on the GX's root NVMe (owner rule 2026-11-02)"))
    elif hd["dev"] and not r["dev"]:
        fails.append(("gx", "pack", src, "device cannot be read, so locality cannot be proven"))
    for bad in ("/mnt", "/net"):
        if under(r["real"], bad):
            fails.append(("gx", "pack", src,
                          f"resolves to {r['real']} under {bad} (mounted from P7 over NFS; packs live on "
                          f"the GX box's own disk)"))
    if remote_fs(r["fstype"]):
        fails.append(("gx", "pack", src, f"sits on {r['fstype']}: a pack must not be served over the network"))
    for name, target in r["links"]:
        if target != "MISSING":
            for bad in ("/mnt", "/net"):
                if under(target, bad):
                    fails.append(("gx", "link", name, f"points at {target} under {bad} (network disk)"))
    cold_sha = cold_placement_sha(cfg["gx_host"], r["real"])
    # One checkpoint split in two: the P7 client half has to declare the same placement_sha256 the cold
    # pack declares, or the pair is serving two halves of different packs. Reading it from the live /cold
    # is what makes a pack switch show up here instead of in a decode run that looks merely slow.
    hot_sha = p7_placement_sha(P7_PACK)
    if cold_sha and hot_sha and cold_sha != hot_sha:
        fails.append(("gx", "placement", src,
                      f"declares placement_sha256={cold_sha[:16]}, but the P7 client half ({P7_PACK}) "
                      f"declares {hot_sha[:16]}: the two halves are different packs"))
    if not quiet:
        print(f"PACK_DISK gx pack=/cold src={src} real={r['real']} fstype={r['fstype']} "
              f"dev={r['dev'] or '-'} root_dev={hd['dev'] or '-'} links={len(r['links'])} "
              f"placement_sha={(cold_sha or '-')[:16]}")


def main(argv):
    args = [a for a in argv[1:] if a != "--quiet"]
    quiet = "--quiet" in argv[1:]
    if len(args) != 1:
        print("usage: pair_pack_disk.py <pair> [--quiet]", file=sys.stderr)
        return 2
    if os.environ.get("PAIR_PACK_DISK") == "0":
        print(f"PACK_DISK_SKIPPED pair={args[0]} PAIR_PACK_DISK=0")
        return 0
    if not P7_PACK:
        print("PACK_DISK_USAGE pair=%s PAIR_PACK_P7 is not set: this check validates the P7 hot half that is"
              " the client of the GX box's /cold pack, and only the caller knows which one that is --"
              " pair_smoke.sh exports its own PAIR_PACK_P7 (pair_ckpt_resolve auto is a logged cross-check,"
              " never the source)" % args[0], file=sys.stderr)
        return 2
    cfg = pc.load_conf(args[0])
    fails = []
    check_p7(cfg, fails, quiet)
    check_gx(cfg, fails, quiet)
    for side, what, path, reason in fails:
        print(f"PACK_DISK_FAIL side={side} what={what} path={path} reason={reason}")
    if fails:
        print(f"PACK_DISK_REFUSED pair={cfg['pair']} fails={len(fails)}")
        return 3
    print(f"PACK_DISK_OK pair={cfg['pair']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
