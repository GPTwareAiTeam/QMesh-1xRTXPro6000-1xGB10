#!/usr/bin/env python3
"""pair_sync_check.py -- is a pair's release actually the release it claims to be?

Two modes.

1) Manifest mode (the pair release system) -- read-only:

       pair_sync_check.py --manifest /path/to/pairs/dev.json
       pair_sync_check.py --manifest ... --quiet          # only the verdict lines
       pair_sync_check.py --manifest ... --no-server      # ignore the running server
       pair_sync_check.py --manifest ... --no-container   # ignore docker state

   For the P7 side (local) and the GX side (ssh to the manifest's gx host) it checks, in one shot:
     * the `current` symlink points at releases/<sha> and `extcache` at releases/<sha>/ext
     * every path of <sha> in `qmesh/` and `exllamav3/` hashes to the commit's blob (the box must
       serve the commit; it is not a git checkout, so this is the only way to know)
     * RELEASE.json agrees with the manifest (sha, build id, tree digest)
     * the native seam lib carries a build stamp for THIS release and its .so hash matches it
     * the torch extension cache is the per-container, per-release cache and carries a stamp for
       THIS release (source sha + arch list), i.e. it was not built from stale sources
     * the container mounts the pair root at /pair, /src/exllamav3 -> /pair/current/src,
       TORCH_EXTENSIONS_DIR=/pair/extcache, PYTHONPATH=/src/exllamav3
     * a running seam server reports the release's build id in its SEAM_SERVER_LOADED line
   Exit 0 = IN_SYNC, 1 = DRIFT, 2 = usage/setup error.

2) Legacy tree mode (drift survey): compare a container's host tree with a reference worktree at a
   given commit for a fixed pair list. Flags --v2/--stg/--pair, same as before.

Why sources and stamps instead of .so hashes: the same C++ does not build byte-identically on
aarch64 and x86_64 (or across toolchains), so a .so hash says nothing across boxes. What must match
is the *source* every piece was built from, plus a stamp that says so; the .so hash is only compared
on the box that built it (against the stamp recorded there).
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys

REPO = os.environ.get("PAIR_REPO", os.path.expanduser("~/recipe-gpu0-gx1-v2"))
PATHS = ["qmesh", "exllamav3"]

# ---- legacy tree mode tables --------------------------------------------------------------------
WORKTREES = {
    "v2": os.environ.get("PAIR_LEGACY_V2_TREE", REPO),
    "stg": os.environ.get("PAIR_LEGACY_STG_TREE", os.path.expanduser("~/recipe-seam-commit")),
}
PAIRS = [
    (name, os.environ.get(f"PAIR_LEGACY_{name.upper()}_HOST", ""), box, container,
     os.environ.get(f"PAIR_LEGACY_{name.upper()}_TREE", REPO), "v2")
    for name, box, container in (("prod", "GX1", "seam-gx1v2"), ("dev", "GX2", "seam-gx2"))
]


def ssh_argv():
    """PAIR_DRY_SSH swaps the transport in the stub-ssh dry run; nothing else changes here."""
    if os.environ.get("PAIR_DRY") == "1" and os.environ.get("PAIR_DRY_SSH"):
        return [os.environ["PAIR_DRY_SSH"]]
    return ["ssh"]


def run(cmd, host=None, stdin=None, timeout=600, binary=False):
    if host:
        if isinstance(cmd, list):
            cmd = " ".join(shlex.quote(c) for c in cmd)
        cmd = ssh_argv() + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd]
    p = subprocess.run(cmd, input=(stdin.encode() if isinstance(stdin, str) else stdin),
                       capture_output=True, timeout=timeout)
    if binary:
        return p.returncode, p.stdout, p.stderr.decode(errors="replace")
    return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")


def md5map(text):
    out = {}
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 32:
            out[parts[1].strip()] = parts[0]
    return out


def ref_md5(repo, commit, paths=None):
    """{path: md5(blob)} for the commit, straight out of git (the box must match the commit)."""
    paths = paths or PATHS
    rc, out, err = run(["git", "-C", repo, "ls-tree", "-r", commit, "--"] + paths)
    if rc:
        raise RuntimeError(f"git ls-tree {commit} in {repo}: {err.strip()}")
    blobs = {}
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        blobs[path] = meta.split()[2]
    if not blobs:
        raise RuntimeError(f"commit {commit} in {repo} tracks none of {paths}")
    batch = "".join(b + "\n" for b in blobs.values())
    rc, outb, err = run(["git", "-C", repo, "cat-file", "--batch"], stdin=batch, binary=True)
    if rc:
        raise RuntimeError(f"git cat-file: {err.strip()}")
    digest, pos = {}, 0
    for path, blob in blobs.items():
        nl = outb.index(b"\n", pos)
        size = int(outb[pos:nl].split()[2])
        digest[path] = hashlib.md5(outb[nl + 1: nl + 1 + size]).hexdigest()
        pos = nl + 1 + size + 1
    return digest


# ---- manifest mode ------------------------------------------------------------------------------
SIDE_SCRIPT = r"""root=%(root)s; sha=%(sha)s; ct=%(ct)s; rel="$root/releases/$sha"; tree="$rel/src"
probe_ct=%(ct_probe)s; probe_srv=%(srv_probe)s
echo "LIVE $(readlink -f "$root/current" 2>/dev/null)"
echo "EXT_LIVE $(readlink -f "$root/extcache" 2>/dev/null)"
echo "RELJSON $(tr -d '\n' < "$rel/RELEASE.json" 2>/dev/null)"
echo "LIBSTAMP $(tr -d '\n' < "$tree/qmesh/native/bin/%(lib)s.%(arch)s.so.stamp" 2>/dev/null)"
echo "LIBSHA $(sha256sum "$tree/qmesh/native/bin/%(lib)s.%(arch)s.so" 2>/dev/null | cut -d' ' -f1)"
echo "CPPSTAMP $(tr -d '\n' < "$tree/qmesh/native/bin/libqmesh_moe_cpp.%(arch)s.so.stamp" 2>/dev/null)"
echo "CPPSHA $(sha256sum "$tree/qmesh/native/bin/libqmesh_moe_cpp.%(arch)s.so" 2>/dev/null | cut -d' ' -f1)"
echo "EXTSTAMP $(tr -d '\n' < "$rel/ext/BUILD_STAMP.json" 2>/dev/null)"
# The stamp is an identity, not a receipt: it says which sources the build was started from. The
# binary is what the container imports, and a stray client import from another tree rebuilds it in
# place (2026-11-02: a probe build's 27,513,952-byte .so became the client's 27,509,856-byte one and
# the pair still read IN_SYNC). Hash the artifact, not just the stamp.
echo "EXTSHA $(sha256sum "$rel/ext/exllamav3_ext/exllamav3_ext.so" 2>/dev/null | cut -d' ' -f1)"
# Which trees have written this cache. torch's version hash covers the include-path spelling, so a
# client importing a second checkout rebuilds exllamav3_ext in place and the stamp no longer
# describes the binary; the ninja file it leaves behind names the tree it built from. Every root in
# it must resolve to this release's src, or a foreign tree owns the cache and pair_serve will DRIFT
# on the next flip (2026-11-03: /artifacts/dspark-tpr/src, .so 5780205b against a stamped 04a61140).
echo "EXTSRC $(grep -ho '/[A-Za-z0-9_./-]*exllamav3' "$rel/ext/exllamav3_ext/build.ninja" 2>/dev/null | sort -u | while read -r r; do printf '%%s->%%s ' "$r" "$(readlink -f "$r" 2>/dev/null)"; done)"
echo "CXSHA $(sha256sum "$tree"/vendor/cuda-exl3/src/cuda_exl3/_C*.so 2>/dev/null | cut -d' ' -f1 | head -1)"
if [ "$probe_ct" = 1 ]; then
echo "CT_STATE $(docker inspect -f '{{.State.Running}}' "$ct" 2>/dev/null)"
echo "CT_ENV $(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$ct" 2>/dev/null | grep -E '^(TORCH_EXTENSIONS_DIR|PYTHONPATH|TORCH_CUDA_ARCH_LIST)=' | sort | tr '\n' ' ')"
echo "CT_CTROOT $(docker exec "$ct" readlink -f /pair/current 2>/dev/null)"
echo "CT_SRC $(docker exec "$ct" readlink -f /src/exllamav3 2>/dev/null)"
echo "CT_PAIRMOUNT $(docker exec "$ct" sh -c 'grep -c " /pair " /proc/mounts' 2>/dev/null)"
fi
if [ "$probe_srv" = 1 ]; then
echo "SRV_PIDS $(docker exec "$ct" pgrep -fc worker_seam_server.py 2>/dev/null)"
echo "SRV_LOADED $(docker exec "$ct" sh -c "grep -m1 '^SEAM_SERVER_LOADED' /tmp/seam_server.log" 2>/dev/null)"
fi
echo "TREEMD5"
cd "$tree" 2>/dev/null && printf '%%s\n' %(paths)s | xargs -d '\n' -r md5sum 2>&1
echo "END"
"""


def side_state(manifest, side, remote, skip_ct=False, skip_srv=False):
    """Run the single state script on one side; return {key: value} plus the md5 map."""
    root = manifest[side]["root"]
    cfg = dict(root=root, sha=manifest["release_sha"], ct=manifest[side]["container"],
               lib="libqmesh_expert_seam", arch=manifest[side]["arch"],
               ct_probe=0 if skip_ct else 1,
               srv_probe=0 if (skip_ct or skip_srv) else 1,
               paths=" ".join(shlex.quote(p) for p in manifest["_paths"]))
    script = SIDE_SCRIPT % cfg
    if remote:
        rc, out, err = run(["bash", "-s"], host=manifest["gx"]["host"], stdin=script)
    else:
        p = subprocess.run(["bash", "-s"], input=script.encode(), capture_output=True)
        rc, out, err = p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")
    if "END" not in out:
        raise RuntimeError(f"{side}: state script did not finish (rc={rc}): "
                           f"{(err or out).strip()[:300]}")
    state, md5, in_tree = {}, {}, False
    for line in out.splitlines():
        if line == "TREEMD5":
            in_tree = True
            continue
        if line == "END":
            break
        if in_tree:
            md5.update(md5map(line))
            continue
        key, _, val = line.partition(" ")
        state[key] = val
    return state, md5, err


def check_side(manifest, side, ref, quiet, skip_ct=False, skip_srv=False):
    label = manifest[side].get("label") or side
    ct = manifest[side]["container"]
    arch = manifest[side]["arch"]
    sha = manifest["release_sha"]
    build_id = manifest["build_id"]
    root = manifest[side]["root"]
    try:
        state, md5, err = side_state(manifest, side, remote=(side == "gx"),
                                     skip_ct=skip_ct, skip_srv=skip_srv)
    except Exception as exc:                                  # noqa: BLE001
        print(f"PAIR {manifest['pair']} {label} {ct}: ERROR unreachable or unreadable: {exc}")
        return False
    rel_dir = f"{root}/releases/{sha}"
    tree = f"{rel_dir}/src"
    problems = []

    # 1. tree == commit
    files = manifest["_paths"]
    diff = sorted(f for f in files if f in md5 and md5[f] != ref[f])
    missing = sorted(f for f in files if f not in md5)
    ok_tree = not diff and not missing
    print(f"PAIR {manifest['pair']} {label} {ct} tree={tree} ref={sha[:8]} files={len(files)} "
          f"{'SYNC' if ok_tree else 'DRIFT'} diff={len(diff)} missing={len(missing)}")
    for f in diff:
        print(f"    DIFF {f} box={md5[f]} ref={ref[f]}")
    for f in missing:
        print(f"    MISSING {f}")
    if not ok_tree:
        problems.append(f"tree_diff={len(diff)} tree_missing={len(missing)}")

    # 2. symlinks: the pair root must be the live one, and the ext cache must be this release's
    if state.get("LIVE") != rel_dir:
        problems.append(f"current={os.path.basename(state.get('LIVE', '')) or '-'} want={sha[:8]}")
    if state.get("EXT_LIVE") != f"{rel_dir}/ext":
        problems.append(f"extcache={state.get('EXT_LIVE') or '-'} want={rel_dir}/ext")

    # 3. RELEASE.json must agree with the manifest
    rj = state.get("RELJSON") or ""
    try:
        rj = json.loads(rj) if rj else {}
    except ValueError:
        rj = {}
    if not rj:
        problems.append("release_json=missing")
    else:
        if rj.get("sha") != sha:
            problems.append(f"release_json_sha={str(rj.get('sha'))[:8]} want={sha[:8]}")
        if rj.get("build_id") != build_id:
            problems.append(f"release_json_build={rj.get('build_id')} want={build_id}")
        if manifest.get("tree_digest") and rj.get("digest") != manifest["tree_digest"]:
            problems.append("release_json_digest_mismatch")

    # 4. native seam lib: built from THIS source, stamped with THIS release
    lib = state.get("LIBSTAMP") or ""
    try:
        lib = json.loads(lib) if lib else {}
    except ValueError:
        lib = {}
    if not lib:
        problems.append("lib_stamp=missing")
    else:
        if lib.get("build_id") != build_id:
            problems.append(f"lib_build_id={lib.get('build_id')} want={build_id}")
        if lib.get("source_sha") != sha:
            problems.append(f"lib_source={str(lib.get('source_sha'))[:8]} want={sha[:8]}")
        if manifest.get("seam", {}).get("fingerprint") and \
                lib.get("seam") != manifest["seam"]["fingerprint"]:
            problems.append(f"lib_seam={lib.get('seam')} want={manifest['seam']['fingerprint']}")
        libsha = state.get("LIBSHA", "")
        if lib.get("sha256") and libsha and lib["sha256"] != libsha:
            problems.append("lib_so_hash_vs_stamp")
        want = manifest.get("lib", {}).get(arch, {}).get("sha256")
        if want and libsha and want != libsha:
            problems.append("lib_so_hash_vs_manifest")

    # 4b. the C moe hook: same rule as the seam lib, because it is a second compiled .so in the
    # release tree. Until this existed a release root carrying a stale or missing hook read IN_SYNC
    # (the check hashed exactly one native artefact), and the hook moves tree_digest, so "the source
    # every piece was built from" would have been untrue of the one piece built last.
    # Only a release that declares the hook is required (RELEASE.json native.moe_cpp, set by the cut
    # when the commit tracks qmesh/native/build_moe.sh) is checked for it: a release cut before the
    # hook had a build script never built one, and demanding it would turn every such root into DRIFT.
    if (rj.get("native") or {}).get("moe_cpp"):
        cpp = state.get("CPPSTAMP") or ""
        try:
            cpp = json.loads(cpp) if cpp else {}
        except ValueError:
            cpp = {}
        if not cpp:
            problems.append("cpp_stamp=missing")
        else:
            if cpp.get("build_id") != build_id:
                problems.append(f"cpp_build_id={cpp.get('build_id')} want={build_id}")
            if cpp.get("source_sha") != sha:
                problems.append(f"cpp_source={str(cpp.get('source_sha'))[:8]} want={sha[:8]}")
            if cpp.get("arch") != arch:
                problems.append(f"cpp_arch={cpp.get('arch')} want={arch}")
            cpp_sha = state.get("CPPSHA", "")
            if not cpp.get("sha256"):
                problems.append("cpp_stamp_no_sha")
            elif not cpp_sha:
                problems.append("cpp_so=unreadable")
            elif cpp["sha256"] != cpp_sha:
                problems.append(f"cpp_so={cpp_sha[:8]} want={str(cpp['sha256'])[:8]}")

    # 5. torch ext cache: per release, per container, built from THIS source, for THIS arch
    ext = state.get("EXTSTAMP") or ""
    try:
        ext = json.loads(ext) if ext else {}
    except ValueError:
        ext = {}
    sm = manifest[side]["sm"]
    if not ext:
        problems.append("ext_stamp=missing")
    else:
        if ext.get("build_id") != build_id:
            problems.append(f"ext_build_id={ext.get('build_id')} want={build_id}")
        if ext.get("source_sha") != sha:
            problems.append(f"ext_source={str(ext.get('source_sha'))[:8]} want={sha[:8]}")
        # the stamp records arch_list as the list torch takes, so compare against the one-entry list
        if ext.get("arch_list") != [manifest[side]["cuda_arch"]]:
            problems.append(f"ext_arch={ext.get('arch_list')} want={[manifest[side]['cuda_arch']]}")
        # ...and the artifact the stamp names must still be the one in the cache
        ext_sha = state.get("EXTSHA") or ""
        if not ext.get("sha256"):
            problems.append("ext_stamp_no_sha")
        elif not ext_sha:
            problems.append("ext_so=unreadable")
        elif ext.get("sha256") != ext_sha:
            problems.append(f"ext_so={ext_sha[:8]} want={str(ext.get('sha256'))[:8]}")
        # A hash mismatch above already fails the side, but by then the pair cannot serve: name the
        # owner. The ninja file in the cache lists the tree it was built from; a root that does not
        # resolve to this release's src is a client that rebuilt the pair's cache in place.
        want_src = f"{rel_dir}/src"
        for item in (state.get("EXTSRC") or "").split():
            src_spelled, _, src_real = item.partition("->")
            if src_real and src_real != want_src:
                problems.append(f"extcache_foreign_src={src_spelled}->{src_real} want={want_src}")
            # The resolve above is blind on a host that has no /pair (both P7 and GX hosts: the
            # mount is inside the container), so a cache built through an *alias of this same tree*
            # passed it silently -- /pair/current/src/exllamav3/exllamav3 resolves to the release's
            # own src, yet torch's version hash covers the spelling and the rebuild replaced the
            # stamped binary in place (2026-11-03, ext_so=c7908310 want=e74d4951, every dev ticket
            # on DRIFT). One link, one spelling: the container's own /src/exllamav3.
            if src_spelled and not src_spelled.startswith("/src/exllamav3"):
                problems.append(f"extcache_spelling={src_spelled} want=/src/exllamav3/...")
        # cuda-exl3 is gitignored and built per checkout, so the GX stamp has to name it: without it
        # the release would verify clean and then serve every layer on the mixed-K module.
        if side == "gx":
            cx = ext.get("cx") or {}
            if not cx:
                problems.append("cx_stamp=missing")
            else:
                if cx.get("build_id") != build_id:
                    problems.append(f"cx_build_id={cx.get('build_id')} want={build_id}")
                if cx.get("source_sha") != sha:
                    problems.append(f"cx_source={str(cx.get('source_sha'))[:8]} want={sha[:8]}")
                if cx.get("arch_list") != [manifest[side]["cuda_arch"]]:
                    problems.append(f"cx_arch={cx.get('arch_list')} want={[manifest[side]['cuda_arch']]}")
                cx_sha = state.get("CXSHA") or ""
                if cx.get("sha256") and cx_sha and cx["sha256"] != cx_sha:
                    problems.append(f"cx_so={cx_sha[:8]} want={str(cx['sha256'])[:8]}")

    # 6. the container must be wired to the pair root, not to some tree somebody copied into it
    if skip_ct:
        pass                                              # --no-container: wiring not probed
    elif state.get("CT_STATE") != "true":
        problems.append("container=not_running")
    else:
        if (state.get("CT_PAIRMOUNT") or "0") == "0":
            problems.append("container=missing_/pair_mount")
        ctr = state.get("CT_CTROOT") or ""
        if os.path.basename(ctr) != sha:
            problems.append(f"container_current={os.path.basename(ctr) or '-'} want={sha[:8]}")
        csrc = state.get("CT_SRC") or ""
        # readlink -f prints the fully resolved chain; /pair/current/src is the same tree
        if csrc not in (f"/pair/releases/{sha}/src", "/pair/current/src"):
            problems.append(f"container_src={csrc or '-'} want=/pair/releases/{sha[:8]}/src")
        env = state.get("CT_ENV") or ""
        # The spec's default is /pair/scratch-ext since 2026-11-04 (see pair_containers.py's
        # docstring): the release's cache is stated by the arm, the server and the build, never
        # inherited. What still has to hold is that the value is a path this pair owns -- an inherited
        # /home/... spelling would put a JIT under the container's $HOME and lose it on recreate.
        tex = ""
        for kv in env.split():
            if kv.startswith("TORCH_EXTENSIONS_DIR="):
                tex = kv.split("=", 1)[1]
        if tex not in ("/pair/scratch-ext", "/pair/extcache"):
            problems.append(f"container_torch_extensions_dir={tex or '-'} want=/pair/scratch-ext")
        for want_kv in ("PYTHONPATH=/src/exllamav3",
                        f"TORCH_CUDA_ARCH_LIST={manifest[side]['cuda_arch']}"):
            if want_kv not in env:
                problems.append(f"container_env_missing={want_kv}")

    # 7. a server that is up must be running THIS release
    if side == "gx" and not skip_srv:
        pids = state.get("SRV_PIDS") or "0"
        try:
            npids = int(pids)
        except ValueError:
            npids = -1
        if npids > 0:
            loaded = state.get("SRV_LOADED") or ""
            got = ""
            for tok in loaded.split():
                if tok.startswith("build="):
                    got = tok.split("=", 1)[1]
            if not loaded:
                problems.append("server_loading")           # no SEAM_SERVER_LOADED line yet
            elif got != build_id:
                problems.append(f"server_build={got} want={build_id}")
        elif npids < 0:
            problems.append("server_state=unknown")

    if problems:
        for p in problems:
            print(f"    DRIFT {p}")
        print(f"RESULT {manifest['pair']} {label} DRIFT {' '.join(problems)}")
        return False
    if not quiet:
        cpp_tok = f"cpp={arch}" if (rj.get("native") or {}).get("moe_cpp") else "cpp=-"
        print(f"STAMP {manifest['pair']} {label} lib={arch} {cpp_tok} ext={sm} build_id={build_id} OK")
    print(f"RESULT {manifest['pair']} {label} IN_SYNC")
    return True


def manifest_mode(args):
    try:
        manifest = json.load(open(args.manifest))
    except Exception as exc:                                  # noqa: BLE001
        print(f"pair_sync_check: cannot read manifest {args.manifest}: {exc}", file=sys.stderr)
        return 2
    if manifest.get("format") != 1:
        print(f"pair_sync_check: manifest {args.manifest} has unsupported format "
              f"{manifest.get('format')!r}", file=sys.stderr)
        return 2
    sha = manifest["release_sha"]
    try:
        ref = ref_md5(REPO, sha + "^{commit}" if not sha.endswith("^{commit}") else sha)
    except Exception as exc:                                  # noqa: BLE001
        print(f"pair_sync_check: reference commit {sha[:8]} not in {REPO}: {exc}", file=sys.stderr)
        return 2
    manifest["_paths"] = sorted(ref)
    sides = [s for s in ("p7", "gx")
             if not (args.p7_only and s != "p7") and not (args.gx_only and s != "gx")]
    ok = True
    for side in sides:
        ok = check_side(manifest, side, ref, args.quiet,
                        skip_ct=args.no_container,
                        skip_srv=args.no_container or args.no_server) and ok
    if args.quiet and ok:
        print(f"RESULT {manifest['pair']} IN_SYNC")
    return 0 if ok else 1


# ---- legacy tree mode ---------------------------------------------------------------------------
def reference(worktree, commit):
    rc, _, err = run(["git", "-C", worktree, "cat-file", "-e", commit + "^{commit}"])
    if rc:
        sys.exit(f"pair_sync_check: {worktree} has no commit {commit}: {err.strip()}")
    rc, out, err = run(["git", "-C", worktree, "ls-tree", "-r", "--name-only", commit, "--"] + PATHS)
    if rc:
        sys.exit(f"pair_sync_check: ls-tree failed in {worktree}: {err.strip()}")
    files = [f for f in out.splitlines() if f]
    if not files:
        sys.exit(f"pair_sync_check: commit {commit} in {worktree} tracks none of {PATHS}")
    return commit, files, ref_md5(worktree, commit)


def check(pair, args):
    name, host, box, container, tree, refkey = pair
    worktree = WORKTREES[refkey]
    commit = getattr(args, refkey)
    commit, files, refmd5 = reference(worktree, commit)
    rc, out, err = run("cd %s && xargs -d '\\n' md5sum 2>&1" % tree, host=host,
                       stdin="\n".join(files) + "\n")
    if rc and not out:
        print(f"PAIR {name} {box} {container}: ERROR unreachable host or tree: {err.strip()[:200]}")
        return False, None
    boxmd5 = md5map(out)
    diff = sorted(f for f in files if f in boxmd5 and boxmd5[f] != refmd5[f])
    missing = sorted(f for f in files if f not in boxmd5)
    ok = not diff and not missing
    print(f"PAIR {name} {box} {container} tree={tree} ref={commit[:8]} files={len(files)} "
          f"{'SYNC' if ok else 'DRIFT'} diff={len(diff)} missing={len(missing)}")
    for f in diff:
        print(f"    DIFF {f} box={boxmd5[f]} ref={refmd5[f]}")
    for f in missing:
        print(f"    MISSING {f}")
    return ok, (name, commit, diff, missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="manifest mode: pairs/<pair>.json")
    ap.add_argument("--quiet", action="store_true", help="one RESULT line per pair when in sync")
    ap.add_argument("--p7-only", action="store_true")
    ap.add_argument("--gx-only", action="store_true")
    ap.add_argument("--no-container", action="store_true")
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--v2", default=None, help="legacy mode: commit the prod v2 pair must serve")
    ap.add_argument("--stg", default=None, help="legacy mode: commit the staging trees must serve")
    ap.add_argument("--pair", default="all", choices=["all", "prod", "dev"], help="legacy mode")
    args = ap.parse_args()
    if args.manifest:
        return manifest_mode(args)
    for key, wt in WORKTREES.items():
        if getattr(args, key) is None:
            rc, out, _ = run(["git", "-C", wt, "rev-parse", "HEAD"])
            if rc:
                sys.exit(f"pair_sync_check: no worktree at {wt}")
            setattr(args, key, out.strip())
    allok = True
    for pair in PAIRS:
        if args.pair not in ("all", pair[0]):
            continue
        if not pair[1]:
            sys.exit(f"pair_sync_check: set PAIR_LEGACY_{pair[0].upper()}_HOST")
        ok, _ = check(pair, args)
        allok = allok and ok
    print("RESULT " + ("ALL_SYNC" if allok else "DRIFT"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
