#!/usr/bin/env bash
# pair_build_ext.sh -- build the native pieces of one release, per container: the exllamav3 torch
# extension on both sides, and on GX the vendored cuda-exl3 cold-expert kernel.
#
#   pair_build_ext.sh <pair> <sha> [--side p7|gx|both]
#   pair_build_ext.sh verify <pair> <sha> [--side p7|gx|both]   (stamp check only, builds nothing)
#
# The torch cache is per release (releases/<sha>/ext) and per container image (P7 torch 2.14, GX torch
# 2.11, and 12.0 vs 12.1 arch), so each side builds its own: a cache shared between containers is how
# one container's stale objects became another container's "fresh" build.
#
# cuda-exl3 is different: it is gitignored, built in place into the release's own tree
# (releases/<sha>/src/vendor/cuda-exl3/src/cuda_exl3/_C.<abi>.so) by the container that will import it,
# so every cut needs this stage before it can serve. Its identity rides in the same BUILD_STAMP.json
# as the torch extension's (`cx`), and verify mode refuses a GX stamp without it.
#
# Guards are the same as pair_build_lib.sh: no server in this pair's container, no compiler already
# running on the box, and on GX enough free memory that no pack is loading. A build killed mid-JIT
# leaves torch's `lock` baton behind, and every later import then sleeps 3 x 600 s on it -- the baton
# is cleared here, before the build, never during someone else's run.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

MODE=build
if [ "${1:-}" = verify ]; then MODE=verify; shift; fi
PAIR_ARG=${1:?usage: pair_build_ext.sh [verify] <pair> <sha> [--side p7|gx|both]}
SHA=${2:?usage: pair_build_ext.sh [verify] <pair> <sha> [--side p7|gx|both]}
shift 2
SIDES=both
while [ $# -gt 0 ]; do
  case "$1" in
    --side) SIDES=${2:?--side needs p7|gx|both}; shift ;;
    *) pair_die "unknown flag $1" ;;
  esac
  shift
done
pair_load "$PAIR_ARG"
# The stamp names the commit, so compare commits: a caller may pass a prefix ("4f299c3a")
# while the build recorded the full sha, and a false mismatch stops the flip.
SHA=$(git -C "$PAIR_REPO" rev-parse --verify "$SHA^{commit}" 2>/dev/null || printf '%s' "$SHA")
REL=$(pair_release_dir "$SHA")
[ -d "$REL/src" ] || pair_die "release ${SHA:0:8} is not cut at $REL"
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json")
JOBS=${MAX_JOBS:-8}; [ "$JOBS" -le 8 ] || JOBS=8
EXT_TIMEOUT=${EXT_TIMEOUT:-1800}
GX_BUILD_AVAIL_GIB=${GX_BUILD_AVAIL_GIB:-60}
MOUNT=${PAIR_MOUNT:-/pair}
# ---- verify: the stamp pair_manifest.py will publish must already be there and be this build -------
if [ "$MODE" = verify ]; then
  for side in p7 gx; do
    [ "$SIDES" = both ] || [ "$SIDES" = "$side" ] || continue
    if [ "$side" = p7 ]; then got=$(cat "$REL/ext/BUILD_STAMP.json" 2>/dev/null)
    else got=$(pair_gxsh "cat '$REL/ext/BUILD_STAMP.json'" 2>/dev/null); fi
    printf '%s' "$got" | python3 "$HERE/pair_ext_stamp_check.py" "$side" "$BUILD_ID" "$SHA" || exit 1
  done
  exit 0
fi

# Dry run (stub-ssh harness only): say what would be built, write nothing (see pair_build_lib.sh).
if [ "$PAIR_DRY" = 1 ]; then
  for side in p7 gx; do
    [ "$SIDES" = both ] || [ "$SIDES" = "$side" ] || continue
    ct=$([ "$side" = p7 ] && echo "$P7CT" || echo "$GXCT")
    sm=$([ "$side" = p7 ] && echo "$P7_ARCH" || echo "$GX_ARCH")
    pair_step "DRY EXT_BUILD $side $ct arch=$(pair_cuda_arch_for "$sm") jobs=$JOBS build=$BUILD_ID"
  done
  exit 0
fi

side_docker() {  # <side> <args...>
  local side=$1; shift
  if [ "$side" = p7 ]; then docker "$@"; else pair_gxsh "docker $(printf '%q ' "$@")"; fi
}
side_hostsh() { local side=$1; if [ "$side" = p7 ]; then bash -c "$2"; else pair_gxsh "$2"; fi; }

build_side() {
  local side=$1 ct sm archlist out rc cxout cxso cxsha cxname
  if [ "$side" = p7 ]; then ct=$P7CT; sm=$P7_ARCH; else ct=$GXCT; sm=$GX_ARCH; fi
  archlist=$(pair_cuda_arch_for "$sm")
  local rel_c="$MOUNT/releases/$SHA"

  [ "$(side_docker "$side" inspect -f '{{.State.Running}}' "$ct" 2>/dev/null)" = true ] \
    || { pair_step "EXT_SKIP $side $ct not_running"; return 1; }
  [ -z "$(side_docker "$side" exec "$ct" pgrep -f worker_seam_server.py 2>/dev/null)" ] \
    || { pair_step "EXT_REFUSE $side server_running_in=$ct"; return 1; }
  # A warm cache is not proof that *this* release built it. build mode used to re-probe whatever .so
  # was in $rel_c/ext and stamp it, so an .so that some other release's sources had produced could be
  # published as this one's (2026-11-04 01:00Z, in the cache that then had to be deleted and rebuilt).
  # The stamp inside the cache is the only self-description it has: if it exists and names another
  # commit, the cache belongs to that commit -- refuse, say how to clear it, and stamp nothing.
  local prev prev_sha
  if [ "$side" = p7 ]; then prev=$(cat "$REL/ext/BUILD_STAMP.json" 2>/dev/null)
  else prev=$(pair_gxsh "cat '$REL/ext/BUILD_STAMP.json'" 2>/dev/null); fi
  if [ -n "$prev" ]; then
    prev_sha=$(printf '%s' "$prev" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("source_sha",""))
except Exception: print("")' 2>/dev/null)
    if [ "$prev_sha" != "$SHA" ]; then
      pair_step "EXT_REFUSE_STAMPED_OTHER $side cache=$rel_c/ext stamp_source_sha=${prev_sha:0:12} want=${SHA:0:12} (the cache in this release's ext dir was blessed for another commit; nothing stamped -- rm -rf the cache inside $ct and re-run the build)"
      return 1
    fi
  fi
  # Count only compilers young enough to still be working: exl3-tp2-moe has kept 223 nvcc and 126
  # cc1plus alive for days, and waiting on those dead ones would block every build forever.
  local builds ignored stale=${PAIR_BUILD_STALE_SECS:-1800}
  compilers() {
    side_hostsh "$1" "ps -eo stat=,etimes=,comm= | awk -v s=$stale '\$1 !~ /^Z/ && \$3 ~ /^(ninja|cc1plus|nvcc|cicc)\$/ {if (\$2 < s) f++; else o++} END {print f+0\" \"o+0}'"
  }
  read -r builds ignored <<<"$(compilers "$side")"
  local waited=0 cap=${PAIR_BUILD_WAIT_SECS:-1800}
  while [ "${builds:-0}" != 0 ] && [ "$waited" -lt "$cap" ]; do
    pair_step "EXT_WAIT $side compiler_running=$builds stale_ignored=${ignored:-0} waited=${waited}s (another build owns the host)"
    sleep 30; waited=$((waited + 30))
    read -r builds ignored <<<"$(compilers "$side")"
  done
  [ "${builds:-0}" = 0 ] || { pair_step "EXT_REFUSE $side compiler_running=$builds waited=${waited}s"; return 1; }
  if [ "$side" = gx ]; then
    local avail
    avail=$(side_hostsh gx "awk '/^MemAvailable:/{print int(\$2/1048576)}' /proc/meminfo")
    [ "${avail:-0}" -ge "$GX_BUILD_AVAIL_GIB" ] \
      || { pair_step "EXT_REFUSE gx mem_avail=${avail:-?}GiB need=$GX_BUILD_AVAIL_GIB"; return 1; }
  fi

  # stale baton: only when nothing is building and no server of ours runs (both checked above)
  side_docker "$side" exec "$ct" sh -c "mkdir -p $rel_c/ext/exllamav3_ext; rm -f $rel_c/ext/exllamav3_ext/lock" \
    || { pair_step "EXT_FAIL $side lock_clear"; return 1; }

  pair_step "EXT_BUILD $side $ct arch=$archlist jobs=$JOBS build=$BUILD_ID"
  # Build through the path the clients import through. torch's JIT version hash covers the include
  # paths, so a cache built from /pair/releases/<sha>/src is a miss for every client -- they import
  # /src/exllamav3 (= /pair/current/src) and would silently rebuild the extension for 10-13 minutes,
  # inside their own run, with the server already loaded (the SIGBUS case). So /src/exllamav3 is
  # pointed at this release for the length of the build and put back afterwards; nothing else runs in
  # the container (the server and compiler guards above saw to that).
  local link=/src/exllamav3 back build_src=$rel_c/src now
  back=$(side_docker "$side" exec "$ct" sh -c "readlink $link" 2>/dev/null)
  if [ -n "$back" ]; then
    # The probe is copied in and run as a *file*. Run from stdin, python puts the cwd on sys.path,
    # and cwd is the resolved path (/pair/releases/<sha>/src, because getcwd() never returns a
    # symlink) -- so the cache got the release spelling while every client, which runs a script and
    # therefore has PYTHONPATH=/src/exllamav3 on sys.path, got the /src spelling. torch's version
    # hash covers the include path, so the two spellings are two caches: each side rebuilt all 145
    # units for the other one (2026-11-02, twice, once inside a smoke).
    inner="ln -sfn $rel_c/src $link
trap 'ln -sfn $back $link' EXIT
cat > /tmp/pair_ext_probe.py
# Clear the baton again in the shell that is about to import: the clear above runs before the
# symlink flip, and the two spellings are not always one inode (/pair/extcache is its own directory
# today, so `rm` on one left the other). A surviving lock never fails the build -- it sleeps in
# file_baton.wait (3 x 600 s) with no compiler running, which looks exactly like a hung host
# (2026-11-03 02:07Z: 20 min, pid 3475093, lock left by a probe killed at 02:02Z).
for lck in $rel_c/ext/exllamav3_ext/lock $MOUNT/extcache/exllamav3_ext/lock; do
  [ -e \$lck ] && rm -f \$lck && echo EXT_LOCK cleared=\$lck
done
cd $link && nice -n 15 env MAX_JOBS=$JOBS NINJAFLAGS=-j$JOBS MAKEFLAGS=-j$JOBS \
PYTHONPATH=$link TORCH_CUDA_ARCH_LIST=$archlist TORCH_EXTENSIONS_DIR=$rel_c/ext \
EXPECT_SRC=$rel_c/src/exllamav3 timeout $EXT_TIMEOUT python3 -u /tmp/pair_ext_probe.py
rc=\$?
# A plain client import -- no find_spec shim, env exactly as the image sets it -- must reuse what we
# just built. Run it from a neutral cwd with PYTHONPATH spelled like the container's: `python3 -c`
# puts '' first on sys.path, importlib resolves '' with getcwd(), and getcwd() returns the release
# src dir with its symlinks resolved -- so from $link the check compiled the /pair spelling while
# every client script gets the /src one (2026-11-02 03:39Z, prod GX: EXT_CLIENT_MISS on a cache that
# was correct). From /tmp, '' resolves to a directory that holds no exllamav3, and the import lands
# on PYTHONPATH=/src/exllamav3 exactly as it does for the arm driver and the GX worker.
if [ \$rc = 0 ]; then
  cd /tmp && PYTHONPATH=$link TORCH_EXTENSIONS_DIR=$rel_c/ext \
    timeout \${EXT_CLIENT_TIMEOUT:-180} python3 -c 'import exllamav3.ext as e; print(\"EXT_CLIENT_HIT\", e.exllamav3_ext.__file__)'
  rc=\$?
fi
exit \$rc"
  else
    pair_step "EXT_LINK $side $link missing (building through $rel_c/src)"
    inner="cat > /tmp/pair_ext_probe.py
cd $rel_c/src && nice -n 15 env MAX_JOBS=$JOBS NINJAFLAGS=-j$JOBS MAKEFLAGS=-j$JOBS \
PYTHONPATH=$rel_c/src TORCH_CUDA_ARCH_LIST=$archlist TORCH_EXTENSIONS_DIR=$rel_c/ext \
EXPECT_SRC=$rel_c/src/exllamav3 timeout $EXT_TIMEOUT python3 -u /tmp/pair_ext_probe.py"
  fi
  if [ "$side" = p7 ]; then
    out=$(docker exec -i "$ct" bash -lc "$inner" < "$HERE/pair_ext_probe.py" 2>&1); rc=$?
  else
    out=$(pair_gxsh "docker exec -i $(printf %q "$ct") bash -lc $(printf %q "$inner")" \
          < "$HERE/pair_ext_probe.py" 2>&1); rc=$?
  fi
  [ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$out" >> "$PAIR_LOG"

  # The link is put back by the parent, not only by the container-side trap. `trap … EXIT` runs only if
  # that bash reaches EXIT: on 2026-11-02 the canary's GX build was killed mid-flight and left
  # seam-gx2-dev wired at an unflipped release -- pair_check read GX DRIFT, and every client in it
  # would have imported a tree that is not `current`. This process is outside the container and
  # outside whatever killed it, so it restores whatever the build's rc, and says where the link went.
  if [ -n "$back" ]; then
    side_docker "$side" exec "$ct" sh -c "ln -sfn $back $link" >/dev/null 2>&1
    now=$(side_docker "$side" exec "$ct" sh -c "readlink -f $link" 2>/dev/null)
    pair_step "EXT_LINK_RESTORE $side -> ${now:-?} (was $back)"
  fi
  # rc=124 from the client check means it started compiling: the cache is not what a client asks for.
  if [ "$rc" = 124 ] && printf '%s' "$out" | grep -q '^EXT_BUILD_OK '; then
    pair_step "EXT_FAIL $side EXT_CLIENT_MISS client_import_rebuilds cache=$rel_c/ext (include path or flags differ from the image's client env)"
    return 1
  fi
  if [ "$rc" != 0 ] || ! printf '%s' "$out" | grep -q '^EXT_BUILD_OK '; then
    pair_step "EXT_FAIL $side rc=$rc $(printf '%s' "$out" | grep -E 'Error|error:|FAILED|EXT_' | tail -4 | tr '\n' ' ')"
    return 1
  fi
  printf '%s\n' "$out" | grep -m1 '^EXT_CLIENT_HIT ' | sed 's/^/  /' | tee -a "${PAIR_LOG:-/dev/null}"
  # The cache dir was created by the container as root; hand it to the host user, which writes the
  # stamp next to it (and cleans old caches later).
  local uid gid; uid=$(id -u); gid=$(id -g)
  side_docker "$side" exec "$ct" sh -c "chown -R $uid:$gid $rel_c/ext" \
    || { pair_step "EXT_FAIL $side chown_cache"; return 1; }

  # cuda-exl3, the cold-expert kernel. Its .so is gitignored on purpose -- it is per checkout and per
  # image -- so a release that was cut but never built has no kernel at all: the worker prints
  # SEAM_MOE_NOEXT and serves every layer on the mixed-K module, which is correct output at a
  # fraction of the rate. Building it here, in the container that will import it, and putting its
  # identity into the same stamp as the torch extension's is what makes "the release passed" mean
  # "the release's kernel passed". (~100 s at MAX_JOBS=8, inside this stage's own timeout.)
  if [ "$side" = gx ]; then
    pair_step "CX_BUILD gx $ct arch=12.1 jobs=$JOBS build=$BUILD_ID"
    cxout=$(pair_gxsh "docker exec $(printf %q "$ct") bash -lc $(printf %q "cd $rel_c/src && env MAX_JOBS=$JOBS TORCH_CUDA_ARCH_LIST=12.1 timeout $EXT_TIMEOUT sh qmesh/build_cudaexl3.sh")" 2>&1)
    rc=$?
    [ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$cxout" >> "$PAIR_LOG"
    # The script's own last line is the import probe's schema assertion, so rc=0 means the .so was
    # built *and* loaded with the shared-accumulator args the seam adapter asks for.
    if [ "$rc" != 0 ] || ! printf '%s' "$cxout" | grep -q '^cuda_exl3 ok:'; then
      pair_step "CX_FAIL gx rc=$rc $(printf '%s' "$cxout" | grep -E 'Error|error:|FAILED|Assertion' | tail -3 | tr '\n' ' ')"
      return 1
    fi
    # rel_c is a path INSIDE the container; the host half of this script has no /pair, so a lookup (or
    # a chown) run through pair_gxsh without docker exec reports the kernel missing on a build that just
    # succeeded -- and the chown would silently do nothing, leaving root-owned files in the release.
    # docker exec runs no shell, so even in the container the `_C*.so` has to be expanded by sh -c.
    cxso=$(pair_gxsh "docker exec $(printf %q "$ct") sh -c $(printf %q "ls -1 $rel_c/src/vendor/cuda-exl3/src/cuda_exl3/_C*.so | head -1")" 2>/dev/null)
    [ -n "$cxso" ] || { pair_step "CX_FAIL gx no_so_in_release_tree"; return 1; }
    cxname=$(basename "$cxso")
    cxsha=$(pair_gxsh "docker exec $(printf %q "$ct") sha256sum '$cxso'" 2>/dev/null | cut -d' ' -f1)
    [ -n "$cxsha" ] || { pair_step "CX_FAIL gx so_unreadable so=$cxname"; return 1; }
    # built as root inside the container; the host user stamps and later prunes the tree
    pair_gxsh "docker exec $(printf %q "$ct") chown -R $(id -u):$(id -g) $rel_c/src/vendor/cuda-exl3" >/dev/null 2>&1 || true
    pair_step "CX_OK gx so=$cxname sha256=$cxsha arch=12.1"
  fi

  local so sha_so
  so=$(printf '%s\n' "$out" | sed -n 's/^EXT_SO //p' | tail -1)
  [ -n "$so" ] || { pair_step "EXT_FAIL $side no_so_path"; return 1; }
  if [ "$side" = p7 ]; then
    sha_so=$(sha256sum "$REL/ext/exllamav3_ext/$(basename "$so")" 2>/dev/null | cut -d' ' -f1)
  else
    sha_so=$(pair_gxsh "sha256sum '$REL/ext/exllamav3_ext/$(basename "$so")' 2>/dev/null" | cut -d' ' -f1)
  fi
  [ -n "$sha_so" ] || { pair_step "EXT_FAIL $side so_not_in_release_cache"; return 1; }
  # A cache that still has a build plan after the build is not a cache: the next client import runs
  # ninja again -- inside a smoke, under a live server -- relinks the .so and leaves the stamp naming
  # a binary nobody serves (2026-11-02: 27,513,952 -> 27,509,856 bytes, and EXT_NINJA_STALE right
  # after a green build). Converge here, out of band, and stamp only what survives `ninja -n`.
  local ndry tries=0
  ninja_plan() {
    side_docker "$side" exec "$ct" sh -c "cd $rel_c/ext/exllamav3_ext && ninja -n 2>&1" 2>/dev/null \
      | grep -E '^\[' | head -3 | tr '\n' ';'
  }
  ndry=$(ninja_plan)
  while [ -n "$ndry" ] && [ "$tries" -lt 2 ]; do
    tries=$((tries + 1))
    pair_step "EXT_NINJA_STALE $side run=$tries would_run=$ndry (running it out of band)"
    side_docker "$side" exec "$ct" \
      sh -c "cd $rel_c/ext/exllamav3_ext && nice -n 15 ninja -j$JOBS >/dev/null 2>&1" || true
    if [ "$side" = p7 ]; then
      sha_so=$(sha256sum "$REL/ext/exllamav3_ext/$(basename "$so")" 2>/dev/null | cut -d' ' -f1)
    else
      sha_so=$(pair_gxsh "sha256sum '$REL/ext/exllamav3_ext/$(basename "$so")' 2>/dev/null" | cut -d' ' -f1)
    fi
    [ -n "$sha_so" ] || { pair_step "EXT_FAIL $side so_vanished_after_ninja"; return 1; }
    ndry=$(ninja_plan)
  done
  [ -z "$ndry" ] || { pair_step "EXT_NINJA_UNSTABLE $side would_run=$ndry"; return 1; }
  # A cache can converge and still hold the wrong build's plan. 2026-11-04 01:00Z: a contract test
  # imported 3593531b's src with TORCH_EXTENSIONS_DIR pointing at 35bfaecb's cache; torch's rebuild
  # decision is textual in build.ninja, so it wrote `-I/pair/releases/3593531b.../src` into that cache
  # and started 132 nvcc units. The build "succeeded", and the client probe below still hit, because
  # the spelling it asks for (/src/exllamav3) was at the front of the same include list. So before
  # stamping, look for a *foreign* release include: clients never spell the cache this way.
  # This release's own spelling is allowed: the `EXT_LINK missing` branch above builds from
  # $rel_c/src on purpose, and such a cache carries `-I/pair/releases/$SHA/src/...` legitimately
  # (checked on the a16465e4 caches: both sides carry `-I/src/exllamav3/...` and zero release paths).
  # Only an include naming *another* commit is evidence that a client pointed this cache at foreign
  # sources.
  local inc foreign=""
  inc=$(side_docker "$side" exec "$ct" \
    sh -c "grep -o -- '-I/pair/releases/[0-9a-f]*/src' $rel_c/ext/exllamav3_ext/build.ninja 2>/dev/null | sort -u" 2>/dev/null)
  for p in $inc; do
    case "$p" in
      "-I/pair/releases/$SHA/src"|"-I/pair/releases/${SHA:0:12}/src") ;;
      *) foreign="$foreign $p" ;;
    esac
  done
  if [ -n "${foreign// /}" ]; then
    side_docker "$side" exec "$ct" sh -c "rm -rf $rel_c/ext/exllamav3_ext" >/dev/null 2>&1
    pair_step "EXT_WIPE_HYBRID $side cache=$rel_c/ext/exllamav3_ext foreign_include=$foreign (a client imported another release's src into this cache; wiped, nothing stamped, re-run the build)"
    return 1
  fi
  # One stamp per release per side, and on GX it carries the kernel too: a release whose cuda-exl3 was
  # not built looks identical otherwise (the .so is gitignored), and the smoke would be the first place
  # that noticed -- as a slow server instead of a missing build.
  # The stamp names the torch that built it: the two halves run different builds (2.14 / 2.11) and an
  # artifact silently moved between them is a mismatch nothing else would catch. A bare torch import,
  # no exllamav3, so it cannot touch the cache it is describing.
  local torchv
  torchv=$(side_docker "$side" exec "$ct" python3 -c 'import torch; print(torch.__version__)' 2>/dev/null | tail -1)
  [ "$torchv" != unknown ] || { pair_step "EXT_FAIL $side torch_version_unreadable ct=$ct"; return 1; }
  stamp=$(python3 - "$BUILD_ID" "$SHA" "$archlist" "$torchv" "$sha_so" "$side" "${cxsha:-}" "${cxname:-}" <<'PY'
import datetime, json, sys
bid, sha, archs, torchv, so, side, cxsha, cxname = sys.argv[1:9]
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
doc = {
    "build_id": bid, "source_sha": sha,
    "arch_list": archs.split(";") if ";" in archs else archs.split(","),
    "torch": torchv, "sha256": so, "host": side,
    "built_at": now,
}
if cxsha:
    doc["cx"] = {"build_id": bid, "source_sha": sha, "arch_list": ["12.1"],
                 "sha256": cxsha, "so": cxname, "built_at": now}
print(json.dumps(doc, indent=2, sort_keys=True))
PY
)
  if [ "$side" = p7 ]; then
    printf '%s\n' "$stamp" > "$REL/ext/BUILD_STAMP.json"
  else
    printf '%s\n' "$stamp" | pair_gxsh "cat > '$REL/ext/BUILD_STAMP.json'"
  fi
  pair_step "EXT_OK $side torch=$torchv arch=$archlist so=$(basename "$so") sha256=$sha_so${cxsha:+ cx=$cxname cx_sha256=$cxsha}"
  pair_step "EXT_NINJA_CLEAN $side ninja_runs=$tries"
  return 0
}

# The GX half claims the box: the ext and the cuda-exl3 build are both CPU-heavy next to whatever else
# has a pack loaded there, and a load that lands mid-compile kills it. A foreign claim is waited out
# (GX_GATE_WAIT_MIN); ours expires on its own, so a killed build cannot leave the box claimed.
rc=0
case "$SIDES" in
  both|gx)
    if ! pair_gx_gate_take "build_ext-$PAIR" $(( (EXT_TIMEOUT + 900) / 60 )) "${GX_GATE_WAIT_MIN:-20}"; then
      pair_die "GX_GATE_BUSY $GXCT $(pair_gx_gate_state): the box is claimed by another job"
    fi
    trap pair_gx_gate_down EXIT ;;
esac
case "$SIDES" in
  both) build_side p7 || rc=1; [ "$rc" = 0 ] && { build_side gx || rc=1; } ;;
  p7) build_side p7 || rc=1 ;;
  gx) build_side gx || rc=1 ;;
  *) pair_die "bad --side $SIDES" ;;
esac
exit $rc
