#!/usr/bin/env bash
# pair_build_lib.sh -- build qmesh/native/bin/libqmesh_expert_seam.<arch>.so for one release.
#
#   pair_build_lib.sh <pair> <sha> [--side p7|gx|both] [--force]
#   pair_build_lib.sh verify <pair> <sha> [--side p7|gx|both]
#
# The library is built *inside* the release's own tree, from the release's own sources, and lands in
# that release's bin/ next to a .so.stamp that names the commit it came from. A shared library is not
# byte-reproducible across containers (compiler, headers, torch), so the stamp -- not a cross-box
# .so hash -- is what pair_check compares. Nothing is ever copied between boxes: each box builds for
# its own arch (x86_64/sm_120 on P7, aarch64/sm_121 on GX).
#
# Both .so files are stripped (--strip-unneeded) before they are stamped. nvcc names its intermediate
# files from the pid (tmpxft_0000d510_...-6_<src>.cudafe1.cpp) and those names are the only bytes two
# builds of the same source disagree on -- six of them, in .strtab -- so an unstripped build never
# hashed the same twice, and every rebuild invalidated the manifest that the previous stamp produced
# (2026-11-03: a prepare rebuilt dev's seam lib on both halves and pair_check read
# lib_so_hash_vs_manifest for a source nobody had changed). Stripping the symbol table leaves .dynsym,
# .nv_fatbin and .rodata alone -- the same 35 exported qes_* symbols resolve -- and makes a rebuild of
# an unchanged source reproduce the cut's bytes, so a rebuild and the manifest written from its stamp
# agree instead of being a second, differently-hashed artifact.
#
# Guards, because an in-place build under a live server has already killed a box (SIGBUS):
#   * no worker_seam_server.py in this pair's own container,
#   * no compiler already active anywhere on the box (ninja/cc1plus/nvcc/cicc),
#   * on GX: MemAvailable above GX_BUILD_AVAIL_GIB (a loading server holds the pack -> low avail).
# The build runs niced with one nvcc; it writes to a temp dir and mv(1)s into place.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

MODE=build
if [ "${1:-}" = verify ]; then MODE=verify; shift; fi
PAIR_ARG=${1:?usage: pair_build_lib.sh [verify] <pair> <sha> [--side p7|gx|both]}
SHA=${2:?usage: pair_build_lib.sh [verify] <pair> <sha> [--side p7|gx|both]}
shift 2
SIDES="both"; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --side) SIDES=${2:?--side needs p7|gx|both}; shift ;;
    --force) FORCE=1 ;;
    *) pair_die "unknown flag $1" ;;
  esac
  shift
done
pair_load "$PAIR_ARG"
# The stamp names the commit, so compare commits: a caller may pass a prefix ("4f299c3a")
# while the build recorded the full sha, and a false mismatch stops the flip.
SHA=$(git -C "$PAIR_REPO" rev-parse --verify "$SHA^{commit}" 2>/dev/null || printf '%s' "$SHA")
REL=$(pair_release_dir "$SHA")
[ -d "$REL/src" ] || pair_die "release ${SHA:0:8} is not cut at $REL (run pair_cut.sh first)"
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json") \
  || pair_die "no RELEASE.json in $REL"
LIB_TIMEOUT=${LIB_TIMEOUT:-900}
MOE_TIMEOUT=${MOE_TIMEOUT:-1200}                # the hook links libtorch: two compiles per side

SEAM_FP=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["seam"]["fingerprint"])' "$REL/RELEASE.json")
GX_BUILD_AVAIL_GIB=${GX_BUILD_AVAIL_GIB:-60}
side_ct() { [ "$1" = p7 ] && echo "$P7CT" || echo "$GXCT"; }
side_sm() { [ "$1" = p7 ] && echo "$P7_ARCH" || echo "$GX_ARCH"; }
side_libdir() { [ "$1" = p7 ] && echo /usr/lib/x86_64-linux-gnu || echo /usr/lib/aarch64-linux-gnu; }
# One transport for "run this docker command on the side's host": P7 is local, GX goes through ssh.
side_docker() {  # <side> <args...>
  local side=$1; shift
  if [ "$side" = p7 ]; then
    docker "$@"
  else
    pair_gxsh "docker $(printf '%q ' "$@")"
  fi
}
side_hostsh() {  # <side> <shell-string>
  local side=$1
  if [ "$side" = p7 ]; then bash -c "$2"; else pair_gxsh "$2"; fi
}

# Verify is read-only: it reads the stamps and hashes both .so files, so it needs no build guard --
# a flip-only ticket verifies while a server loads on GX and while other builds own the box.
# Two artefacts carry two stamps with one rule: a stamp must name THIS commit and this arch, and the
# .so beside it must be the one the stamp hashed. A release cut before the hook had a build script
# declares native.moe_cpp=false in RELEASE.json and is not asked for the hook (it never had one).
verify_one() {  # <side> <arch> <name> <tag>
  local side=$1 arch=$2 name=$3 tag=$4
  local path="src/qmesh/native/bin/$name.$arch.so" got want
  if [ "$side" = p7 ]; then
    got=$(cat "$REL/$path.stamp" 2>/dev/null)
    want=$(sha256sum "$REL/$path" 2>/dev/null | cut -d' ' -f1)
  else
    got=$(pair_gxsh "cat '$REL/$path.stamp' 2>/dev/null")
    want=$(pair_gxsh "sha256sum '$REL/$path' 2>/dev/null" | cut -d' ' -f1)
  fi
  python3 - "$got" "$want" "$BUILD_ID" "$SHA" "$arch" "$side" "$tag" <<'PY'
import json, sys
got, want, bid, sha, arch, side, tag = sys.argv[1:8]
try:
    d = json.loads(got)
except Exception:
    print(f"{tag}_STAMP_MISSING {side} {arch}"); sys.exit(1)
bad = [k for k, v in (("build_id", bid), ("source_sha", sha), ("arch", arch)) if d.get(k) != v]
if bad:
    print(f"{tag}_STAMP_MISMATCH {side} {' '.join(f'{k}={d.get(k)}' for k in bad)}"); sys.exit(1)
if want and d.get("sha256") != want:
    print(f"{tag}_SHA_MISMATCH {side} on_disk={want} stamp={d.get('sha256')}"); sys.exit(1)
print(f"{tag}_OK {side} {arch} sha256={d.get('sha256')} build={d.get('build_id')}")
PY
}

verify_side() {
  local side=$1 sm arch cpp_req
  sm=$(side_sm "$side"); arch=$(pair_arch_for "$sm")
  cpp_req=$(python3 -c 'import json,sys;print(str(bool((json.load(open(sys.argv[1])).get("native") or {}).get("moe_cpp"))).lower())' \
    "$REL/RELEASE.json" 2>/dev/null)
  verify_one "$side" "$arch" "$PAIR_LIB" LIB || return 1
  if [ "$cpp_req" = true ]; then
    verify_one "$side" "$arch" "$PAIR_CPP_LIB" CPP || return 1
  else
    pair_step "CPP_SKIP $side $arch release declares native.moe_cpp=false"
  fi
  return 0
}

if [ "$MODE" = verify ]; then
  for side in p7 gx; do
    [ "$SIDES" = both ] || [ "$SIDES" = "$side" ] || continue
    verify_side "$side" || exit 1
  done
  exit 0
fi

# Dry run (stub-ssh harness only): say what would be built, write nothing. A manifest is refused
# without real stamps, so a PAIR_DRY run can never leave a pair looking updated.
if [ "$PAIR_DRY" = 1 ]; then
  for side in p7 gx; do
    [ "$SIDES" = both ] || [ "$SIDES" = "$side" ] || continue
    ct=$([ "$side" = p7 ] && echo "$P7CT" || echo "$GXCT")
    sm=$([ "$side" = p7 ] && echo "$P7_ARCH" || echo "$GX_ARCH")
    pair_step "DRY LIB_BUILD $side $ct arch=$(pair_arch_for "$sm") sm=$sm build=$BUILD_ID"
    pair_step "DRY CPP_BUILD $side $ct arch=$(pair_arch_for "$sm") sm=$sm build=$BUILD_ID"
  done
  exit 0
fi

build_side() {
  local side=$1 ct arch sm libdir inner out
  # Both artefacts share one guarded stage: the guards are about the box, not about one library.
  ct=$(side_ct "$side"); sm=$(side_sm "$side")
  arch=$(pair_arch_for "$sm"); libdir=$(side_libdir "$side")
  local lib="src/qmesh/native/bin/$PAIR_LIB.$arch.so"


  # ---- guards ------------------------------------------------------------------------------------
  local running
  running=$(side_docker "$side" inspect -f '{{.State.Running}}' "$ct" 2>/dev/null)
  [ "$running" = true ] || { pair_step "LIB_SKIP $side $ct not_running"; return 1; }
  if [ -n "$(side_docker "$side" exec "$ct" pgrep -f worker_seam_server.py 2>/dev/null)" ]; then
    pair_step "LIB_REFUSE $side server_running_in=$ct (never build under a live server)"; return 1
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
    pair_step "LIB_WAIT $side compiler_running=$builds stale_ignored=${ignored:-0} waited=${waited}s (another build owns the host)"
    sleep 30; waited=$((waited + 30))
    read -r builds ignored <<<"$(compilers "$side")"
  done
  [ "${builds:-0}" = 0 ] || { pair_step "LIB_REFUSE $side compiler_running=$builds waited=${waited}s"; return 1; }
  if [ "$side" = gx ]; then
    local avail
    avail=$(side_hostsh gx "awk '/^MemAvailable:/{print int(\$2/1048576)}' /proc/meminfo")
    [ "${avail:-0}" -ge "$GX_BUILD_AVAIL_GIB" ] \
      || { pair_step "LIB_REFUSE gx mem_avail=${avail:-?}GiB need=$GX_BUILD_AVAIL_GIB (a server is loading)"; return 1; }
  fi


  # ---- build -------------------------------------------------------------------------------------
  local tmp="/pair/releases/$SHA/.lib-$arch-$$"
  local flags="-O2 -std=c++17 -arch=$sm -Xcompiler -fPIC -shared -x cu qmesh_expert_seam.cu -o $tmp/$PAIR_LIB.$arch.so -libverbs -lcuda -lcudart -Xlinker -rpath=$libdir -I."
  # bin/ must end up owned by the host user: the container writes the .so as root, and the stamp
  # next to it is written here (a root-owned bin/ made the stamp write fail with "Permission denied").
  local uid gid; uid=$(id -u); gid=$(id -g)
  local bindir="$REL/src/qmesh/native/bin"
  if [ "$side" = p7 ]; then mkdir -p "$bindir"
  else pair_gxsh "mkdir -p '$bindir'"; fi
  inner="set -e; cd /pair/releases/$SHA/src/qmesh/native; mkdir -p bin $tmp; rm -rf $tmp/*; \
         nice -n 15 env MAX_JOBS=8 /usr/local/cuda/bin/nvcc $flags; \
         strip --strip-unneeded $tmp/$PAIR_LIB.$arch.so; \
         mv -f $tmp/$PAIR_LIB.$arch.so bin/$PAIR_LIB.$arch.so; rmdir $tmp; \
         chown $uid:$gid bin bin/$PAIR_LIB.$arch.so; \
         sha256sum bin/$PAIR_LIB.$arch.so"
  pair_step "LIB_BUILD $side $ct arch=$arch sm=$sm build=$BUILD_ID"
  out=$(side_docker "$side" exec "$ct" bash -lc "$inner" 2>&1)
  local rc=$?
  if [ "$rc" != 0 ]; then
    pair_step "LIB_FAIL $side rc=$rc $(printf '%s' "$out" | tail -5 | tr '\n' ' ')"
    return 1
  fi
  local sha_lib
  sha_lib=$(printf '%s\n' "$out" | awk '/^[0-9a-f]{64} /{print $1}' | tail -1)
  [ -n "$sha_lib" ] || { pair_step "LIB_FAIL $side no_sha_in_output"; return 1; }

  local cc
  cc=$(side_docker "$side" exec "$ct" /usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -1 | tr -s ' ')
  stamp_write() {  # <name> <flags> <sha256>
    local name=$1 sflags=$2 sha_lib=$3 stamp stamp_path
    stamp=$(python3 - "$BUILD_ID" "$SHA" "$arch" "$sm" "$SEAM_FP" "$sha_lib" "$cc" "$sflags" "$side" "$name" <<'PY'
import datetime, json, sys
bid, sha, arch, sm, fp, libsha, cc, flags, side, name = sys.argv[1:11]
print(json.dumps({
    "build_id": bid, "source_sha": sha, "arch": arch, "sm": sm, "lib": name,
    "seam": fp, "sha256": libsha, "compiler": cc, "flags": flags,
    "host": side, "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, indent=2, sort_keys=True))
PY
)
    stamp_path="$REL/src/qmesh/native/bin/$name.$arch.so.stamp"
    if [ "$side" = p7 ]; then
      printf '%s\n' "$stamp" > "$stamp_path" || return 1
      chmod 644 "$stamp_path"
    else
      printf '%s\n' "$stamp" | pair_gxsh "cat > '$stamp_path'" || return 1
      pair_gxsh "chmod 644 '$stamp_path'"
    fi
  }
  stamp_write "$PAIR_LIB" "$flags" "$sha_lib" || { pair_step "LIB_FAIL $side stamp_write"; return 1; }
  pair_step "LIB_OK $side arch=$arch sha256=$sha_lib build=$BUILD_ID stamp=$PAIR_LIB.$arch.so.stamp"

  # The declaration, not the file list, decides: a release cut before the hook existed tracks no
  # build_moe.sh and has nothing to build, and prepare must not fail on a tree that never had it.
  local cpp_req
  cpp_req=$(python3 -c 'import json,sys;print(str(bool((json.load(open(sys.argv[1])).get("native") or {}).get("moe_cpp"))).lower())' \
    "$REL/RELEASE.json" 2>/dev/null)
  if [ "$cpp_req" != true ]; then
    pair_step "CPP_SKIP $side $arch release declares native.moe_cpp=false"
    return 0
  fi

  # ---- the C moe hook -----------------------------------------------------------------------------
  # A second compiled artefact from the same tree, built under the same guards and stamped the same
  # way. It links libtorch on purpose (the hook calls the vendored cuda-exl3 kernels through the
  # dispatcher, so the python and C paths share one implementation), which is why its build is not
  # one nvcc line: build_moe.sh, tracked at the same commit, owns the flags. It is built even while
  # SEAM_MOE_CPP is unset in the pair env -- a default nobody has compiled is not a default.
  local moe="$PAIR_CPP_LIB.$arch.so" moe_tmp="/pair/releases/$SHA/.moe-$arch-$$"
  # A client image can carry a partial toolkit at /usr/local/cuda -- nvcc and cuda_runtime.h are there
  # but cusparse.h is not, and torch's ATen/cuda/CUDAContextLight.h includes it from the host compiler,
  # so the hook died on a missing header three files deep in torch's headers (2026-11-03, CPP_FAIL p7,
  # while the same stage built on GX where the header exists). Find a directory that has it and hand it
  # to g++ the way g++ reads it, CPLUS_INCLUDE_PATH; refuse if no directory does, because an include
  # path that only works on one box is not a build.
  inc_probe='if [ -f /usr/local/cuda/include/cusparse.h ]; then echo default; exit 0; fi
for c in /opt/cuda/include /opt/cuda/targets/*/include /usr/local/cuda*/targets/*/include; do
  [ -f "$c/cusparse.h" ] && { echo "$c"; exit 0; }
done
exit 1'
  incdir=$(side_docker "$side" exec "$ct" bash -c "$inc_probe" 2>/dev/null | tail -1)
  if [ -z "$incdir" ]; then
    pair_step "CPP_FAIL $side no_cusparse_h (no include dir under /opt/cuda* or /usr/local/cuda*/targets carries cusparse.h, which ATen/cuda/CUDAContextLight.h includes)"
    return 1
  fi
  local incenv=""
  [ "$incdir" = default ] || incenv="CPLUS_INCLUDE_PATH=$incdir"
  pair_step "CPP_TOOLKIT $side cusparse=$incdir"
  local moeflags="build_moe.sh: nvcc -O2 -std=c++17 -arch=$sm (kernels), g++ -O2 -std=c++20 -fPIC -pthread (torch ABI), link libtorch${incenv:+ inc=$incdir}"
  inner="set -e; cd /pair/releases/$SHA/src/qmesh/native; mkdir -p bin; rm -rf $moe_tmp; mkdir -p $moe_tmp; \
         nice -n 15 env MAX_JOBS=8 CUDA_ARCH=$sm MOE_OUT=$moe_tmp MOE_OBJ=$moe_tmp/obj $incenv bash build_moe.sh; \
         strip --strip-unneeded $moe_tmp/$moe; \
         mv -f $moe_tmp/$moe bin/$moe; rm -rf $moe_tmp; \
         chown $uid:$gid bin/$moe; \
         sha256sum bin/$moe"
  pair_step "CPP_BUILD $side $ct arch=$arch sm=$sm build=$BUILD_ID"
  out=$(side_docker "$side" exec "$ct" bash -lc "$inner" 2>&1)
  rc=$?
  if [ "$rc" != 0 ]; then
    pair_step "CPP_FAIL $side rc=$rc $(printf '%s' "$out" | tail -5 | tr '\n' ' ')"
    return 1
  fi
  sha_lib=$(printf '%s\n' "$out" | awk '/^[0-9a-f]{64} /{print $1}' | tail -1)
  [ -n "$sha_lib" ] || { pair_step "CPP_FAIL $side no_sha_in_output"; return 1; }
  stamp_write "$PAIR_CPP_LIB" "$moeflags" "$sha_lib" || { pair_step "CPP_FAIL $side stamp_write"; return 1; }
  pair_step "CPP_OK $side arch=$arch sha256=$sha_lib build=$BUILD_ID stamp=$moe.stamp"
  return 0
}

# The GX half claims the box for the same reason the ext build does: the transport is compiled there
# with 8 jobs next to whatever else has a pack loaded, and a load that lands mid-compile kills it. The
# claim can only be taken inside a queue ticket, so a GX build outside one is refused rather than run
# unclaimed. P7-only builds need no ticket.
rc=0
case "$SIDES" in
  both|gx)
    # the claim covers the whole stage: the seam lib and the hook are built back to back
    if ! pair_gx_gate_take "build_lib-$PAIR" $(( (LIB_TIMEOUT + MOE_TIMEOUT + 300) / 60 )) "${GX_GATE_WAIT_MIN:-20}"; then
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
