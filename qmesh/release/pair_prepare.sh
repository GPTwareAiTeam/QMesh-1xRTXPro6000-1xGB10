#!/usr/bin/env bash
# pair_prepare.sh -- cut, wire and build a release for one pair, OUTSIDE its queue.
#
#   pair_prepare.sh <pair> [sha] [--side p7|gx|both]
#
# Why this exists: the torch JIT build of a release takes 5-13 minutes per side, and doing it inside
# the PairUpdate ticket kept the pair's GPU idle for that whole time. Everything expensive happens
# here instead -- niced, MAX_JOBS<=8, writing only into <root>/releases/<sha> -- while the pair keeps
# running whatever it is running. The ticket that follows (`pair_update.sh <pair> [sha] --flip-only`)
# only flips `current`, restarts the server, smokes and verifies: minutes, not an hour.
#
# What it does NOT do: it never moves `current`/`extcache` toward the new release (the one exception
# is the first-release bootstrap that container wiring needs to exist at all), never starts, stops or
# recreates a serving container, and never touches another pair's containers.
#
# Guards (all of them refuse, none of them "work around"):
#   * a container of this pair that is serving (worker_seam_server.py inside it) -- the build runs in
#     that container and a rebuild under a live server has already SIGBUSed a box,
#   * a container name that belongs to another pair in pairs.conf, or is named by a live ticket,
#   * an existing container whose spec differs: creating containers is allowed, recreating is not,
#   * GX MemAvailable below GX_BUILD_AVAIL_GIB (checked by the builders themselves).
#
# Prints, in order: CUT / PLAN / CREATE|EXISTS / LINK_UNPROMOTED? / LIB_* / EXT_* and finally
#   PREPARED <pair> sha=<sha> build=<build_id> sides=<sides>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_prepare.sh <pair> [sha] [--side p7|gx|both]}
SHA_ARG=${2:-}
[ "${SHA_ARG:-}" = "--side" ] && SHA_ARG=""
SIDES=both
while [ $# -gt 0 ]; do
  case "$1" in
    --side) SIDES=${2:?--side needs p7|gx|both}; shift ;;
    -*) pair_die "unknown flag $1" ;;
  esac
  shift
done
pair_load "$PAIR_ARG"

# A prepare must never leave a pair container wired away from `current`. The ext build has to point
# /src/exllamav3 at the release it is building for the cache path to match a client's import path, and
# on 2026-11-02 the canary's GX build was killed mid-flight: the restore was a trap inside the
# container's `bash -lc`, so seam-gx2-dev stayed wired at the unflipped release and pair_check read GX
# DRIFT with nothing able to import /pair/current/src. The build restores from its parent process now,
# and this runs on every exit of prepare -- success, refusal, kill of the build -- as the last word.
# It also *creates* the wiring: a container recreated from a spec that carries no /src mount (someone
# `docker run`s one by hand beside pair_containers.py) comes back with no /src at all, and then the old
# `if [ -L /src/exllamav3 ]` guard was false on both sides, so this printed RELINK_MISMATCH and left
# the pair unrunnable -- `seam_server_ensure`'s `cd /src/exllamav3` fails, /tmp/seam_server.log is
# never created, and pair_check reads container_src=- for every client (2026-11-03 16:32Z). mkdir and
# repoint: a link of any other target is ours to overwrite on exit, that is the contract above.
relink_cmd='mkdir -p /src && ln -sfn /pair/current/src /src/exllamav3; readlink -f /src/exllamav3'
relink_current() {
  local side ct want got
  for side in p7 gx; do
    case "${SIDES:-both}" in both|"$side") ;; *) continue ;; esac
    ct=$([ "$side" = p7 ] && echo "$P7CT" || echo "$GXCT")
    if [ "$side" = p7 ]; then
      want=$(docker exec "$ct" readlink -f /pair/current/src 2>/dev/null)
      got=$(docker exec "$ct" sh -c "$relink_cmd" 2>/dev/null)
    else
      want=$(pair_gxsh "docker exec $ct readlink -f /pair/current/src 2>/dev/null")
      got=$(pair_gxsh "docker exec $ct sh -c '$relink_cmd' 2>/dev/null")
    fi
    if [ -z "$want" ]; then
      pair_step "RELINK_SKIP $side $ct (no /pair/current/src: unpromoted pair, nothing to restore)"
    elif [ "$got" = "$want" ]; then
      pair_step "RELINK $side $ct -> $got"
    else
      pair_step "RELINK_MISMATCH $side $ct want=$want got=${got:-?}"
    fi
  done
  return 0
}
trap relink_current EXIT

# ---- guards --------------------------------------------------------------------------------------
python3 "$HERE/pair_containers.py" guard "$PAIR" || exit 2
for side in p7 gx; do
  ct=$([ "$side" = p7 ] && echo "$P7CT" || echo "$GXCT")
  running=$([ "$side" = p7 ] && docker inspect -f '{{.State.Running}}' "$ct" 2>/dev/null \
                                || pair_gxsh "docker inspect -f '{{.State.Running}}' $ct 2>/dev/null")
  [ "$running" = true ] || continue
  n=$([ "$side" = p7 ] && docker exec "$ct" pgrep -fc worker_seam_server.py 2>/dev/null \
                       || pair_gxsh "docker exec $ct pgrep -fc worker_seam_server.py 2>/dev/null")
  [ "${n:-0}" = 0 ] || pair_die "PREPARE_REFUSED $side $ct is serving ($n worker_seam_server.py): " \
      "prepare needs a container of this pair that is not serving"
done

# Report a container that cannot reach the driver; do not repair it here. Prepare runs out of band,
# with no gate held, so a restart would kill whatever is running in it (2026-11-02). The ticket's smoke
# restarts it once it owns the box.
for side in p7 gx; do
  ct=$([ "$side" = p7 ] && echo "$P7CT" || echo "$GXCT")
  pair_nvml_ok "$side" "$ct" || echo "PREPARE_NVML_WARN $side ct=$ct (docker exec nvidia-smi -L fails; the ticket's smoke will restart it under the gate)" >&2
done

# ---- pack locality (refuse before cutting anything onto a wrong disk) ----------------------------
# Owner rule 2026-11-02: an expert pack lives on the local drive of the box that serves it. A pack
# reached over NFS (or through a symlink that lands on someone else's drive) makes serving speed
# depend on the network, so it is refused here -- before a release is cut or a library built -- and
# again by the smoke, which is the last gate before `current` moves.
# The gate refuses to *run* without PAIR_PACK_P7, because only the pair knows which client half of the
# split checkpoint serves GX's /cold. RM10 2026-11-05 (Main): it is NAMED, never derived -- `pair_ckpt_resolve
# auto` went ambiguous at 08:32 (a second exl3-p7hot-* on a live pack's sha) and voided three runs, so naming
# none refuses here, before anything is cut, rather than resolving to a half that may be the wrong one.
# `auto` still runs, as one PAIR_CKPT_XCHECK line; it never decides.
PAIR_PACK_P7=$(pair_ckpt_require "${PAIR_PACK_P7:-}") \
  || pair_die "PREPARE_FAILED $PAIR phase=packs reason=no_PAIR_PACK_P7 (export PAIR_PACK_P7=/models/dsv41/exl3-p7hot-<pack>)"
export PAIR_PACK_P7
pair_step "PREPARE_PACK_P7 ${PAIR_PACK_P7:-unnamed} cold=$(pair_cold_src "$GXCT" 2>/dev/null) sha=$(pair_cold_placement "$GXCT" 2>/dev/null | cut -c1-12)"
pack_out=$(python3 "$HERE/pair_pack_disk.py" "$PAIR" 2>&1); pack_rc=$?
printf '%s\n' "$pack_out" | sed 's/^/  /'
[ "$pack_rc" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=packs rc=$pack_rc (a pack is not on the serving box's own disk)"

# ---- 1. cut (or reuse an existing release of this sha) -------------------------------------------
cut_out=$("$HERE/pair_cut.sh" "$PAIR" $SHA_ARG 2>&1)
cut_rc=$?
printf '%s\n' "$cut_out" | sed 's/^/  /'
[ "$cut_rc" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=cut rc=$cut_rc"
SHA=$(printf '%s\n' "$cut_out" | tail -1)
REL=$(pair_release_dir "$SHA")
[ -n "$SHA" ] && [ -d "$REL/src" ] || pair_die "PREPARE_FAILED $PAIR phase=cut no_release_dir"
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json") \
  || pair_die "PREPARE_FAILED $PAIR phase=cut no_build_id"
pair_step "PREPARE_RELEASE pair=$PAIR sha=${SHA:0:12} build=$BUILD_ID"

# ---- 2. containers (create-only) -----------------------------------------------------------------
plan_out=$(python3 "$HERE/pair_containers.py" plan "$PAIR" 2>&1)
printf '%s\n' "$plan_out" | sed 's/^/  /'
case "$plan_out" in
*CREATE*|*RECREATE*)
  # A pair that has never run (prod on 2026-11-02) has no containers at all: the plan says CREATE and
  # apply makes them. Saying "unchanged" there left the release cut and the lib unbuilt, because the
  # build runs inside those containers. RECREATE still needs the ticket's PAIR_ALLOW_RECREATE, and the
  # apply refuses that here on its own.
  why=$(printf '%s\n' "$plan_out" | grep -o 'CREATE\|RECREATE' | head -1)
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  python3 "$HERE/pair_containers.py" apply "$PAIR" "$SHA" --ts "$ts" 2>&1 | sed 's/^/  /'
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=containers rc=$rc ($why: creating needs the " \
      "containers to be absent, recreating needs PAIR_ALLOW_RECREATE=1 inside a queued PairUpdate)"
  pair_step "PREPARE_CONTAINERS pair=$PAIR $why wired ts=$ts"
  ;;
*)
  pair_step "PREPARE_CONTAINERS pair=$PAIR unchanged"
  ;;
esac

# ---- 3. native lib -------------------------------------------------------------------------------
pair_step "PREPARE_LIB pair=$PAIR sides=$SIDES"
"$HERE/pair_build_lib.sh" "$PAIR" "$SHA" --side "$SIDES" 2>&1 | sed 's/^/  /'
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=lib rc=$rc"

# ---- 4. torch ext --------------------------------------------------------------------------------
pair_step "PREPARE_EXT pair=$PAIR sides=$SIDES"
"$HERE/pair_build_ext.sh" "$PAIR" "$SHA" --side "$SIDES" 2>&1 | sed 's/^/  /'
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=ext rc=$rc"

# ---- 5. the pair manifest must describe what we just built ---------------------------------------
# A lib or ext rebuild changes the artifact's sha256 and its stamp, and the pair manifest is what
# pair_sync_check.py compares those against -- so a prepare that rebuilds and stops leaves the pair
# permanently DRIFT on a hash nobody can fix by rebuilding (2026-11-03 17:04Z: the /src repair rebuilt
# the seam lib both sides and ext on P7, stamps said 17:07, the manifest still named 13:15's build, and
# both halves read `lib_so_hash_vs_manifest`). pair_update_run.sh writes the manifest after its own
# build+flip; prepare is the other path that builds, so it refreshes the same way -- but only for the
# release the pair actually points at: a candidate sha prepared beside a serving pair must not rewrite
# the description of what is being served.
cur=$(pair_current_sha "$PAIR_ROOT" 2>/dev/null)
if [ "$cur" = "$SHA" ]; then
  ref="$PAIR_STATE_DIR/refs-$SHA.json"
  python3 "$HERE/pair_manifest.py" write "$PAIR" "$SHA" ${ref:+"--ref" "$ref"} --by pair_prepare 2>&1 \
    | sed 's/^/  /'
  [ "${PIPESTATUS[0]:-1}" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=manifest rc=${PIPESTATUS[0]}"
  pair_step "PREPARE_MANIFEST pair=$PAIR sha=${SHA:0:12} refreshed"
else
  pair_step "PREPARE_MANIFEST pair=$PAIR skipped (built ${SHA:0:12} is not current=${cur:-none})"
fi

pair_step "PREPARED pair=$PAIR sha=$SHA build=$BUILD_ID sides=$SIDES"
echo "$SHA"
