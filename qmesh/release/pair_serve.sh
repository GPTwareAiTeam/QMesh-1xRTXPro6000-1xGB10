#!/usr/bin/env bash
# pair_serve.sh -- load the pair's current release and leave its server serving.
#
#   queue/run_prod.sh ServerUp 20 "scripts/pair_serve.sh prod"
#   queue/run_dev.sh  ServerUp 20 "scripts/pair_serve.sh dev"
#
# The smoke and the update both start a server and both stop it when they end, so a pair that is not
# in the middle of a ticket has nothing serving -- and "the pair is down" is a state nobody chose. This
# is the one explicit way to reach the steady state instead of the middle of an operation: it loads
# from the release `current` names (never a working tree), records the worker in the pair's pidfile so
# a later ticket can stop exactly that process, and drops its box claim as soon as the load is done --
# the claim protects the load, the server outliving it is the pair's job, not a leak.
#
# It refuses when the release is not IN_SYNC: serving a tree whose artifact came from somewhere else
# is the ad-hoc delivery this whole system exists to end.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

pair_load "${1:?usage: pair_serve.sh <prod|dev>}"
SHA=${2:-$(pair_current_sha "$PAIR_ROOT")}
[ -n "$SHA" ] || pair_die "PAIR_SERVE_NO_RELEASE pair=$PAIR (no $PAIR_ROOT/current)"
REL=$(pair_release_dir "$SHA")
[ -d "$REL" ] || pair_die "PAIR_SERVE_NO_RELEASE pair=$PAIR sha=${SHA:0:12} dir=$REL"
[ -f "$REL/RELEASE.json" ] || pair_die "PAIR_SERVE_NO_RELEASE pair=$PAIR sha=${SHA:0:12} (no RELEASE.json)"
BUILD_ID=$(python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("build_id",""))
except Exception: print("")' "$REL/RELEASE.json")
ENSURE=$PAIR_SCRIPTS/seam_server_ensure.sh
[ -x "$ENSURE" ] || pair_die "PAIR_SERVE_NO_ENSURE $ENSURE"
pair_step "PAIR_SERVE_START pair=$PAIR sha=${SHA:0:12} ct=$GXCT build=${BUILD_ID:-unknown}"

# IN_SYNC first: the sync check is the only thing that knows whether the on-disk artifact is the one
# the stamp names, and a server loading a mismatched extension is exactly the 08:03Z smoke.
chk=$("$PAIR_SCRIPTS/pair_check.sh" "$PAIR" 2>&1)
printf '%s\n' "$chk" | sed 's/^/  /'
case "$chk" in
  *DRIFT*|*MISSING*) pair_die "PAIR_SERVE_DRIFT pair=$PAIR (build it first: scripts/pair_build_ext.sh $PAIR --side p7|gx)" ;;
esac
case "$chk" in
  *"RESULT $PAIR P7 IN_SYNC"*"RESULT $PAIR GX IN_SYNC"*) ;;
  *) pair_die "PAIR_SERVE_NOT_IN_SYNC pair=$PAIR (pair_check did not report both halves IN_SYNC)" ;;
esac

if [ "$(pair_gx_foreign_server)" != "" ]; then
  pair_die "PAIR_SERVE_FOREIGN $(pair_gx_foreign_server) (one pack load per box)"
fi
if [ "$(pair_gxsh "docker exec $GXCT pgrep -fc '^python3 -u qmesh/worker_seam_server.py'" 2>/dev/null)" != 0 ]; then
  serving=$(pair_gx_serving_build)
  # Something already serves here. If it is this exact build, the pair is already resting where this
  # ticket wanted to put it: a restore queued as a safety net behind an evicting ticket must be a no-op
  # there, not a second 116 GiB load that evicts the next ticket's measurement. Anything else is
  # somebody's work and the caller has to decide to stop it first; silently reusing it would mean the
  # pair serves whatever the last ticket left loaded.
  if [ -n "$BUILD_ID" ] && [ "$serving" = "$BUILD_ID" ]; then
    pair_server_record "$BUILD_ID" || true
    pair_step "PAIR_SERVE_ALREADY_UP pair=$PAIR build=$BUILD_ID sha=${SHA:0:12} ct=$GXCT $(cat "$(pair_server_pidfile)" 2>/dev/null)"
    exit 0
  fi
  pair_die "PAIR_SERVE_ALREADY $(pair_server_pidfile) $(cat "$(pair_server_pidfile)" 2>/dev/null) serving=${serving:-unknown}"
fi

pair_gx_gate_take "serve-$PAIR" $(( ${SERVE_TIMEOUT_MIN:-25} )) 0 \
  || pair_die "PAIR_SERVE_GX_BUSY $(pair_gx_gate_state)"

# The box is ours now, so the two things the smoke checks are checked here too, in the same order: a
# container whose device cgroup was re-derived by a systemd reload is alive and cannot see the GPU, and
# restarting it is seconds against a whole ticket; and a JIT lock left by a build killed mid-link blocks
# every import at 512 MB with an empty log. Prod's first serve died exactly this way ("No CUDA GPUs are
# available" in the worker's init, 30 s in), which is a restart, not a release problem.
pair_ensure_nvml p7 "$P7CT" || { pair_gx_gate_down; pair_die "PAIR_SERVE_NVML p7=$P7CT"; }
pair_ensure_nvml gx "$GXCT" || { pair_gx_gate_down; pair_die "PAIR_SERVE_NVML gx=$GXCT"; }
locks=$(pair_torch_locks "$GXCT")
if [ -n "$locks" ]; then
  if [ "$(pair_torch_build_live "$GXCT")" = 0 ]; then
    pair_gxsh "docker exec $GXCT sh -c 'rm -f /root/.cache/torch_extensions/*/*/lock /tmp/torch_extensions/*/*/lock'" >/dev/null 2>&1
    # One line per path, in the token GxPort2's driver greps: "which import was blocked" is the question
    # a post-mortem asks, and a comma-joined list in the cleared= line answers only for one of them.
    for _f in $locks; do pair_step "STALE_LOCK_REMOVED path=$_f side=gx ct=$GXCT"; done
    pair_step "PAIR_SERVE_TORCH_LOCK cleared=$(printf '%s' "$locks" | tr '\n' ',')"
  else
    pair_gx_gate_down
    pair_die "PAIR_SERVE_TORCH_LOCK kept (a build or server is alive beside it): $(printf '%s' "$locks" | tr '\n' ' ')"
  fi
fi

# The pack is 113 GiB and a previous server's memory is not returned the instant it exits, so the load
# waits for the box to cover the pack plus the margin instead of failing the serve on a transient.
PACK_MIB=$(pair_gxsh "docker exec $GXCT sh -c 'du -bc /cold/cold-L*.safetensors 2>/dev/null | tail -1'" 2>/dev/null | cut -f1)
case "${PACK_MIB:-}" in ''|*[!0-9]*) PACK_MIB=0 ;; esac
if [ "$PACK_MIB" -gt 0 ]; then
  want=$(( PACK_MIB / 1048576 + LOAD_MARGIN ))
  waited=0
  while :; do
    avail=$(pair_gxsh "docker exec $GXCT sh -c 'awk \"/MemAvailable/{print int(\\\$2/1024)}\" /proc/meminfo'" 2>/dev/null | tr -dc 0-9)
    [ -n "${avail:-}" ] && [ "$avail" -ge "$want" ] && break
    [ "$waited" -lt "${SERVE_MEM_WAIT_S:-180}" ] || { pair_gx_gate_down; pair_die "PAIR_SERVE_MEM avail=${avail:-?} want=$want (pack $((PACK_MIB/1048576)) + margin $LOAD_MARGIN MiB)"; }
    sleep 10; waited=$(( waited + 10 ))
  done
  pair_step "PAIR_SERVE_MEM avail=${avail} want=$want waited=${waited}s"
fi

out=$(GX=$GX CT=$GXCT GX_DEV=${GX_DEV:-rocep1s0f0} GX_GID=$GX_GID PORT=$PORT \
      SEAM_GATE_DIR="$(dirname "$(pair_gx_gate_file)")" \
      SERVER_BUCKETS=$BUCKETS SEAM_SERVER_TIMEOUT=${SERVE_SERVER_TIMEOUT:-600} \
      SEAM_SLAB_SLOTS=${SERVE_SLAB_SLOTS:-1} \
      SEAM_ENV="$(pair_server_env "$BUILD_ID")" \
      "$ENSURE" --fresh --buckets "$BUCKETS" 2>&1)
rc=$?
[ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$out" >> "$PAIR_LOG"
printf '%s\n' "$out" | grep -E '^(SEAM_ENSURE|SEAM_SLACK|SEAM_NO_MEMORY|SEAM_NO_PACK|SEAM_SERVER)' | sed 's/^/  /'
if [ "$rc" != 0 ]; then
  pair_gx_gate_down
  pair_die "PAIR_SERVE_FAILED pair=$PAIR build=${BUILD_ID:-unknown} rc=$rc"
fi
pair_server_record "$BUILD_ID" || { pair_gx_gate_down; pair_die "PAIR_SERVE_NO_PID pair=$PAIR (the load returned 0 but no worker argv was found)"; }
pair_gx_gate_down
pair_step "PAIR_SERVE_UP pair=$PAIR sha=${SHA:0:12} build=${BUILD_ID:-unknown} $(cat "$(pair_server_pidfile)")"
