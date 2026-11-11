#!/usr/bin/env bash
# pair_update_run.sh -- move one pair onto a release, under the pair's queue, and verify it there.
#
#   pair_update_run.sh <pair> [sha]
#
# Run by the queue as agent PairUpdate (see pair_update.sh); running it by hand is allowed only with
# the GPU idle. Phases, in this order, each stopping the job on real failure:
#
#   1. cut      the release exists on P7 and on the pair's GX box (git archive, same path both sides)
#   2. containers  recreate any container whose wiring differs from qmesh/release/containers/*.json
#               (mounts /pair + /src/exllamav3 -> /pair/current/src); the old one is kept as .bak-<ts>
#   3. libs     native seam lib for both arches, built from the release tree, stamped beside the .so
#   4. ext      torch JIT cache for the release, built from the release tree (not from `current`)
#   5. flip     current -> releases/<sha> on both sides, and the in-container symlink
#   6. smoke    start the server from that release, check GX memory, run one decode row
#   7. manifest write pairs/<pair>.json, then pair_check must return 0 -- the same check the queue
#               guard runs before it starts anybody's job
#
# Nothing here is allowed to stop a container or the server on interruption: a wrapper that did that
# once outlived its job and took down the next arm's server. The TERM/INT trap flips the symlink back
# (so the next job sees the release it saw before) and exits; explicit failures are rolled back by
# hand in restore().
set -uo pipefail
# bash reads a script lazily, and this file lives in the installed directory that pair_install.sh replaces in
# place: a replace or a hand edit mid-run makes the rest of the file parse as garbage, and an update runs for
# an hour (2026-11-03: a canary smoke died at line 580 with "syntax error near unexpected token `('" after the
# arm had been measured, because the installed copy was edited underneath it). So run from a private copy; the
# copy takes HERE from PAIR_SELF_HERE, because its own $0 is in /tmp.
if [ "${PAIR_UPD_SELF:-0}" != 1 ]; then
  PAIR_SELF_COPY=$(mktemp "${TMPDIR:-/tmp}/pair_update_run.XXXXXX") || exit 2
  PAIR_SELF_HERE=$(cd "$(dirname "$0")" && pwd)
  if cp "$0" "$PAIR_SELF_COPY"; then
    PAIR_UPD_SELF=1 PAIR_SELF_HERE=$PAIR_SELF_HERE PAIR_SELF_COPY=$PAIR_SELF_COPY \
      exec bash "$PAIR_SELF_COPY" "${@}"
  fi
  echo "UPDATE_FAILED no_self_copy $0"; exit 2
fi
HERE=${PAIR_SELF_HERE:-$(cd "$(dirname "$0")" && pwd)}
# Until on_exit/on_term (which restore the pair) exist, a refusal in between must still drop the copy.
trap 'rm -f "${PAIR_SELF_COPY:-}"' EXIT
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_update_run.sh <pair> [sha]}
SHA_ARG=${2:-}
pair_load "$PAIR_ARG"
# RM10 2026-11-05 (Main): the P7 client half is named, never derived (see pair_ref.sh:33). The release
# gate this runs (pair_smoke.sh) refuses without it, and the canary is the wrong place to find out.
PAIR_PACK_P7=$(pair_ckpt_require) || exit 1
export PAIR_PACK_P7

# The script that actually flips checks the canary too, so queueing it directly is not a way around
# pair_update.sh/pair_promote.sh. It needs the sha up front: dev evidence is recorded per sha, and a
# prod flip of "whatever current happens to point at" is exactly the drift this exists to stop.
if [ "$PAIR" = prod ] && [ "${PAIR_DRY:-0}" != 1 ]; then
  [ -n "$SHA_ARG" ] || pair_die "prod needs an explicit sha: dev canary evidence is recorded per sha" \
    "(use scripts/pair_promote.sh <sha>)"
  canary=$(pair_canary_gate "$SHA_ARG" "$PAIR_STATE/dev/evidence/$SHA_ARG.json")
  [ "$canary" = ok ] || pair_die "prod refused: dev canary for ${SHA_ARG:0:12} does not qualify: " \
    "$canary (use scripts/pair_promote.sh <sha>)"
  pair_step "UPDATE_CANARY pair=prod sha=${SHA_ARG:0:12} dev_evidence=$PAIR_STATE/dev/evidence/$SHA_ARG.json ok"
fi

mkdir -p "$PAIR_STATE_DIR/logs"
PAIR_LOG=${PAIR_LOG:-$PAIR_STATE_DIR/logs/update-$PAIR-$(date -u +%Y%m%dT%H%M%SZ).log}
export PAIR_LOG
ln -sfn "$(basename "$PAIR_LOG")" "$PAIR_STATE_DIR/logs/latest-update-$PAIR.log" 2>/dev/null || true
PHASE=start; TS=""; OLD_SHA=""; FLIPPED=""; VERIFIED=""
# PAIR_DRY only: the stub harness sets PAIR_DRY_FAIL_AT=<phase> to force a failure there and exercise
# the restore path. A real run never sets PAIR_DRY, so this can not fire on the boxes.
dry_gate() {
  if [ "${PAIR_DRY:-0}" = 1 ] && [ "${PAIR_DRY_FAIL_AT:-}" = "$1" ]; then
    fail "PAIR_DRY_FAIL_AT=$1 (harness-injected)"
  fi
  return 0
}
phase() { PHASE=$1; dry_gate "$1"; }
status_write() {  # <eta_min> <doing>
  [ -n "${GPU_RUN_STATUS:-}" ] && echo "eta_min=$1 doing=$2" > "$GPU_RUN_STATUS" 2>/dev/null
  return 0
}
# The release's evidence: what a promotion (pair_promote.sh, and the prod gate in pair_update.sh)
# reads to decide whether this pair may be copied to the other one. It is written per sha, not per
# pair, because the pair's manifest moves on with the next release while a promotion may come later.
#   evidence <ok|failed> <why>
evidence() {
  local ok=$1 why=${2:-}
  [ "${PAIR_DRY:-0}" = 1 ] && return 0
  python3 - "$PAIR_STATE_DIR/evidence/${SHA:-none}.json" "$PAIR" "${SHA:-}" "${BUILD_ID:-}" \
           "$ok" "$why" "$PHASE" "${PAIR_LOG:-}" "${SMOKE_ROW:-}" "${SMOKE_IDS:-}" \
           "$P7CT" "$GXCT" "${REL:-}" "${SMOKE_MIX:-}" "${SMOKE_PREFILL_STATE:-}" <<'PY' 2>&1 | sed 's/^/  /' | tee -a "${PAIR_LOG:-/dev/null}"
import datetime, json, os, sys
path, pair, sha, build_id, ok, why, phase, log, row, ids, p7ct, gxct, rel, mix, pfstate = sys.argv[1:16]
if not sha:
    print("EVIDENCE_SKIP no release sha yet")
    sys.exit(0)
seam = ""
try:
    seam = json.load(open(os.path.join(rel, "RELEASE.json"))).get("seam", {}).get("fingerprint", "")
except Exception:                                                       # noqa: BLE001
    pass
ids_doc = {}
for tok in ids.split():
    k, _, v = tok.partition("=")
    if k == "n":
        ids_doc["n"] = int(v)
    elif k == "sha256":
        ids_doc["sha256"] = v
# Which layers the cuda-exl3 restack refused on the smoke's load: the digest is only meaningful
# together with it (one release drew 38/2 on one load and 39/1 on the next, 2026-11-02).
mix_doc = {}
for tok in mix.split():
    k, _, v = tok.partition("=")
    if k in ("cx_layers", "mixedk_layers") and v.isdigit():
        mix_doc[k] = int(v)
    elif k == "nofit_layers":
        mix_doc[k] = v
doc = {
    "format": 1,
    "pair": pair,
    "release_sha": sha,
    "build_id": build_id,
    "seam": seam,
    "ok": ok == "ok",
    "failed_at_phase": "" if ok == "ok" else phase,
    "why": why,
    # prefill: "judged" means the 32k prefill rate was within the box's memory floor and cleared the
    # threshold; "unjudged" means the box was starved (see SMOKE_PREFILL_MIN_AVAIL_MIB) and the rate
    # says nothing about the release -- a promotion of this sha must see which one it was.
    "smoke": {"row": row, "ids": ids_doc, "kernel": mix_doc, "prefill": pfstate or "none"},
    "containers": {"p7": p7ct, "gx": gxct},
    "log": log,
    "written_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
os.makedirs(os.path.dirname(path), exist_ok=True)
# The admission record is not a receipt for the latest attempt: an ok file is what admits the sha to
# prod, so a failed re-run of a release that already passed must not erase it (2026-11-02: a dev
# re-smoke of the live release wrote ok=false over the admission prod had been promoted on). A failed
# attempt is still recorded -- under its own build id -- and if there is no admission yet it takes the
# plain name, so a promotion attempt sees exactly why the sha does not qualify.
admitted = False
if os.path.exists(path):
    try:
        admitted = bool(json.load(open(path)).get("ok"))
    except Exception:                                                   # noqa: BLE001
        admitted = False
fate = "admission"
if not doc["ok"] and admitted:
    dest = path[:-5] + f".{build_id or 'failed'}.failed.json"
    fate = "kept"
else:
    dest = path
tmp = f"{dest}.tmp.{os.getpid()}"
json.dump(doc, open(tmp, "w"), indent=2, sort_keys=True)
open(tmp, "a").write("\n")
os.replace(tmp, dest)
print(f"EVIDENCE {dest} pair={pair} sha={sha[:12]} ok={doc['ok']} ids={ids_doc.get('n', 0)}"
      f" admission={fate}")
PY
}
# Who else is on this pair: tickets in its queue (not our own) or a live client/server process in
# either of its containers. Recreating a container SIGKILLs an arm someone else is running (2026-11-01
# 09:39: an 12-minute prefill arm died with CLIENT_RC=137 and no traceback), so this refuses instead.
# Only processes matter: the queue already serialises tickets, so a queued ticket behind ours can not
# race us, but an arm exec'd into the container (or a server still up) will be killed by the recreate.
pair_others_busy() {
  local n
  # pgrep -f counts a *defunct* process, and a defunct process holds nothing -- no memory, no GPU, no port. A
  # bench that was killed leaves a zombie whose parent is the container's PID 1, and PID 1 in these containers
  # never reaps, so the zombie is permanent. On 2026-11-03 10:14 the a16465e4 canary was refused with
  # `pair_busy p7_procs=1` while the only match was `[qes_bench.x86_6] <defunct>` from a ticket that had
  # already ended -- a refusal with no live process behind it, and it would have refused every ticket after it.
  # So the count comes from ps state+args rather than pgrep: a Z state is skipped, and the awk that does the
  # skipping excludes its own command line.
  n=$(docker exec "$P7CT" ps -eo stat=,args= 2>/dev/null |
        awk '$1 !~ /^Z/ && /worker_seam_server|pair_arm|pair_p7_smoke|qes/ && !/awk/ {n++} END{print n+0}')
  if [ "${n:-0}" != 0 ]; then printf 'p7_procs=%s ' "$n"; fi
  n=$(pair_gxsh "docker exec $GXCT ps -eo stat=,args= 2>/dev/null | awk '\$1 !~ /^Z/ && /worker_seam_server|pair_arm|qes/ && !/awk/ {n++} END{print n+0}'")
}
gpu_free() {
  # A dry run has no GPU to look at -- on a shared box it would be looking at somebody else's job (a
  # colleague's A/B on the same card) and refusing for a reason that has nothing to do with this
  # release. Only PAIR_DRY may state the answer; a real ticket always asks nvidia-smi.
  if [ "$PAIR_DRY" = 1 ] && [ -n "${PAIR_DRY_GPU:-}" ]; then
    [ "$PAIR_DRY_GPU" = free ]
    return
  fi
  local used
  used=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
  [ -z "$used" ]
}
on_term() {
  pair_step "PAIR_UPDATE_INTERRUPTED pair=$PAIR phase=$PHASE"
  status_write 1 "interrupted"
  # Only the symlink: cheap, and it keeps the next job on the release it was checked against.
  if [ -n "$FLIPPED" ] && [ -n "$OLD_SHA" ] && [ -z "$VERIFIED" ]; then
    pair_step "RESTORE_LINK pair=$PAIR -> ${OLD_SHA:0:12} (interrupted before verification)"
    python3 "$HERE/pair_containers.py" link "$PAIR" "$OLD_SHA" >/dev/null 2>&1 \
      || pair_step "RESTORE_LINK_FAILED pair=$PAIR old=${OLD_SHA:0:12}"
  fi
  [ -n "$TS" ] && echo "recreated at ts=$TS; undo with: $HERE/pair_containers.py rollback $PAIR $TS" \
      > "$PAIR_STATE_DIR/restore-$PAIR-$TS.txt"
  # The ticket's own client and the server it loaded go with it. This is not the old pattern-kill that
  # took down the next arm's server: pair_server_stop addresses the pid this ticket recorded, after
  # checking that pid still runs worker_seam_server.py, and does nothing when none was recorded (a
  # ticket interrupted before the smoke, or after a verified flip, keeps the serving server).
  cli_stop=$(pair_client_stop 2>/dev/null || true)
  [ -n "$cli_stop" ] && pair_step "RESTORE_$cli_stop"
  srv_stop=$(pair_server_stop 2>&1 || true)
  [ -n "$srv_stop" ] && pair_step "RESTORE_$srv_stop"
  exit 143
}
trap on_term TERM INT

restore() {  # only on explicit failure, never from a trap
  pair_step "RESTORE_BEGIN pair=$PAIR phase=$PHASE"
  if [ -n "$FLIPPED" ] && [ -n "$OLD_SHA" ]; then
    python3 "$HERE/pair_containers.py" link "$PAIR" "$OLD_SHA" >/dev/null 2>&1 \
      && pair_step "RESTORE_LINK pair=$PAIR -> ${OLD_SHA:0:12}" \
      || pair_step "RESTORE_LINK_FAILED pair=$PAIR old=${OLD_SHA:0:12}"
  fi
  if [ -n "$TS" ]; then
    python3 "$HERE/pair_containers.py" rollback "$PAIR" "$TS" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
    pair_step "RESTORE_CONTAINERS pair=$PAIR ts=$TS"
  fi
  if [ "${SERVER_BEFORE:-0}" = 0 ]; then
    # Nothing was serving before this update, so the box goes back the way it was found. The smoke
    # started a server from the release under test; leaving it loaded holds the pack (116 GiB) and
    # blocks the next attempt exactly like a leak would. On success the server stays up: there it is
    # the update's own result, not leftover state.
    # Anchored: the launcher's shell carries the script name in its cmdline, and stopping the shell
    # leaves the worker holding the pack -- the leak this line exists to prevent.
    if pair_gxsh "docker exec $GXCT pkill -f '^python3 -u qmesh/worker_seam_server.py'" >/dev/null 2>&1; then
      pair_step "RESTORE_SERVER_STOP pair=$PAIR (the failed smoke's server stopped; none was serving before)"
    else
      pair_step "RESTORE_SERVER_SKIP pair=$PAIR (no server was serving before this update)"
    fi
  elif [ -n "$OLD_SHA" ] && [ -d "$(pair_release_dir "$OLD_SHA")" ]; then
    local old_build
    old_build=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("build_id",""))' \
                "$(pair_release_dir "$OLD_SHA")/RELEASE.json" 2>/dev/null)
    # The restore server starts only if this ticket can claim the box: the ensure refuses a launch
    # without a self/ancestor gate claim, and a box that somebody else has taken is exactly the case
    # where the restored server must wait rather than load 113 GiB next to a running experiment.
    if ! pair_gx_gate_take "restore-$PAIR" 15 0; then
      pair_step "RESTORE_SERVER_SKIP pair=$PAIR (gx box is claimed: $(pair_gx_gate_state))"
      pair_step "RESTORE_END pair=$PAIR"
      return 0
    fi
    pair_step "RESTORE_SERVER pair=$PAIR build=${old_build:-unknown}"
    GX=$GX CT=$GXCT GX_DEV=${GX_DEV:-rocep1s0f0} GX_GID=$GX_GID PORT=$PORT SERVER_BUCKETS=$BUCKETS \
      SEAM_SERVER_TIMEOUT=${RESTORE_TIMEOUT:-600} SEAM_SLAB_SLOTS=${SMOKE_SLAB_SLOTS:-1} \
      SEAM_ENV="$(pair_server_env "${old_build:-}")" \
      "$PAIR_SCRIPTS/seam_server_ensure.sh" --fresh --buckets "$BUCKETS" 2>&1 \
      | grep -E '^(SEAM_ENSURE|SEAM_SLACK|SEAM_SERVER_READY)' | sed 's/^/  /' | tee -a "$PAIR_LOG"
    pair_step "RESTORE_SERVER_DONE pair=$PAIR rc=${PIPESTATUS[0]}"
    pair_gx_gate_down
  fi
  pair_step "RESTORE_END pair=$PAIR"
}
fail() {
  pair_step "PAIR_UPDATE_FAILED pair=$PAIR phase=$PHASE why=$*"
  status_write 1 "rollback ($PHASE)"
  # The failed attempt is evidence too: a later promotion of this sha must see it and say why.
  [ -n "${SHA:-}" ] && evidence failed "$*"
  restore
  exit 1
}
# memory_fail: the smoke could not load a server because the GX box had no room for the pack, which
# says nothing about the release. The code stays installed (current stays moved), the evidence says
# why, and the manifest is not written -- so the pair fails closed and its queue asks a human instead
# of running jobs on an unverified release. Rolling the code back here would only re-hide the box.
memory_fail() {
  local why=$*
  pair_step "PAIR_UPDATE_FAILED pair=$PAIR phase=$PHASE why=memory:$why"
  printf '%s\n' "${smoke_out:-}" | grep -E '^SMOKE_MEM|^SMOKE_SERVER_(SLAB|SWAP_WARN|LOADED)' \
    | sed 's/^/  /' | tee -a "$PAIR_LOG"
  [ -n "${SHA:-}" ] && evidence failed "memory: $why"
  status_write 0 "smoke failed (box memory); release stays installed, no manifest"
  pair_step "PAIR_UPDATED pair=$PAIR sha=$SHA build=$BUILD_ID smoke=FAILED_MEMORY rollback=none manifest=none"
  pair_step "PAIR_UPDATE_RECOURSE pair=$PAIR (release installed and current, server never loaded:" \
    "free $GX, then scripts/pair_update.sh $PAIR $SHA --flip-only to re-smoke and write the manifest)"
  exit 1
}

pair_step "UPDATE_START pair=$PAIR gpu=$GPU ct=$P7CT/$GXCT root=$PAIR_ROOT log=$PAIR_LOG"
if ! gpu_free && [ "${PAIR_UPDATE_FORCE:-0}" != 1 ]; then
  pair_step "PAIR_UPDATE_REFUSED pair=$PAIR gpu_busy (GPU $GPU has a compute app; queue it instead)"
  exit 3
fi
for t in "$PAIR_QUEUE"/gpu_queue/[0-9]*-*; do
  [ -e "$t" ] || continue
  base=$(basename "$t")
  case "$base" in *-PairUpdate|*-PairUpdateRun) continue ;; esac
  pair_step "UPDATE_QUEUE_NOTE pair=$PAIR other_ticket=$base (queued behind us; the queue serialises)"
done
# An orphan this pair's own ticket left behind is not somebody else's work: a ticket killed before its
# trap ran (SIGKILL, a container restart) leaves the pack resident, and refusing there is how the 23:54
# smoke spent the GX2 window on `pair_busy gx_procs=2` while a colleague waited. The pid file names it,
# pair_server_stop checks the pid still is that worker, so a foreign server is still refused below.
orphan=$(pair_server_stop 2>&1 || true)
case "$orphan" in
  SERVER_STOP\ *) pair_step "UPDATE_ORPHAN_${orphan#SERVER_STOP } (this pair's interrupted ticket left it; stopped)" ;;
  *)              [ -n "$orphan" ] && pair_step "UPDATE_ORPHAN_NOTE $orphan" ;;
esac
others=$(pair_others_busy)
if [ -n "$others" ] && [ "${PAIR_UPDATE_FORCE:-0}" != 1 ]; then
  pair_step "PAIR_UPDATE_REFUSED pair=$PAIR pair_busy $others (a client or ticket owns this pair; wait for it)"
  exit 3
fi
[ -n "$others" ] && pair_step "UPDATE_FORCED pair=$PAIR pair_busy $others (PAIR_UPDATE_FORCE=1)"
# Whether this pair was actually serving before the update decides what the restore has to put back.
# A pair that was idle (the normal case: the busy check above refuses a serving one) must be left
# idle on failure -- restarting a server for a release that never served costs a full 113 GiB pack
# load on GX2, i.e. minutes of the very box the smoke just proved is memory-tight.
foreign=$(pair_gx_foreign_server)
if [ -n "$foreign" ] && [ "${PAIR_UPDATE_FORCE:-0}" != 1 ]; then
  # Before anything is cut, flipped or re-created: a pack load in another container owns the box.
  # On 2026-11-02 02:41 a ticket re-created P7 and flipped before its smoke found this, then had to
  # roll both back -- a busy box must cost a refusal, not a rollback.
  pair_step "PAIR_UPDATE_REFUSED pair=$PAIR gx_busy ${foreign} (one pack load per box; wait for it)"
  exit 3
fi
[ -n "$foreign" ] && pair_step "UPDATE_FORCED pair=$PAIR gx_busy ${foreign} (PAIR_UPDATE_FORCE=1)"
SERVER_BEFORE=$(pair_gxsh "docker exec $GXCT pgrep -fc worker_seam_server.py 2>/dev/null" | tail -1)
SERVER_BEFORE=${SERVER_BEFORE:-0}
pair_step "UPDATE_SERVER_BEFORE pair=$PAIR serving=$SERVER_BEFORE"
status_write 120 "cutting release"

# ---- 1..4: release, containers, libs, ext --------------------------------------------------------
# A flip-only ticket is what pair_update.sh enqueues after `pair_prepare.sh <pair> <sha>` has already
# cut, wired and built the release outside the queue. Nothing is built here: the stamps must already
# name this exact sha and build, and the containers must already be wired -- otherwise the ticket
# stops before the flip and says which prepare to run, instead of holding the GPU for an hour of JIT.
if [ "${PAIR_FLIP_ONLY:-0}" = 1 ]; then
  phase prepared
  [ -n "$SHA_ARG" ] || fail "PAIR_FLIP_ONLY needs an explicit sha (nothing would be prepared)"
  # A flip ticket may name the release by a prefix (that is what the operator typed); the
  # stamps were written with the full sha by prepare, so resolve before comparing them.
  SHA=$(git -C "$PAIR_REPO" rev-parse --verify "$SHA_ARG^{commit}" 2>/dev/null) \
    || fail "no such commit: $SHA_ARG"
  REL=$(pair_release_dir "$SHA")
  [ -d "$REL/src" ] || fail "release ${SHA:0:12} is not cut at $REL (run scripts/pair_prepare.sh $PAIR $SHA)"
  BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json") \
    || fail "release ${SHA:0:12} has no build_id (run scripts/pair_prepare.sh $PAIR $SHA)"
  pair_step "UPDATE_RELEASE pair=$PAIR sha=${SHA:0:12} build=$BUILD_ID (prepared out of band)"
  plan_out=$(python3 "$HERE/pair_containers.py" plan "$PAIR" 2>&1)
  printf '%s\n' "$plan_out" | sed 's/^/  /' | tee -a "$PAIR_LOG" >/dev/null
  # Any delta at all: a flip-only ticket may not cut, wire or re-create a container on its own, and a
  # plan line is the statement that one of those is needed (RECREATE for a container that exists,
  # CREATE for one that does not). Matching only RECREATE once let a pair with no containers at all
  # walk straight through, so both are checked every time.
  #
  # The one exception is a spec change the operator queued deliberately with --recreate: the ticket
  # then builds the containers from the spec (same apply path as a full update, old container renamed
  # to <name>.bak-<ts>) and keeps going into the flip and the smoke, which is what proves the new
  # spec. Nothing else may set PAIR_ALLOW_RECREATE.
  case "$plan_out" in
    *" CREATE "*)
      fail "containers do not exist: run scripts/pair_prepare.sh $PAIR $SHA first" ;;
    *" RECREATE "*)
      [ "${PAIR_ALLOW_RECREATE:-0}" = 1 ] || fail "the container spec changed: enqueue this release with --recreate (pair_promote.sh $SHA --recreate for prod) so the queue re-creates the containers; a flip-only ticket will not"
      phase containers
      # Refuse before anything is built or renamed: a create on a GPU that somebody is using kills
      # their CUDA context (see pair_p7_gpu_users). PAIR_GPU_WAIT_MIN covers an arm that is already
      # ending; the default 0 refuses at once and leaves the pair exactly as it was found.
      gpu_foreign=$(pair_gpu_foreign_users 0 1 "$P7CT")
      gpu2_users=$(pair_p7_gpu_users 2)
      [ -z "$gpu2_users" ] || pair_step "UPDATE_GPU2_IN_USE $(printf '%s' "$gpu2_users" | paste -sd';' -) (not this recipe, not touched)"
      if [ -n "$gpu_foreign" ]; then
        for i in $(seq 1 $(( ${PAIR_GPU_WAIT_MIN:-0} * 6 ))); do
          sleep 10
          gpu_foreign=$(pair_gpu_foreign_users 0 1 "$P7CT")
          [ -z "$gpu_foreign" ] && break
        done
      fi
      [ -z "$gpu_foreign" ] || fail "gpu_busy p7_gpus=0,1 users=$(printf '%s' "$gpu_foreign" | paste -sd';' -)"
      TS=$(date -u +%Y%m%dT%H%M%SZ)
      status_write 100 "recreating containers (ts=$TS)"
      python3 "$HERE/pair_containers.py" apply "$PAIR" "$SHA" --ts "$TS" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
      rc=${PIPESTATUS[0]}
      [ "$rc" = 0 ] || fail "containers apply rc=$rc"
      pair_step "UPDATE_CONTAINERS pair=$PAIR recreated ts=$TS"
      ;;
    *)
      pair_step "UPDATE_CONTAINERS pair=$PAIR unchanged (prepared)"
      ;;
  esac
  lib_out=$("$HERE/pair_build_lib.sh" verify "$PAIR" "$SHA" --side both 2>&1); lib_rc=$?
  printf '%s\n' "$lib_out" | sed 's/^/  /' | tee -a "$PAIR_LOG"
  [ "$lib_rc" = 0 ] || fail "lib verify rc=$lib_rc (run scripts/pair_prepare.sh $PAIR $SHA)"
  ext_out=$("$HERE/pair_build_ext.sh" verify "$PAIR" "$SHA" --side both 2>&1); ext_rc=$?
  printf '%s\n' "$ext_out" | sed 's/^/  /' | tee -a "$PAIR_LOG"
  [ "$ext_rc" = 0 ] || fail "ext verify rc=$ext_rc (run scripts/pair_prepare.sh $PAIR $SHA)"
else
# ---- 1. cut -------------------------------------------------------------------------------------
phase cut
cut_out=$("$HERE/pair_cut.sh" "$PAIR" $SHA_ARG 2>&1)
cut_rc=$?
printf '%s\n' "$cut_out" | sed 's/^/  /' | tee -a "$PAIR_LOG"
[ "$cut_rc" = 0 ] || fail "cut rc=$cut_rc"
SHA=$(printf '%s\n' "$cut_out" | tail -1)
[ -n "$SHA" ] && [ -d "$(pair_release_dir "$SHA")/src" ] || fail "cut produced no release dir"
REL=$(pair_release_dir "$SHA")
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json") \
  || fail "release ${SHA:0:12} has no build_id"
pair_step "UPDATE_RELEASE pair=$PAIR sha=${SHA:0:12} build=$BUILD_ID"

# ---- 2. containers ------------------------------------------------------------------------------
phase containers
status_write 110 "checking container wiring"
TS=$(date -u +%Y%m%dT%H%M%SZ)
plan_out=$(python3 "$HERE/pair_containers.py" plan "$PAIR" 2>&1)
printf '%s\n' "$plan_out" | sed 's/^/  /' | tee -a "$PAIR_LOG" >/dev/null
case "$plan_out" in
*RECREATE*)
  status_write 100 "recreating containers (ts=$TS)"
  python3 "$HERE/pair_containers.py" apply "$PAIR" "$SHA" --ts "$TS" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || fail "containers apply rc=$rc"
  pair_step "UPDATE_CONTAINERS pair=$PAIR recreated ts=$TS"
  ;;
*)
  TS=""
  pair_step "UPDATE_CONTAINERS pair=$PAIR unchanged"
  ;;
esac

# ---- 3. native libs -----------------------------------------------------------------------------
phase lib
status_write 85 "building seam lib (x86_64 + aarch64)"
"$HERE/pair_build_lib.sh" "$PAIR" "$SHA" --side both 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || fail "lib build rc=$rc"

# ---- 4. torch ext -------------------------------------------------------------------------------
phase ext
status_write 60 "building torch ext (P7, then GX)"
"$HERE/pair_build_ext.sh" "$PAIR" "$SHA" --side both 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || fail "ext build rc=$rc"

fi

# ---- 5. flip ------------------------------------------------------------------------------------
phase flip
OLD_SHA=$(pair_current_sha "$PAIR_ROOT")
GX_OLD=$(pair_current_sha "$PAIR_ROOT" 1)
[ "$OLD_SHA" = "$GX_OLD" ] || pair_step "UPDATE_NOTE old releases differ p7=${OLD_SHA:0:12} gx=${GX_OLD:0:12}; restoring ${OLD_SHA:0:12} on both"
[ -n "$OLD_SHA" ] && pair_step "UPDATE_FLIP from=${OLD_SHA:0:12} to=${SHA:0:12}" || pair_step "UPDATE_FLIP from=- to=${SHA:0:12} (first release)"
status_write 30 "flipping current -> ${SHA:0:12}"
python3 "$HERE/pair_containers.py" link "$PAIR" "$SHA" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || fail "link flip rc=$rc"
FLIPPED=1
python3 "$HERE/pair_containers.py" inner "$PAIR" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || fail "inner link rc=$rc"

# ---- 6. smoke -----------------------------------------------------------------------------------
phase smoke
status_write 20 "server start + one decode row"
# The smoke the canary runs is the *installed* one (this file is $PAIR_SCRIPTS/pair_update_run.sh), so a
# change to the release tree's smoke scripts is not exercised until it is installed: on 2026-11-03 a dev
# canary of 90186b6c passed its prefill at 814.4 tok/s with the box at avail_mib=507 under the installed
# smoke, which predated SMOKE_PREFILL_MIN_AVAIL_MIB -- the release's new gate never ran. Compare the
# copy this ticket reads with the release tree and say which it is, so a canary is never read as proof
# that a test change works. A mismatch is a warning, not a failure: script-only differences do not make
# the release worse than the one it replaces.
smoke_files="pair_smoke.sh pair_arm.sh pair_server_smoke.sh pair_server.sh pair_lib.sh"
diffs=""
for f in $smoke_files; do
  [ -f "$REL/qmesh/release/$f" ] || continue
  [ "$(md5sum "$HERE/$f" 2>/dev/null | cut -d' ' -f1)" = \
    "$(md5sum "$REL/qmesh/release/$f" 2>/dev/null | cut -d' ' -f1)" ] || diffs="$diffs $f"
done
if [ -n "$diffs" ]; then
  pair_step "SMOKE_SCRIPT_STALE pair=$PAIR differs=$diffs installed=$HERE release=$REL this canary exercises the installed test scripts, not the release's"
else
  pair_step "SMOKE_SCRIPT_CHECK pair=$PAIR match=yes files=$(printf '%s' $smoke_files | tr ' ' ',')"
fi
# The smoke stops the worker it loaded, pass or fail: a server that outlives its ticket is a leak, and
# the pair starts one again through the ensure when a client needs it.
smoke_out=$("$HERE/pair_smoke.sh" "$PAIR" "$SHA" 2>&1)
smoke_rc=$?
printf '%s\n' "$smoke_out" | sed 's/^/  /' | tee -a "$PAIR_LOG"
if [ "$smoke_rc" != 0 ]; then
  case "$smoke_out" in
    *"SMOKE_FAIL memory "*) memory_fail "$(printf '%s' "$smoke_out" | grep -m1 'SMOKE_FAIL memory')" ;;
    *) fail "smoke rc=$smoke_rc $(printf '%s' "$smoke_out" | grep -m1 SMOKE_FAIL)" ;;
  esac
fi
row=$(printf '%s\n' "$smoke_out" | sed -n 's/^SMOKE_ROW //p' | head -1)
SMOKE_ROW=$row
# n= and sha256= of the ids the arms generated (total across arms, and a commitment over every
# ARM_IDS line); the promotion gate reads this, not the text. Fall back to the first arm's line for
# a smoke that predates the total.
SMOKE_IDS=$(printf '%s\n' "$smoke_out" | sed -n 's/^SMOKE_ARM_IDS_TOTAL n=\([0-9]*\) sha256=\([0-9a-f]*\).*/n=\1 sha256=\2/p' | head -1)
[ -n "$SMOKE_IDS" ] || SMOKE_IDS=$(printf '%s\n' "$smoke_out" | sed -n 's/^SMOKE_ARM_IDS .* n=\([0-9]*\) sha256=\([0-9a-f]*\).*/n=\1 sha256=\2/p' | head -1)
# The layer set the smoke was served by, carried into the evidence next to the ids: a digest alone
# cannot say whether a mismatch came from the release or from a different restack.
SMOKE_MIX=$(printf '%s\n' "$smoke_out" | sed -n 's/^SMOKE_KERNEL_MIX //p' | head -1)
# Whether the prefill rate was judged at all: "unjudged" when the box was starved of memory at either
# end of the prefill window, so the evidence never claims a speed check that did not happen.
SMOKE_PREFILL_STATE=$(printf '%s\n' "$smoke_out" | sed -n 's/^SMOKE_PREFILL_OK .*/judged/p;s/^SMOKE_PREFILL_UNJUDGED .*/unjudged/p' | head -1)
pair_step "UPDATE_SMOKE pair=$PAIR $row ${SMOKE_IDS:-ids=none} mix=${SMOKE_MIX:-unknown} prefill=${SMOKE_PREFILL_STATE:-none}"

# prod's decode ids must equal the ids dev's smoke of this same sha produced: same packs, same code,
# so the engine has to emit the same tokens on both boxes. The gate already refuses a sha dev never
# passed; this refuses one where prod's own smoke did not reproduce dev's ids -- a different pack
# placement, a silently swapped pack, or a client that came up on another checkpoint. Both numbers are
# kept whatever happens, since an ids mismatch is the only record of which side moved.
if [ "$PAIR" = prod ] && [ -n "$SMOKE_IDS" ]; then
  dev_ids=$(python3 - "$PAIR_STATE/dev/evidence/$SHA.json" <<'PY' 2>/dev/null
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
ids = (doc.get("smoke") or {}).get("ids") or {}
n, digest = ids.get("n"), ids.get("sha256")
if isinstance(n, int) and digest:
    print(f"n={n} sha256={digest}")
PY
)
  dev_mix=$(python3 - "$PAIR_STATE/dev/evidence/$SHA.json" <<'PY' 2>/dev/null
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
k = ((doc.get("smoke") or {}).get("kernel") or {})
if k:
    print(" ".join(f"{n}={v}" for n, v in sorted(k.items())))
PY
)
  # n=0 on either side is the constant hash of [] and would "match" vacuously: refuse it.
  case "$SMOKE_IDS $dev_ids" in
    "n=0 "*|*" n=0 "*) fail "ids_zero prod=$SMOKE_IDS dev=${dev_ids:-none} (a zero-token digest is not identity)" ;;
  esac
  case "$dev_ids" in
    "") pair_step "UPDATE_IDS_CROSSCHECK pair=prod skipped (dev evidence for $SHA records no ids)" ;;
    "$SMOKE_IDS")
      # The mix is printed, not compared: the control release held one digest across 37/3, 38/2
      # and 39/1, so a differing mix is context for a mismatch, not a mismatch by itself.
      if [ -n "$dev_mix" ] && [ "$dev_mix" != "${SMOKE_MIX:-}" ]; then
        pair_step "UPDATE_IDS_CROSSCHECK pair=prod ok $SMOKE_IDS (dev mix '$dev_mix', prod mix '${SMOKE_MIX:-unknown}')"
      else
        pair_step "UPDATE_IDS_CROSSCHECK pair=prod ok $SMOKE_IDS"
      fi
      ;;
    *) fail "ids_mismatch prod=$SMOKE_IDS dev=$dev_ids" ;;
  esac
fi

# the smoke row is the release's reference row: same release must reproduce it (pair_ref.sh)
python3 - "$PAIR_STATE_DIR/refs-$SHA.json" "$SHA" "$BUILD_ID" "$REL" "$row" <<'PY' | tee -a "$PAIR_LOG"
import datetime, json, sys
path, sha, build_id, rel, row = sys.argv[1:6]
kv = dict(tok.split("=", 1) for tok in row.split() if "=" in tok)
seam = json.load(open(f"{rel}/RELEASE.json")).get("seam", {}).get("fingerprint")
doc = {"format": 1, "release_sha": sha, "build_id": build_id, "seam": seam,
       "rows": [{"i": int(kv.get("i", 0)), "text_sha256": kv.get("text_sha256", "")}],
       "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
json.dump(doc, open(path, "w"), indent=2, sort_keys=True)
open(path, "a").write("\n")
print(f"REF {path} rows=1")
PY

# ---- 7. manifest + the guard's own check --------------------------------------------------------
if [ "$PAIR_DRY" = 1 ]; then
  # the dry builders write no stamps on purpose, so a manifest here would be fiction
  pair_step "DRY MANIFEST pair=$PAIR sha=$SHA (skipped: no stamps outside a real build)"
  pair_step "DRY VERIFY pair=$PAIR (skipped: nothing to verify against)"
  VERIFIED=1
  pair_step "PAIR_UPDATED_DRY pair=$PAIR sha=$SHA build=$BUILD_ID (dry: no manifest, no verify)"
  echo "$SHA"
  exit 0
fi
phase manifest
status_write 8 "writing manifest and verifying"
python3 "$HERE/pair_manifest.py" write "$PAIR" "$SHA" --ref "$PAIR_STATE_DIR/refs-$SHA.json" \
  --log "$PAIR_LOG" --by "PairUpdate" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || fail "manifest write rc=$rc"

phase verify
check_out=$("$HERE/pair_check.sh" "$PAIR" 2>&1)
check_rc=$?
printf '%s\n' "$check_out" | sed 's/^/  /' | tee -a "$PAIR_LOG"
[ "$check_rc" = 0 ] || fail "post-update check rc=$check_rc"
VERIFIED=1
# The smoke stopped its own worker and dropped its record; nothing of this ticket's is still running
# on the box, which is what lets the next ticket take the gate without inheriting a stranger's server.
status_write 1 "done"

# Evidence last, so the file only exists once the pair is really on the release and verified.
evidence ok
pair_step "PAIR_UPDATED pair=$PAIR sha=$SHA build=$BUILD_ID check=IN_SYNC ${SMOKE_IDS:-ids=none}"
echo "$SHA"
