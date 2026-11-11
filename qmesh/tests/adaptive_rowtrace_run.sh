#!/usr/bin/env bash
# CSRelease queue only; isolated dev instrumentation, never a performance measurement.
set -uo pipefail
ART_ROOT=${PAIR_ART_MOUNT:?set PAIR_ART_MOUNT to the host artifact mount}
ROOT=${ROOT:-$ART_ROOT/gpu0-gx1}
HERE=$ROOT/scripts
WIN_PAIR=dev WIN_ROOT=$ROOT/adaptive-batch/rowtrace WIN_BUDGET_MIN=${WIN_BUDGET_MIN:-90}
PAIR_PACK_P7=${PAIR_PACK_P7:?set PAIR_PACK_P7 to the hot checkpoint}
CHECKPOINT=$PAIR_PACK_P7 SRV_PORT=8001 MOE_DET=1
# SRC selects the staged source under test; COMPARATOR_SRC selects the comparator, defaulting to
# the same source (record layouts must match).
SRC=${SRC:?set SRC to the staged container source}
COMPARATOR_SRC=${COMPARATOR_SRC:-$SRC}
CACHE_TOKENS=${TRACE_CACHE_TOKENS:-1048576}
export WIN_PAIR WIN_ROOT WIN_BUDGET_MIN PAIR_PACK_P7 CHECKPOINT SRV_PORT MOE_DET SRC COMPARATOR_SRC CACHE_TOKENS
TRACE_WIDTH=${TRACE_WIDTH:-1} TRACE_POSITION=${TRACE_POSITION:-0}
export TRACE_WIDTH TRACE_POSITION
[[ $TRACE_WIDTH =~ ^[0-9]+$ ]] && ((TRACE_WIDTH > 0)) || { echo 'dev: trace width must be positive' >&2; exit 1; }
[[ $TRACE_POSITION =~ ^[0-9]+$ ]] || { echo 'dev: trace position must be nonnegative' >&2; exit 1; }
# NO_DRAFT=1 fires one request at a time with no drafter: every traced forward is the prompt's own,
# so a difference is prefill/prefix behaviour rather than batched verification.
NO_DRAFT=${NO_DRAFT:-0}
export NO_DRAFT
MATCH_BATCH=${MATCH_BATCH:-0}
export MATCH_BATCH
if [ "$MATCH_BATCH" != 0 ]; then
  [[ $MATCH_BATCH =~ ^[1-8]$ ]] || { echo 'invalid MATCH_BATCH' >&2; exit 1; }
  [ "$NO_DRAFT" = 0 ] || { echo 'matched trace requires draft' >&2; exit 1; }
fi
[[ $NO_DRAFT = 0 || $NO_DRAFT = 1 ]] || { echo 'dev: NO_DRAFT must be 0 or 1' >&2; exit 1; }
if [ "$NO_DRAFT" = 1 ] && [ "$TRACE_WIDTH" != 1 ]; then echo 'dev: NO_DRAFT traces one token' >&2; exit 1; fi
. "$HERE/pair_window_lib.sh"
win_init dspark-rowtrace
win_checks
[ "$SRV_PORT" = 8001 ] || win_die 'dev port must be 8001'
[[ $CACHE_TOKENS =~ ^[0-9]+$ ]] && ((CACHE_TOKENS > 0 && CACHE_TOKENS % 256 == 0)) || win_die 'bad shared cache capacity'
if "$HERE/pair_server.sh" dev status > "$WIN_DEST/preexisting-server.log" 2>&1; then
  win_die 'dev server already active; refusing ownership'
fi
# Never evict an existing worker, even when the gate is free.
[ "$(pair_gx_foreign_server)" = '' ] || win_die 'foreign GX worker exists'
[ "$(pair_gx_serving_build)" = '' ] || win_die 'GX worker already active'
win_take_gate dspark-rowtrace "$((WIN_BUDGET_MIN + 3))"
server_attempted=0 server_recorded=0 client_attempted=0 client_owned=0 monitor=
record_owned_worker() {
  pair_gate_is_ours || return 1
  local live_build
  live_build=$(pair_gx_serving_build)
  [ -n "$live_build" ] && [ "$live_build" = "$WIN_BUILD_ID" ] || return 1
  pair_server_record "$WIN_BUILD_ID"
}
cleanup() {
  if [ -n "${monitor:-}" ]; then kill "$monitor" 2>/dev/null || :; wait "$monitor" 2>/dev/null || :; fi
  if [ "$client_owned" = 1 ]; then
    "$HERE/pair_server.sh" dev stop || { echo 'TRACE_FAIL client teardown failed; gate retained' >&2; return 1; }
  fi
  if [ "$client_attempted" = 1 ] && [ "$client_owned" = 0 ]; then
    echo 'TRACE_FAIL client ownership unresolved; gate retained for release-manager cleanup' >&2
    return 1
  fi
  if [ "$server_attempted" = 1 ] && [ "$server_recorded" = 0 ]; then
    if record_owned_worker; then server_recorded=1; else
      echo 'TRACE_FAIL worker ownership unresolved; gate retained for release-manager cleanup' >&2
      return 1
    fi
  fi
  win_free
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ws_checkpoint() {
  local phase=$1 ws
  ws=$(pair_gx_ws) || win_die "GX observation unavailable ($phase)"
  printf '%s\t%s\t%s\n' "$(date +%s)" "$phase" "$ws" >> "$WIN_DEST/gx-memory.tsv"
  echo "TRACE_GX phase=$phase $ws"
}
printf 'epoch\tphase\tfields\n' > "$WIN_DEST/gx-memory.tsv"
ws_checkpoint before
server_attempted=1
server_rc=0
win_server '' 'SEAM_MOE_CPP_DET=1' || server_rc=$?
if record_owned_worker; then server_recorded=1; else win_die 'worker ownership unresolved; gate retained'; fi
[ "$server_rc" -eq 0 ] || win_die "worker start rc=$server_rc"
[ "$(pair_gx_worker_det)" = 1 ] || win_die 'worker deterministic mode missing'
ws_checkpoint worker_start
printf 'epoch\tarm\tp7_free_mib\n' > "$WIN_DEST/gpu-memory.tsv"
echo startup > "$WIN_DEST/active-arm"
(
  while :; do
    arm=$(cat "$WIN_DEST/active-arm" 2>/dev/null || echo unknown)
    free=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
    printf '%s\t%s\t%s\n' "$(date +%s)" "$arm" "${free:-error}" >> "$WIN_DEST/gpu-memory.tsv"
    ws=$(pair_gx_ws 2>&1) || ws="OBSERVATION_FAILED $ws"
    printf '%s\t%s\t%s\n' "$(date +%s)" "$arm-sample" "$ws" >> "$WIN_DEST/gx-memory.tsv"
    printf 'epoch=%s arm=%s\n' "$(date +%s)" "$arm" >> "$WIN_DEST/gx-vmstat-samples.txt"
    pair_gxsh "cat /proc/vmstat" >> "$WIN_DEST/gx-vmstat-samples.txt" 2>/dev/null || echo 'OBSERVATION_FAILED vmstat' >> "$WIN_DEST/gx-memory.tsv"
    sleep 1
  done
) & monitor=$!
tags='B1 B2'
[ "$MATCH_BATCH" = 0 ] || tags='R1 C1 C2 R2'
for tag in $tags; do
  batch=1 spec=A reference= rows=8
  if [ "$MATCH_BATCH" != 0 ]; then
    batch=$MATCH_BATCH
    spec=$(printf 'A,%.0s' $(seq 1 "$batch")); spec=${spec%,}
    [ "$tag" = C1 ] || [ "$tag" = C2 ] && rows=48
    [ "$tag" = R1 ] || reference="DSPARK_ROWTRACE_REFERENCE=/artifacts/${WIN_DEST#"$ART_ROOT"/}/R1/trace.pt"
  elif [ "$tag" = B2 ]; then
    batch=2 spec=A,A
    reference="DSPARK_ROWTRACE_REFERENCE=/artifacts/${WIN_DEST#"$ART_ROOT"/}/B1/trace.pt"
  fi
  # The prefix cache is off in every arm: it would let the two arms share cached pages, which
  # would make a difference (or its absence) unattributable.
  draft="DSPARK=1 EXL3_PREFIX_CACHE=0"
  if [ "$NO_DRAFT" = 1 ]; then
    # No drafter: every traced forward is the prompt's own, one token at a time.
    draft="DSPARK=0 EXL3_PREFIX_CACHE=0"
  fi
  trace_ct="/artifacts/${WIN_DEST#"$ART_ROOT"/}/$tag"
  echo "$tag" > "$WIN_DEST/active-arm"
  extra="BATCH=$batch QMESH_ALLOW_BATCH_UNSAFE=1 SEAM_MOE_CPP_DET=1 EXL3_WHOLE_GRAPH=0 EXL3_DSV4_BATCH_GRAPH=0 $draft SERVE_WIDTH_LOG=1 DSPARK_ROWTRACE_OUT=$trace_ct DSPARK_ROWTRACE_BATCH=$batch DSPARK_ROWTRACE_WIDTH=$TRACE_WIDTH DSPARK_ROWTRACE_POSITION=$TRACE_POSITION DSPARK_ROWTRACE_SAME_BATCH=1 EXL3_MOE_COOP_ROWS=$rows $reference"
  client_attempted=1
  PAIR_WHOLE_GRAPH=0 PAIR_SRV_EXTRA_ENVS="$extra" "$HERE/pair_server.sh" dev start > "$WIN_DEST/$tag-launch.log" 2>&1
  rc=$?
  # Claim only an UP emitted by this start, never ALREADY_UP. The launcher checks socket PID ownership.
  if grep -q "PAIR_SRV_UP pair=dev .* port=8001 " "$WIN_DEST/$tag-launch.log" && ! grep -q 'PAIR_SRV_ALREADY_UP' "$WIN_DEST/$tag-launch.log"; then client_owned=1; fi
  [ "$rc" -eq 0 ] && [ "$client_owned" = 1 ] || win_die "$tag client launch/ownership failure"
  grep -q 'DSPARK_ROWTRACE_ARMED' "$PAIR_STATE_DIR/server/pair-server.log" || win_die "$tag trace not armed"
  if [ "$MATCH_BATCH" != 0 ]; then
    pid=$(python3 -c 'import sys; print(next(x[4:] for x in open(sys.argv[1]).read().split() if x.startswith("pid=")))' "$PAIR_STATE_DIR/server/pair-server.pid") || win_die "$tag missing pid"
    docker exec "$P7CT" sh -c 'tr "\000" "\n" < /proc/"$1"/environ; cat /proc/"$1"/maps' sh "$pid" > "$WIN_DEST/$tag-process.txt" || win_die "$tag process observation failed"
    grep -qx "EXL3_MOE_COOP_ROWS=$rows" "$WIN_DEST/$tag-process.txt" || win_die "$tag threshold mismatch"
    grep -Fq "$SRC/exllamav3_ext.so" "$WIN_DEST/$tag-process.txt" || win_die "$tag extension mismatch"
  fi
  driver_rc=0
  python3 "$HERE/b2_token_drive.py.pending" http://127.0.0.1:8001 128 "$spec" "$tag" > "$WIN_DEST/$tag-driver.log" 2>&1 || driver_rc=$?
  cp "$PAIR_STATE_DIR/server/pair-server.log" "$WIN_DEST/$tag-server.log" || win_die "$tag log copy failed"
  [ "$driver_rc" = 0 ] || win_die "$tag driver failed rc=$driver_rc"
  ws_checkpoint "$tag-after"
  "$HERE/pair_server.sh" dev stop || win_die "$tag client stop failed"
  client_owned=0 client_attempted=0
  [ -s "$WIN_DEST/$tag/trace.pt" ] || win_die "$tag no aligned trace; inspect candidate logs"
  grep -q 'PREFIX_CACHE enabled=0' "$WIN_DEST/$tag-server.log" || win_die "$tag prefix cache enabled"
  if grep -q 'DSPARK_ROWTRACE_ADMISSION_SHORT' "$WIN_DEST/$tag-server.log"; then win_die "$tag incomplete coordinated admission"; fi
  grep -q 'DSPARK_ROWTRACE_SAVED' "$WIN_DEST/$tag-server.log" || win_die "$tag trace marker missing"
done
if [ "$MATCH_BATCH" = 0 ]; then
  compare_rc=0
  docker exec "$P7CT" python3 "$COMPARATOR_SRC/qmesh/tests/compare_dspark_rowtrace.py" "/artifacts/${WIN_DEST#"$ART_ROOT"/}/B1/trace.pt" "/artifacts/${WIN_DEST#"$ART_ROOT"/}/B2/trace.pt" > "$WIN_DEST/compare.log" 2>&1 || compare_rc=$?
  cat "$WIN_DEST/compare.log"
  grep -q '^ALIGNMENT OK:' "$WIN_DEST/compare.log" || win_die 'invalid/missing trace alignment'
  echo "TRACE_DONE dir=$WIN_DEST compare_rc=$compare_rc; differences/ambiguous coverage are characterization, not isolation proof or performance"
else
  docker exec "$P7CT" python3 "$COMPARATOR_SRC/qmesh/tests/uniform_trace_compare.py" "/artifacts/${WIN_DEST#"$ART_ROOT"/}" > "$WIN_DEST/compare.log" 2>&1 || win_die 'matched trace comparison failed'
  cat "$WIN_DEST/compare.log"
  echo "TRACE_DONE dir=$WIN_DEST matched diagnostic only"
fi
