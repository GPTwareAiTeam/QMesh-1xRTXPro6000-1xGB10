#!/usr/bin/env bash
# Release manager only, inside the dev GPU queue. Never use on production.
set -uo pipefail
ROOT=${ROOT:-${PAIR_ART_MOUNT:?set PAIR_ART_MOUNT to the host artifact mount}/gpu0-gx1}
HERE=$ROOT/scripts
WIN_PAIR=dev WIN_ROOT=$ROOT/uniform-moe/pair-ab WIN_BUDGET_MIN=${WIN_BUDGET_MIN:-180}
PAIR_PACK_P7=${PAIR_PACK_P7:?set PAIR_PACK_P7 to the hot checkpoint}
CHECKPOINT=$PAIR_PACK_P7 SRV_PORT=8001 MOE_DET=1
SRC=${SRC:?set SRC to frozen fbb source with matching extension}
REPORT=${REPORT:?set REPORT to frozen host reporter path}
MODE=${MODE:-diagnostic}
TIMED_DRIVER=${TIMED_DRIVER:-}
SMOKE=${SMOKE:-0}
[[ $SMOKE = 0 || $SMOKE = 1 ]] || { echo 'invalid SMOKE' >&2; exit 2; }
[ "$MODE" = diagnostic ] || [ "$MODE" = timing ] || { echo 'invalid MODE' >&2; exit 2; }
[ "$MODE" = diagnostic ] || [ -f "$TIMED_DRIVER" ] || { echo 'missing TIMED_DRIVER' >&2; exit 2; }
CACHE_TOKENS=${CACHE_TOKENS:-1048576}
export WIN_PAIR WIN_ROOT WIN_BUDGET_MIN PAIR_PACK_P7 CHECKPOINT SRV_PORT MOE_DET SRC CACHE_TOKENS
. "$HERE/pair_window_lib.sh"
win_init uniform-pair-ab
win_checks
[ "$SRV_PORT" = 8001 ] || win_die 'dev port required'
if "$HERE/pair_server.sh" dev status > "$WIN_DEST/preexisting-server.log" 2>&1; then win_die 'dev client already active'; fi
[ "$(pair_gx_foreign_server)" = '' ] || win_die 'foreign GX worker exists'
[ "$(pair_gx_serving_build)" = '' ] || win_die 'GX worker already active'
win_take_gate uniform-pair-ab "$((WIN_BUDGET_MIN + 3))"
server_attempted=0 server_recorded=0 client_attempted=0 client_owned=0
record_owned_worker() {
  pair_gate_is_ours || return 1
  local live_build
  live_build=$(pair_gx_serving_build)
  [ -n "$live_build" ] && [ "$live_build" = "$WIN_BUILD_ID" ] || return 1
  pair_server_record "$WIN_BUILD_ID"
}
cleanup() {
  if [ "$client_owned" = 1 ]; then
    "$HERE/pair_server.sh" dev stop || { echo 'AB_FAIL client teardown failed; gate retained' >&2; return 1; }
  fi
  if [ "$client_attempted" = 1 ] && [ "$client_owned" = 0 ]; then
    echo 'AB_FAIL client ownership unresolved; gate retained' >&2; return 1
  fi
  if [ "$server_attempted" = 1 ] && [ "$server_recorded" = 0 ]; then
    if record_owned_worker; then server_recorded=1; else echo 'AB_FAIL worker ownership unresolved; gate retained' >&2; return 1; fi
  fi
  win_free
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
server_attempted=1
server_rc=0
win_server '' 'SEAM_MOE_CPP_DET=1' || server_rc=$?
if record_owned_worker; then server_recorded=1; else win_die 'worker ownership unresolved'; fi
[ "$server_rc" -eq 0 ] || win_die "worker start rc=$server_rc"
[ "$(pair_gx_worker_det)" = 1 ] || win_die 'worker deterministic mode missing'
dsparks='0 1'; batches='1 2 4 8'
if [ "$SMOKE" = 1 ]; then dsparks='1'; batches='1 4'; fi
for dspark in $dsparks; do
  for batch in $batches; do
    for run in 1 2 3 4; do
      case $run in 1|4) rows=8;; *) rows=48;; esac
      tag=d${dspark}-b${batch}-r${rows}-n${run}
      win_status 120 "$tag"
      load=$(cut -d' ' -f1 /proc/loadavg)
      python3 -c 'import sys; sys.exit(float(sys.argv[1]) > 12)' "$load" || win_die "$tag host load $load > 12"
      ps -e -o stat=,comm= | python3 -c 'import sys; sys.exit(1 if any(len(p)==2 and not p[0].startswith("Z") and p[1] in {"nvcc", "ptxas", "cc1plus", "cicc"} for p in (line.split() for line in sys.stdin)) else 0)' || win_die "$tag live build compiler active"
      before=$(pair_gx_ws) || win_die "$tag GX observation failed"
      printf '%s\tbefore\t%s\t%s\n' "$tag" "$load" "$before" >> "$WIN_DEST/observations.tsv"
      extra="BATCH=$batch QMESH_ALLOW_BATCH_UNSAFE=1 SEAM_MOE_CPP_DET=1 EXL3_PREFIX_CACHE=0 DSPARK=$dspark EXL3_MOE_COOP_ROWS=$rows"
      if [ "$MODE" = diagnostic ]; then extra="$extra SERVE_WIDTH_LOG=1"; else extra="$extra SERVE_JOB_TIMING=1 SERVE_WIDTH_LOG=0"; fi
      client_attempted=1
      PAIR_SRV_EXTRA_ENVS="$extra" "$HERE/pair_server.sh" dev start > "$WIN_DEST/$tag-launch.log" 2>&1
      rc=$?
      if grep -q 'PAIR_SRV_UP pair=dev .* port=8001 ' "$WIN_DEST/$tag-launch.log" && ! grep -q 'PAIR_SRV_ALREADY_UP' "$WIN_DEST/$tag-launch.log"; then client_owned=1; fi
      [ "$rc" -eq 0 ] && [ "$client_owned" = 1 ] || win_die "$tag client launch/ownership failure"
      pid=$(python3 -c 'import sys; print(next(x[4:] for x in open(sys.argv[1]).read().split() if x.startswith("pid=")))' "$PAIR_STATE_DIR/server/pair-server.pid") || win_die "$tag missing pid"
      docker exec "$P7CT" sh -c 'tr "\000" "\n" < /proc/"$1"/environ; cat /proc/"$1"/maps' sh "$pid" > "$WIN_DEST/$tag-process.txt" || win_die "$tag process observation failed"
      grep -qx "EXL3_MOE_COOP_ROWS=$rows" "$WIN_DEST/$tag-process.txt" || win_die "$tag coop setting missing"
      grep -qx "DSPARK=$dspark" "$WIN_DEST/$tag-process.txt" || win_die "$tag dspark setting missing"
      grep -qx 'EXL3_WHOLE_GRAPH=1' "$WIN_DEST/$tag-process.txt" || win_die "$tag whole graph setting missing"
      if [ "$MODE" = timing ]; then
        grep -qx 'SERVE_JOB_TIMING=1' "$WIN_DEST/$tag-process.txt" || win_die "$tag timing setting missing"
        grep -qx 'SERVE_WIDTH_LOG=0' "$WIN_DEST/$tag-process.txt" || win_die "$tag width log enabled"
      fi
      grep -Fq "$SRC/exllamav3_ext.so" "$WIN_DEST/$tag-process.txt" || win_die "$tag extension mapping mismatch"
      same=$(printf 'A,%.0s' $(seq 1 "$batch")); same=${same%,}
      mixed=$(for ((i=0;i<batch;i++)); do if ((i % 2)); then printf 'B,'; else printf 'A,'; fi; done); mixed=${mixed%,}
      if [ "$MODE" = diagnostic ]; then
        for prompt in A B; do
          python3 "$HERE/b2_token_drive.py.pending" http://127.0.0.1:8001 128 "$prompt" "$tag-single-$prompt" > "$WIN_DEST/$tag-single-$prompt.log" 2>&1 || win_die "$tag singleton $prompt failed"
          python3 "$REPORT" check "$WIN_DEST/$tag-single-$prompt.log" 1 || win_die "$tag singleton $prompt token data missing"
        done
        for kind in same mixed; do
          spec=$same; [ "$kind" = mixed ] && spec=$mixed
          offset=$(wc -c < "$PAIR_STATE_DIR/server/pair-server.log") || win_die "$tag log offset failed"
          python3 "$HERE/b2_token_drive.py.pending" http://127.0.0.1:8001 128 "$spec" "$tag-$kind" > "$WIN_DEST/$tag-$kind.log" 2>&1 || win_die "$tag $kind driver failed"
          python3 "$REPORT" check "$WIN_DEST/$tag-$kind.log" "$batch" || win_die "$tag $kind token data missing"
          python3 "$REPORT" admission "$PAIR_STATE_DIR/server/pair-server.log" "$offset" "$batch" > "$WIN_DEST/$tag-$kind-admission.json" || win_die "$tag $kind admission not established"
        done
      else
        python3 "$TIMED_DRIVER" http://127.0.0.1:8001 "$same" > "$WIN_DEST/$tag-warmup.json" || win_die "$tag warmup failed"
        python3 "$TIMED_DRIVER" http://127.0.0.1:8001 "$same" > "$WIN_DEST/$tag-timing.json" || win_die "$tag timing failed"
      fi
      cp "$PAIR_STATE_DIR/server/pair-server.log" "$WIN_DEST/$tag-server.log" || win_die "$tag log copy failed"
      if [ "$MODE" = diagnostic ]; then grep -q 'SERVE_DECODE_WIDTH active_rows=' "$WIN_DEST/$tag-server.log" || win_die "$tag no width evidence"; fi
      after=$(pair_gx_ws) || win_die "$tag GX observation failed"
      load=$(cut -d' ' -f1 /proc/loadavg)
      python3 -c 'import sys; sys.exit(float(sys.argv[1]) > 12)' "$load" || win_die "$tag after host load $load > 12"
      printf '%s\tafter\t%s\t%s\n' "$tag" "$load" "$after" >> "$WIN_DEST/observations.tsv"
      python3 -c 'import sys
def fields(s): return dict(x.split("=", 1) for x in s.split() if "=" in x)
a,b=map(fields,sys.argv[1:])
for k in ("box_swap_used_mib", "vm_swap_mib", "pid"):
    if not a.get(k, "").isdigit() or not b.get(k, "").isdigit(): raise SystemExit("missing " + k)
if a["pid"] != b["pid"] or int(b["box_swap_used_mib"]) - int(a["box_swap_used_mib"]) >= 128 or int(a["vm_swap_mib"]) != 0 or int(b["vm_swap_mib"]) != 0: raise SystemExit("paging or worker change")' "$before" "$after" || win_die "$tag paging gate failed"
      "$HERE/pair_server.sh" dev stop || win_die "$tag client stop failed"
      client_owned=0 client_attempted=0
    done
  done
done
if [ "$MODE" = diagnostic ]; then python3 "$REPORT" summarize "$WIN_DEST" "$SMOKE" > "$WIN_DEST/summary.jsonl" || win_die 'summary failed'; fi
echo "AB_DONE dir=$WIN_DEST"
