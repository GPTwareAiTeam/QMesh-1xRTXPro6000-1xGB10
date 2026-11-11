#!/usr/bin/env bash
# ws_rows sweep on the dev pair: 1024 / 1536 / 2048 rows per cold-expert call, one --fresh pack load
# per arm, each arm accepted on the same two numbers (32k prefill completeness, and the MemAvailable
# that is left after it). The point is not "bigger is faster": with the 113.65 GiB pack resident the
# GX has ~1 GiB to give, so a larger row piece buys fewer calls per token and pays for them in the
# transient the 32k prefill needs. An arm measured below CX_SLACK_MIN MiB of free memory is not
# evidence of anything except that the box was paging, so it is reported and marked counts=0 rather
# than dropped -- all three arms are in the answer, including the ones that do not count.
#
#   CX_STAGE_SHA=d330dcf5 CX_WT=<worktree> \
#     queue/run_dev.sh SeamRobust 45 bash qmesh/seamtools/seam_rows_sweep.sh
#
# Knobs (env): CX_WS_LIST="1024 1536 2048", CX_SLACK_MIN=256, CX_WT, CX_STAGE_SHA, CX_CT, CX_P7CT,
# CX_CUDA_VISIBLE (0 inside a pair container), CX_SWEEP_BASE (evidence root).
# Exit: 3 gate never free, 13 the container is not the commit under test, 15 wrong commit, 16 dirty
# worktree, 24 an arm failed (its rc is printed; every arm still runs), 7 the box was not released.
set -uo pipefail
. ${PAIR_SCRIPTS:?Set PAIR_SCRIPTS to the pair scripts directory}/pair_lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cx_pair_server.sh"
pair_load dev                       # GX=<GX host>, GXCT=seam-gx2-dev, PORT=47311, BUCKETS=1,6,64,8192
CT=${CX_CT:-$GXCT}
P7CT=${CX_P7CT:-seam-p7-dev}
WT=${CX_WT:-.}
GPU=${CX_CUDA_VISIBLE:-0}           # a pair container renumbers its single card to 0
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARM_SH="$HERE/seam_direct_arm3.sh"
WS_LIST=${CX_WS_LIST:-1024 1536 2048}
SLACK_MIN=${CX_SLACK_MIN:-256}
STAMP=$(date +%m%d-%H%M%S)
SW=${CX_SWEEP_BASE:-./evidence/sweep}/$STAMP
LOG=$SW/sweep.log
mkdir -p "$SW"
: > "$LOG"

say() { echo "$(date +%T) SW $*" | tee -a "$LOG"; }
status() { [ -n "${GPU_RUN_STATUS:-}" ] && printf 'eta_min=%s doing=%s\n' "$1" "$2" > "$GPU_RUN_STATUS"; return 0; }
gx_mem() {  # -> "avail_mib swapfree_mib"
  timeout 20 ssh -o BatchMode=yes "$GX" \
    'awk "/^MemAvailable:/{a=\$2} /^SwapFree:/{s=\$2} END{printf \"%d %d\", a/1024, s/1024}" /proc/meminfo' \
    2>/dev/null | tr -d '\r'
}
cleanup() {  # the kill-safe handback; the same three steps as the other tickets
  local rc=$? i av sw st
  trap - EXIT INT TERM
  timeout 12 ssh -o BatchMode=yes "$GX" \
    "docker exec $CT sh -c 'pkill -f \"^python3 -u qmesh/worker_seam_server.py\" || exit 0; sleep 4; pkill -KILL -f \"^python3 -u qmesh/worker_seam_server.py\" || true'" \
    >/dev/null 2>&1
  st=$(pair_gx_gate_state)
  pair_gx_gate_down
  for i in 1 2 3; do
    read -r av sw < <(gx_mem)
    [ -n "${av:-}" ] && [ "$av" -ge "${CX_MIN_AVAIL:-118400}" ] && break
    sleep 3
  done
  say "CLEANUP rc=$rc gx_avail_mib=${av:-?} swap_free_mib=${sw:-?} gate_was=${st:-none}"
  exit "$rc"
}
trap cleanup EXIT INT TERM

[ $# = 0 ] || { echo "usage: $0   (no arguments: every knob is an env var, see the header)" >&2; exit 2; }
status 44 "gx2 gate: ws_rows sweep $WS_LIST, one fresh load per arm"
pair_gx_gate_take SeamRobust-rows 44 15 || { say "SW FAIL gate_unresolved"; exit 3; }
say "gate file=$(pair_gx_gate_file) owner=SeamRobust-rows"
say "script=$0 sha=$(sha256sum "$0" | cut -c1-16) arm_sha=$(sha256sum "$ARM_SH" | cut -c1-16) wt=$WT arms=$WS_LIST slack_min=$SLACK_MIN"
say "box before stop: gx=[$(gx_mem)]"

cx_pair_server_stop | sed "s/^/SW pair   /" | tee -a "$LOG"
say "box after stop: gx=[$(gx_mem)]  (this is the number each arm's SEAM_PACK_LOAD should match)"
# Attribute every arm to a commit and prove the container holds those bytes. No rsync: the container's
# /src/exllamav3 is a symlink into the pair's release, and writing another commit into a release
# directory makes pair_check call the pair drifted (cx_verify_tree.sh has the long version).
SRC=$WT
sha=$(git -C "$SRC" rev-parse HEAD 2>/dev/null)
dirty=$(git -C "$SRC" status --porcelain 2>/dev/null | grep -cv '^?? ')
say "STAGED src=$SRC branch=$(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null) sha=${sha:-none} dirty=$dirty want=${CX_STAGE_SHA:-any}"
[ -z "${CX_STAGE_SHA:-}" ] || [ "${CX_STAGE_SHA:0:8}" = "${sha:0:8}" ] ||
  { say "SW FAIL $SRC is at ${sha:-none}, not at the requested ${CX_STAGE_SHA}"; exit 15; }
[ "$dirty" = 0 ] || { say "SW FAIL $SRC has $dirty uncommitted files; no arm can be attributed to ${sha:-none}"; exit 16; }
status 42 "verifying the container runs $SRC"
SRC="$SRC" GX="$GX" CX_CT="$CT" bash "$HERE/cx_verify_tree.sh" 2>&1 | tee -a "$SW/verify.log" >> "$LOG" ||
  { say "SW FAIL the container is not running $sha; nothing it prints would be evidence"; exit 13; }
grep -ah '^VERIFY lib \|^VERIFY target ' "$SW/verify.log" | sed 's/^/SW /' | tee -a "$LOG"

# One arm per ws_rows, in the order given. arm3 does the load (stop_server -> ensure -> SEAM_PACK_LOAD),
# the client run, the 32k prefill and the host census itself; this ticket only sequences the arms and
# puts the three numbers side by side.
armrc=""
i=0
nws=$(printf '%s\n' $WS_LIST | wc -l)
for ws in $WS_LIST; do
  i=$((i + 1))
  E=$SW/w$ws
  mkdir -p "$E"
  left=$(( nws - i + 1 ))
  status $(( 8 + left * 11 )) "arm $i/$nws ws_rows=$ws: fresh pack load, client run, 32k prefill"
  say "ARM START ws=$ws evidence=$E"
  CX_WT="$WT" CX_GATE=own CX_ARM_TAG="W$ws" CX_EVIDENCE="$E" CX_LOADS=1 CX_IDS_RUNS=1 CX_WS_ROWS=$ws \
  CX_IDS_MAX_NEW=${CX_IDS_MAX_NEW:-128} CX_IDS_WARMUP_NEW=${CX_IDS_WARMUP_NEW:-8} \
  CX_DETERMINISTIC=0 CX_PREFILL=1 CX_WANT_DIGEST=any \
  CX_SERVER_PLACEMENT=/cold/placement.exl3.json \
  CX_PLACEMENT="${CX_PLACEMENT:?Set CX_PLACEMENT}" \
  CX_RANKED=0 CX_ORACLE=0 CX_REFUSAL=0 \
    bash "$ARM_SH" 2>&1 | tee -a "$SW/arm_W$ws.log" >> "$LOG"
  rc=${PIPESTATUS[0]}
  say "ARM END ws=$ws rc=$rc"
  armrc="$armrc $ws:$rc"
  # Report the arm either way: a failed arm is a data point about this geometry, and the whole point
  # of the sweep is that the reader sees all three.
  SRV=$E/server_load1.log
  PL=$E/pf32k_load1.log
  AL=$SW/arm_W$ws.log
  slab=$(grep -ah 'SEAM_SERVER_SLAB ' "$SRV" 2>/dev/null | tail -1)
  wsmib=$(printf '%s' "$slab" | sed -n 's/.*ws_mib=\([0-9.]*\).*/\1/p')
  rmax=$(printf '%s' "$slab" | sed -n 's/.*rows_max=\([0-9]*\).*/\1/p')
  fits=$(printf '%s' "$slab" | sed -n 's/.*fits=\([0-9]*\).*/\1/p')
  loadav=$(printf '%s' "$slab" | sed -n 's/.*avail_mib=\([0-9]*\).*/\1/p')
  av=$(grep -ah 'SLACK load1 ' "$AL" 2>/dev/null | tail -1 | sed -n 's/.*avail_mib=\([0-9]*\).*/\1/p')
  swp=$(grep -ah 'SLACK load1 ' "$AL" 2>/dev/null | tail -1 | sed -n 's/.*swap_free_mib=\([0-9]*\).*/\1/p')
  wsw=$(grep -ac '^SEAM_SERVER_SWAP_WARN' "$SRV" 2>/dev/null)
  pf=$(grep -ah 'PREFILL_ARM depth=32768 ' "$PL" 2>/dev/null | tail -1)
  pfs=$(printf '%s' "$pf" | sed -n 's/.*prefill_s=\([0-9.]*\).*/\1/p')
  pfr=$(printf '%s' "$pf" | sed -n 's/.*tokens=\([0-9]*\).*/\1/p')
  # The decode number and the Option-B swap window, both from this arm's own log and this load: a
  # width is only interesting if it is faster per token without paging inside the decode window.
  dec=$(grep -ah 'DECODE tag=' "$AL" 2>/dev/null | tail -1)
  dms=$(printf '%s' "$dec" | sed -n 's/.*decode_ms_per_tok=\([0-9.]*\).*/\1/p')
  dps=$(printf '%s' "$dec" | sed -n 's/.*decode_tok_per_s=\([0-9.]*\).*/\1/p')
  dnt=$(printf '%s' "$dec" | sed -n 's/.*new_tokens=\([0-9]*\).*/\1/p')
  read -r swapb swapa <<< "$(grep -ah 'DIRECT PHASESAMP ' "$AL" 2>/dev/null | sed 's/.*DIRECT PHASESAMP //' |
    awk '{ ph=$1; for (i=2;i<=NF;i++) { split($i,kv,"="); f[kv[1]]=kv[2] }
           if (ph=="after_load1") b=f["vm_swap_mib"]
           if (ph=="after_decode1") a=f["vm_swap_mib"] }
         END{ print (b==""?"?":b), (a==""?"?":a) }')"
  swapg="?"; swapok=0
  case "${swapb}${swapa}" in
    ''|*[!0-9]*) ;;
    *) swapg=$(( swapa - swapb )); [ "$swapg" -le "${CX_SWAP_GROW_MAX:-128}" ] && swapok=1 ;;
  esac
  counts=0; why=""
  if [ "$rc" != 0 ]; then why="arm_rc_$rc"
  elif [ -z "${av:-}" ]; then why="no_slack_line"
  elif [ "$av" -lt "$SLACK_MIN" ]; then why="below_slack_$SLACK_MIN"
  elif [ -z "${dms:-}" ]; then why="no_decode_line"
  elif [ "$swapok" != 1 ]; then why="swap_decode_window_${swapg}_over_${CX_SWAP_GROW_MAX:-128}"
  else counts=1; fi
  say "SWEEP ROW ws=$ws rc=$rc load_avail_mib=${loadav:-?} ws_mib=${wsmib:-?} rows_max=${rmax:-?} fits=${fits:-?} decode_ms_per_tok=${dms:-?} decode_tok_per_s=${dps:-?} decode_tokens=${dnt:-?} swap_before_arm_mib=${swapb:-?} swap_after_decode_mib=${swapa:-?} swap_growth_mib=$swapg prefill_avail_mib=${av:-?} swap_free_mib=${swp:-?} swap_warn=$wsw prefill_tokens=${pfr:-?} prefill_s=${pfs:-?} prefill_tok_per_s=$(awk -v t="${pfr:-0}" -v s="${pfs:-0}" 'BEGIN{ printf (s>0)?"%.1f":"?", t/s }') counts=$counts why=${why:-ok}"
done

status 4 "verdict"
say "SWEEP ARMS rc=$armrc"
say "SWEEP TABLE sorted by ws_rows:"
# Read the whole table into a variable first: `grep ... "$LOG" | tee -a "$LOG"` appends to the file it
# is reading, and the arm lines it appends are themselves SWEEP ROW lines.
tbl=$(grep -ah 'SWEEP ROW ' "$LOG" | sort -t= -k2 -n)
printf '%s\n' "$tbl" | tee -a "$LOG"
say "FINAL gx=[$(gx_mem)]"
for kv in $armrc; do
  [ "${kv#*:}" = 0 ] || { say "SW FAIL ws=${kv%:*} rc=${kv#*:}; every arm is reported above, but a failed arm is not a measurement"; exit 24; }
done
say "SW PASS arms=$WS_LIST slack_min=$SLACK_MIN: rows above the slack floor are the ones that count"
