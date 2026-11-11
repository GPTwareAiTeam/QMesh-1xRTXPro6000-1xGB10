#!/usr/bin/env bash
# Shared by the seam-robust dev tickets: the dev pair's own server is PairRelease-2's steady state,
# and one pack load per GX box is the rule -- so a ticket that needs the box either fails on
# "box busy" or stops that server and puts it back when it is done. Stopping it is recorded here so
# the restore is a `pair_serve` from the release, not an ad-hoc relaunch.
#
#   . "$HERE/cx_pair_server.sh"
#   pair_load dev
#   cx_pair_server_stop            # sets CX_PAIR_RESTORE=1 when it actually stopped something
#   ... box work ...
#   cx_pair_server_restore         # pair_serve dev, after the gate is down (it takes its own)
#
# Needs from the caller: PAIR, GX, GXCT, and the pair_lib helpers (pair_gxsh, pair_server_*).

cx_pair_server_pid() {  # anchored worker pid in the pair's own container, if one is serving
  timeout 20 ssh -o BatchMode=yes "$GX" \
    "docker exec $GXCT pgrep -f '^python3 -u qmesh/worker_seam_server.py' | head -1" 2>/dev/null | tr -d '\r'
}

cx_pair_server_stop() {
  CX_PAIR_RESTORE=0
  local pid
  pid=$(cx_pair_server_pid)
  [ -n "$pid" ] || return 0
  # Nobody stops another job's server, and the claim is what makes that checkable: a stop outside a
  # claim is how seam-gx2-dev lost its server at 07:53:43 on 2026-11-02 (my ticket's prologue TERMed
  # pid 17825 while GxPort2 held the gate, and pair_serve then had to reload 116 GiB).
  if declare -F pair_gate_is_ours >/dev/null 2>&1; then
    pair_gate_is_ours || { echo "PAIR_SERVER_STOP_REFUSED ct=$GXCT (this ticket holds no gx gate)"; return 1; }
  fi
  echo "PAIR_SERVER_STOP pid=$pid ct=$GXCT (one pack load per box; the pair's own restore brings it back)"
  CX_PAIR_RESTORE=1
  # Record the live pid so pair_server_stop TERMs exactly the worker (never the wrapper shell).
  pair_server_record "stopped-by-seam-robust"
  pair_server_stop
  # An unrecorded pid (a server somebody started by hand, or a stale pidfile): TERM the anchored argv.
  if [ -n "$(cx_pair_server_pid)" ]; then
    timeout 20 ssh -o BatchMode=yes "$GX" \
      "docker exec $GXCT pkill -f '^python3 -u qmesh/worker_seam_server.py'" >/dev/null 2>&1
    echo "PAIR_SERVER_STOP pkill=by_argv ct=$GXCT"
  fi
  return 0
}
cx_pair_server_restore() {  # <logdir> <stamp>; the pair's owner queues pair_serve itself, so this
                            # is only for tickets that stop the server without a restore ticket
  [ "${CX_PAIR_RESTORE:-0}" = 1 ] || return 0
  local dir=${1:-/tmp} stamp=${2:-$(date +%m%d-%H%M%S)} out rc=0
  out=$dir/pair_serve_restore_$stamp.log
  echo "PAIR_SERVER_RESTORE start pair=$PAIR (restoring the server this ticket stopped)"
  "${PAIR_SCRIPTS:?Set PAIR_SCRIPTS to the pair scripts directory}/pair_serve.sh" "$PAIR" > "$out" 2>&1 || rc=$?
  echo "PAIR_SERVER_RESTORE rc=$rc $(grep -aE 'PAIR_SERVE_READY|PAIR_SERVE_DONE|SEAM_ENSURE state=|PAIR_SERVE_GX_BUSY|PAIR_SERVE_MEM|PAIR_SERVE_DRIFT' "$out" | tail -2 | tr '\n' ' ')"
  echo "PAIR_SERVER restored=$([ "$rc" = 0 ] && echo yes || echo no) log=$out"
  return $rc
}
