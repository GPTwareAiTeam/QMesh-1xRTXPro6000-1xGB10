#!/usr/bin/env bash
# seam_pair_serve.sh -- evict whatever the pair is serving, then leave it serving the release
# `current` names at the ws_rows THIS CONF names.
#
# ws_rows is fixed by the load (it sizes the server's workspace slab: 235 MiB at 1024, 58.8 MiB at
# 256, 1880 MiB at auto), so "serve 256 for one timeline, then back to the shipping 1024" is two
# 116 GiB loads with an eviction between them -- and the second half cannot be done by
# scripts/pair_serve.sh alone: it compares build_id, not width, so with the same release already
# serving it prints PAIR_SERVE_ALREADY_UP and exits 0, leaving the wrong width up. The eviction is
# therefore here, inside one ticket, so there is no window in which the box serves nothing.
#
#   PAIR_CONF=<pairs.conf> \
#     queue/run_dev.sh ServerUp 25 bash <this script>          # 256: the decode-shaped timeline load
#   queue/run_dev.sh ServerUp 25 bash <this script>            # restore the shipping width (1024)
#
# Exit: 3 the GX gate never came free, 4 the eviction was refused (this ticket holds no gate),
# anything else is pair_serve.sh's own (it loads, records the worker and drops the gate itself).
set -uo pipefail
. ${PAIR_SCRIPTS:?Set PAIR_SCRIPTS to the pair scripts directory}/pair_lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cx_pair_server.sh"
pair_load dev
status() { [ -n "${GPU_RUN_STATUS:-}" ] && printf 'eta_min=%s doing=%s\n' "$1" "$2" > "$GPU_RUN_STATUS"; return 0; }
say() { echo "$(date +%T) SERVE $*"; }
gx_mem() {  # -> "avail_mib swap_free_mib" from the GX container's own /proc/meminfo
  timeout 30 ssh -o BatchMode=yes "$GX" \
    "docker exec $GXCT sh -c 'awk \"/MemAvailable|SwapFree|SwapTotal/{printf \\\"%s=%d \\\", \\\$1, \\\$2/1024}\" /proc/meminfo'" \
    2>/dev/null | tr -d '\r' | awk '{for(i=1;i<=NF;i++){split($i,kv,"=");v[kv[1]]=kv[2]}
      printf "avail_mib=%s swap_free_mib=%s box_swap_mib=%d", v["MemAvailable:"], v["SwapFree:"], v["SwapTotal:"]-v["SwapFree:"]}'
}

[ $# = 0 ] || { echo "usage: $0   (no arguments: PAIR_CONF names the width, see the header)" >&2; exit 2; }
say "conf=$PAIR_CONF ws_rows=$WS_ROWS ct=$GXCT current=$(pair_current_sha "$PAIR_ROOT") before=[$(gx_mem)]"
status 22 "evicting the current worker, then a fresh load at ws_rows=$WS_ROWS"
pair_gx_gate_take SeamRobust-serve ${SERVE_WAIT_MIN:-20} 5 || { say "FAIL gate_unresolved $(pair_gx_gate_state)"; exit 3; }
say "gate file=$(pair_gx_gate_file) owner=SeamRobust-serve"
cx_pair_server_stop | sed 's/^/SERVE /' || { pair_gx_gate_down; say "FAIL stop_refused (this ticket holds no gate)"; exit 4; }
pair_gx_gate_down
sleep 5   # the pack's memory is not returned the instant the worker exits; pair_serve waits too
say "after stop=[$(gx_mem)]"
status 16 "fresh load, pack 113 GiB at ws_rows=$WS_ROWS"
exec bash ${PAIR_SCRIPTS:?Set PAIR_SCRIPTS to the pair scripts directory}/pair_serve.sh dev
