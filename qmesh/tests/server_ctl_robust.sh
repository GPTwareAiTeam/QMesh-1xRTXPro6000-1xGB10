#!/usr/bin/env bash
# Control-plane robustness baseline: a seam server must survive every client it does not expect.
#
# 2026-11-01/02 the server aborted the whole process on a hello it did not like ("qes: peer layout
# mismatch"), so a stray or stale client killed the cold-expert transport for both hybrid pairs.
# The C++ side now answers bad control hellos with a typed rejection and closes that connection;
# this script is the regression test, and it needs no pack, no GPU memory and no second host:
# a --stub server binds the control port and the per-bucket data ports on 0.0.0.0, and
# qmesh/tests/ctl_robust.py throws handshakes at it over loopback.
#
#   CT=seam-p7v2-robust ./qmesh/tests/server_ctl_robust.sh
#   CT=seam-gx2stg DEV=rocep1s0f0 GID=3 PORT=47911 ./qmesh/tests/server_ctl_robust.sh
#
# Stub only. It maps no pack and prints `stub=1` in its LOADED line, so it can run on a box whose
# real server is down without touching the 113 GiB placement; do not point it at a loaded server's
# port, the second bind fails and this script says so.
set -uo pipefail

CT=${CT:?set CT to the container to run the stub server in}
PORT=${PORT:-47911}
DEV=${DEV:-rocep142s0f1}
GID=${GID:-5}
BUCKETS=${BUCKETS:-1,6,64,8192}
# The stub still parses the placement file to build the bucket geometry (it maps no pack), so on a
# box without /cold point this at any copy of the same JSON: a few hundred KiB, read once.
PLACEMENT=${PLACEMENT:-/cold/placement.exl3.json}
ROOT=${ROOT:-/src/exllamav3}
HERE=$(cd "$(dirname "$0")" && pwd)
SRVLOG=/tmp/ctl_robust_srv.log
fails=0
say() { echo "$*"; }
fail() { say "FAIL $*"; fails=$((fails + 1)); }

dx() { docker exec "$CT" bash -lc "$1"; }

# A stub left behind by a crashed run owns exactly the ports the check below is about to test, so
# clear it first; the port check then only sees processes this script does not own.
dx "pkill -f '[w]orker_seam_server.py.*--stub' >/dev/null 2>&1; rm -f $SRVLOG"

# Nothing else may hold the control port or any per-bucket data port: one bucket would bind-fail in
# qes_server_open and the probe would then be talking to somebody else's process. A bind attempt is
# the only honest test -- `ss -ltn` misses both a TIME_WAIT hold and the ephemeral range, which is
# how a "free" 49116 then failed to listen (Linux hands outbound ports out of 32768-60999 and a
# local proxy had just used one). So keep PORT below 32768, and set SO_REUSEADDR: without it this
# check reports the previous run's closed sockets as busy and the test can never run twice.
need=$(($(tr ',' ' ' <<< "$BUCKETS" | wc -w) + 1))
busy=$(dx "python3 - $PORT $need <<'EOF'
import socket, sys
p0, n = int(sys.argv[1]), int(sys.argv[2])
for p in range(p0, p0 + n):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((\"0.0.0.0\", p))
    except OSError:
        print(p)
    finally:
        s.close()
EOF" 2>/dev/null | tr -d '\r')
if [ -n "$busy" ]; then
  say "PORT_BUSY ctl=${PORT}: $busy cannot be bound; pick another PORT (this script starts its own stub)"
  exit 3
fi

docker cp "$HERE/ctl_robust.py" "$CT:/tmp/ctl_robust.py" >/dev/null || { say "COPY_FAILED"; exit 2; }
start_stub() {   # $1 = extra server args, $2 = extra probe args
  dx "pkill -f '[w]orker_seam_server.py.*--stub' >/dev/null 2>&1; sleep 0.3; rm -f $SRVLOG"
  say "STUB ctl=${PORT} buckets=${BUCKETS} dev=${DEV} gid=${GID} ct=${CT}$1"
  docker exec -d "$CT" bash -lc "cd $ROOT && exec python3 -u qmesh/worker_seam_server.py --stub \
    --placement $PLACEMENT --dev $DEV --gid $GID --port $PORT --buckets $BUCKETS $1 > $SRVLOG 2>&1"
  pid=""
  listening=0
  for _ in $(seq 1 60); do
    pid=$(dx "pgrep -f '[w]orker_seam_server.py.*--stub' | head -1")
    if dx "grep -aq SEAM_LISTEN $SRVLOG" 2>/dev/null; then listening=1; break; fi
    [ -n "$pid" ] || break
    sleep 0.5
  done
  if [ "$listening" != 1 ]; then
    say "STUB_DID_NOT_START (see $CT:$SRVLOG)"
    dx "tail -6 $SRVLOG" | sed 's/^/  /'
    exit 4
  fi
  say "STUB_PID pid=${pid}"
  dx "python3 /tmp/ctl_robust.py 127.0.0.1 $PORT $BUCKETS $2" | sed 's/^/  /'
  rc=${PIPESTATUS[0]}
  # The server must still be the same live process, with every rejection logged rather than fatal.
  alive=$(dx "kill -0 $pid 2>/dev/null && echo yes || echo no")
  [ "$alive" = yes ] || fail "server pid $pid is gone after the probe"
  rejects=$(dx "grep -ac 'SEAM_REJECT\|SEAM_QUEUE' $SRVLOG" 2>/dev/null | tr -dc 0-9)
  datarej=$(dx "grep -ac 'stale-data-conn' $SRVLOG" 2>/dev/null | tr -dc 0-9)
  say "SERVER_ALIVE pid=${pid} alive=${alive} reject_lines=${rejects:-0} stale_data=${datarej:-0} probe_rc=${rc}"
  [ "${rejects:-0}" -ge 5 ] || fail "expected >=5 rejected/queued control connections in the log, saw ${rejects:-0}"
  # The 12 bad data hellos must have been refused, not ignored: this is the path that used to
  # abort the process, and a silent no-op here would make the case above meaningless.
  [ "${datarej:-0}" -ge 12 ] || fail "expected >=12 stale-data-conn rejections, saw ${datarej:-0}"
  dx "grep -aE 'Fatal Python error|Aborted|assert' $SRVLOG" | head -3 | sed 's/^/  LEAK /'
  dx "grep -aE 'Fatal Python error|Aborted|assert' $SRVLOG" | grep -q . && fail "server log shows a fatal error"
  [ "$rc" = 0 ] || fail "probe reported failures (rc=$rc)"
  dx "pkill -f '[w]orker_seam_server.py.*--stub'" >/dev/null 2>&1
  sleep 0.5
  left=$(dx "pgrep -f '[w]orker_seam_server.py.*--stub' | wc -l" | tr -dc 0-9)
  [ "${left:-0}" = 0 ] || fail "stub still running after cleanup"
}

# Two lifetimes, because the concurrent client has two legitimate outcomes and both have to be
# survivable: refused (ST_BUSY, the default) and queued then admitted (--wait, SEAM_WAIT=1).
say "== refuse mode"
start_stub "" ""
say "== queue mode"
start_stub "--wait" "wait"

if [ "$fails" = 0 ]; then
  say "RESULT ctl_robust=pass stub_pid_survived=yes modes=refuse,queue"
  exit 0
fi
say "RESULT ctl_robust=fail failures=${fails}"
exit 1
