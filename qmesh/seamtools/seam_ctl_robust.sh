#!/usr/bin/env bash
# The regression test for the crash the owner is paying for, run against the release the dev pair
# actually serves: a seam server must survive every client it does not expect.
#
# qmesh/tests/server_ctl_robust.sh starts a --stub server (no pack, no GPU memory, no second host) on
# a port of its own inside the pair's GX container and throws handshakes at it over loopback: bad
# magic, wrong protocol version, wrong layer count, malformed bucket count, a disjoint bucket list, a
# peer that says nothing, 12 bad data connections (the path that used to abort the process with
# "qes: peer layout mismatch"), one legitimate session, two sessions in a row with different bucket
# subsets, and a client that arrives while somebody else's session is live -- in both reject and
# queue modes. After each, the same pid must still be answering status and the rejections must be in
# the log rather than a core file.
#
# It runs on the GX *host* because that is where the docker daemon is; only the two test files are
# copied there. Nothing is loaded, so this ticket never needs the pack gate and never disturbs a real
# server: the stub binds 47911..47914, four ports clear of 47311.
#
#   queue/run_dev.sh SeamRobust 6 bash <this script>
#
# Exit: 2 the files would not copy, 24 the battery failed, 0 pass.
set -uo pipefail
. ${PAIR_SCRIPTS:?Set PAIR_SCRIPTS to the pair scripts directory}/pair_lib.sh
pair_load dev                       # GX=<GX host>, GXCT=seam-gx2-dev
CT=${CX_CT:-$GXCT}
WT=${CX_WT:-.}
PORT=${CX_CTL_PORT:-47911}
STAMP=$(date +%m%d-%H%M%S)
D=${CX_CTL_BASE:-./evidence/ctl}/$STAMP
LOG=$D/ctl.log
mkdir -p "$D"
: > "$LOG"

say() { echo "$(date +%T) CTL $*" | tee -a "$LOG"; }
status() { [ -n "${GPU_RUN_STATUS:-}" ] && printf 'eta_min=%s doing=%s\n' "$1" "$2" > "$GPU_RUN_STATUS"; return 0; }

[ $# = 0 ] || { echo "usage: $0   (no arguments: every knob is an env var)" >&2; exit 2; }
say "script=$0 sha=$(sha256sum "$0" | cut -c1-16) src=$WT ct=$CT port=$PORT"
say "probe_sha=$(sha256sum "$WT/qmesh/tests/ctl_robust.py" | cut -c1-16) runner_sha=$(sha256sum "$WT/qmesh/tests/server_ctl_robust.sh" | cut -c1-16)"

# The files, not a tree: the container already holds the release under /src/exllamav3, and this test
# is about the bytes that release serves. (cx_verify_tree.sh covers "same tree"; it needs the release
# directory, which is not on this host.)
status 5 "copying the two test files to $GX"
scp -q -o BatchMode=yes "$WT/qmesh/tests/ctl_robust.py" "$WT/qmesh/tests/server_ctl_robust.sh" \
  "$GX:/tmp/" || { say "CTL FAIL scp"; exit 2; }
say "copied=$(timeout 20 ssh -o BatchMode=yes "$GX" "sha256sum /tmp/ctl_robust.py /tmp/server_ctl_robust.sh" 2>/dev/null | tr -d '\r' | cut -c1-16 | tr '\n' ' ')"

status 4 "stub server + 12 bad hellos + 2 lifetimes in $CT"
# DEV/GID are the GX side's own verbs device, not the pair's P7 client device: the runner's defaults
# (rocep142s0f1 / 5) are the P7 NIC, which does not exist in this container. GX2's serving worker
# runs `--dev rocep1s0f0 --gid 3` (ps, pid 3260, 20:45), so the stub is opened the same way; pair_load
# exports GX_GID, and the device name is overridable because GX1 and GX2 are not guaranteed equal.
GX_DEV=${CX_GX_DEV:-rocep1s0f0}
say "gxverbs dev=$GX_DEV gid=$GX_GID (the running worker's own --dev/--gid)"
timeout 240 ssh -o BatchMode=yes "$GX" \
  "CT='$CT' PORT='$PORT' DEV='$GX_DEV' GID='$GX_GID' PLACEMENT=/cold/placement.exl3.json ROOT=/src/exllamav3 \
     bash /tmp/server_ctl_robust.sh" 2>&1 | tr -d '\r' | tee -a "$D/battery.log" | sed 's/^/CTL   /' >> "$LOG"
rc=${PIPESTATUS[0]}
say "battery rc=$rc"

status 2 "verdict"
grep -ah '^STUB \|^STUB_PID \|^SERVER_ALIVE \|^  REJECT\|^  WAIT\|^  OK\|^FAIL \|^PORT_BUSY\|^RESULT ' "$D/battery.log" |
  sed 's/^/CTL /' | tee -a "$LOG"
say "FINAL rc=$rc"
[ "$rc" = 0 ] || { say "CTL FAIL battery rc=$rc; the server under test did not survive a client it did not expect"; exit 24; }
say "CTL PASS the shipped transport rejects unexpected clients instead of aborting"
