#!/usr/bin/env bash
# pair_server_stop: an interrupted ticket puts the box back, and only what it loaded.
#
# On 2026-11-02 a dev ticket was killed at 23:32 while its prefill ran. The smoke's trap dropped the
# gate but left `worker_seam_server.py` holding the 116 GiB pack, so the 23:54 smoke refused
# `PAIR_UPDATE_REFUSED pair_busy gx_procs=2` and a colleague's window was lost. The other direction is
# worse: a wrapper that stopped the server "by pattern" once killed the next arm's server.
set -u
WORK=$(mktemp -d "${TMPDIR:-/tmp}/pairown.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
HERE=$(cd "$(dirname "$0")/.." && pwd)
fails=0
ok()   { printf 'OK   %s\n' "$*"; }
bad()  { printf 'FAIL %s\n' "$*"; fails=$((fails + 1)); }
check(){ local want=$1 got=$2 what=$3; [ "$want" = "$got" ] && ok "$what" || bad "$what: wanted '$want' got '$got'"; }

# A GX box that is a dict of /proc pids: the stub ssh answers the three commands pair_server_stop and
# pair_server_record use, and records every kill it is asked for.
mkdir -p "$WORK/bin" "$WORK/state/dev"
cat > "$WORK/bin/stub_ssh" <<'EOS'
#!/usr/bin/env bash
# stub_ssh <host> <cmd...>: the last argument is the remote shell line
cmd=${!#}
echo "$cmd" >> "$STUB_LOG"
case "$cmd" in
  # pair_server_record anchors the pattern (^) since 2026-11-02 so a wrapper process cannot come back
  # from head -1; the stub follows the caller, not the other way round.
  "docker exec "*"pgrep -f '^python3 -u qmesh/worker_seam_server.py' | head -1"*)
    printf '%s\n' "${STUB_PID:-}" ;;
  *"/proc/"*"/cmdline"*)
    pid=$(printf '%s' "$cmd" | sed -n 's#.*/proc/\([0-9][0-9]*\)/cmdline.*#\1#p')
    [ "$pid" = "${STUB_PID:-}" ] && [ "${STUB_ALIVE:-1}" = 1 ] && printf '%s ' "${STUB_CMDLINE:-python3 -u qmesh/worker_seam_server.py}"
    ;;
  *"kill -TERM"*)
    pid=$(printf '%s' "$cmd" | sed -n 's/.*kill -TERM \([0-9][0-9]*\).*/\1/p')
    [ "$pid" = "${STUB_PID:-}" ] && [ "${STUB_ALIVE:-1}" = 1 ] && echo sent
    ;;
esac
EOS
chmod +x "$WORK/bin/stub_ssh"
export STUB_LOG="$WORK/cmds" PAIR_SSH="$WORK/bin/stub_ssh" PAIR_DRY=0
export PAIR_STATE="$WORK/state" PAIR_ROOT_OVERRIDE="$WORK/pair" PAIR_DRY_SSH="" PAIR_SSH_OPTS=() 
source "$HERE/pair_lib.sh" 2>/dev/null
PAIR=dev GX=gx10-2 GXCT=seam-gx2-dev P7CT=seam-p7-dev PAIR_DRY=0
PAIR_STATE_DIR=$WORK/state/dev  # pair_load would set this; the test does not want pairs.conf

# 1. The record names the worker the ensure started.
STUB_PID=4242 export STUB_PID
pair_server_record dev-build-1 && ok "the worker is recorded" || bad "the worker is recorded"
f=$(cat "$PAIR_STATE_DIR/server-dev.pid")
check "pair=dev build=dev-build-1 pid=4242" "$(printf '%s' "$f" | cut -d' ' -f1-3)" "the record names pair, build and pid"

# 2. stop sends TERM to exactly that pid, and clears the record.
out=$(pair_server_stop); check "SERVER_STOP pid=4242 build=dev-build-1" "$out" "stop reports the pid it stopped"
[ -e "$PAIR_STATE_DIR/server-dev.pid" ] && bad "the record is cleared" || ok "the record is cleared"
grep -q "kill -TERM 4242" "$STUB_LOG" && ok "one TERM, addressed by pid" || bad "one TERM, addressed by pid"

# 3. A pid that no longer runs the worker is never killed: the box may have moved on to somebody else.
: > "$STUB_LOG"
export STUB_PID=4242 STUB_ALIVE=0
printf 'pair=dev build=dev-build-1 pid=4242 at=x\n' > "$PAIR_STATE_DIR/server-dev.pid"
out=$(pair_server_stop)
check "SERVER_STOP_SKIP pid=4242 (gone or not the worker)" "$out" "a gone worker is skipped, not killed"
grep -q "kill -TERM" "$STUB_LOG" && bad "no kill was sent" || ok "no kill was sent"

unset STUB_ALIVE
# 4. No record, no action: a ticket interrupted before the smoke leaves a serving pair alone.
: > "$STUB_LOG"
out=$(pair_server_stop); check "" "$out" "no record means no stop"
grep -q "kill -TERM" "$STUB_LOG" && bad "no kill without a record" || ok "no kill without a record"

# 5. A dry run touches nothing.
printf 'pair=dev build=dev-build-1 pid=4242 at=x\n' > "$PAIR_STATE_DIR/server-dev.pid"
: > "$STUB_LOG"
out=$(PAIR_DRY=1 pair_server_stop)
case "$out" in SERVER_STOP_DRY*) ok "a dry run only says what it would stop" ;; *) bad "a dry run only says what it would stop: '$out'" ;; esac
grep -q "kill" "$STUB_LOG" && bad "dry run sent no kill" || ok "dry run sent no kill"

echo "RESULT $([ "$fails" = 0 ] && echo PASS || echo FAIL) fails=$fails"
[ "$fails" = 0 ]
