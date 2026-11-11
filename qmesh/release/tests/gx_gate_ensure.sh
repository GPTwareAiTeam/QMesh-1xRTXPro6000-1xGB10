#!/bin/bash
# The shared ensure must refuse to touch a GX box it does not hold a claim on.
#
# Owner rule (2026-11-02): the gate is binding for every agent's GX server launch. Until then the gate
# was only taken by pair_smoke.sh, so a bench that called seam_server_ensure.sh directly could kill or
# start a server on a box a smoke owned (a GX2 bench started 01:38, the dev smoke took the gate at
# 01:56, found the port and the pack taken and failed 900 s later). The helper therefore checks the
# claim and refuses with rc 3, naming the state, and the dispatch passes that status through.
#
# No ssh, docker, container, GPU or queue is touched: the gate check runs before the first remote call.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
W=$(mktemp -d /tmp/gxgate.XXXXXX); trap 'rm -rf "$W"' EXIT
export SEAM_GATE_DIR="$W/queue" SEAM_GATE_REQUIRED=1 CT=seam-gx2-dev
mkdir -p "$SEAM_GATE_DIR"
F=$SEAM_GATE_DIR/gx2.gate
now=$(date +%s)
fails=0
ck() {  # ck <name> <want_rc> <want_substr> ; runs seam_gate_check in this shell
  local name=$1 want=$2 sub=$3 out rc
  out=$(seam_gate_check 2>&1); rc=$?
  if [ "$rc" = "$want" ] && { [ -z "$sub" ] || case "$out" in *"$sub"*) true ;; *) false ;; esac; }; then
    echo "OK   $name"
  else
    echo "FAIL $name | rc=$rc want=$want out=$out"
    fails=$((fails + 1))
  fi
}
. "$REL/seam_server_ensure.sh"

ck "no gate file: refused, named" 3 "gate_free"

printf 'owner=smoke-dev' >"$F"
ck "a claim without until is malformed, refused" 3 "gate_bad"

printf 'owner=smoke-dev pid=%s until=%s\n' "$$" "$((now - 5))" >"$F"
ck "an expired claim is refused" 3 "gate_expired"

printf 'owner=smoke-dev pid=999999 until=%s\n' "$((now + 600))" >"$F"
ck "a claim whose pid is gone is refused as stale" 3 "gate_stale"

sleep 300 & foreign=$!
printf 'owner=bench pid=%s until=%s\n' "$foreign" "$((now + 600))" >"$F"
ck "a live foreign claim is refused by name" 3 "gate_foreign"
kill "$foreign" 2>/dev/null

# The holder's own children are allowed: the smoke takes the gate, launches the ensure, and the ensure
# must see the smoke as an ancestor (this is the path every ticket takes). The gate pid here is the
# test shell's, and the helper runs as its real child process (not a subshell of it), so the check has
# to walk ps -o ppid= rather than compare $$.
printf 'owner=smoke-dev pid=%s until=%s\n' "$$" "$((now + 600))" >"$F"
ck "our own claim is accepted" 0 ""
out=$(SEAM_GATE_DIR="$SEAM_GATE_DIR" CT=seam-gx2-dev bash -c \
      '. "$1/seam_server_ensure.sh"; seam_gate_check; echo "rc=$?"' _ "$REL" 2>&1)
case "$out" in
  *"rc=0"*) echo "OK   a child process inherits its parent's claim" ;;
  *) echo "FAIL a child process inherits its parent's claim | $out"; fails=$((fails + 1)) ;;
esac

# SEAM_GATE_REQUIRED=0 is for stub harnesses only, and must not read anything.
rm -f "$F"
out=$(SEAM_GATE_REQUIRED=0 seam_gate_check 2>&1); rc=$?
if [ "$rc" = 0 ] && [ -z "$out" ]; then echo "OK   SEAM_GATE_REQUIRED=0 bypasses the check"; else
  echo "FAIL SEAM_GATE_REQUIRED=0 bypasses the check | rc=$rc out=$out"; fails=$((fails + 1)); fi

# Direct use as a command: the refusal must leave as rc 3, not as the old blanket 2 (pair_smoke.sh
# classifies on 3 = preflight refused versus 2 = server died, and the blanket 2 is why a busy box read
# as "server_timeout_or_died").
out=$(SEAM_GATE_DIR="$SEAM_GATE_DIR" CT=seam-gx2-dev GX=127.0.0.1 "$REL/seam_server_ensure.sh" --stub 2>&1); rc=$?
case "$rc:$out" in
  3:*gate_free*) echo "OK   the dispatch exits with the preflight status (3) and the reason" ;;
  *) echo "FAIL the dispatch exits with the preflight status | rc=$rc out=$out"; fails=$((fails + 1)) ;;
esac

echo
echo "RESULT gx_gate_ensure $([ "$fails" = 0 ] && echo PASS || echo FAIL) fails=$fails"
[ "$fails" = 0 ]
