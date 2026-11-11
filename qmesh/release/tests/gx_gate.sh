#!/bin/bash
# gx_gate.sh -- the GX box claim, as the release scripts and an experiment both read it.
#
#   ./tests/gx_gate.sh
#
# A GX box carries one cold pack. A release build, a release smoke and an experiment that has the pack
# loaded all want it, and two of them at once has already OOM-killed a build and buried a measurement
# under somebody else's hundreds of MiB. The claim is a file: "<owner> pid=… until=<epoch>", free when
# missing or past. The properties that matter:
#   * a stale claim must not wedge the box (until is authoritative, not existence),
#   * a foreign live claim is waited out and never overwritten (two jobs must not both think they won),
#   * a job drops the claim only if the line is still its own (it must not release somebody else's).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
W=$(mktemp -d /tmp/pairgate.XXXXXX)
export PAIR_GATE_DIR=$W
# shellcheck source=../pair_lib.sh
. "$REL/pair_lib.sh"
PAIR_HOME=$W; PAIR_DRY=0; GXCT=seam-gx2-dev; PAIR=dev
# A claim may only be taken inside a queue ticket (pair_ticket_hardcap); these cases are the claim's
# own battery, so they stand in for the wrapper and declare the cap it would export.
export GPU_TICKET_HARDCAP=$(( $(date +%s) + 3600 ))
f=$W/gx2.gate
fails=0
want() {  # <case> <expected-substring> <actual>
  if printf '%s' "$3" | grep -q -- "$2"; then echo "OK   $1"; else echo "FAIL $1 (wanted '$2', got '$3')"; fails=$((fails + 1)); fi
}
eq() { if [ "$2" = "$3" ]; then echo "OK   $1"; else echo "FAIL $1 (wanted '$2', got '$3')"; fails=$((fails + 1)); fi; }

eq "the gate names the box from the container" "$W/gx2.gate" "$(pair_gx_gate_file)"
eq "no file means free" free "$(pair_gx_gate_state)"
# An unresolvable GXCT must not answer "free": queue/gx.gate never exists, so a claimed box read free
# and a hand-run check took GX2 while a smoke held it (2026-11-02 03:1x). Unresolvable is an error.
_gxct=$GXCT; unset GXCT
out=$(pair_gx_gate_file 2>/dev/null); rc=$?
eq "an unset GXCT resolves no gate" 2 "$rc"
eq "an unset GXCT has no path" "" "$out"
eq "an unset GXCT reads unresolved, not free" gate_unresolved "$(pair_gx_gate_state)"
if pair_gx_gate_take nope 5 0; then echo "FAIL unresolvable_refused (took a box it cannot name)"; fails=$((fails + 1));
else echo "OK   unresolvable_refused"; fi
[ -f "$f" ] && { echo "FAIL unresolvable_writes_nothing (wrote $f)"; fails=$((fails + 1)); } || echo "OK   unresolvable_writes_nothing"
GXCT=bogus
eq "an unparsable GXCT reads unresolved" gate_unresolved "$(pair_gx_gate_state)"
GXCT=$_gxct
eq "a resolvable GXCT names its own box again" "$W/gx2.gate" "$(pair_gx_gate_file)"

# A live foreign claim: a real process holds it (a dead pid is free, which the next case pins).
sleep 300 & foreign=$!
printf 'owner=Experiment pid=%s until=%s\n' "$foreign" "$(( $(date +%s) + 600 ))" > "$f"
want "a live foreign claim reads as taken" '^Experiment until=[0-9]*$' "$(pair_gx_gate_state)"
if pair_gx_gate_take mine 5 0; then echo "FAIL take_foreign (took a live foreign claim)"; fails=$((fails + 1));
else echo "OK   take_foreign (refused)"; fi
want "take_foreign left the foreign line alone" '^owner=Experiment ' "$(cat "$f" | tr '\n' ' ')"

printf 'owner=Experiment pid=%s until=%s\n' "$foreign" "$(( $(date +%s) - 1 ))" > "$f"
eq "an expired claim is free" free "$(pair_gx_gate_state)"
if pair_gx_gate_take mine 5 0; then echo "OK   take_expired"; else echo "FAIL take_expired"; fails=$((fails + 1)); fi
want "take_expired wrote our line" '^owner=mine pid='"$$"' host='"$(hostname -s)"' until=[0-9]* ' "$(cat "$f" | tr '\n' ' ')"

printf 'owner=mine pid=1 until=%s\n' "$(( $(date +%s) + 600 ))" > "$f"
pair_gx_gate_down
[ -f "$f" ] && echo "OK   down_foreign_kept" || { echo "FAIL down_foreign_kept (released somebody else's claim)"; fails=$((fails + 1)); }

printf 'owner=mine pid=%s until=%s\n' "$$" "$(( $(date +%s) + 600 ))" > "$f"
pair_gx_gate_down
[ -f "$f" ] && { echo "FAIL down_own_removed (claim still there)"; fails=$((fails + 1)); } || echo "OK   down_own_removed"

# A claim written on another box carries a pid this shell cannot see; that is not a dead holder. The
# 2026-11-02 02:26 race: a runner read a live claim, its `kill -0` failed, and it took the box under a
# load in flight. Host is part of the claim, and a foreign host is authoritative until `until`.
printf 'owner=OtherBox pid=999999 host=gx10-9 until=%s\n' "$(( $(date +%s) + 600 ))" > "$f"
want "a foreign-host claim is held even though its pid is invisible here" '^OtherBox until=[0-9]*$' "$(pair_gx_gate_state)"
if pair_gx_gate_take mine 5 0; then echo "FAIL take_foreign_host (took another host's live claim)"; fails=$((fails + 1));
else echo "OK   take_foreign_host (refused)"; fi
rm -f "$f"

# Two takers on the same free box: exactly one may win, the other has to see the claim.
( pair_gx_gate_take racer-a 5 0 && echo A ) > "$W/a.out" & ra=$!
( pair_gx_gate_take racer-b 5 0 && echo B ) > "$W/b.out" & rb=$!
wait "$ra" "$rb"   # never bare wait: the foreign sleep above is a background job too
won=$(cat "$W/a.out" "$W/b.out" | grep -c .)
[ "$won" = 1 ] && echo "OK   one_of_two_racers" || { echo "FAIL one_of_two_racers (won=$won)"; fails=$((fails + 1)); }
rm -f "$f"

# A build that is killed cannot clean up: the stamp has to expire on its own.
printf 'owner=build pid=777 until=%s\n' "$(( $(date +%s) + 1 ))" > "$f"
sleep 2
eq "a dead job's claim expires by itself" free "$(pair_gx_gate_state)"
pair_gx_gate_down

# A dry run writes nothing: the dry-run harness must not claim a real box.
rm -f "$f"; PAIR_DRY=1; pair_gx_gate_take dry 5 0; PAIR_DRY=0
[ -f "$f" ] && { echo "FAIL dry_writes_nothing (a dry run claimed the box)"; fails=$((fails + 1)); } || echo "OK   dry_writes_nothing"

kill "$foreign" 2>/dev/null
rm -rf "$W"
echo "RESULT $( [ "$fails" = 0 ] && echo PASS || echo FAIL ) fails=$fails"
[ "$fails" = 0 ]
