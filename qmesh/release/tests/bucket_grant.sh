#!/usr/bin/env bash
# pair_bucket_grant.py: the smoke's verdict on the buckets the seam granted.
#
# The client adopts the intersection of its request and the server's grant; the release's fallback
# default (1,6,64,256,512,1024,2048,4096) does not contain the shipping 8192, so the 23:32 dev arm
# negotiated accepted=1,6,64 against a server granting 1,6,64,8192 and every prefill chunk went 128
# calls deep. The smoke has to notice that, and the notice has to fail in a test.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
G=$HERE/pair_bucket_grant.py
fails=0
ok()   { printf 'OK   %s\n' "$*"; }
bad()  { printf 'FAIL %s\n' "$*"; fails=$((fails + 1)); }
check(){ local want=$1 got=$2 what=$3; [ "$want" = "$got" ] && ok "$what" || bad "$what: wanted '$want' got '$got'"; }

out=$(python3 "$G" "1,6,64,8192" "1,6,64,8192"); rc=$?
check 0 "$rc" "the pair's own grant passes"
check "OK accepted=1,6,64,8192 want=1,6,64,8192" "$out" "and says so in one line"

# The 23:32 failure, verbatim: the grant was the client's fallback cut down by the server.
out=$(python3 "$G" "1,6,64" "1,6,64,8192"); rc=$?
check 1 "$rc" "a 64-row cap fails"
check "MISS accepted=1,6,64 want=1,6,64,8192 missing=8192" "$out" "and names the missing width"

# A server that grants extra buckets is not a fault: the pair's list is the requirement, not a census.
out=$(python3 "$G" "1,6,64,256,8192" "1,6,64,8192"); rc=$?
check 0 "$rc" "an extra granted bucket does not fail"
check "OK accepted=1,6,64,256,8192 want=1,6,64,8192" "$out" "and the line reports the full grant"

# No negotiated line at all (client never connected): rc 2, so the caller keeps the arm's own reason.
out=$(python3 "$G" "" "1,6,64,8192"); rc=$?
check 2 "$rc" "an unread grant is not a grant fault"
check "MISS accepted=unknown want=1,6,64,8192" "$out" "and is labelled unknown"

# Garbage in the list is a miss, never a pass.
out=$(python3 "$G" "1,6,64,wide" "1,6,64,8192"); rc=$?
check 1 "$rc" "an unparsable list fails"

check 2 "$(python3 "$G" >/dev/null 2>&1; echo $?)" "a wrong arity is a usage exit"

echo "RESULT $([ "$fails" = 0 ] && echo PASS || echo FAIL) fails=$fails"
[ "$fails" = 0 ]
