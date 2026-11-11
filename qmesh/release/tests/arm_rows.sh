#!/usr/bin/env bash
# pair_arm_rows.py: the smoke's verdict on an arm log. This exists because the first version could not
# fail in a test -- it was an inline heredoc that read the pipe as its own source, and it reported
# "arm_no_tokens NONE" for a complete row (dev, 23:29:19). Fixtures here are the arm's real byte shapes.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
PARSER=$HERE/pair_arm_rows.py
tmp=$(mktemp -d "${TMPDIR:-/tmp}/armrows.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
fails=0
ok()   { printf 'OK   %s\n' "$*"; }
bad()  { printf 'FAIL %s\n' "$*"; fails=$((fails + 1)); }
check(){ local want=$1 got=$2 what=$3; [ "$want" = "$got" ] && ok "$what" || bad "$what: wanted '$want' got '$got'"; }

# The row the dev arm actually printed, verbatim in shape (a JSON row between ARM_START and ARM_IDS,
# wrapped in the paging chatter of the same log).
cat >"$tmp/real.log" <<'EOF'
ARM_START dspark1 prompts=1 max_new=8 num_draft=5
 -- CPU split experts (worker, static): layers.0.ffn [133..384) of 384, placement from saved profile
{"i": 0, "label": "p0", "arm": "dspark1", "prompt_tokens": 22, "new_tokens": 8, "ttft_ms": 3156.3, "decode_ms_per_tok": 1669.86, "wall_s": 14.846, "text": "This\nShowing live article 12 of"}
SUMMARY {"arm": "dspark1", "n": 1, "tokens": 8}
ARM_END dspark1
EOF
out=$($PARSER "$tmp/real.log")
check "i=0 rows=1 new_tokens=8" "$(printf '%s\n' "$out" | head -1 | cut -d' ' -f1-3)" "a real arm row is read"
check "NEW 8" "$(printf '%s\n' "$out" | grep '^NEW ')" "the token total is the row's"
case "$(printf '%s\n' "$out" | head -1)" in
  *text_sha256=*[0-9a-f]) ok "the row hash is present" ;;
  *) bad "the row hash is present" ;;
esac

# The log is read even when stdout was never captured: the parser takes the path, so a heredoc in the
# caller cannot swallow it. Two rows, out of order, sum without a row being dropped.
printf '%s\n' '{"i": 1, "new_tokens": 4, "text": "b"}' '{"i": 0, "new_tokens": 4, "text": "a"}' >"$tmp/two.log"
out=$(printf 'not this\n' | $PARSER "$tmp/two.log")
check "i=0 rows=2 new_tokens=8" "$(printf '%s\n' "$out" | head -1 | cut -d' ' -f1-3)" "two rows are summed, first index wins"
# ... and the hash is over (i|text) in index order, so a re-ordered log hashes the same.
$PARSER "$tmp/two.log" >"$tmp/h1"
printf '%s\n' '{"i": 0, "new_tokens": 4, "text": "a"}' '{"i": 1, "new_tokens": 4, "text": "b"}' >"$tmp/two2.log"
$PARSER "$tmp/two2.log" >"$tmp/h2"
check "$(cat "$tmp/h1")" "$(cat "$tmp/h2")" "the row hash does not depend on log order"

# No rows: a log with only chatter, and a missing log. Both must say NONE rather than fail the smoke.
printf 'SEAMPROF bucket rows\nARM_END dspark1\n' >"$tmp/empty.log"
check "NONE" "$($PARSER "$tmp/empty.log")" "chatter alone is NONE"
check "NONE" "$($PARSER "$tmp/does-not-exist.log")" "a missing log is NONE"
# A JSON object without text (a SUMMARY line is not JSON on its own; this one is) is not a row.
printf '%s\n' '{"arm": "dspark1", "n": 1}' >"$tmp/summary.log"
check "NONE" "$($PARSER "$tmp/summary.log")" "an object without text is not a row"

printf 'RESULT %s fails=%d\n' "$([ $fails = 0 ] && echo PASS || echo FAIL)" "$fails"
[ $fails = 0 ]
