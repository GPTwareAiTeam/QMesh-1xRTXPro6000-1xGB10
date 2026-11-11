#!/bin/bash
# update_enqueue.sh -- a queued update actually carries the command that runs it.
#
#   ./tests/update_enqueue.sh
#
# pair_update.sh once enqueued `"${cmd[@]}"` with cmd never built: the queue got a ticket with no
# command, the runner dutifully "ran" it for zero seconds and reported the pair updated. Nothing
# was updated. This test reads the argv the queue runner receives and the environment the update
# run would inherit, and asserts an empty ticket is refused rather than queued.
#
# Swept with stubs: the runner here is a stub that records its argv and writes the ticket file the
# real runner writes, so pair_enqueue's wait for the ticket succeeds without a GPU or a queue.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
export PAIR_CONF="$HERE/pairs.conf" PAIR_COLD_PACK=/home/test/cold-pack PAIR_ART_MOUNT=/home/test/artifacts
SHA=$(git -C "$(dirname "$(dirname "$(dirname "$HERE")")")" rev-parse HEAD)
WORK=$(mktemp -d /tmp/pairq.XXXXXX)
mkdir -p "$WORK/queue/dev/gpu_queue" "$WORK/state"
cat > "$WORK/queue/run_dev.sh" <<EOF
#!/usr/bin/env bash
agent=\$1
printf '%s\n' "\$@" > "$WORK/argv"
printf 'PAIR_FLIP_ONLY=%s\n' "\${PAIR_FLIP_ONLY:-}" > "$WORK/env"
: > "$WORK/queue/dev/gpu_queue/01-1-\$BASHPID-\$agent"
EOF
chmod +x "$WORK/queue/run_dev.sh"
cd "$REL"
export PAIR_ROOT_OVERRIDE="$WORK/pair" PAIR_QUEUE="$WORK/queue" PAIR_STATE_OVERRIDE="$WORK/pairs" \
       PAIR_DRY=1 PAIR_DRY_GPU=free
unset PAIR_FLIP_ONLY
fails=0
want() {  # <case> <expected-substring> <text>
  if printf '%s' "$3" | grep -q -- "$2"; then echo "OK   $1"
  else echo "FAIL $1 (wanted '$2')"; printf '%s\n' "$3" | tail -6 | sed 's/^/     /'; fails=$((fails + 1)); fi
}

out=$(./pair_update.sh dev "$SHA" --flip-only --max-min 240 2>&1); rc=$?
want "a flip ticket is enqueued" "ENQUEUED dev agent=PairUpdate" "$out"
[ "$rc" = 0 ] && echo "OK   enqueue exit=0" || { echo "FAIL enqueue exit=$rc"; fails=$((fails + 1)); }
argv=$(cat "$WORK/argv" 2>/dev/null)
want "the ticket runs pair_update_run.sh for this pair" "pair_update_run.sh" "$argv"
want "the ticket names the pair" "^dev$" "$argv"
want "the ticket names the sha" "^$SHA$" "$argv"
want "flip-only reaches the run" "PAIR_FLIP_ONLY=1" "$(cat "$WORK/env" 2>/dev/null)"

# A full update (no --flip-only) must not leave the flip-only flag set for the run it starts.
rm -f "$WORK/argv" "$WORK/env"
out=$(./pair_update.sh dev "$SHA" --max-min 60 2>&1); rc=$?
want "a full update is enqueued" "ENQUEUED dev agent=PairUpdate" "$out"
want "a full update does not flip only" "PAIR_FLIP_ONLY=$" "$(cat "$WORK/env" 2>/dev/null)"

# The queue itself refuses a ticket with nothing to run, so the empty-command bug can not come back
# through another caller either.
rm -f "$WORK/argv"
out=$(. ./pair_lib.sh; pair_load dev; pair_enqueue PairUpdate 60 2>&1); rc=$?
want "an empty ticket is refused" "no command given" "$out"
[ "$rc" != 0 ] && echo "OK   empty ticket exit=$rc" || { echo "FAIL empty ticket exit=0"; fails=$((fails + 1)); }
[ ! -s "$WORK/argv" ] && echo "OK   nothing was started" || { echo "FAIL the stub runner was called"; fails=$((fails + 1)); }

# The runner names its ticket with its own $$, and setsid(1) may fork, so the pid in the ticket is
# not the pid pair_enqueue got from $!. Matching on it reported ENQUEUE_FAILED for jobs that were
# queued and did run (2026-11-01: a probe ran on dev while its caller was told nothing had).
cat > "$WORK/queue/run_dev.sh" <<EOF2
#!/usr/bin/env bash
agent=\$1
printf '%s\n' "\$@" > "$WORK/argv"
: > "$WORK/queue/dev/gpu_queue/01-1-999999-\$agent"
EOF2
chmod +x "$WORK/queue/run_dev.sh"
rm -f "$WORK/argv"
out=$(./pair_update.sh dev "$SHA" --flip-only --max-min 240 2>&1); rc=$?
want "a ticket carrying another pid is still reported as queued" "ENQUEUED dev agent=PairUpdate" "$out"
want "the reported ticket is the file the runner wrote" "ticket=01-1-999999-PairUpdate" "$out"

echo "RESULT $([ "$fails" = 0 ] && echo PASS || echo FAIL) fails=$fails"
rm -rf "$WORK"
[ "$fails" = 0 ]
