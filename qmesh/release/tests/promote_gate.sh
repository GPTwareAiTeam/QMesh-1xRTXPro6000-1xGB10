#!/bin/bash
# promote_gate.sh -- prod can only be reached for a sha dev proved, and only through pair_promote.sh.
#
#   ./tests/promote_gate.sh
#
# Swept with stubs: no container, no queue, no GPU. Each case asserts the refusal message, because a
# gate that refuses for the wrong reason is worse than none -- it hides the real blocker.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
export PAIR_CONF="$HERE/pairs.conf" PAIR_COLD_PACK=/home/test/cold-pack PAIR_ART_MOUNT=/home/test/artifacts
SHA=$(git -C "$(dirname "$(dirname "$(dirname "$HERE")")")" rev-parse HEAD)
WORK=$(mktemp -d /tmp/pairgate.XXXXXX); export PATH="$WORK/bin:$PATH"
mkdir -p "$WORK/bin" "$WORK/state" "$WORK/pairs/dev/evidence"
cp "$HERE/stub_docker.py" "$WORK/bin/docker"; chmod +x "$WORK/bin/docker"
cp "$HERE/stub_ssh.sh" "$WORK/bin/stub_ssh"; chmod +x "$WORK/bin/stub_ssh"
sed -i "s#/tmp/stub/state.json#$WORK/state/state.json#; s#/tmp/stub/exec.log#$WORK/state/exec.log#" "$WORK/bin/docker"
cd "$REL"
# PAIR_QUEUE (not PAIR_QUEUE_ROOT) is what pair_lib.sh reads for the queue: pointing it at a directory
# with no runner is what makes an accidental enqueue impossible in a stub sweep.
export PAIR_DRY=1 PAIR_DRY_SSH="$WORK/bin/stub_ssh" PAIR_ROOT_OVERRIDE="$WORK/pair" \
       PAIR_QUEUE="$WORK/noq" PAIR_DRY_GPU=free PAIR_STATE_OVERRIDE="$WORK/pairs"
unset PAIR_PROMOTE
EV="$WORK/pairs/dev/evidence/$SHA.json"
fails=0
want() {  # <case> <expected-substring> <log>
  if printf '%s' "$3" | grep -q -- "$2"; then echo "OK   $1"
  else echo "FAIL $1 (wanted '$2')"; printf '%s\n' "$3" | tail -6 | sed 's/^/     /'; fails=$((fails + 1)); fi
}

# 1. prod straight from the update script: refused before anything is read or queued.
out=$(./pair_update.sh prod "$SHA" 2>&1); rc=$?
want "prod without pair_promote refused" "use scripts/pair_promote.sh" "$out"
[ "$rc" = 2 ] && echo "OK   prod refusal exit=2" || { echo "FAIL prod refusal exit=$rc"; fails=$((fails + 1)); }

# 2. the same, carrying the flag a promotion carries, but with no dev evidence for this sha.
out=$(PAIR_PROMOTE=1 ./pair_update.sh prod "$SHA" 2>&1); rc=$?
want "prod without evidence refused" "no dev canary evidence" "$out"
[ "$rc" = 2 ] && echo "OK   evidence refusal exit=2" || { echo "FAIL evidence refusal exit=$rc"; fails=$((fails + 1)); }

# 3. evidence that exists but is a failed attempt: a failure must not pass as a canary.
printf '{"format":1,"pair":"dev","release_sha":"%s","ok":false,"failed_at_phase":"smoke","why":"smoke rc=1 SMOKE_FAIL arm rc=124","smoke":{"row":"","ids":{}}}\n' \
  "$SHA" > "$EV"
out=$(PAIR_PROMOTE=1 ./pair_update.sh prod "$SHA" 2>&1); rc=$?
want "failed dev attempt refused" "ok=false at smoke" "$out"
[ "$rc" = 2 ] && echo "OK   failed-evidence refusal exit=2" || { echo "FAIL failed-evidence exit=$rc"; fails=$((fails + 1)); }

# 4. a passing attempt with a row but no ids: exactly the case the supervisor called out -- text is
#    not proof that a decode ran.
printf '{"format":1,"pair":"dev","release_sha":"%s","ok":true,"smoke":{"row":"i=0 rows=1","ids":{}}}\n' \
  "$SHA" > "$EV"
out=$(PAIR_PROMOTE=1 ./pair_update.sh prod "$SHA" 2>&1); rc=$?
want "dev evidence without ids refused" "no_decode_ids" "$out"
[ "$rc" = 2 ] && echo "OK   no-ids refusal exit=2" || { echo "FAIL no-ids refusal exit=$rc"; fails=$((fails + 1)); }

# 5. a passing attempt with ids: the gate opens, and the next thing that stops the run is the queue
#    runner being absent in this harness -- i.e. nothing about the canary gate.
printf '{"format":1,"pair":"dev","release_sha":"%s","ok":true,"smoke":{"row":"i=0 rows=1","ids":{"n":8,"sha256":"deadbeef"}}}\n' \
  "$SHA" > "$EV"
out=$(PAIR_PROMOTE=1 ./pair_update.sh prod "$SHA" 2>&1); rc=$?
want "qualified evidence opens the gate" "UPDATE_CANARY pair=prod sha=${SHA:0:12}" "$out"
if printf '%s' "$out" | grep -q "queue runner missing"; then
  echo "OK   stopped after the gate (no queue in the harness)"
else
  echo "FAIL expected the missing queue runner after the gate"; printf '%s\n' "$out" | tail -6 | sed 's/^/     /'
  fails=$((fails + 1))
fi

# 6. pair_promote.sh is the stated way in: without --force it refuses on the dev manifest gap.
out=$(./pair_promote.sh "$SHA" 2>&1); rc=$?
want "promote refuses a non-dev sha without --force" "not ${SHA:0:12}" "$out"

# 7. with --force (manifest/seam relaxed) it still refuses without evidence.
rm -f "$EV"
out=$(./pair_promote.sh "$SHA" --force 2>&1); rc=$?
want "promote refuses a sha without dev evidence" "no dev canary evidence" "$out"
[ "$rc" = 2 ] && echo "OK   promote evidence refusal exit=2" || { echo "FAIL promote evidence exit=$rc"; fails=$((fails + 1)); }

# 8. The ticket's own script refuses prod without a canary, so queueing pair_update_run.sh by hand is
#    not a way around the gate. Both cases refuse before any phase: no sha, and a sha dev never ran.
out=$(PAIR_DRY=0 ./pair_update_run.sh prod 2>&1); rc=$?
want "prod ticket without a sha refused" "prod needs an explicit sha" "$out"
FAKE=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef
out=$(PAIR_DRY=0 ./pair_update_run.sh prod "$FAKE" 2>&1); rc=$?
want "prod ticket with an unproven sha refused" "dev canary for deadbeefdead does not qualify" "$out"
[ "$rc" = 2 ] && echo "OK   prod ticket bypass exit=2" || { echo "FAIL prod ticket bypass exit=$rc"; fails=$((fails + 1)); }
if [ -d "$WORK/state" ] && [ -s "$WORK/state/exec.log" ]; then
  echo "FAIL the refused ticket ran a docker command"; fails=$((fails + 1))
else
  echo "OK   the refused ticket touched nothing"
fi

echo
if [ "$fails" = 0 ]; then echo "RESULT PASS fails=0"; else echo "RESULT FAIL fails=$fails"; fi
exit $((fails != 0))
