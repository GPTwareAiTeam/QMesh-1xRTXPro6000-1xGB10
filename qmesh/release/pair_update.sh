#!/usr/bin/env bash
# pair_update.sh -- queue an update of one pair, to land right after the job that is running now.
#
#   pair_update.sh <pair> [sha] [--max-min N] [--flip-only] [--recreate] [--build]
#
# --flip-only: the release was already cut, wired and built out of band by pair_prepare.sh, so the
# ticket only flips `current`, restarts the server, smokes and verifies (minutes, not the hour a
# torch JIT build takes). Without it the ticket still does everything itself.
#
# --recreate: the container spec changed (2026-11-02: the P7 containers no longer mount the 12 TB
# drive, because a pack lives on the serving box's own disk), so the ticket is allowed one recreate.
# Implies --flip-only, unless --build is also given: the release is prepared as usual, and the queue
# is where a container is re-built from its spec -- never an operator with docker rm. The old
# container is renamed to <name>.bak-<ts>, so the flip is still recoverable by hand if the smoke fails.
#
# --build: the opposite of --flip-only, and the only way to combine "build" with "--recreate". A pair
# whose container spec changed *and* whose torch extension was therefore never built (prod before a
# promotion that also changes the spec) has no order that works in two passes: prepare refuses to
# build before the container is recreated, and a flip ticket refuses to flip before the extension is
# built. With --build the ticket takes the full path -- cut, containers (recreating when the plan says
# RECREATE), lib build, ext build in the recreated container, flip, smoke -- which is the order that
# does work. Flag order does not matter.
#
# prod is not addressable from here directly: it is promoted only through pair_promote.sh, which
# requires the exact sha to have passed on dev (a PAIR_UPDATED evidence file with a smoke that
# recorded generated token ids and a post-update check that came back IN_SYNC). A bare
# `pair_update.sh prod …` refuses, so a bad dev release cannot reach the production pair by
# accident -- and the refusal is a hard exit before the queue is touched, not a warning.
#
# The update is not run here: it is a ticket on the pair's own GPU queue, so it waits for the pair's
# current job to finish and never collides with one (one GPU job per pair, always through the queue).
# PairUpdate is put on line 1 of gpu_priority.txt first, so it goes next even if other agents are
# already waiting -- their tickets keep the rank they were created with, so nobody is reordered.
#
# Prints ENQUEUED <pair> agent=PairUpdate pid=... ticket=... log=...
# Follow it with:  scripts/pair_status.sh <pair>   and   tail -f .../pairs/logs/latest-update-<pair>.log
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_update.sh <pair> [sha] [--max-min N] [--recreate] [--build]}
shift
SHA=""; MINS=${PAIR_UPDATE_MAX_MIN:-120}; FLIP_ONLY=${PAIR_FLIP_ONLY:-0}; RECREATE=0
BUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-min) MINS=$2; shift ;;
    --flip-only) FLIP_ONLY=1 ;;
    --recreate) FLIP_ONLY=1; RECREATE=1 ;;
    --build) BUILD=1 ;;
    -*) pair_die "unknown flag $1" ;;
    *) SHA=$1 ;;
  esac
  shift
done
# Order-independent on purpose: --recreate implies a flip, --build is the full path, and a pair that
# needs both (a spec change plus an extension that has never been built against the new spec) must end
# up on the full path whatever order the two were typed in.
[ "$BUILD" = 1 ] && FLIP_ONLY=0
pair_load "$PAIR_ARG"

if [ "$PAIR" = prod ] && [ "${PAIR_PROMOTE:-0}" != 1 ]; then
  pair_die "prod is promoted, not updated: use scripts/pair_promote.sh <sha> (dev canary first)"
fi
if [ "$PAIR" = prod ] && [ -z "$SHA" ]; then
  pair_die "prod promotion needs an explicit sha (dev evidence is recorded per sha)"
fi
if [ "$PAIR" = prod ]; then
  # The canary evidence: written by the dev ticket itself (pair_update_run.sh evidence()), so it
  # can not be produced by the run that wants to consume it.
  ev=$PAIR_STATE/dev/evidence/$SHA.json
  if [ ! -f "$ev" ]; then
    pair_die "no dev canary evidence for ${SHA:0:12}: $ev (dev must reach PAIR_UPDATED first)"
  fi
  gate=$(pair_canary_gate "$SHA" "$ev")
  [ "$gate" = ok ] || pair_die "dev canary for ${SHA:0:12} does not qualify: $gate ($ev)"
  pair_step "UPDATE_CANARY pair=$PAIR sha=${SHA:0:12} dev_evidence=$ev ok"
fi
if [ "$RECREATE" = 1 ]; then
  # The permission travels in the environment of the queued ticket: the queue is the only place a
  # container is built from its spec, and the ticket is the only process that holds it.
  export PAIR_ALLOW_RECREATE=1
  pair_step "UPDATE_RECREATE pair=$PAIR allowed (spec changed; the ticket re-creates once)"
fi

pair_ensure_priority PairUpdate
cmd=("$HERE/pair_update_run.sh" "$PAIR" "$SHA")
[ "$FLIP_ONLY" = 1 ] && export PAIR_FLIP_ONLY=1
pair_enqueue PairUpdate "$MINS" "${cmd[@]}"
