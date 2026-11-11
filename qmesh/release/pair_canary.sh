#!/usr/bin/env bash
# pair_canary.sh -- finish a prepare inside the queue and then flip, as one ticket.
#
#   pair_canary.sh <pair> <sha>
#
# Why this exists: `pair_prepare.sh <pair> <sha> --side gx` builds the GX native transport lib, and
# that build may only run under the box's claim (`pair_gx_gate_up` refuses outside a queue ticket with
# PAIR_GATE_REFUSED outside_ticket), so the GX half of a prepare cannot finish out of band while any
# other job has ever held gx<N>.gate. Running the prepare and the flip as two tickets does not work
# either: both are rank 01 (`pair_update.sh` puts PairUpdate on line 1), the flip's earlier enqueue
# time wins, and it stops at `phase=prepared` because the GX ext it verifies is not there yet.
#
# So the ticket does both, in order, and stops the moment the prepare refuses -- a flip over a
# half-built release is what `--flip-only`'s verify exists to prevent.
#
# The caller still has to prepare the P7 side out of band first (it needs no claim and no gate time).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_canary.sh <pair> <sha>}
SHA_ARG=${2:?usage: pair_canary.sh <pair> <sha>}

pair_load "$PAIR_ARG"

# The queue orders tickets by rank, but a rank-01 ticket starts as soon as it is at the head -- a job
# that is already running does not hold the head, it only holds the *claim*. So a canary queued ahead
# of a benchmark runs beside it unless it waits here. 2026-11-02 11:20:43: a rank-01
# `pair_update.sh … --flip-only` started while G6Bound's bench held gx2 and died at `phase=prepared`;
# harmless only because the GX half was not built yet. Wait for the claim to be released, bounded, and
# then let the prepare take its own.
waited=0
while :; do
  state=$(pair_gx_gate_state)
  [ "$state" = free ] && break
  if [ "$waited" -ge "${PAIR_CANARY_GATE_WAIT_MIN:-40}" ]; then
    pair_die "CANARY_GATE_WAIT_TIMEOUT pair=$PAIR state=$state waited=${waited}m"
  fi
  pair_step "CANARY_GATE_WAIT pair=$PAIR state=$state waited=${waited}m"
  sleep 60; waited=$((waited + 1))
done

# A ticket is the only place a container may be repaired, so the first thing it does is put both halves
# back on `current`. On 2026-11-02 the previous GX build was killed mid-flight and left seam-gx2-dev
# wired at the unflipped release (`pair_check dev` = GX DRIFT, nothing importing /pair/current/src).
# This is idempotent: it only rewrites a symlink that resolves elsewhere, and it never touches a real
# directory or a container without /pair/current/src (an unpromoted pair has nothing to restore to).
relink_side() {
  local side=$1 ct=$2 got
  local inner='if [ -L /src/exllamav3 ] && [ -e /pair/current/src ]; then ln -sfn /pair/current/src /src/exllamav3; fi; readlink -f /src/exllamav3'
  if [ "$side" = p7 ]; then
    got=$(docker exec "$ct" sh -c "$inner" 2>/dev/null)
  else
    got=$(pair_gxsh "docker exec $(printf %q "$ct") sh -c $(printf %q "$inner")" 2>/dev/null)
  fi
  pair_step "CANARY_RELINK $side $ct -> ${got:-?}"
}
relink_side p7 "$P7CT"
relink_side gx "$GXCT"

nice -n 15 "$HERE/pair_prepare.sh" "$PAIR_ARG" "$SHA_ARG" --side gx || exit 2
PAIR_FLIP_ONLY=1 exec "$HERE/pair_update_run.sh" "$PAIR_ARG" "$SHA_ARG"
