#!/usr/bin/env bash
# pair_check.sh -- is this pair's P7 half and GX half serving one and the same release?
#
#   pair_check.sh <pair> [--quiet]
#
# Exit: 0 IN_SYNC, 4 PAIR_DRIFT (a job must not start), 5 PAIR_CHECK_ERROR (could not tell).
# This is the guard the GPU queue calls before it starts a job (qmesh/release/gpu_run_pair.sh); it is
# also what pair_status.sh and the update job call. It never inspects by hand: the comparison lives in
# pair_sync_check.py (manifest mode), so the queue, the status page and the updater cannot disagree.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_check.sh <pair> [--quiet]}
shift
pair_load "$PAIR_ARG"

if [ ! -r "$PAIR_MANIFEST" ]; then
  echo "PAIR_DRIFT $PAIR no_manifest ($PAIR_MANIFEST)"
  exit 4
fi
out=$(python3 "$HERE/pair_sync_check.py" --manifest "$PAIR_MANIFEST" "$@" 2>&1)
rc=$?
printf '%s\n' "$out"
case "$rc" in
  0) exit 0 ;;
  1) echo "PAIR_DRIFT $PAIR"; exit 4 ;;
  *) [ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$out" >> "$PAIR_LOG"
     echo "PAIR_CHECK_ERROR $PAIR rc=$rc"; exit 5 ;;
esac
