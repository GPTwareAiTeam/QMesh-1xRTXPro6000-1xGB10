#!/usr/bin/env bash
# pair_status.sh -- one page per pair: what it claims to run, what it really runs, and why not.
#
#   pair_status.sh <pair>            # verbose
#   pair_status.sh <pair> --quiet    # the guard's view only
#   pair_status.sh all               # both pairs, one after the other
#
# Exit: 0 both halves agree on the release, 4 drift, 5 could not tell (same as pair_check.sh).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

WHICH=${1:?usage: pair_status.sh <prod|dev|all> [--quiet]}
QUIET=${2:-}
rc=0
for p in $([ "$WHICH" = all ] && echo "prod dev" || echo "$WHICH"); do
  pair_load "$p"
  echo "== pair $p =="
  echo "P7CT=$P7CT GPU=$GPU  GX=$GX CT=$GXCT  PEER=$SEAM_PEER:$PORT buckets=$BUCKETS  ROOT=$PAIR_ROOT"
  cur=$(pair_current_sha "$PAIR_ROOT"); gxcur=$(pair_current_sha "$PAIR_ROOT" 1)
  echo "CURRENT p7=${cur:--} gx=${gxcur:--}"
  if [ -r "$PAIR_MANIFEST" ]; then
    echo "MANIFEST release=$(pair_manifest_get release_sha) build=$(pair_manifest_get build_id) " \
         "seam=$(pair_manifest_get seam fingerprint) updated=$(pair_manifest_get updated_at)"
  else
    echo "MANIFEST none ($PAIR_MANIFEST)"
  fi
  python3 "$HERE/pair_containers.py" ls "$p" 2>&1 | sed 's/^/  /'
  if [ "$QUIET" != "--quiet" ]; then
    pids=$(pair_gxsh "docker exec $GXCT pgrep -fc worker_seam_server.py" 2>/dev/null | tail -1)
    echo "SERVER $GXCT pids=${pids:-?}"
    pair_gxsh "docker exec $GXCT sh -c \"grep -a '^SEAM_SERVER_\\(LOADED\\|READY\\|SLAB\\)' /tmp/seam_server.log | tail -3\"" 2>/dev/null \
      | sed 's/^/  /'
  fi
  echo "-- release check --"
  "$HERE/pair_check.sh" "$p" ${QUIET:+--quiet} | sed 's/^/  /'
  crc=${PIPESTATUS[0]}
  [ "$crc" = 0 ] || rc=$crc
done
exit $rc
