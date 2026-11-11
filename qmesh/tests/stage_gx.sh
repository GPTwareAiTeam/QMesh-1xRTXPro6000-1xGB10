#!/usr/bin/env bash
# Stage this worktree onto the box that runs the cold experts.
#
# The GX trees are *copies* (rsynced, no .git), not shares: a change committed on P7 is invisible to a
# container until someone stages it. That is how a ranking change can be "landed" and still not be
# running on the GX -- and why every driver that trusts container code stages first and then proves,
# by sha, that the container is looking at the same bytes.
#
#   GX=<ssh-host> CX_CT=<container> DST=<remote-path> bash qmesh/tests/stage_gx.sh
#
# Not copied: .git, qmesh/native/obj (build intermediates), qmesh/native/bin (the aarch64 libraries
# are built in the container's own CUDA image, and a stale P7-side binary must never overwrite them),
# __pycache__. Exit 0 staged and verified, 1 rsync failed, 2 a staged file does not match.
set -uo pipefail
GX=${GX:?set GX to the server SSH host}
CT=${CX_CT:?set CX_CT to the server container}
SRC=${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
DST=${DST:?set DST to the remote checkout path}
CTWT=${CTWT:-/src/exllamav3}

rsync -a --exclude .git --exclude qmesh/native/obj --exclude qmesh/native/bin --exclude __pycache__ \
  "$SRC/" "$GX:$DST/" || { echo "STAGE FAIL rsync $SRC/ -> $GX:$DST/"; exit 1; }

# The files whose behaviour the seam's two halves have to agree on. A stale one of these is exactly
# the class of bug this script exists for: the transport cannot tell an old numbering from a new one.
bad=0
for f in qmesh/worker_seam_server.py qmesh/placement.py qmesh/native/qes.py \
         qmesh/tests/cx_direct_check.py qmesh/tests/placement_sha_refusal.py \
         qmesh/tests/refusal_stub.sh qmesh/native/qmesh_expert_seam.cu; do
  l=$(sha256sum "$SRC/$f" | cut -c1-16)
  r=$(ssh -o BatchMode=yes "$GX" "docker exec $CT sha256sum $CTWT/$f" 2>/dev/null | cut -c1-16)
  if [ "$l" = "$r" ]; then
    echo "STAGE $f sha=$l"
  else
    echo "STAGE FAIL $f local=$l ct=${r:-missing}"
    bad=1
  fi
done
# The libraries are built in the container, so their shas are the container's business; print them so
# a log can be matched against the run that produced a number.
ssh -o BatchMode=yes "$GX" "docker exec $CT bash -lc 'cd $CTWT/qmesh/native/bin && sha256sum * 2>/dev/null'" \
  | awk 'NF == 2 {printf "STAGE lib %s sha=%.16s\n", $2, $1}'
# The library is built in the container from the .cu on disk, so after a stage it can legitimately
# be one revision behind (a comment or a stderr line costs no behaviour, and rebuilding it changes
# the build id for every container sharing the tree). Say which it is: BUILD_INFO records the sha the
# lib was compiled from, and a log that shows state=stale next to a passing run tells the next reader
# that the source on disk is newer on purpose, not that the stage silently failed.
cu=$(sha256sum "$SRC/qmesh/native/qmesh_expert_seam.cu" | cut -c1-16)
built=$(ssh -o BatchMode=yes "$GX" "docker exec $CT cat $CTWT/qmesh/native/bin/BUILD_INFO.aarch64" 2>/dev/null |
  sed -n 's/^cu_sha256_full=//p' | cut -c1-16)
echo "STAGE lib_src seam_cu=$cu built_from=${built:-unknown} state=$([ "$cu" = "$built" ] && echo match || echo stale)"
[ "$bad" = 0 ] || exit 2
echo "STAGE OK host=$GX ct=$CT tree=$CTWT"
