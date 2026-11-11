#!/usr/bin/env bash
# Prove that a container is running the tree under test -- without writing a byte into it.
#
# Until the 2026-11-02 release rollout a ticket "staged" its worktree onto the GX (rsync into
# .) and then checked seven files by sha. Both halves of that broke
# with the rollout: the box no longer keeps a scratch copy (the container's /src/exllamav3 is a
# symlink to <pair root>/current/src, a release directory), and rsyncing a different commit into a
# release directory makes pair_check report PAIR_DRIFT, after which the next release prepare refuses
# to run. Staging an identical commit is silent but pointless.
#
# A census or an A/B arm does not need the box to be mutated; it needs proof that the bytes executing
# in the container are the bytes in the worktree the number will be attributed to. So this compares
# the whole tree by content, one dry-run, and prints what it found:
#
#   GX=${GX:?Set GX to the remote host} CX_CT=seam-gx2-dev SRC=<worktree> \
#     DST=${PAIR_ROOT:?Set PAIR_ROOT}/current/src bash cx_verify_tree.sh
#
# Exit: 0 same bytes, 2 a file differs (the first ten are printed), 3 the container does not resolve
# /src/exllamav3 to DST (then the comparison would be about the wrong directory), 4 rsync itself
# failed (no proof either way).
set -uo pipefail
: "${GX:?Set GX to the remote host}"
CT=${CX_CT:-seam-gx2-dev}
SRC=${SRC:-.}
DST=${CX_DST:-${PAIR_ROOT:?Set PAIR_ROOT}/current/src}
CTWT=${CTWT:-/src/exllamav3}

# The two sides do not share a path space: rsync writes the host's $DST, while the container sees the
# same directory as /src/exllamav3, reached through the /pair mount. So resolve the destination on the
# host, resolve /src/exllamav3 inside the container, map the container's /pair back to the host's pair
# root, and require that both name one directory -- otherwise this would be a proof about bytes nobody
# is executing.
: "${PAIR_ROOT:?Set PAIR_ROOT}"
res=$(ssh -o BatchMode=yes "$GX" "docker exec $CT readlink -f $CTWT" 2>/dev/null | tr -d '\r')
want=$(ssh -o BatchMode=yes "$GX" "readlink -f $DST" 2>/dev/null | tr -d '\r')
case "$res" in
  /pair/*) got="$PAIR_ROOT/${res#/pair/}" ;;
  *)       got="" ;;
esac
if [ -z "$got" ] || [ "$got" != "$want" ]; then
  echo "VERIFY FAIL ct=$CT $CTWT=$res host=$want want=$DST"
  exit 3
fi
echo "VERIFY target ct=$CT tree=$res host=$want"

# -c checksums instead of size+mtime (a release cut and a worktree checkout of one commit rarely
# share mtimes), --dry-run so nothing is written, --itemize-changes so every difference names itself.
out=$(rsync -a -c --dry-run --itemize-changes \
        --exclude .git --exclude qmesh/native/obj --exclude qmesh/native/bin --exclude __pycache__ \
        "$SRC/" "$GX:$DST/" 2>&1)
rc=$?
if [ "$rc" -ge 24 ]; then
  printf 'VERIFY FAIL rsync rc=%s %s\n' "$rc" "$(printf '%s' "$out" | tail -3 | tr '\n' ' ')"
  exit 4
fi
nd=$(printf '%s\n' "$out" | grep -c '^[*>]' || true)
if [ "$nd" -gt 0 ]; then
  printf '%s\n' "$out" | grep '^[*>]' | head -10 | sed 's/^/VERIFY diff /'
  echo "VERIFY FAIL src=$SRC files_diff=$nd"
  exit 2
fi
# The aarch64 transport library is built inside the container, never copied onto it: report the .cu it
# was built from so a log can tell "the lib is the commit under test" from "the lib is one edit
# behind" (a comment or a print costs no behaviour and rebuilding it changes every container's build
# id). The release ships a .stamp next to the .so; a tree staged by hand may ship BUILD_INFO. Read it
# through $CTWT, not $DST: inside the container only the /pair path exists.
stamp=$(ssh -o BatchMode=yes "$GX" "docker exec $CT sh -c 'cat $CTWT/qmesh/native/bin/libqmesh_expert_seam.aarch64.so.stamp 2>/dev/null || cat $CTWT/qmesh/native/bin/BUILD_INFO.aarch64 2>/dev/null'" 2>/dev/null | tr '\n' ' ')
cu=$(ssh -o BatchMode=yes "$GX" "docker exec $CT sha256sum $CTWT/qmesh/native/qmesh_expert_seam.cu 2>/dev/null" | cut -c1-16)
echo "VERIFY lib cu_sha=${cu:-missing} stamp=${stamp:-none}"
echo "VERIFY OK src=$SRC files_diff=0"
