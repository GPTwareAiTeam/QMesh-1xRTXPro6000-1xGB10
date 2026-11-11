#!/bin/bash
# server_gate.sh -- the GX seam server refuses to load the cold pack outside the box gate.
#
#   ./qmesh/tests/server_gate.sh                 # python3 on PATH
#   PY=/usr/bin/python3 ./qmesh/tests/server_gate.sh
#
# A GX box has one cold pack and one server, and two kinds of job want it: the pair release, and a
# bench that measures against a loaded pack. They are serialised by <queue>/gx<N>.gate, and the only
# launcher that checks the claim is seam_server_ensure.sh, which passes SEAM_GATE_OK=1 through the
# docker exec. A server started any other way must refuse before it maps the 113.65 GiB pack: the
# direct launch of 2026-11-02 01:43 on GX2 held the pack outside the claim and blocked the dev
# update for its whole run.
#
# Every case is cheap and safe on a box that is serving: the refused case stops at the guard, the
# accepted case stops in SeamServer's own SEAM_NO_PACK check on a directory that has no pack, and
# the stub case returns before the pack and CUDA by construction (SeamServer.__init__), so no case
# maps a page of the pack or allocates device memory.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$(dirname "$HERE")")
PY=${PY:-python3}
NOPACK=${NOPACK:-/tmp/server-gate-test-nopack}
mkdir -p "$NOPACK"
# SeamServer parses the placement before it looks at --stub, and this directory holds no pack: the
# stub case below only needs that parse to succeed.
printf '{}\n' > "$NOPACK/placement.json"
fails=0
ok()  { echo "OK   $1"; }
bad() { echo "FAIL $1"; printf '%s\n' "$2" | tail -6 | sed 's/^/     /'; fails=$((fails + 1)); }

cd "$ROOT" || exit 2
# A tree that is not the release whose torch cache is in TORCH_EXTENSIONS_DIR must never import under
# that cache: torch would rebuild the objects from this tree and relink the release's .so, leaving a
# binary that no longer matches BUILD_STAMP.json (2026-11-02: this test did that to dev's release and
# cost a rebuild). Refuse instead of silently rebuilding, and say which release the cache belongs to.
if [ -n "${TORCH_EXTENSIONS_DIR:-}" ] && [ -f "$(dirname "$TORCH_EXTENSIONS_DIR")/BUILD_STAMP.json" ]; then
  cache_sha=$("$PY" - "$(dirname "$TORCH_EXTENSIONS_DIR")/BUILD_STAMP.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("source_sha", ""))
PY
)
  tree_sha=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)
  if [ "$cache_sha" != "$tree_sha" ]; then
    echo "SKIP server_gate.sh: TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR belongs to release" \
         "${cache_sha:-?}, the tree under test is ${tree_sha:-<not a git checkout>}:" \
         "a foreign tree would relink the release's .so. Point TORCH_EXTENSIONS_DIR at a scratch" \
         "dir (the accepted case then pays a full JIT build) or run it in that release's tree."
    exit 2
  fi
fi
if ! "$PY" -c 'import torch' 2>/dev/null; then
  echo "SKIP server_gate.sh: $PY has no torch (run it in a client/server container: PY=python3)"
  exit 0
fi

run() {  # <env NAME=VAL | -> ... --; prints stdout+stderr, returns the exit status
  local -a envs=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift || true
  env -u SEAM_GATE_OK "${envs[@]}" timeout 120 "$PY" -m qmesh.worker_seam_server \
    --model-dir "$NOPACK" --placement "$NOPACK/placement.json" --port 47199 "$@" 2>&1
}

echo "== a pack load without the gate"
out=$(run --); rc=$?
if printf '%s' "$out" | grep -q '^SEAM_SERVER_REFUSED SEAM_GATE_OK missing: start through seam_server_ensure.sh'; then
  ok "refused with the gate line"
else
  bad "no SEAM_SERVER_REFUSED line" "$out"
fi
[ "$rc" = 3 ] && ok "exit 3" || bad "exit was $rc, wanted 3" "$out"
if printf '%s' "$out" | grep -q 'SEAM_NO_PACK'; then
  bad "reached the pack load anyway" "$out"
else
  ok "stopped before SeamServer"
fi

echo "== the same launch with a value that is not 1"
for v in 0 yes ''; do
  out=$(run SEAM_GATE_OK="$v" --); rc=$?
  if printf '%s' "$out" | grep -q 'SEAM_SERVER_REFUSED'; then ok "SEAM_GATE_OK='$v' refused"; else
    bad "SEAM_GATE_OK='$v' was accepted" "$out"; fi
  [ "$rc" = 3 ] && ok "SEAM_GATE_OK='$v' exit 3" || bad "SEAM_GATE_OK='$v' exit was $rc" "$out"
done

echo "== the gate's answer, SEAM_GATE_OK=1"
out=$(run SEAM_GATE_OK=1 --); rc=$?
if printf '%s' "$out" | grep -q 'SEAM_SERVER_REFUSED'; then
  bad "SEAM_GATE_OK=1 still refused" "$out"
elif printf '%s' "$out" | grep -q 'SEAM_NO_PACK'; then
  # Past the guard, into SeamServer's own pre-load check: exactly the path the ensure takes.
  ok "passed the guard into the load (SEAM_NO_PACK, rc=$rc)"
else
  bad "SEAM_GATE_OK=1 neither refused nor reached the load" "$out"
fi

echo "== --stub maps no pack, so it is exempt"
# --stub loads no pack, so it must pass the guard and reach qes.Server. No device is opened either:
# the RDMA name is one that cannot exist, so this case costs no device and no memory wherever it
# runs, and the pass condition is "got past the guard", which is what the exemption claims.
out=$(run -- --stub --dev nosuchrdma0); rc=$?
if printf '%s' "$out" | grep -q 'SEAM_SERVER_REFUSED'; then
  bad "--stub was refused" "$out"
elif printf '%s' "$out" | grep -qE 'SEAM_SERVER_LOADED .*stub=1|qes_server_open: RDMA device nosuchrdma0'; then
  ok "--stub passed the guard without the gate (rc=$rc)"
else
  bad "--stub neither refused nor reached qes.Server" "$out"
fi

if [ "$fails" = 0 ]; then echo "SERVER_GATE_OK"; exit 0; fi
echo "SERVER_GATE_FAIL n=$fails"; exit 1
