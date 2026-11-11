#!/bin/bash
# Dry sweep of the whole pair-update flow: stub ssh + stub docker, no container, queue or GPU touched.
#   ./tests/dry_run.sh [pair]
# Prints CREATE/LINK_UNPROMOTED/INNER/APPLY/LIB/EXT/FLIP/SMOKE lines and ends with PAIR_UPDATED_DRY.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
export PAIR_CONF="$HERE/pairs.conf" PAIR_COLD_PACK=/home/test/cold-pack PAIR_ART_MOUNT=/home/test/artifacts
PAIR=${1:-dev}
SHA=${2:-$(git -C "$(dirname "$(dirname "$(dirname "$HERE")")")" rev-parse HEAD)}
WORK=$(mktemp -d /tmp/pairdry.XXXXXX); export PATH="$WORK/bin:$PATH"
mkdir -p "$WORK/bin" "$WORK/state"
cp "$HERE/stub_docker.py" "$WORK/bin/docker"; chmod +x "$WORK/bin/docker"
cp "$HERE/stub_ssh.sh" "$WORK/bin/stub_ssh"; chmod +x "$WORK/bin/stub_ssh"
sed -i "s#/tmp/stub/state.json#$WORK/state/state.json#; s#/tmp/stub/exec.log#$WORK/state/exec.log#" "$WORK/bin/docker"
mkdir -p "$WORK/pair/releases/$SHA/src" "$WORK/pair/releases/$SHA/ext"
echo '{"build_id":"dry-demo","seam":{"fingerprint":"fp:dry","proto":"3"}}' > "$WORK/pair/releases/$SHA/RELEASE.json"
cd "$REL"
PAIR_DRY=1 PAIR_DRY_SSH="$WORK/bin/stub_ssh" PAIR_ROOT_OVERRIDE="$WORK/pair" \
  PAIR_QUEUE="$WORK/noq" PAIR_DRY_GPU=free PAIR_STATE_OVERRIDE="$WORK/pairs" \
  ./pair_update_run.sh "$PAIR" "$SHA"
