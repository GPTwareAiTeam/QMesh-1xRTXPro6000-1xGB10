#!/usr/bin/env bash
# pair_env.sh -- the environment a client half needs, in eval-able form.
#
#   eval "$(pair_env.sh dev)"
#   eval "$(pair_env.sh prod)"
#
# Every client invocation (a bench script, a one-off arm, the smoke row) gets the seam address, the
# RDMA device/gid, the GPU index and the release root from here instead of from whatever a script
# remembered to export. Values are shell-quoted, one K=V per line, no `export` (add it yourself).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

pair_load "${1:?usage: pair_env.sh <pair> [sha]}"
SHA=${2:-$(pair_current_sha "$PAIR_ROOT")}
[ -n "$SHA" ] || pair_die "pair $PAIR has no release yet (no $PAIR_ROOT/current)"

printf 'PAIR=%q\n' "$PAIR"
printf 'P7CT=%q\n' "$P7CT"
printf 'GX=%q\n' "$GX"
printf 'GXCT=%q\n' "$GXCT"
printf 'PAIR_ROOT=%q\n' "$PAIR_ROOT"
printf 'PAIR_RELEASE=%q\n' "$PAIR_ROOT/releases/$SHA"
printf 'PAIR_BUILD_ID=%q\n' "$(python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1]))["build_id"])
except Exception: print("")' "$PAIR_ROOT/releases/$SHA/RELEASE.json")"
# The container is device-scoped (spec --gpus device=<host index>), so the one card it can see is
# device 0 inside it: anything exec'ing into $P7CT must export 0, not the host index. PAIR_GPU_HOST
# is the host index, for the few host-side probes (nvidia-smi -i, --gpus) that need it.
printf 'CUDA_VISIBLE_DEVICES=%q\n' 0
printf 'PAIR_GPU_HOST=%q\n' "$GPU"
printf 'SEAM_DEV=%q\n' "$SEAM_DEV"
printf 'SEAM_GID=%q\n' "$SEAM_GID"
printf 'GX_GID=%q\n' "$GX_GID"
printf 'SEAM_PEER=%q\n' "$SEAM_PEER"
printf 'SEAM_PORT=%q\n' "$PORT"
# SEAM_BUCKETS is deliberately not exported: the client asks for its own superset and the server
# narrows to the list it granted. Exporting the server's list here caps prefill chunks at its top row.
