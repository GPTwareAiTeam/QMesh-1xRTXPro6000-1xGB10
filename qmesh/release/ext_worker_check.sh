#!/usr/bin/env bash
# ext_worker_check.sh -- which exllamav3 extension did the dev pair's GX worker actually import?
#
#   ext_worker_check.sh <tag>
#
# Prints one EXT_WORKER line (tag, worker pid, the live process's TORCH_EXTENSIONS_DIR/PYTHONPATH, the
# .so behind /pair/extcache and the sha its BUILD_STAMP claims) and exits 0 only when the worker's
# cache is the release cache and the .so matches its stamp.
#
# Why it exists: the containers' default TORCH_EXTENSIONS_DIR became /pair/scratch-ext on 2026-11-04, so
# a worker launched without a pin silently JIT-builds its own exllamav3_ext.so into scratch and the arm
# measures a build made inside the measurement window instead of the release artifact. That is exactly
# what the 02:07Z cand arm did (scratch .so dbaf5447... built 02:07 vs release 12ce7498... stamped
# 01:00:08Z), and a number from such an arm cannot be quoted. An arm that prints EXT_WORKER_OK with
# matching shas is the evidence that it did not.
set -uo pipefail
S=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$S/pair_lib.sh"

TAG=${1:?usage: ext_worker_check.sh <tag>}
pair_load dev

wpid=$(pair_gxsh "docker exec $GXCT pgrep -f '^python3 -u qmesh/worker_seam_server[.]py' | head -1" 2>/dev/null | tr -d '\r')
if [ -z "$wpid" ]; then
  echo "EXT_WORKER tag=$TAG no_worker (nothing is serving -- the arm cannot have measured anything)" >&2
  exit 2
fi

# Read the environment of the process that is actually holding the GPUs, not of a shell that inherited
# it: a `docker exec` without -e shows the container default and would make a pinned worker look unpinned.
wenv=$(pair_gxsh "docker exec $GXCT sh -c 'tr \"\\0\" \"\\n\" < /proc/$wpid/environ | grep -E \"^(TORCH_EXTENSIONS_DIR|PYTHONPATH)=\"'" 2>/dev/null \
       | tr -d '\r' | tr '\n' ' ')

# The artifact: what /pair/extcache resolves to, its sha, and what the release's stamp says it should be.
wso=$(pair_gxsh "docker exec $GXCT readlink -f /pair/extcache/exllamav3_ext/exllamav3_ext.so" 2>/dev/null | tr -d '\r')
wsha=$(pair_gxsh "docker exec $GXCT sha256sum '$wso' 2>/dev/null" 2>/dev/null | cut -c1-16)
# The stamp carries two sha256 keys: the nested `cx` object's (the _C...so) comes first in the file, so a
# plain `grep -o '"sha256"' | head -1` compared exllamav3_ext.so against _C's sha and printed
# EXT_WORKER_DRIFT on a healthy worker (03:31:58Z tag=serve_x so=7cd31e1e stamp=0438e3e9 -- those are the
# two artifacts of the one build dev-3cd80749f4db). Take the top-level key, parsed, not grepped.
stamp=$(pair_gxsh "docker exec $GXCT cat /pair/extcache/BUILD_STAMP.json" 2>/dev/null | tr -d '\r' \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("sha256","")[:16])' 2>/dev/null)
[ -n "$stamp" ] || { echo "EXT_WORKER_STAMP_UNREADABLE tag=$TAG (BUILD_STAMP.json missing or unparsable; cannot judge the cache)" >&2; exit 12; }

echo "EXT_WORKER tag=$TAG pid=$wpid env=$wenv so=$wsha stamp=${stamp:-none}"
case "$wenv" in
  *TORCH_EXTENSIONS_DIR=/pair/extcache*) echo "EXT_WORKER_OK tag=$TAG cache=release" ;;
  *)
    echo "EXT_WORKER_FOREIGN tag=$TAG env=$wenv (the arm imported a build from outside the release cache)" >&2
    exit 11
    ;;
esac
case "$wsha" in
  "$stamp"*) echo "EXT_WORKER_STAMP_OK tag=$TAG sha=$wsha" ;;
  *)
    echo "EXT_WORKER_DRIFT tag=$TAG so=$wsha stamp=${stamp:-none} (the cache moved under the stamp)" >&2
    exit 11
    ;;
esac
exit 0
