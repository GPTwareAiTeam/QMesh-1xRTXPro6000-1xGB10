#!/usr/bin/env bash
# pair_promote.sh -- move the OTHER pair onto the release dev is already serving.
#
#   pair_promote.sh <sha> [--force] [--max-min N] [--build] [--recreate]
#
# The queued ticket only flips and smokes: the release must already be cut, wired and built on prod
# by pair_prepare.sh, so a promote never compiles inside the queue (2026-11-02: a full promote left
# the queue rebuilt both libs and the torch extension at flip time, which is exactly what prepare is
# for). `--build` is the escape hatch for a pair that was never prepared; expect it to queue the
# compilers. Pass it *with* `--recreate` when prod's containers need a new spec and the extension has
# never been built against that spec (the promotion of a spec-changing release): on its own,
# `--recreate` queues a flip-only ticket, which would recreate the container and then stop at the ext
# verify, because an extension built for the old TORCH_EXTENSIONS_DIR does not exist in the new one.
#
# --recreate: prod's spec changed too (the containers are re-created from containers/*.json inside
# the queued ticket, old one renamed to <name>.bak-<ts>). Pass it whenever `pair_containers.py plan
# prod` reports RECREATE, otherwise the flip-only ticket refuses before the flip. Combine with
# `--build` as above; `--build` wins over the implicit flip in either order.
#
# Rollout order is not a preference: dev is the pair whose failures are cheap, so a release reaches
# prod only after dev runs it and its check says IN_SYNC. This script refuses otherwise. It also
# refuses to promote a release whose seam fingerprint differs from what dev is serving -- a wire
# change has to be promoted as its own release, never mixed into a version bump.
#
# It then queues pair_update.sh prod <sha>, which lands behind whatever prod is running.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

SHA_ARG=${1:?usage: pair_promote.sh <sha> [--force] [--max-min N] [--build]}
shift
FORCE=0; MINS=""; FLIP_ONLY=1; RECREATE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1 ;;
    --max-min) MINS=$2; shift ;;
    --build) FLIP_ONLY=0 ;;
    --recreate) RECREATE=1 ;;
    *) pair_die "unknown flag $1" ;;
  esac
  shift
done

pair_load dev
SHA=$(git -C "$PAIR_REPO" rev-parse --verify "$SHA_ARG^{commit}" 2>/dev/null) \
  || pair_die "no such commit: $SHA_ARG"
dev_sha=$(pair_manifest_get release_sha)
dev_seam=$(pair_manifest_get seam fingerprint)
dev_build=$(pair_manifest_get build_id)
pair_step "PROMOTE_CHECK sha=${SHA:0:12} dev_manifest=${dev_sha:0:12} dev_build=${dev_build:-none}"

if [ "$dev_sha" != "$SHA" ]; then
  [ "$FORCE" = 1 ] || pair_die "dev is at ${dev_sha:0:12}, not ${SHA:0:12}; promote what dev ran (or --force)"
  echo "PROMOTE_WARN dev is at ${dev_sha:0:12}; promoting ${SHA:0:12} anyway (--force)"
fi
dev_check=$("$HERE/pair_check.sh" dev --quiet 2>&1)
crc=$?
printf '%s\n' "$dev_check"
[ "$crc" = 0 ] || { [ "$FORCE" = 1 ] || pair_die "dev does not check out (rc=$crc); not promoting"; }

# The manifest says what dev serves NOW; the evidence says the sha passed there. They are different
# questions -- dev's manifest moves on with every release, while the promotion may come later -- and
# prod refuses to move without the evidence anyway (pair_update.sh). Check it here first, so the
# refusal names the missing piece instead of a gate deep inside the prod ticket.
ev=$PAIR_STATE/dev/evidence/$SHA.json
[ -f "$ev" ] || pair_die "no dev canary evidence for ${SHA:0:12}: $ev (dev must reach PAIR_UPDATED)"
ev_gate=$(pair_canary_gate "$SHA" "$ev")
[ "$ev_gate" = ok ] || pair_die "dev evidence for ${SHA:0:12} does not qualify: $ev_gate ($ev)"
pair_step "PROMOTE_EVIDENCE sha=${SHA:0:12} ids=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print((d.get("smoke") or {}).get("ids",{}).get("n",0))' "$ev") file=$ev"

rel=$(pair_release_dir "$SHA")
new_seam=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["seam"]["fingerprint"])' \
           "$rel/RELEASE.json" 2>/dev/null)
[ -n "$new_seam" ] || pair_die "release ${SHA:0:12} has no RELEASE.json at $rel (cut it on dev first)"
if [ "$new_seam" != "$dev_seam" ]; then
  echo "PROMOTE_WARN seam ${new_seam} differs from dev ${dev_seam}"
  [ "$FORCE" = 1 ] || pair_die "seam fingerprint changed; that is a different protocol, promote it deliberately (--force)"
fi
pair_step "PROMOTE_OK sha=${SHA:0:12} seam=$new_seam dev=IN_SYNC evidence=$ev"

pair_load prod
cmd=("$HERE/pair_update.sh" prod "$SHA")
[ -n "$MINS" ] && cmd+=(--max-min "$MINS")
# The ticket flips and smokes; it does not compile. --build is the deliberate exception, and it is
# forwarded as its own flag: `--recreate` implies a flip inside pair_update.sh, so a promote that has
# to compile *and* recreate its containers would otherwise queue a flip-only ticket and stop at the
# ext verify with nothing built. `--build` wins over `--recreate` there, in either order.
if [ "$FLIP_ONLY" = 1 ]; then cmd+=(--flip-only); else cmd+=(--build); fi
[ "$RECREATE" = 1 ] && cmd+=(--recreate)
# pair_update.sh prod refuses without this: the evidence check has already happened here, and the
# flag is what says so. Nothing else sets it.
PAIR_PROMOTE=1 "${cmd[@]}"
