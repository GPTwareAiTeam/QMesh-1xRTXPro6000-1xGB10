#!/usr/bin/env bash
# pair_cut.sh -- materialise a release from a git commit on P7 and on the pair's GX box.
#
#   pair_cut.sh <pair> [sha] [--no-gx] [--force]
#
# A release is a directory, not a checkout: <root>/releases/<sha>/src is `git archive <sha>` and
# <root>/releases/<sha>/RELEASE.json says which commit, which branch, which seam fingerprint and
# which build id it is. Nothing is ever copied into it by hand; the boxes have no git, so this is
# the only way a box can name the commit it runs.
#
# The release root is the same path on P7 and on the GX box, so the containers can mount it at /pair
# and the two halves of a pair read the same layout. Cutting does not change what runs: `current`
# only moves in pair_update_run.sh, under the pair's queue lock.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_cut.sh <pair> [sha] [--no-gx] [--force]}
shift
SHA_ARG=""; DO_GX=1; FORCE=0
for a in "$@"; do
  case "$a" in
    --no-gx) DO_GX=0 ;;
    --force) FORCE=1 ;;
    -*) pair_die "unknown flag $a" ;;
    *) SHA_ARG=$a ;;
  esac
done
pair_load "$PAIR_ARG"

# ---- resolve the commit ---------------------------------------------------------------------------
# every tree here is a git worktree, so .git is a file, not a directory: ask git, don't stat
git -C "$PAIR_REPO" rev-parse --git-dir >/dev/null 2>&1 || pair_die "no repo at $PAIR_REPO"
if [ -z "$SHA_ARG" ]; then
  SHA=$(git -C "$PAIR_REPO" rev-parse --verify "$PAIR_BRANCH^{commit}" 2>/dev/null) \
    || pair_die "branch $PAIR_BRANCH not in $PAIR_REPO"
else
  SHA=$(git -C "$PAIR_REPO" rev-parse --verify "$SHA_ARG^{commit}" 2>/dev/null) \
    || pair_die "no such commit: $SHA_ARG"
fi
git -C "$PAIR_REPO" merge-base --is-ancestor "$SHA" "$PAIR_BRANCH" 2>/dev/null \
  || pair_die "$SHA is not on $PAIR_BRANCH; releases may only come from that branch"

BRANCH_TIP=$(git -C "$PAIR_REPO" rev-parse "$PAIR_BRANCH")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BUILD_ID="$PAIR-$(printf %s "$SHA" | cut -c1-12)-$STAMP"
TMPNAME=".$SHA.cut-$STAMP"
REL=$(pair_release_dir "$SHA")

# ---- seam identity + tree digest, read straight from the commit -------------------------------------
seam_of_commit() {  # <path>
  git -C "$PAIR_REPO" cat-file blob "$SHA:$1" 2>/dev/null
}
SEAM_FP=$(for f in "${PAIR_SEAM_FILES[@]}"; do printf '%s\0' "$f"; seam_of_commit "$f"; done | sha256sum | cut -c1-16)
[ -n "$SEAM_FP" ] && [ "$SEAM_FP" != "$(printf 'x' | sha256sum | cut -c1-16)" ] || pair_die "cannot read seam sources at $SHA"
SEAM_VER=$(seam_of_commit qmesh/native/qes.py | sed -n 's/^PROTO_VERSION *= *\([0-9][0-9]*\).*/\1/p' | head -1)
[ -z "$SEAM_VER" ] && SEAM_VER=$(seam_of_commit qmesh/native/qmesh_expert_seam.h \
  | sed -n 's/^#define QES_PROTO_VERSION *\([0-9][0-9]*\).*/\1/p' | head -1)
SEAM_VER=${SEAM_VER:--}
# A release that tracks the hook's build script is one whose hook must be built and hashed; the
# declaration travels in RELEASE.json so pair_check can tell it from a release cut before the hook
# had a build (those roots carry no .so and cannot be made to grow one without touching the tree).
if git -C "$PAIR_REPO" cat-file -e "$SHA:qmesh/native/build_moe.sh" 2>/dev/null; then
  CPP_REQ=true
else
  CPP_REQ=false
fi
DIGEST=$(git -C "$PAIR_REPO" ls-tree -r "$SHA" -- "${PAIR_VERIFY_PATHS[@]}" | python3 -c '
import hashlib, sys
rows = []
for line in sys.stdin:
    meta, _, path = line.rstrip("\n").partition("\t")
    mode, _typ, blob = meta.split()
    rows.append("%s %s %s" % (mode, blob, path))
print(hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest())
')
FILES=$(git -C "$PAIR_REPO" ls-tree -r "$SHA" -- "${PAIR_VERIFY_PATHS[@]}" | wc -l)
[ "$FILES" -gt 0 ] || pair_die "commit $SHA tracks none of ${PAIR_VERIFY_PATHS[*]}"

# A release is one object with two copies. If either side already carries this sha, its RELEASE.json
# is the truth for both: rendering a second one would stamp a different build_id/cut_at into the same
# release, and the smoke compares the build id the client reads with the one the server reads -- a
# mismatched pair with identical code (2026-11-01). Only --force re-stamps. A dry run is a rehearsal
# over a seeded stand-in release and never ships, so it keeps its stand-in.
adopt=""
if [ "$FORCE" != 1 ] && [ "$PAIR_DRY" != 1 ]; then
  rel_p7="$PAIR_ROOT/releases/$SHA/RELEASE.json"
  if [ -f "$rel_p7" ]; then
    adopt=$rel_p7
  elif [ -n "$(pair_gxsh "[ -f '$PAIR_ROOT/releases/$SHA/RELEASE.json' ] && echo yes" 2>/dev/null)" ]; then
    adopt=$(pair_gxsh "cat '$PAIR_ROOT/releases/$SHA/RELEASE.json'" 2>/dev/null)
  fi
fi
if [ -n "$adopt" ]; then
  RELEASE_JSON=$(cat "$adopt" 2>/dev/null || printf '%s\n' "$adopt")
  build_of=$(
    printf '%s\n' "$RELEASE_JSON" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    sys.exit("unreadable RELEASE.json: %s" % e)
if d.get("sha") != sys.argv[1]:
    sys.exit("RELEASE.json names %s, not %s" % (d.get("sha"), sys.argv[1]))
if d.get("digest") != sys.argv[2] or d.get("files") != int(sys.argv[3]):
    sys.exit("RELEASE.json was cut from a different tree (digest %s files %s, this commit is %s/%s)"
             % (d.get("digest"), d.get("files"), sys.argv[2], sys.argv[3]))
print(d.get("build_id") or "")
' "$SHA" "$DIGEST" "$FILES"
  ) || pair_die "release $SHA already exists but $adopt does not describe this commit: $build_of"
  BUILD_ID=$build_of
  [ -n "$BUILD_ID" ] || pair_die "release $SHA exists but its RELEASE.json has no build_id"
  pair_step "CUT_ADOPT $PAIR sha=${SHA:0:8} build_id=$BUILD_ID (the pair already has this release)"
else
RELEASE_JSON=$(
  python3 - "$SHA" "$BRANCH_TIP" "$BUILD_ID" "$DIGEST" "$SEAM_FP" "$SEAM_VER" "$FILES" "$PAIR" "$STAMP" "$CPP_REQ" <<'PY'
import json, sys
sha, tip, bid, digest, fp, ver, files, pair, stamp, cpp = sys.argv[1:]
print(json.dumps({
    "format": 1,
    "sha": sha,
    "branch": "tools/pair-release",
    "branch_tip_at_cut": tip,
    "build_id": bid,
    "digest": digest,
    "files": int(files),
    "seam": {"fingerprint": fp, "version": None if ver == "-" else int(ver)},
    "native": {"moe_cpp": cpp == "true"},
    "pair": pair,
    "cut_at": stamp,
    "cut_from": "git archive",
}, indent=2, sort_keys=True))
PY
)
[ -n "$RELEASE_JSON" ] || pair_die "failed to render RELEASE.json"
fi

pair_step "CUT $PAIR sha=${SHA:0:8} build_id=$BUILD_ID seam=fp:$SEAM_FP/proto:$SEAM_VER files=$FILES cpp=$CPP_REQ"

cut_side() {  # <p7|gx> <root>
  local side=$1 root=$2 rel="$2/releases/$SHA" tmp="$2/releases/$TMPNAME" log
  # the existence probe has to run on the side that owns the tree: both boxes use the same path,
  # so a local test would see the other side's copy
  local have=0
  if [ "$side" = p7 ]; then
    [ -d "$rel" ] && have=1
  else
    [ -n "$(pair_gxsh "[ -d '$rel' ] && echo yes")" ] && have=1
  fi
  if [ "$have" = 1 ] && [ ! "$FORCE" = 1 ]; then
    # EXISTING does not mean stamped. A 0-byte RELEASE.json is what a cut that died between
    # creating the file and writing its content leaves behind -- and this branch used to keep it
    # forever, so on 2026-11-02 gx2 carried an empty stamp for 9fc3ca8a and dev verify failed with
    # `DRIFT release_json=missing` *after* a green smoke. Rewrite the stamp only (same sha, same
    # digest, same build id on both sides, the pair-level truth); the tree is not touched.
    local got
    if [ "$side" = p7 ]; then got=$(wc -c < "$rel/RELEASE.json" 2>/dev/null || echo 0)
    else got=$(pair_gxsh "wc -c < '$rel/RELEASE.json' 2>/dev/null || echo 0"); fi
    if [ "${got:-0}" -le 0 ]; then
      echo "STAMP_REPAIR $PAIR $side $rel (RELEASE.json was ${got:-0} bytes, tree kept)"
      if [ "$side" = p7 ]; then
        printf '%s\n' "$RELEASE_JSON" > "$rel/RELEASE.json" || return 1
      else
        pair_gxsh "cat > '$rel/RELEASE.json'" <<<"$RELEASE_JSON" || return 1
        got=$(pair_gxsh "wc -c < '$rel/RELEASE.json' 2>/dev/null || echo 0")
        [ "${got:-0}" -gt 0 ] || { echo "STAMP_REPAIR_FAILED $PAIR gx $rel"; return 1; }
      fi
      echo "CUT $PAIR $side $rel (stamp repaired)"
      return 0
    fi
    echo "EXISTS $PAIR $side $rel"
    return 0
  fi
  if [ "$side" = p7 ]; then
    rm -rf "$tmp"; mkdir -p "$tmp/src" || return 1
    git -C "$PAIR_REPO" archive --format=tar "$SHA" | tar -x -C "$tmp/src" || return 1
    # The client half has no .git and no SEAM_BUILD_ID in its container, so without this file it
    # reports build=unknown to the server's session handshake and a pair can be mixed without
    # anybody noticing (2026-11-01: `SEAM_SESSION_START ... build=unknown proto=3`, accepted).
    # qes.build_id() reads <tree>/qmesh/BUILD_ID as its last fallback; write it, never hand-copy.
    printf '%s\n' "$BUILD_ID" > "$tmp/src/qmesh/BUILD_ID" || return 1
    printf '%s\n' "$RELEASE_JSON" > "$tmp/RELEASE.json" || return 1
    # a cut never ships a stamp it cannot read back (2026-11-02: gx2 was left with 0 bytes)
    [ -s "$tmp/RELEASE.json" ] || { echo "STAMP_WRITE_FAILED $PAIR p7"; return 1; }
    rm -rf "$rel"
    mv -T "$tmp" "$rel" || return 1
  else
    # one round trip for the extraction, one for the stamp; `tar -x` reads the archive from stdin
    local out
    out=$(pair_gxsh "set -e; mkdir -p '$root/releases'; rm -rf '$tmp'; mkdir -p '$tmp/src'; \
               tar -x -C '$tmp/src'; echo EXTRACTED" \
          < <(git -C "$PAIR_REPO" archive --format=tar "$SHA")) || return 1
    case "$out" in *EXTRACTED*) ;; *) echo "EXTRACT_FAILED $PAIR gx: $out"; return 1 ;; esac
    # The stamp travels inside the command, never on stdin. `out=$(pair_gxsh "…" ) < <(printf …)`
    # binds the redirect to this shell, so the remote `cat` read the *host's* stdin and wrote a
    # 0-byte RELEASE.json -- 2026-11-02: gx2 carried one for 9fc3ca8a, dev smoked green and then
    # failed verify with `DRIFT release_json=missing`. base64 keeps the JSON out of the quoting;
    # a ~470-byte stamp is a safe argument.
    local out b64
    b64=$(printf '%s\n' "$RELEASE_JSON" | base64 -w0) || return 1
    out=$(pair_gxsh "set -e; printf '%s\n' '$BUILD_ID' > '$tmp/src/qmesh/BUILD_ID'; \
               printf %s '$b64' | base64 -d > '$tmp/RELEASE.json'; \
               [ -s '$tmp/RELEASE.json' ] || { echo STAMP_EMPTY; exit 1; }; \
               rm -rf '$rel'; mv -T '$tmp' '$rel'; echo CUT_OK") || return 1
    case "$out" in *CUT_OK*) ;; *) echo "CUT_FAILED $PAIR gx: $out"; return 1 ;; esac
  fi
  local n
  if [ "$side" = p7 ]; then
    n=$(find "$rel/src" -type f | wc -l)
  else
    n=$(pair_gxsh "find '$rel/src' -type f | wc -l")
  fi
  [ "${n:-0}" -ge "$FILES" ] || { echo "CUT_SHORT $PAIR $side files=$n want>=$FILES"; return 1; }
  # One release, one stamp: the two copies must be byte-identical, or the client and the server
  # name different builds for the same code and the pair is mixed without anybody noticing.
  if [ "$side" = gx ] && [ -f "$PAIR_ROOT/releases/$SHA/RELEASE.json" ]; then
    local sum_p7 sum_gx
    sum_p7=$(sha256sum "$PAIR_ROOT/releases/$SHA/RELEASE.json" | cut -d' ' -f1)
    sum_gx=$(pair_gxsh "sha256sum '$rel/RELEASE.json' 2>/dev/null | cut -d' ' -f1")
    [ "${sum_p7:-x}" = "${sum_gx:-y}" ] || {
      echo "STAMP_MISMATCH $PAIR p7=${sum_p7:-none} gx=${sum_gx:-none}"; return 1; }
  fi
  echo "CUT $PAIR $side $rel files=$n"
}

rc=0
cut_side p7 "$PAIR_ROOT" || rc=1
if [ "$DO_GX" = 1 ] && [ "$rc" = 0 ]; then
  cut_side gx "$PAIR_ROOT" || rc=1
fi
if [ "$rc" != 0 ]; then
  pair_step "CUT_FAILED $PAIR sha=${SHA:0:8}"
  exit 1
fi
pair_step "CUT_OK $PAIR sha=$SHA build_id=$BUILD_ID"
echo "$SHA"
