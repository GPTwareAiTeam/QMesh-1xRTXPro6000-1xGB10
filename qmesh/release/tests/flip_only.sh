#!/bin/bash
# flip_only.sh -- the split between preparing a release and flipping onto it, swept with stubs only.
#
#   ./tests/flip_only.sh [pair]
#
# What a flip-only ticket must never do: build anything, create or recreate a container, or flip onto
# a release whose stamps are missing/foreign. Each case below fails (or passes) without touching a
# container, the queue or a GPU.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
export PAIR_CONF="$HERE/pairs.conf" PAIR_COLD_PACK=/home/test/cold-pack PAIR_ART_MOUNT=/home/test/artifacts
PAIR=${1:-dev}
SHA=${2:-$(git -C "$(dirname "$(dirname "$(dirname "$HERE")")")" rev-parse HEAD)}
WORK=$(mktemp -d /tmp/pairflip.XXXXXX); export PATH="$WORK/bin:$PATH"
mkdir -p "$WORK/bin" "$WORK/state"
cp "$HERE/stub_docker.py" "$WORK/bin/docker"; chmod +x "$WORK/bin/docker"
cp "$HERE/stub_ssh.sh" "$WORK/bin/stub_ssh"; chmod +x "$WORK/bin/stub_ssh"
sed -i "s#/tmp/stub/state.json#$WORK/state/state.json#; s#/tmp/stub/exec.log#$WORK/state/exec.log#" "$WORK/bin/docker"
REL_DIR="$WORK/pair/releases/$SHA"
mkdir -p "$REL_DIR/src" "$REL_DIR/ext" "$REL_DIR/src/qmesh/native/bin"
BID="flip-demo"
printf '{"build_id":"%s","seam":{"fingerprint":"fp:dry","proto":"3"}}' "$BID" > "$REL_DIR/RELEASE.json"
for arch in x86_64 aarch64; do
  so="$REL_DIR/src/qmesh/native/bin/libqmesh_expert_seam.$arch.so"
  printf 'so-%s\n' "$arch" > "$so"
  sha=$(sha256sum "$so" | cut -d' ' -f1)
  printf '{"build_id":"%s","source_sha":"%s","arch":"%s","sha256":"%s"}' "$BID" "$SHA" "$arch" "$sha" \
    > "$so.stamp"
done
printf 'ext\n' > "$REL_DIR/ext/exllamav3_ext.so"
# The GX stamp also has to name the cuda-exl3 kernel (its .so is gitignored and built per checkout),
# so a prepared GX release is only "prepared" with both halves stamped.
stamp_json() {  # <build_id> <so-sha> <cx-sha|""> -- one BUILD_STAMP.json as pair_build_ext.sh writes it
  local bid=$1 so_sha=$2 cx_sha=$3
  if [ -n "$cx_sha" ]; then
    printf '{"build_id":"%s","source_sha":"%s","sha256":"%s","cx":{"build_id":"%s","source_sha":"%s","arch_list":["12.1"],"sha256":"%s","so":"_C.so","built_at":"2026-11-01T00:00:00Z"}}' \
      "$bid" "$SHA" "$so_sha" "$bid" "$SHA" "$cx_sha"
  else
    printf '{"build_id":"%s","source_sha":"%s","sha256":"%s"}' "$bid" "$SHA" "$so_sha"
  fi
}
stamp_json "$BID" "$(sha256sum "$REL_DIR/ext/exllamav3_ext.so" | cut -d' ' -f1)" "cx-$BID" \
  > "$REL_DIR/ext/BUILD_STAMP.json"
cd "$REL"
export PAIR_DRY=1 PAIR_DRY_SSH="$WORK/bin/stub_ssh" PAIR_ROOT_OVERRIDE="$WORK/pair" \
       PAIR_QUEUE="$WORK/noq" PAIR_DRY_GPU=free PAIR_STATE_OVERRIDE="$WORK/pairs"
run() { ./pair_update_run.sh "$PAIR" "$@" 2>&1; }
fails=0
want() {  # <case> <expected-substring> <log>
  if printf '%s' "$3" | grep -q -- "$2"; then echo "OK   $1"
  else echo "FAIL $1 (wanted '$2')"; printf '%s\n' "$3" | tail -6 | sed 's/^/     /'; fails=$((fails + 1)); fi
}
want_not() {  # <case> <forbidden-substring> <log>
  if printf '%s' "$3" | grep -q -- "$2"; then echo "FAIL $1 (found '$2')"; fails=$((fails + 1))
  else echo "OK   $1"; fi
}

built=$(grep -cE '^(  )?(LIB_BUILD|EXT_BUILD)' "$WORK/state/exec.log" 2>/dev/null || true)

# 1. a flip-only ticket with no sha has nothing prepared to flip to
out=$(PAIR_FLIP_ONLY=1 run); rc=$?
want "no sha refused" "PAIR_UPDATE_FAILED pair=$PAIR phase=prepared why=PAIR_FLIP_ONLY needs an explicit sha" "$out"
[ "$rc" != 0 ] && echo "OK   no sha exit=$rc" || { echo "FAIL no sha exited 0"; fails=$((fails + 1)); }

# 2. containers that are not wired yet: prepare's job, never the ticket's
out=$(PAIR_FLIP_ONLY=1 run "$SHA"); rc=$?
want "unwired containers refused" "containers do not exist: run scripts/pair_prepare.sh" "$out"
want_not "no container created by the ticket" "CREATE $PAIR" "$out"

# 3. wired containers, but the ext stamp of another build
python3 pair_containers.py apply "$PAIR" "$SHA" --ts T >/dev/null 2>&1
printf '{"build_id":"other","source_sha":"%s","sha256":"x"}' "$SHA" > "$REL_DIR/ext/BUILD_STAMP.json"
out=$(PAIR_FLIP_ONLY=1 run "$SHA"); rc=$?
want "foreign ext stamp refused" "EXT_STAMP_MISMATCH" "$out"
want "foreign ext stamp fails before the flip" "PAIR_UPDATE_FAILED pair=$PAIR phase=prepared" "$out"

# 3b. the torch extension is stamped for this build but the cuda-exl3 kernel is not: the release would
# serve every layer on the mixed-K module, so the ticket must refuse it before the flip.
stamp_json "$BID" "$(sha256sum "$REL_DIR/ext/exllamav3_ext.so" | cut -d' ' -f1)" "" \
  > "$REL_DIR/ext/BUILD_STAMP.json"
out=$(PAIR_FLIP_ONLY=1 run "$SHA"); rc=$?
want "kernel-less gx stamp refused" "CX_STAMP_MISSING gx" "$out"
want "kernel-less gx stamp fails before the flip" "PAIR_UPDATE_FAILED pair=$PAIR phase=prepared" "$out"
[ "$rc" != 0 ] && echo "OK   kernel-less exit=$rc" || { echo "FAIL kernel-less gx stamp exited 0"; fails=$((fails + 1)); }

# 4. prepared: stamps of this build, nothing to build, flip and smoke happen
stamp_json "$BID" "$(sha256sum "$REL_DIR/ext/exllamav3_ext.so" | cut -d' ' -f1)" "cx-$BID" \
  > "$REL_DIR/ext/BUILD_STAMP.json"
out=$(PAIR_FLIP_ONLY=1 run "$SHA"); rc=$?
want "prepared release flips" "PAIR_UPDATED_DRY pair=$PAIR sha=$SHA build=$BID" "$out"
want "stamp verify reported" "LIB_OK p7 x86_64" "$out"
want_not "flip-only builds nothing" "LIB_BUILD" "$out"
want_not "flip-only builds no ext" "EXT_BUILD" "$out"
[ "$rc" = 0 ] && echo "OK   prepared exit=0" || { echo "FAIL prepared exit=$rc"; fails=$((fails + 1)); }

# 5. A container spec delta -- here an extra network mount -- is an
#    ADVISORY, not a stop and not a recreate. The owner's rule is that an existing container is never
#    recreated or renamed: a release changes what runs, not the box. So a delta can only be reported,
#    and the earlier shape of this test -- delta stops the flip, --recreate applies it, the old
#    container is renamed and the new one loses the mount -- was withdrawn on 2026-11-02. What is
#    asserted now: the flip proceeds, says the mount is not in the spec, and changes no container,
#    with or without PAIR_ALLOW_RECREATE.
python3 - "$WORK/state/state.json" "$PAIR" <<'PY'
import json, sys
p, pair = sys.argv[1], sys.argv[2]
st = json.load(open(p))
for name in st:
    if name.startswith("seam-p7-") and name.endswith("-" + pair):
        st[name]["Mounts"].append({"Source": "/mnt/network/models",
                                   "Destination": "/mnt/network/models"})
json.dump(st, open(p, "w"))
PY
out=$(PAIR_FLIP_ONLY=1 run "$SHA"); rc=$?
# The plan is what an operator reads the delta in: the flip itself sends it to the pair log only.
plan=$(python3 ./pair_containers.py plan "$PAIR" 2>&1)
want "a spec delta is reported as an advisory, not a reason to stop" \
     "advisory=mount:-/mnt/network/models=/mnt/network/models (not in spec)" "$plan"
want "and the flip carries on over the prepared release" \
     "UPDATE_CONTAINERS pair=$PAIR unchanged (prepared)" "$out"
want_not "and it recreates nothing" "UPDATE_CONTAINERS pair=$PAIR recreated" "$out"
want "and the release still lands" "PAIR_UPDATED_DRY pair=$PAIR sha=$SHA" "$out"
[ "$rc" = 0 ] && echo "OK   spec delta exit=$rc" || { echo "FAIL spec delta exited $rc"; fails=$((fails + 1)); }
out=$(PAIR_FLIP_ONLY=1 PAIR_ALLOW_RECREATE=1 run "$SHA"); rc=$?
want_not "PAIR_ALLOW_RECREATE does not recreate an existing container either" \
         "UPDATE_CONTAINERS pair=$PAIR recreated" "$out"
want "and the release still lands with it set" "PAIR_UPDATED_DRY pair=$PAIR sha=$SHA" "$out"
[ "$rc" = 0 ] && echo "OK   recreate-flag exit=$rc" || { echo "FAIL recreate-flag exit=$rc"; fails=$((fails + 1)); }
python3 - "$WORK/state/state.json" "$PAIR" <<'PY'
import json, sys
st = json.load(open(sys.argv[1]))
pair = sys.argv[2]
p7 = [n for n in st if n.startswith("seam-p7-") and n.endswith("-" + pair) and ".bak-" not in n]
old = [n for n in st if n.startswith("seam-p7-") and ".bak-" in n]
mnt = [m for n in p7 for m in st[n]["Mounts"] if m["Destination"].startswith("/mnt")]
ok = len(p7) == 1 and not old and bool(mnt)
print(("OK   " if ok else "FAIL ") + "the P7 container is the same one, renamed to nothing, mount untouched"
      + ("" if ok else f" (p7={p7} bak={old} mnt={mnt})"))
sys.exit(0 if ok else 1)
PY
[ $? = 0 ] || fails=$((fails + 1))
echo "RESULT $([ "$fails" = 0 ] && echo PASS || echo FAIL) fails=$fails"
rm -rf "$WORK"
[ "$fails" = 0 ]
