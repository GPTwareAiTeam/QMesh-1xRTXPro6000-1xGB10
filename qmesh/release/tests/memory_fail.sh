#!/bin/bash
# memory_fail.sh -- a smoke that the GX box refused for lack of memory must NOT roll the release back.
#
#   ./tests/memory_fail.sh
#
# GX2 ran with ~800 MiB available tonight: the pack load can be refused by the box while the release
# itself is perfectly good. The ticket then keeps the code installed, records evidence saying why,
# writes no manifest (so the pair fails closed) and prints `smoke=FAILED_MEMORY rollback=none`.
# Run with PAIR_DRY=1 and a stubbed docker/ssh; only pair_smoke.sh is replaced, everything else is the
# real ticket.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
SHA=$(git -C "$(dirname "$(dirname "$(dirname "$HERE")")")" rev-parse HEAD)
WORK=$(mktemp -d /tmp/pairmem.XXXXXX); export PATH="$WORK/bin:$PATH"
export PAIR_CONF="$HERE/pairs.conf" PAIR_COLD_PACK=/home/test/cold-pack PAIR_ART_MOUNT=/home/test/artifacts
mkdir -p "$WORK/bin" "$WORK/state" "$WORK/run"
cp "$HERE/stub_docker.py" "$WORK/bin/docker"; chmod +x "$WORK/bin/docker"
cp "$HERE/stub_ssh.sh" "$WORK/bin/stub_ssh"; chmod +x "$WORK/bin/stub_ssh"
sed -i "s#/tmp/stub/state.json#$WORK/state/state.json#; s#/tmp/stub/exec.log#$WORK/state/exec.log#" "$WORK/bin/docker"
# a copy of the tooling so pair_smoke.sh can be replaced without touching the tree under test
cp -a "$REL"/. "$WORK/run/"
REL_DIR="$WORK/pair/releases/$SHA"
mkdir -p "$REL_DIR/src/qmesh/native/bin" "$REL_DIR/ext"
BID="mem-demo"
printf '{"build_id":"%s","seam":{"fingerprint":"fp:dry","proto":"3"}}' "$BID" > "$REL_DIR/RELEASE.json"
for arch in x86_64 aarch64; do
  so="$REL_DIR/src/qmesh/native/bin/libqmesh_expert_seam.$arch.so"
  printf 'so-%s\n' "$arch" > "$so"
  printf '{"build_id":"%s","source_sha":"%s","arch":"%s","sha256":"%s"}' "$BID" "$SHA" "$arch" \
    "$(sha256sum "$so" | cut -d' ' -f1)" > "$so.stamp"
done
printf 'ext\n' > "$REL_DIR/ext/exllamav3_ext.so"
# both halves of the GX stamp: the torch extension and the per-checkout cuda-exl3 kernel
printf '{"build_id":"%s","source_sha":"%s","sha256":"%s","cx":{"build_id":"%s","source_sha":"%s","arch_list":["12.1"],"sha256":"cx-mem","so":"_C.so","built_at":"2026-11-01T00:00:00Z"}}' \
  "$BID" "$SHA" "$(sha256sum "$REL_DIR/ext/exllamav3_ext.so" | cut -d' ' -f1)" "$BID" "$SHA" \
  > "$REL_DIR/ext/BUILD_STAMP.json"

fails=0
want() {  # <case> <expected-substring> <log>
  if printf '%s' "$3" | grep -q -- "$2"; then echo "OK   $1"
  else echo "FAIL $1 (wanted '$2')"; printf '%s\n' "$3" | tail -8 | sed 's/^/     /'; fails=$((fails + 1)); fi
}
want_not() {
  if printf '%s' "$3" | grep -q -- "$2"; then echo "FAIL $1 (found '$2')"; fails=$((fails + 1)); else echo "OK   $1"; fi
}

export PAIR_DRY=1 PAIR_DRY_SSH="$WORK/bin/stub_ssh" PAIR_ROOT_OVERRIDE="$WORK/pair" \
       PAIR_QUEUE="$WORK/noq" PAIR_DRY_GPU=free PAIR_STATE_OVERRIDE="$WORK/pairs" PAIR_PACK_DISK=0
cd "$WORK/run"
python3 pair_containers.py apply dev "$SHA" --ts T >/dev/null 2>&1 || { echo "FAIL could not wire containers"; exit 1; }

run_smoke_stub() {  # <SMOKE_FAIL line>
  printf '#!/bin/bash\ncat <<EOF\nSMOKE_MEM_BEFORE avail_mib=795 swap_free_mib=12 swap_total_mib=3352\nSMOKE_FAIL %s\nEOF\nexit 1\n' \
    "$1" > "$WORK/run/pair_smoke.sh"
  chmod +x "$WORK/run/pair_smoke.sh"
  PAIR_FLIP_ONLY=1 ./pair_update_run.sh dev "$SHA" 2>&1
}

out=$(run_smoke_stub "memory server_preflight PREFLIGHT_FAIL gx-mem avail=1GiB need=115GiB host=gx-dev.invalid")
rc=$?
want "box memory refusal is reported as such" \
  "PAIR_UPDATED pair=dev sha=$SHA build=$BID smoke=FAILED_MEMORY rollback=none manifest=none" "$out"
want "the memory lines are kept" "SMOKE_MEM_BEFORE avail_mib=795" "$out"
want "the reason names memory" "why=memory:SMOKE_FAIL memory server_preflight" "$out"
want "recourse names the box, not the release" "free gx-dev.invalid" "$out"
want_not "no rollback of the code" "RESTORE_BEGIN" "$out"
want_not "no manifest is claimed" "check=IN_SYNC" "$out"
[ "$rc" = 1 ] && echo "OK   memory failure exit=1" || { echo "FAIL memory failure exit=$rc"; fails=$((fails + 1)); }

# A real regression still rolls back: the same stub, one word different.
out=$(run_smoke_stub "server_preflight_refused PREFLIGHT_FAIL server_in=1 host=gx-dev.invalid")
rc=$?
want "a non-memory smoke failure still fails the update" "PAIR_UPDATE_FAILED pair=dev phase=smoke" "$out"
want "and rolls the release back" "RESTORE_BEGIN pair=dev phase=smoke" "$out"
want_not "and does not claim the release is installed" "smoke=FAILED_MEMORY" "$out"

echo "RESULT $([ "$fails" = 0 ] && echo PASS || echo FAIL) fails=$fails"
rm -rf "$WORK"
[ "$fails" = 0 ]
