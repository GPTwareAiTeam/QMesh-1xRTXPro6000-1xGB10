#!/bin/bash
# smoke_memory.sh -- the smoke tells a full box apart from a bad release.
#
#   ./tests/smoke_memory.sh
#
# The worker refuses a pack that does not fit in MemAvailable and exits 3, but the ensure wrapper
# reports "died" (rc 2) for that too. Reading only the exit code turned "GX2 has 800 MiB free"
# into "the release is broken" and rolled a good release back. pair_smoke.sh now reads the verdict
# from SEAM_NO_MEMORY / SEAM_NO_PACK in the ensure output, whatever the wrapper's rc was.
#
# Swept with stubs: no container, no GPU, no server. Only pair_smoke.sh runs; the ensure it would
# start is replaced by a script that prints the lines the real one printed tonight.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
export PAIR_CONF="$HERE/pairs.conf" PAIR_COLD_PACK=/home/test/cold-pack PAIR_ART_MOUNT=/home/test/artifacts
SHA=$(git -C "$(dirname "$(dirname "$(dirname "$HERE")")")" rev-parse HEAD)
WORK=$(mktemp -d /tmp/pairsm.XXXXXX)
mkdir -p "$WORK/bin" "$WORK/state" "$WORK/pair/releases/$SHA/src" "$WORK/pairs/dev"
cp "$HERE/stub_ssh.sh" "$WORK/bin/stub_ssh"; chmod +x "$WORK/bin/stub_ssh"
printf '{"build_id":"sm-demo","seam":{"fingerprint":"fp:dry","proto":"3"}}' > "$WORK/pair/releases/$SHA/RELEASE.json"
cd "$REL"
export PAIR_ROOT_OVERRIDE="$WORK/pair" PAIR_STATE_OVERRIDE="$WORK/pairs" PAIR_SSH="$WORK/bin/stub_ssh" \
       PAIR_QUEUE="$WORK/noq" PAIR_GATE_DIR="$WORK/noq" PAIR_PACK_DISK=0
# The GX gate lives next to the live queue, not in PAIR_QUEUE: without PAIR_GATE_DIR this stub smoke
# waited out the real dev smoke's claim (SMOKE_GATE_WAIT_MIN, 20 min) before it could fail -- the
# battery looked hung on 2026-11-02 while dev was actually running.
mkdir -p "$WORK/noq"
# pair_smoke.sh takes the box claim before it starts anything (pair_gx_gate_take), and a claim may only
# be taken by a queue ticket: pair_ticket_hardcap looks for GPU_TICKET_HARDCAP or a ticket of an
# ancestor pid. Swept by hand there is neither, so the smoke stopped at `PAIR_GATE_REFUSED
# outside_ticket label=smoke-dev` + `SMOKE_FAIL gx_busy free` and the verdict logic these four cases
# are about was never reached (4 red on 2026-11-02). The cap here is what a real ticket exports; the
# gate itself is still this work dir's, so nothing on a box is touched.
export GPU_TICKET_HARDCAP=$(( $(date +%s) + 1800 ))
# Every question the smoke asks of a container goes through stub ssh -> `docker` *here*, and there is
# no seam-p7-dev or seam-gx2-dev on this workstation: before this stub existed the smoke stopped at
# `SMOKE_FAIL nvml gx=` or `SMOKE_FAIL ext_client_miss p7` and never reached its own logic. The stub
# answers the two questions that must be "yes" (the container sees the driver, and a client import
# reuses the stamped artifact) and fails everything else, so a missing container still reads as one.
export PATH="$WORK/bin:$PATH"
cat > "$WORK/bin/docker" <<'SH'
#!/bin/bash
case "$*" in
  "exec "*" nvidia-smi -L")
    echo "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-stub)" ;;
  *import\ exllamav3.ext*)
    echo "SMOKE_EXT_HIT /pair/extcache/exllamav3_ext/stub.so" ;;
  *)
    exit 1 ;;
esac
SH
chmod +x "$WORK/bin/docker"
unset PAIR_DRY PAIR_DRY_SSH
fails=0
want() {  # <case> <expected-substring> <text>
  if printf '%s' "$3" | grep -q -- "$2"; then echo "OK   $1"
  else echo "FAIL $1 (wanted '$2')"; printf '%s\n' "$3" | tail -6 | sed 's/^/     /'; fails=$((fails + 1)); fi
}
want_not() {
  if printf '%s' "$3" | grep -q -- "$2"; then echo "FAIL $1 (found '$2')"; fails=$((fails + 1)); else echo "OK   $1"; fi
}
ensure_stub() {  # <rc> <lines...>
  local rc=$1; shift
  { echo '#!/bin/bash'
    for l in "$@"; do printf 'echo %q\n' "$l"; done
    printf 'exit %s\n' "$rc"
  } > "$WORK/bin/ensure"; chmod +x "$WORK/bin/ensure"
}

# 1. The worker's own refusal, seen through a wrapper that calls it "died".
ensure_stub 2 \
  "SEAM_ENSURE state=died after=120s" \
  "SEAM_NO_MEMORY need_mib=116362+1024 avail_mib=116574 swapped_mib=834 - refusing to load"
out=$(SEAM_ENSURE="$WORK/bin/ensure" ./pair_smoke.sh dev "$SHA" 2>&1); rc=$?
want "a worker memory refusal is a memory verdict" "SMOKE_FAIL memory server_no_memory SEAM_NO_MEMORY need_mib=116362" "$out"
want_not "and is not reported as a dead server" "server_timeout_or_died" "$out"
[ "$rc" = 1 ] && echo "OK   smoke exits 1" || { echo "FAIL smoke exit=$rc"; fails=$((fails + 1)); }

# 2. A missing pack is a release fault, not a full box: it must not take the memory path.
ensure_stub 3 "SEAM_ENSURE state=died after=3s" "SEAM_NO_PACK dir=/pair/current/model - expected cold-L00.safetensors"
out=$(SEAM_ENSURE="$WORK/bin/ensure" ./pair_smoke.sh dev "$SHA" 2>&1); rc=$?
want "a missing pack is its own reason" "SMOKE_FAIL no_pack SEAM_NO_PACK" "$out"
want_not "and is not called memory" "SMOKE_FAIL memory" "$out"

# 3. A wrapper that dies without a verdict stays a timeout, and does not pretend to be memory.
ensure_stub 2 "SEAM_ENSURE state=died after=120s"
out=$(SEAM_ENSURE="$WORK/bin/ensure" ./pair_smoke.sh dev "$SHA" 2>&1); rc=$?
want "a silent death stays a timeout" "SMOKE_FAIL server_timeout_or_died" "$out"
want_not "with no memory verdict invented" "SMOKE_FAIL memory" "$out"

# 4. The smoke grants the pair's own list (pairs.conf), never a narrower one: the release was measured
# on 1,6,64,8192, and a grant of 1,6,64 is a 64-row prefill cap rather than the same instrument.
ensure_stub 0 "SEAM_ENSURE state=started port=1"
out=$(SEAM_ENSURE="$WORK/bin/ensure" ./pair_smoke.sh dev "$SHA" 2>&1)
want "the server is asked for the pair's buckets" "buckets=1,6,64,8192" "$out"

echo "RESULT $([ "$fails" = 0 ] && echo PASS || echo FAIL) fails=$fails"
rm -rf "$WORK"
[ "$fails" = 0 ]
