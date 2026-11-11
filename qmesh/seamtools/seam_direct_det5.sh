#!/usr/bin/env bash
# Determinism acceptance: can the cold-expert digests be pinned at all?
#
# The 10:33 ticket (evidence/ab/1102-103357) left one question it could not answer with the code it
# ran: two --fresh loads of the *same* table produced two digests, and the ranked table produced
# both of those outcomes too. The two observed outcomes were
#   load1=67158aeb688585b5371ae509eccd5c673df99005970f416555c3e5f373d03ddd
#   load2=3a05b6c3ab197dee332e905417c27e86ca659b95b4eeceaad34e02f10eccafdd
# in both orders, and they agree on the first six ids and differ at position seven (736 vs 24).
# Two candidate causes were on the table, and they are distinguishable:
#
#   * the client's own MoE epilogue. qmesh/exl3_moe_cuda.py:438 sums the down-gemm into acc with
#     fp32 atomics when the kernel can write acc out directly (HAS_ACC_OUT and not DETERMINISTIC),
#     and :50 reads CUDA_EXL3_DETERMINISTIC; with it set the module calls exl3_moe_combine instead
#     and the summation order is fixed.
#   * the GX-side scatter. qmesh/native/qmesh_moe_kernels.cu:10-13 says the scatter's within-expert
#     order *is* atomicAdd order, "the only difference is the order of the fp32 atomics in the
#     down-projection epilogue (<= 1 ulp per row)", and that kernel has no determinism switch.
#
# So the test is: pin the switch on both halves and see whether the ids stop moving. If they do,
# the client's atomics were the whole story and "identical digests" becomes a usable criterion. If
# they still move, the residual is the GX scatter's 1 ulp -- unavoidable without a deterministic
# scatter on the GX -- and the honest criterion stays "the digests are drawn from a small set, and
# the top-2 margin says the flip is a tie-break". Either outcome is a result; only an untested
# switch would have been a waste of the window.
#
#   A  plain table, TWO --fresh loads x TWO client runs, CUDA_EXL3_DETERMINISTIC=1. Its four digests
#      must collapse to one. Four, not two, because the within-load pair costs ~45 s and separates
#      "the load changed something" from "the decode is not stable at all".
#   B  ranked table, same shape, plus the Hello-sha refusal test: a --stub server advertising the
#      *other* table, and the shipping client pointed at it. It must be refused at the Hello with
#      the server left alive (qmesh/tests/refusal_stub.sh), which is the half of the two-table rule
#      that keeps a mismatched client from taking the process down. Its single digest must equal A's:
#      the ranked file is a pure reordering of the same experts, so with the summation order pinned
#      the outputs are bit-identical. That is the strongest form of "the numbering changed nothing".
#
# No rate is measured here, and no --ws-rows above the 1024 default is asked for: the owner's rule
# is that ws_rows > 1024 does not run on any box until the 32k-prefill headroom is understood (ws
# 2048 measured 1123.9 tok/s at 129 MiB slack with swap_warn=1, which is not shippable).
#
#   queue/run_dev.sh SeamRobust 45 bash <this script>
set -uo pipefail
. ${PAIR_SCRIPTS:?Set PAIR_SCRIPTS to the pair scripts directory}/pair_lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cx_pair_server.sh"
pair_load dev                       # GX=<GX host>, GXCT=seam-gx2-dev, PORT=47311, BUCKETS=1,6,64,8192
CT=${CX_CT:-$GXCT}                 # the pair's own GX container: /cold pack + the release tree
P7CT=${CX_P7CT:-seam-p7-dev}        # pairs.conf's dev client (same release as the GX half)
WT=${CX_WT:-.}
# The pair container renumbers its single card to 0 (`--gpus device=1`), so this is the container's
# index; det5 passes it to both arms, and each arm's client imports it as CUDA_VISIBLE_DEVICES.
GPU=${CX_CUDA_VISIBLE:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARM_SH="$HERE/seam_direct_arm3.sh"
STAMP=$(date +%m%d-%H%M%S)
AB=${CX_AB_BASE:-./evidence/ab}/$STAMP
E_A=$AB/A
E_B=$AB/B
mkdir -p "$E_A" "$E_B"
LOG=$AB/ab.log
: > "$LOG"

PLAIN_SERVER=/cold/placement.exl3.json
PLAIN_CLIENT=${CX_PLAIN_CLIENT:?Set CX_PLAIN_CLIENT}
RANKED_SERVER=/cold/placement.exl3.ranked.json
RANKED_CLIENT=${CX_RANKED_CLIENT:?Set CX_RANKED_CLIENT}
WANT=${CX_WANT_DIGEST:-"67158aeb 3a05b6c3"}   # the two outcomes the default path has produced
RUNS=${CX_IDS_RUNS:-2}
LOADS=${CX_LOADS:-2}

say() { echo "$(date +%T) AB $*" | tee -a "$LOG"; }
status() { [ -n "${GPU_RUN_STATUS:-}" ] && printf 'eta_min=%s doing=%s\n' "$1" "$2" > "$GPU_RUN_STATUS"; return 0; }
gx_mem() {  # -> "avail_mib swapfree_mib"
  timeout 20 ssh -o BatchMode=yes "$GX" \
    'awk "/^MemAvailable:/{a=\$2} /^SwapFree:/{s=\$2} END{printf \"%d %d\", a/1024, s/1024}" /proc/meminfo' \
    2>/dev/null | tr -d '\r'
}
cleanup() {  # always hand the box back clear: a stranded pack load is what broke 11:03-11:16
  local rc=$? i av sw st
  trap - EXIT INT TERM       # a signal must not run this twice, and must not resume the arms after
  # A kill must not leave a pack load behind. At 11:16 a SIGTERM skipped this script's EXIT trap
  # and the server kept pulling 113.65 GiB into seam-gx2-4460 until I killed it by hand; the box
  # was left with 676 MiB and the next agent's ticket started into a swap storm. TERM first,
  # because a server between requests unmaps its slab cleanly; KILL after 4 s for one that is
  # inside a load and ignores it. The pattern is anchored: an unanchored match also hits the
  # `sh -c` that spawned it and reports a kill that did not happen. Budget: the queue kills the
  # whole process group with TERM, waits 30 s, then KILLs it, so everything below has to fit in
  # 30 s and the claim must be dropped before the one step that can be slow (the server's exit).
  timeout 12 ssh -o BatchMode=yes "$GX" \
    "docker exec $CT sh -c 'pkill -f \"^python3 -u qmesh/worker_seam_server.py\" || exit 0; sleep 4; pkill -KILL -f \"^python3 -u qmesh/worker_seam_server.py\" || true'" \
    >/dev/null 2>&1
  st=$(pair_gx_gate_state)
  pair_gx_gate_down            # first, because it is what keeps the next ticket out of my mess
  # 118400, the same floor the arm uses: the 11:03 load started at 119630 and still ended at
  # avail 867 with 948 MiB swapped, so "clear" has to mean the number that load actually needed,
  # not the server's bare preflight of 118268.
  for i in 1 2 3; do           # then give the pack's pages a moment to come back, best effort
    read -r av sw < <(gx_mem)
    [ -n "${av:-}" ] && [ "$av" -ge "${CX_MIN_AVAIL:-118400}" ] && break
    sleep 3
  done
  if [ "${av:-0}" -lt "${CX_MIN_AVAIL:-118400}" ]; then
    # The next ticket would start into 676 MiB. It can see the memory itself, but say it loudly.
    say "CLEANUP WARN the box is not clear: avail_mib=${av:-?} -- do not start a load here yet"
    timeout 12 ssh -o BatchMode=yes "$GX" "docker exec $CT pkill -KILL -f \"^python3 -u qmesh/worker_seam_server.py\" || true" >/dev/null 2>&1
  fi
  say "CLEANUP rc=$rc gx_avail_mib=${av:-?} swap_free_mib=${sw:-?} gate_was=${st:-none}"
  exit "$rc"
}
trap cleanup EXIT INT TERM

[ $# = 0 ] || { echo "usage: $0   (no arguments: every knob is an env var, see the header)" >&2; exit 2; }
status 40 "taking the gx2 gate: determinism, 2 loads x 2 runs, plain then ranked"
pair_gx_gate_take SeamRobust-det5 35 15 || { say "AB FAIL gate_unresolved"; exit 3; }
say "gate file=$(pair_gx_gate_file) owner=SeamRobust-det5"
say "script=$0 sha=$(sha256sum "$0" | cut -c1-16) arm_sha=$(sha256sum "$ARM_SH" | cut -c1-16) want={$WANT} runs=$RUNS loads=$LOADS"
say "tables plain server=$PLAIN_SERVER client=$PLAIN_CLIENT"
say "tables ranked server=$RANKED_SERVER client=$RANKED_CLIENT"
say "arms A plain DET=1 L=$LOADS R=$RUNS | B ranked DET=1 L=$LOADS R=$RUNS refusal  (no rate arm, ws_rows default)"

# Stop the pair's own server once: no arm may load a pack beside it. No copy is made -- the container's
# tree is proven equal to this worktree instead (cx_verify_tree.sh): a release directory must not be
# rewritten, and an A/B arm only needs the proof that both halves run the commit it is testing.
cx_pair_server_stop | sed "s/^/AB pair   /" | tee -a "$LOG"
status 33 "verifying the container runs $WT"
SRC="$WT" GX="$GX" CX_CT="$CT" bash "$HERE/cx_verify_tree.sh" 2>&1 | tee -a "$AB/verify.log" >> "$LOG"
stg=${PIPESTATUS[0]}
say "VERIFY rc=$stg $(grep -ac '^VERIFY ' "$AB/verify.log") lines from cx_verify_tree.sh"
grep -ah '^VERIFY lib \|^VERIFY target ' "$AB/verify.log" | sed 's/^/AB /' | tee -a "$LOG"
[ "$stg" = 0 ] || { say "AB FAIL the container is not running this commit; nothing it prints would be evidence"; exit 13; }

# Arm A: does pinning the summation order make two loads agree?
say "ARM A start plain placement, DET=1, $LOADS loads x $RUNS runs"
status 26 "arm A: plain table, DET=1, $LOADS loads x $RUNS runs"
CX_WT="$WT" CX_GATE=own CX_ARM_TAG=A CX_EVIDENCE="$E_A" CX_LOADS=$LOADS CX_IDS_RUNS=$RUNS \
CX_DETERMINISTIC=1 CX_PREFILL=0 CX_WANT_DIGEST=any \
CX_SERVER_PLACEMENT=$PLAIN_SERVER CX_PLACEMENT=$PLAIN_CLIENT CX_RANKED=0 CX_ORACLE=0 CX_REFUSAL=0 \
  bash "$ARM_SH" 2>&1 | tee -a "$AB/armA.log" >> "$LOG"
rcA=${PIPESTATUS[0]}
say "ARM A rc=$rcA"

# Arm B: the same, on the ranked table, plus the refusal test that the two-table rule depends on.
say "ARM B start ranked placement, DET=1, $LOADS loads x $RUNS runs, refusal"
status 14 "arm B: ranked table, DET=1, $LOADS loads x $RUNS runs, refusal"
CX_WT="$WT" CX_GATE=own CX_ARM_TAG=B CX_EVIDENCE="$E_B" CX_LOADS=$LOADS CX_IDS_RUNS=$RUNS \
CX_DETERMINISTIC=1 CX_PREFILL=0 CX_WANT_DIGEST=any \
CX_SERVER_PLACEMENT=$RANKED_SERVER CX_PLACEMENT=$RANKED_CLIENT CX_RANKED=1 CX_ORACLE=0 CX_REFUSAL=1 \
  bash "$ARM_SH" 2>&1 | tee -a "$AB/armB.log" >> "$LOG"
rcB=${PIPESTATUS[0]}
say "ARM B rc=$rcB"

status 4 "verdict"
# Everything below reads the arm logs, not the evidence directories: arm3 prints DIGESTS/IDSET/DET/
# SLACK through its own `say`, so they land in the file this wrapper tees it into. Grepping the
# evidence dir for them returns nothing and would read as "the run produced no digest".
mapof() {  # <arm log> -> "load1r1=<digest>" lines, from that log's last DIGESTS line
  # Two shapes exist in the record: the earlier tickets labelled one run per load ("load1="), the
  # current arm labels every run ("load1r1=", "load1r2="). Both are accepted. The 64 is not
  # decoration: the 10:33 verdict compared a 12-character prefix against a 16-character one and
  # called a mismatch a match, so a digest that is not a full sha256 does not count here.
  grep -ah 'DIGESTS ' "$1" 2>/dev/null | tail -1 | sed 's/.*DIGESTS //' \
    | tr ' ' '\n' | grep -E '^load[0-9]+(r[0-9]+)?=[0-9a-f]{64}$'
}
digs() { mapof "$1" | cut -d= -f2 | sort -u; }
nof() { printf '%s\n' "$1" | grep -c .; }
gapmin() {  # the closest top-2 call the arm's own probe saw; a tie-break flip needs a tiny margin
  # The probe prints from inside the client (pair_arm_ids_gap.py wraps the generator), so its lines
  # are in the ids_<tag>.log files, which arm3 prefixes with "DIRECT <tag>   ".
  grep -ah 'ARM_GAP ' "$1" 2>/dev/null | sed -nE 's/.* gap=([0-9.eE+-]+).*/\1/p' | sort -g | head -1
}
A_LOG=$AB/armA.log; B_LOG=$AB/armB.log
A_D=$(digs "$A_LOG"); B_D=$(digs "$B_LOG")
A_N=$(nof "$A_D"); B_N=$(nof "$B_D")
say "DIGESTS A {$(printf '%s ' $A_D)}| B {$(printf '%s ' $B_D)}"
say "LOADMAP A $(mapof "$A_LOG" | tr '\n' ' ')"
say "LOADMAP B $(mapof "$B_LOG" | tr '\n' ' ')"
say "GAPMIN A $(gapmin "$A_LOG" | cut -d. -f1-4) B $(gapmin "$B_LOG" | cut -d. -f1-4)"
say "DETFLAGS $(grep -ah 'DET load' "$A_LOG" "$B_LOG" 2>/dev/null | sed 's/^.*DET //' | tr '\n' ';')"
# The arm prefixes its own lines with a clock and "DIRECT ", so these are unanchored on purpose: an
# anchored '^IDSET ' grep silently reported "no line" about a line that was there.
grep -ah 'IDSET ' "$A_LOG" 2>/dev/null | sed 's/^.*IDSET /A IDSET /' | tee -a "$LOG"
grep -ah 'IDSET ' "$B_LOG" 2>/dev/null | sed 's/^.*IDSET /B IDSET /' | tee -a "$LOG"

[ "$rcA" = 0 ] || { say "AB FAIL arm A rc=$rcA (plain table, DET=1)"; exit 20; }
[ "$rcB" = 0 ] || { say "AB FAIL arm B rc=$rcB (ranked table, DET=1)"; exit 21; }
# The switch has to be on the serving process, not just in my environment: the arm reads it from
# /proc on the GX and prints DET load<n> server_deterministic=1. A zero here means the comparison
# below is being run without the thing it is testing and proves nothing.
# One grep per arm: asking the two logs together let arm A's line vouch for a B load that ran
# without the switch, which is the one thing this gate exists to prevent.
for n in $(seq 1 "$LOADS"); do
  for a in A B; do
    eval "l=\${${a}_LOG}"
    grep -ah "DET load$n server_deterministic=1 " "$l" >/dev/null 2>&1 ||
      { say "AB FAIL arm $a load$n did not run with CUDA_EXL3_DETERMINISTIC=1 on the GX (see $l)"; exit 27; }
  done
done
[ "$A_N" -gt 0 ] || { say "AB FAIL arm A left no digest to compare (see $A_LOG)"; exit 28; }
[ "$B_N" -gt 0 ] || { say "AB FAIL arm B left no digest to compare (see $B_LOG)"; exit 29; }
[ "$A_N" = 1 ] ||
  { say "AB FAIL the plain table gave $A_N outcomes {$A_D} across $LOADS loads: CUDA_EXL3_DETERMINISTIC=1 did not fix the ids, so the residual is the GX scatter (qmesh_moe_kernels.cu:10-13: atomicAdd order, <= 1 ulp per row, no switch)"; exit 30; }
[ "$B_N" = 1 ] ||
  { say "AB FAIL the ranked table gave $B_N outcomes {$B_D} across $LOADS loads with the switch on"; exit 31; }
[ "$A_D" = "$B_D" ] ||
  { say "AB FAIL plain gave $A_D but ranked gave $B_D with the summation order pinned: the renumbering changed the computation"; exit 32; }
say "AB PASS: $LOADS plain loads x $RUNS runs and $LOADS ranked loads x $RUNS runs all reached $A_D"
