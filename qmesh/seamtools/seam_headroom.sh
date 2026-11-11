#!/usr/bin/env bash
# The headroom census: one --fresh load per workspace shape (ws_rows 1024, the shipping geometry, and
# 256, a decode-sized piece), each with a decode arm and the 32k prefill, and every line that says how
# much memory the box and the worker had left -- plus a 10 s trajectory across all three phases.
#
# Why a separate ticket: the ws 2048 arm measured 1123.9 tok/s and then printed post-32k
# avail_mib=129 with swap_warn=1, which is fast and not shippable. Before any ws_rows above 1024
# is tested again, the owner wants the actual per-layer balance on a fresh load of the release we
# ship -- the pack is 113.65 GiB of a 121.6 GiB pool, so the whole question is what the load and
# the prefill each take out of the ~8 GiB that is not the pack.
#
# The lines this leaves in the evidence dir, all from the server's own log (arm3 tees it, prefixed
# with "DIRECT load1   ", so none of the greps are anchored):
#   SEAM_PACK_LOAD    need_mib, avail_mib before the pack, margin, fadvise, swapped
#   layer <N>:        per layer: GiB allocated, avail_mib, cached_mib, rss_peak, host rss/swap/trim
#   SEAM_HOST_MEM     phase=load_done rss/swap/anon/trim + avail_mib + box swap (the host bill)
#   SEAM_PACK_KERNEL  how many layers went to cudaexl3 and how many stayed mixed-K, avail/min
#   SEAM_SERVER_SLAB  slab_mib, ws_rows, ws_mib, rows_max, avail_mib, swapped_mib, fits
#   SEAM_SERVER_SWAP_WARN  present only when the load paged; its swapped_mib is the pack sliding
#   SLACK load1       after the 32k prefill: avail_mib, swap_free_mib, ws_rows, prefill, swap_warn
#
#   queue/run_dev.sh SeamRobust 40 bash <this script>     (two loads plus two prefills)
set -uo pipefail
. ${PAIR_SCRIPTS:?Set PAIR_SCRIPTS to the pair scripts directory}/pair_lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cx_pair_server.sh"
pair_load dev                       # GX=<GX host>, GXCT=seam-gx2-dev, PORT=47311, BUCKETS=1,6,64,8192
# The pair's own container (pairs.conf via `pair_load dev`): /cold is the pack, /src/exllamav3 is the
# release it serves from. seam-gx2-4460 no longer exists on either GX10 after the 11-02 reboot.
CT=${CX_CT:-$GXCT}
P7CT=${CX_P7CT:-seam-p7-dev}        # pairs.conf's dev client: /models, /artifacts, same release tree
WT=${CX_WT:-.}   # after the release lands, point this at its tip
# A pair container renumbers its one card to 0 (`--gpus device=1` at start-up), so the client index is
# the container's, not the host's.
GPU=${CX_CUDA_VISIBLE:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARM_SH="$HERE/seam_direct_arm3.sh"
STAMP=$(date +%m%d-%H%M%S)
AB=${CX_AB_BASE:-./evidence/ab}/$STAMP
LOG=$AB/headroom.log
mkdir -p "$AB"; : > "$LOG"
# Two load shapes, because the question is whether the workspace geometry is what makes the worker
# swap: 1024 is today's shipping shape (measured fast, measured tight), 256 is a decode-sized piece.
# Each shape gets its own --fresh load, so the comparison is load-to-load and not arm-to-arm.
ROWS=${CX_HR_ROWS:-1024 256}
WS=                             # set per iteration; kept for the log lines that name the shape

# say owns the prefix: a call site that spells "HR" itself (the NOTE/GRANT/PASS lines) must not print
# it twice -- the 21:56 census read "HR HR GRANT ws=1024 ..." for exactly that reason.
say() { local m="$*"; echo "$(date +%T) HR ${m#HR }" | tee -a "$LOG"; }
status() { [ -n "${GPU_RUN_STATUS:-}" ] && printf 'eta_min=%s doing=%s\n' "$1" "$2" > "$GPU_RUN_STATUS"; return 0; }
gx_mem() {  # -> "avail_mib swapfree_mib"
  timeout 20 ssh -o BatchMode=yes "$GX" \
    'awk "/^MemAvailable:/{a=\$2} /^SwapFree:/{s=\$2} END{printf \"%d %d\", a/1024, s/1024}" /proc/meminfo' \
    2>/dev/null | tr -d '\r'
}
cleanup() {  # the kill-safe handback; see seam_direct_det5.sh for why each step is here
  local rc=$? i av sw st
  trap - EXIT INT TERM
  timeout 12 ssh -o BatchMode=yes "$GX" \
    "docker exec $CT sh -c 'pkill -f \"^python3 -u qmesh/worker_seam_server.py\" || exit 0; sleep 4; pkill -KILL -f \"^python3 -u qmesh/worker_seam_server.py\" || true'" \
    >/dev/null 2>&1
  st=$(pair_gx_gate_state)
  pair_gx_gate_down
  for i in 1 2 3; do
    read -r av sw < <(gx_mem)
    [ -n "${av:-}" ] && [ "$av" -ge "${CX_MIN_AVAIL:-118400}" ] && break   # the arm's own floor
    sleep 3
  done
  say "CLEANUP rc=$rc gx_avail_mib=${av:-?} swap_free_mib=${sw:-?} gate_was=${st:-none}"
  exit "$rc"
}
trap cleanup EXIT INT TERM

[ $# = 0 ] || { echo "usage: $0   (no arguments: every knob is an env var, see the header)" >&2; exit 2; }
status 20 "gx2 gate: headroom census, one fresh load per shape (ws_rows $ROWS)"
# Two loads plus two prefills need more than one: 45 min, and 10 min of waiting for the box to drain.
pair_gx_gate_take SeamRobust-hr "${CX_HR_GATE_MIN:-45}" 10 || { say "HR FAIL gate_unresolved"; exit 3; }
say "gate file=$(pair_gx_gate_file) owner=SeamRobust-hr"
say "script=$0 sha=$(sha256sum "$0" | cut -c1-16) arm_sha=$(sha256sum "$ARM_SH" | cut -c1-16) wt=$WT ws_rows=[$ROWS]"
say "box before stop: gx=[$(gx_mem)]"

cx_pair_server_stop | sed "s/^/HR pair   /" | tee -a "$LOG"
say "box after stop: gx=[$(gx_mem)]  (this is the number the load's SEAM_PACK_LOAD should match)"
# Attribute the census to a commit, not to "the tree on disk". The container has no .git (it holds a
# copy of a release), so the only way a number from this ticket can be matched to a release is to print
# the sha here, refuse a dirty worktree, and let cx_verify_tree.sh prove afterwards that every file in
# the container equals this worktree -- not seven of them, and without rewriting the release. A census taken from a half-applied worktree is exactly the kind of
# evidence that reads as measured and is not.
SRC=$WT
sha=$(git -C "$SRC" rev-parse HEAD 2>/dev/null)
dirty=$(git -C "$SRC" status --porcelain 2>/dev/null | grep -cv '^?? ')
say "STAGED src=$SRC branch=$(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null) sha=${sha:-none} dirty=$dirty want=${CX_STAGE_SHA:-any}"
[ -z "${CX_STAGE_SHA:-}" ] || [ "${CX_STAGE_SHA:0:8}" = "${sha:0:8}" ] ||
  { say "HR FAIL $SRC is at ${sha:-none}, not at the requested ${CX_STAGE_SHA}; the canary shipped another commit"; exit 15; }
[ "$dirty" = 0 ] || { say "HR FAIL $SRC has $dirty uncommitted files; a census cannot be attributed to ${sha:-none}"; exit 16; }
# Verification, not a copy: the container's /src/exllamav3 is a symlink into the pair's release, and
# writing another commit into a release directory is what makes pair_check call the pair drifted.
status 17 "verifying the container runs $SRC at $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null)"
SRC="$SRC" GX="$GX" CX_CT="$CT" bash "$HERE/cx_verify_tree.sh" 2>&1 | tee -a "$AB/verify.log" >> "$LOG" ||
  { say "HR FAIL the container is not running $sha; nothing it prints would be evidence"; exit 13; }
grep -ah '^VERIFY lib \|^VERIFY target ' "$AB/verify.log" | sed 's/^/HR /' | tee -a "$LOG"

# Two things a reader of this table would otherwise have to know from elsewhere.
#  - d330dcf5 has no prefix cache (a fresh Cache per client process, e2e_prompts.py:271), so the
#    second shape's load is not helped by the first one's prefixes and a faster load2 is not a
#    warm-cache artifact. Equally, nothing here benefits from one.
#  - the two shapes do not answer a prefill in the same number of pieces (see HR GRANT): ws_rows is
#    a ceiling on the *piece*, the client chunks a prefill at the largest granted bucket, so 256
#    rows turns one 8192-row chunk into 32 pieces and 1024 into 8. Prefill seconds therefore compare
#    weight re-reads, not throughput; the comparison this census is for is VmRSS/VmSwap/Pss and
#    SUnreclaim at each phase.
say "HR NOTE prefix_cache=none(d330dcf5) piece_split=min(ws_rows,rows_max) prefill_compare=swap_only"
rcs=""
fails=0
for WS in $ROWS; do
  E=$AB/H$WS
  mkdir -p "$E"
  status 12 "--fresh load at ws_rows $WS: load, decode arm, 32k prefill, 10 s sampling"
  # The shape here comes from SEAM_ENV, so it is unpinned by construction -- but pair_serve.sh,
  # pair_smoke.sh, pair_update_run.sh and the decode/roundprof tickets all take their shape from the
  # pair line's WS_ROWS (pairs.conf, default 1024) through pair_server_env. A row here can therefore
  # never be compared against a server that came up from one of those paths unless the pair line said
  # 256; HR GRANT is the line that settles which shape a load actually had.
  # CX_WT is passed, not inherited: arm3 verifies the container against ${CX_WT:-<robust worktree>},
  # so a driver that verified one tree and an arm that verified another would compare the container
  # twice against different bytes and the arm's exit 13 would blame the wrong commit.
  CX_WT="$WT" CX_GATE=own CX_ARM_TAG="H$WS" CX_EVIDENCE="$E" CX_LOADS=1 CX_IDS_RUNS=1 CX_WS_ROWS="$WS" \
  CX_DETERMINISTIC=0 CX_PREFILL=1 CX_WANT_DIGEST=any \
  CX_SERVER_PLACEMENT=/cold/placement.exl3.json \
  CX_PLACEMENT="${CX_PLACEMENT:?Set CX_PLACEMENT}" \
  CX_RANKED=0 CX_ORACLE=0 CX_REFUSAL=0 \
    bash "$ARM_SH" 2>&1 | tee -a "$AB/armH$WS.log" >> "$LOG"
  rcH=${PIPESTATUS[0]}
  rcs="$rcs $WS:$rcH"
  [ "$rcH" = 0 ] || fails=$((fails + 1))
  say "ARM H$WS rc=$rcH"

  status 3 "verdict ws_rows $WS"
  SRV=$E/server_load1.log            # server_log()'s archive of /tmp/seam_server.log; column 0
  PL=$E/pf32k_load1.log
  say "SOURCES ws_rows=$WS server=$(basename "$SRV") prefill=$(basename "$PL")"
  # Unanchored anyway: the copy in the ticket log is prefixed with "DIRECT load1   ", this one is not.
  grep -ah 'SEAM_PACK_LOAD ' "$SRV" | sed "s/^/HR PACKLOAD ws=$WS /" | tee -a "$LOG"
  grep -acE 'layer [0-9]+:' "$SRV" | sed "s/^/HR LAYERLINES ws=$WS /" | tee -a "$LOG"
  grep -ah 'SEAM_HOST_MEM ' "$SRV" | sed "s/^/HR HOSTMEM ws=$WS /" | tee -a "$LOG"
  grep -ah 'SEAM_HOST_MAP ' "$SRV" | sed "s/^/HR HOSTMAP ws=$WS /" | tee -a "$LOG"
  grep -ah 'SEAM_PACK_KERNEL ' "$SRV" | sed "s/^/HR KERNEL ws=$WS /" | tee -a "$LOG"
  # The two lines the workspace question is answered from: the pack's own slab/workspace bill, and the
  # allocator-internal split (free/total of the unified pool, allocated vs reserved) once per load.
  grep -ah 'SEAM_SERVER_SLAB ' "$SRV" | sed "s/^/HR SRVSLAB ws=$WS /" | tee -a "$LOG"
  # The grant, stated instead of inferred: with SEAM_WS_ROWS_AUTO=0 the server serves one request in
  # pieces of min(ws_rows, rows_max), so a 256-row server answers an 8192-row prefill chunk in 32
  # pieces where a 1024-row one answers it in 8. That is a different prefill shape as well as a
  # different workspace (the client's chunk_rows is the largest granted bucket, not ws_rows), which
  # is why the two arms' prefill seconds are not a throughput A/B. Printed so no reader has to hold
  # the bucket list in their head to notice it.
  read -r SRVWS RMAX <<< "$(grep -ah 'SEAM_SERVER_SLAB ' "$SRV" | tail -1 |
    sed -n 's/.* ws_rows=\([0-9]*\) .* rows_max=\([0-9]*\) .*/\1 \2/p')"
  PPC=$(( (${RMAX:-1} + ${SRVWS:-1} - 1) / ${SRVWS:-1} ))
  say "HR GRANT ws=$WS server_ws_rows=${SRVWS:-?} rows_max=${RMAX:-?} pieces_per_prefill_chunk=$PPC auto=0"
  grep -ah 'SEAM_SERVER_DEVMEM ' "$SRV" | sed "s/^/HR DEVMEM ws=$WS /" | tee -a "$LOG"
  grep -ah 'SLAB_SHIP \|SAMPLE_SHIP ' "$AB/armH$WS.log" | sed "s/.*DIRECT /HR /" | tee -a "$LOG"
  # Host samples around the load (clear box, 113 GiB resident) and after the prefill; SUnreclaim is
  # the slab nobody can hand back, which is the memory the pack load leaves unaccounted for.
  grep -ah 'DIRECT SLAB phase=' "$AB/armH$WS.log" | sed "s/.*SLAB /HR CENSUS ws=$WS /" | tee -a "$LOG"
  # The three phase points, each a synchronous sample: at load, after the decode arm, after 32k.
  grep -ah 'DIRECT PHASESAMP ' "$AB/armH$WS.log" | sed "s/.*DIRECT PHASESAMP /HR PHASESAMP ws=$WS /" | tee -a "$LOG"
  # The 10 s trajectory across load+decode+prefill: which consumer grows (Cached = page cache,
  # SUnreclaim = kernel slab, VmRSS/Pss = inside the worker, VmSwap = the worker's own pages).
  grep -ah 'SAMPLE_START \|SAMPLES load' "$AB/armH$WS.log" | sed "s/.*DIRECT /HR /;s/$/ ws=$WS/" | tee -a "$LOG"
  grep -ah 'DIRECT SAMP load1 ' "$AB/armH$WS.log" | sed "s/.*DIRECT SAMP load1 SAMP /HR SAMP ws=$WS /" | tee -a "$LOG"
  # From this load's own SEAM_SERVER_READY, never the tail: the container log is truncated by the
  # ensure path, but a restore that starts a server with >> would leave the previous load's warn at
  # the top of the file and a tail-based grep would quote it as this shape's.
  awk '/^SEAM_SERVER_READY/{n=NR} {a[NR]=$0} END{ if (n) for (i=n;i<=NR;i++) print a[i]; else for (i=1;i<=NR;i++) print a[i] }' "$SRV" |
    grep -ah 'SEAM_SERVER_SWAP_WARN ' | sed "s/^/HR SWAPWARN ws=$WS /" | tee -a "$LOG"
  # The piece count rides on the prefill line: the seconds there are 8192 tokens over PPC weight
  # passes, so nobody divides tokens by seconds and calls the result a shape comparison.
  grep -ah 'PREFILL_ARM ' "$PL" | sed "s/^/HR PREFILL ws=$WS pieces_per_chunk=$PPC /" | tee -a "$LOG"
done
# The comparison the owner reads first: worker swap and RSS at the three phases, per shape.
say "SHAPE TABLE (worker, from the phase samples)"
for WS in $ROWS; do
  grep -ah 'DIRECT PHASESAMP ' "$AB/armH$WS.log" 2>/dev/null |
    sed 's/.*DIRECT PHASESAMP //' |
    awk -v w="$WS" '
      { ph=$1; rss="-"; vsw="-"; av="-"
        for (i = 2; i <= NF; i++) if ($i ~ /^vm_rss_mib=/) { split($i, a, "="); rss = a[2] }
        for (i = 2; i <= NF; i++) if ($i ~ /^vm_swap_mib=/) { split($i, a, "="); vsw = a[2] }
        for (i = 2; i <= NF; i++) if ($i ~ /^avail_mib=/) { split($i, a, "="); av = a[2] }
        printf "HR SHAPE ws=%s phase=%s vm_rss_mib=%s vm_swap_mib=%s avail_mib=%s\n", w, ph, rss, vsw, av }' |
    tee -a "$LOG"
done
# Option-B swap gate (owner decision, 2026-11-03): a worker VmSwap level is not a refusal -- an idle
# GX2 with no pack already holds ~1.6 GiB of box swap -- so the only refusal here is growth inside the
# measured decode window, after_load1 -> after_decode1. Prefill growth (~356 MiB at ws 256, ~734 at
# 1024 after a 32k prefill) is expected and printed as a warning. A post-load reading whose avail_mib
# is still above ~2000 MiB cannot be post-residency, so it is withheld from judgement, not judged.
# Levels are printed (worker VmSwap, box swap, avail, pid, and the load's own swapped_mib) beside
# every verdict so a reader can see the pressure the number was taken at.
num() { case "${1:-}" in ''|*[!0-9]*) return 1 ;; esac; }
say "SWAPGATE rule=decode_window_growth>${CX_SWAP_GROW_MAX:-128}MiB:fail levels:warn prefill_growth:warn post_load_avail>${CX_SWAP_JUDGE_AVAIL_MAX:-2000}MiB:withhold"
swapfail=0
for WS in $ROWS; do
  E=$AB/H$WS
  vals=$(grep -ah 'DIRECT PHASESAMP ' "$AB/armH$WS.log" 2>/dev/null | sed 's/.*DIRECT PHASESAMP //' |
    awk -v tot="${CX_SWAP_TOTAL_MIB:-16384}" '
      { ph=$1
        for (i=2;i<=NF;i++) { split($i,kv,"="); f[kv[1]]=kv[2] }
        if (ph=="after_load1")    { b=f["vm_swap_mib"]; a0=f["avail_mib"]; box=tot-f["swapfree_mib"]; pid=f["pid"] }
        if (ph=="after_decode1")  { ad=f["vm_swap_mib"]; a1=f["avail_mib"]; pid1=f["pid"] }
        if (ph=="after_prefill1") { ap=f["vm_swap_mib"]; a2=f["avail_mib"] } }
      END{ print (b==""?"?":b), (ad==""?"?":ad), (ap==""?"?":ap), (a0==""?"?":a0), (a1==""?"?":a1),
                 (a2==""?"?":a2), (box==""?"?":box), (pid==""?"?":pid), (pid1==""?"?":pid1) }')
  read -r B AD AP A0 A1 A2 BOX PID PID1 <<< "${vals:-? ? ? ? ? ? ? ? ?}"
  LSW=$(grep -ah 'SEAM_PACK_LOAD ' "$E/server_load1.log" 2>/dev/null | tail -1 |
    sed -n 's/.* swapped_mib=\([0-9]*\).*/\1/p')
  if ! num "$B" || ! num "$AD"; then
    say "SWAPGATE ws=$WS before_arm_vm_swap_mib=${B:-?} after_decode_vm_swap_mib=${AD:-?} box_swap_mib=${BOX:-?} avail_mib=${A0:-?} pid=${PID:-?} load_swapped_mib=${LSW:-?} verdict=withheld reason=no_phase_sample"
  elif num "$A0" && [ "$A0" -gt "${CX_SWAP_JUDGE_AVAIL_MAX:-2000}" ]; then
    say "SWAPGATE ws=$WS before_arm_vm_swap_mib=$B after_decode_vm_swap_mib=$AD growth_mib=$(( AD - B )) box_swap_mib=$BOX avail_mib=$A0 pid=$PID load_swapped_mib=${LSW:-?} verdict=withheld reason=avail_mib_over_${CX_SWAP_JUDGE_AVAIL_MAX:-2000}_not_post_residency WARNING=level_only"
  else
    g=$(( AD - B ))
    if [ "$g" -gt "${CX_SWAP_GROW_MAX:-128}" ]; then
      v="FAIL(growth_over_${CX_SWAP_GROW_MAX:-128}MiB)"; swapfail=$((swapfail + 1))
    else
      v=pass
    fi
    say "SWAPGATE ws=$WS before_arm_vm_swap_mib=$B after_decode_vm_swap_mib=$AD growth_mib=$g box_swap_mib=$BOX avail_mib=$A0 after_decode_avail_mib=$A1 pid=$PID/$PID1 load_swapped_mib=${LSW:-?} verdict=$v"
  fi
  if num "$AD" && num "$AP"; then
    say "SWAPGATE-PREFILL ws=$WS after_decode_vm_swap_mib=$AD after_prefill_vm_swap_mib=$AP growth_mib=$(( AP - AD )) avail_after_prefill_mib=${A2:-?} WARNING=level_only"
  fi
done
say "SWAPGATE-SUMMARY shapes_failing_growth=$swapfail of ws_rows=[$ROWS] bar=${CX_SWAP_GROW_MAX:-128}MiB"
say "FINAL gx=[$(gx_mem)] rc=$rcs"
[ "$fails" = 0 ] || { say "FAIL arms failed:$rcs"; exit 24; }
[ "${swapfail:-0}" = 0 ] || { say "FAIL swap growth inside the decode window exceeded ${CX_SWAP_GROW_MAX:-128} MiB on ${swapfail} shape(s) (Option B: the only level that refuses)"; exit 26; }
say "PASS ws_rows=[$ROWS] census printed above; nothing above the shipping shape is tested until the owner reads it"
