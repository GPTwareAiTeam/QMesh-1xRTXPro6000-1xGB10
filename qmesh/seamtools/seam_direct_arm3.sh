#!/usr/bin/env bash
# Acceptance for the direct from-pack cold-expert path on the dev pair (P7 GPU1 + GX2, 4460 packs).
#
# One arm of the two-arm acceptance (see seam_direct_ab.sh): two consecutive --fresh loads of the
# same release, each with SEAM_MOE_DIRECT=1, and after each load:
#   1. a decode arm with --ids (pair_arm_ids.py, rows=1 max_new=8, the shipping shape), whose
#      ARM_IDS sha256 is the ids digest -- 67158aeb on the plain table *and* on the ranked one,
#      because a reordering of the cold experts must not change a token, and
#   2. a 32k prefill in 8192-row calls (mem_budget.py, PREFILL_ARMS=32768:8192).
#
# What each load has to show, in the server log:
#   SEAM_PACK_KERNEL cudaexl3_layers=40        every layer on the cuda-exl3 kernel -- no mixed-K tail,
#                                             which is what the restack could not do on this box
#                                             (38/2 at the default margin, 37/3 at 2600 MiB)
#   SEAM_MOE_DIRECT layer=.. n=..              one line per layer, no SEAM_MOE_DIRECT_FAIL
#   SEAM_SERVER_SLACK / SEAM_SERVER_SLAB        avail_mib, avail_min_mib, slack, swapped_mib after load
#
# References on this pair (32k chunk 8192): 957.7 tok/s (canary ae4fe322 run B, 39/1 cudaexl3),
# 881.8 tok/s (control 9fc3ca8a, 38/2), 818.2 (9fc3ca8a, 37/3).
#
# What it does before either load, in this order, all of it inside the same gate (so the rsync onto
# the container's live bind mount waits until the box is this ticket's and the pair's server is
# down):
#   STAGE            rsync this worktree onto the GX and prove by sha that the container sees it
#                    (qmesh/tests/stage_gx.sh; exit 13 if it does not)
#   REFUSAL          a --stub server on this ticket's table and the shipping client pointed at the
#                    other one: it must be refused at the Hello and the server must survive
#                    (qmesh/tests/refusal_stub.sh; exit 12 if it is not)
#   ORACLE           cx_direct_check.py on two real layers, both ways, byte for byte (exit 10)
#   PREFILL          32k/8192, the rate the references were taken at (exit 14 if it did not finish)
#   SHAPE            after each load: the server's own SEAM_SERVER_SLAB ws_rows must equal the shape
#                    this arm asked for, and the grant (rows_max, pieces per prefill chunk, which is
#                    min(ws_rows, rows_max) when SEAM_WS_ROWS_AUTO=0) is printed beside the numbers
#                    (exit 25 if the server was started at another shape -- a load measured under
#                    the wrong label is worse than no load)
#
#   queue/run_dev.sh SeamRobust 40 bash <this script>
# The gate is taken for gx2 and dropped in the trap; my server is stopped at the end. The dev pair's
# own server is a pack load too, so if it is up it is stopped for the run and reloaded afterwards
# from the release (cx_pair_server.sh) -- the pair ends the ticket in the state it started in.
set -uo pipefail
. "${PAIR_SCRIPTS:?Set PAIR_SCRIPTS to the pair scripts directory}/pair_lib.sh"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cx_pair_server.sh"
pair_load dev                       # GX=<GX host>, GXCT=seam-gx2-dev, PORT=47311, BUCKETS=1,6,64,8192
# The pair's own cold-expert container: it mounts /cold (the 113 GiB pack) and /src/exllamav3, which
# after the 2026-11-02 release rollout is a symlink to $PAIR_ROOT/current/src rather than a scratch
# copy. seam-gx2-4460 was the pre-release box and is gone on both GX10s; the pair config is the only
# place a container name should come from, so the default is what `pair_load dev` just read.
CT=${CX_CT:-$GXCT}
SEAM_GATE_CT=${SEAM_GATE_CT:-$GXCT}
# The pair's own client container (pairs.conf), not a hand-made one: it has /models and /artifacts,
# and its /src/exllamav3 is the same release the GX half serves from.
P7CT=${CX_P7CT:-seam-p7-dev}
: "${CKPT:?Set CKPT to the container checkpoint path}"
# Inside a pair container the card is renumbered: the container is started with `--gpus device=1`, so
# its one visible GPU is 0 there (PairRelease-2, 2026-11-02 pinning fix). A client that exports the
# host index would either fail to find a device or sit on the wrong card.
GPU=${CX_CUDA_VISIBLE:-0}
E=${CX_EVIDENCE:-./evidence/directpack}
STAMP=$(date +%m%d-%H%M%S)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The host-side census (SUnreclaim + the ten biggest slab caches) is a file of its own because it
# has to run on the GX, where /proc/slabinfo needs root and the service user has no sudo: it reads through
# `docker exec`, which is uid 0 on the same kernel. Shipped per ticket and sha-printed, so a SLAB
# line can be traced to the script that produced it.
HC_SH=${CX_HC_SH:-$HERE/host_census.sh}
SAM_SH=${CX_SAM_SH:-$HERE/gx_sampler.sh}
mkdir -p "$E"
# Evidence the *container* writes has to land on a mount that container has rw. /artifacts is ro
# here (09:57 run: ARM_IDS_NOTE Errno 30), /models is rw on both sides, so the ids json is written
# there and copied into $E afterwards; the host copy is the one the digest checks read.
ARM_IDS_DIR=${CX_ARM_IDS_DIR:-/models/.seam-evidence/$(basename "$E")}
mkdir -p "$ARM_IDS_DIR"
LOG=$E/direct_$STAMP.log
: > "$LOG"
# The workspace geometry the client asks the server for. WS_ENV used to be an inherited variable
# with no default, which under `set -u` would have killed the arm at this line, before the gate.
WS_ENV=${CX_WS_ENV:-SEAM_WS_ROWS=${CX_WS_ROWS:-1024} SEAM_WS_ROWS_AUTO=0}
SEAM_ENV_DIRECT="SEAM_MOE_KERNEL=cudaexl3 SEAM_MOE_PRELOAD=1 SEAM_MOE_DIRECT=1 SEAM_CX_MARGIN_MIB=1536 $WS_ENV"
# The digest of one arm (dspark1, n=8), which is what the ARM_IDS line below carries: 67158aeb.
# f96fc065 is a different metric -- SMOKE_ARM_IDS_TOTAL, the same sequence summed over all rounds --
# and comparing the arm line against it turns every good run into "DIRECT WARN".
# The ids are not bitwise reproducible across loads: two --fresh loads of the *same* table on
# 11-02 gave 67158aeb and 3a05b6c3, agreeing on the first six tokens and differing at position
# seven (736 vs 24), and the ranked table produced exactly those two outcomes as well. The digest
# is therefore a member check against the set this checkpoint produces, plus a requirement that
# the canonical one appears at least once per arm, plus the per-step top-2 margin that says a flip
# is a tie-break and not a different computation. Cross-arm equality is set equality, not
# element-wise equality, and that is checked by the caller.
WANT_DIGEST=${CX_WANT_DIGEST:-"67158aeb 3a05b6c3"}
WANT_ANY=${CX_WANT_ANY:-67158aeb}
IDS_RUNS=${CX_IDS_RUNS:-1}   # client runs against one loaded server: repeatability inside a load
# Both halves of the pair, named, not resolved: the server numbers its cold experts by the file at
# $SERVER_PLACEMENT and advertises its sha in the Hello; the client loads the file at
# $CLIENT_PLACEMENT and refuses the session if the shas differ. Empty CLIENT_PLACEMENT = whatever
# the pack's manifest / repo default resolves to (the unranked baseline).
SERVER_PLACEMENT=${CX_SERVER_PLACEMENT:-/cold/placement.exl3.ranked.json}
CLIENT_PLACEMENT=${CX_PLACEMENT:?Set CX_PLACEMENT to the client placement path}
RANKED=${CX_RANKED:-1}
LOADS=${CX_LOADS:-2}      # 2 = the two-load acceptance arm; 1 = a single-load arm (C, ws geometry)

# say owns the prefix: call sites that spell "DIRECT" themselves (FAIL/PASS/rc lines) must not print
# it twice -- the 21:56 census logged "DIRECT DIRECT FAIL load1r1 left no digest to compare".
say() { local m="$*"; echo "$(date +%T) DIRECT ${m#DIRECT }" | tee -a "$LOG"; }
status() { [ -n "${GPU_RUN_STATUS:-}" ] && printf 'eta_min=%s doing=%s\n' "$1" "$2" > "$GPU_RUN_STATUS"; return 0; }

gx_mem() {  # -> "avail_mib swapfree_mib"
  timeout 20 ssh -o BatchMode=yes "$GX" \
    'awk "/^MemAvailable:/{a=\$2} /^SwapFree:/{s=\$2} END{printf \"%d %d\", a/1024, s/1024}" /proc/meminfo' \
    2>/dev/null | tr -d '\r'
}

gx_holders() {  # -> "name=used" for whoever is holding the box, so a refusal names a culprit
  timeout 12 ssh -o BatchMode=yes "$GX" \
    'docker stats --no-stream --format "{{.Name}}={{.MemUsage}}" 2>/dev/null | head -4' \
    2>/dev/null | tr -d '\r' | tr '\n' ' '
}
host_census() {  # <phase> -> one SLAB line: available/slab/unreclaimable + the ten biggest caches
  local tag=$1 out
  out=$(timeout 30 ssh -o BatchMode=yes "$GX" "CX_CT='$CT' bash /tmp/qm_host_census.sh '$tag'" \
          2>/dev/null | tr -d '\r' | tail -1)
  [ -n "$out" ] || out="phase=$tag unavailable"
  say "SLAB $out"
}

# The prefill is where the server's host memory goes to swap (worker VmSwap 0 at load_done, 820 MiB
# during a 32k+512k prefill), so the trajectory has to be sampled while it happens, not only before
# and after: a 10 s series from just before the load to just after the prefill separates page cache
# (Cached rises, process does not), kernel slab (SUnreclaim/SReclaimable rise), CUDA/pinned growth
# inside the worker (VmRSS/Pss rise) and plain paging of the worker (VmSwap rises).
sampler_start() {  # <tag>: background trajectory on the GX, ended by sampler_stop
  timeout 25 ssh -o BatchMode=yes "$GX" \
    "rm -f /tmp/qm_sampler.stop /tmp/qm_sampler_$1.out; setsid nohup bash /tmp/qm_gx_sampler.sh '${CX_SAMPLE_S:-10}' 400 '$CT' /tmp/qm_sampler_$1.out >/dev/null 2>&1 < /dev/null &" \
    >/dev/null 2>&1
  sleep 0.5
  say "SAMPLE_START tag=$1 interval_s=${CX_SAMPLE_S:-10}"
}
sampler_stop() {  # <tag> <loadN>: stop it, fetch the series, log every sample
  local tag=$1 ln=$2 f=$E/samples_$1.log n
  timeout 15 ssh -o BatchMode=yes "$GX" 'touch /tmp/qm_sampler.stop' >/dev/null 2>&1
  sleep 2
  timeout 20 ssh -o BatchMode=yes "$GX" "pkill -f '[q]m_gx_sampler.sh' 2>/dev/null; true" >/dev/null 2>&1
  scp -q -o BatchMode=yes "$GX:/tmp/qm_sampler_$tag.out" "$f" 2>/dev/null
  n=$(grep -ac '^SAMP ' "$f" 2>/dev/null | tr -dc 0-9)
  say "SAMPLES $ln n=${n:-0} interval_s=${CX_SAMPLE_S:-10} file=$f"
  [ "${n:-0}" -gt 0 ] && grep -a '^SAMP ' "$f" | sed "s/^/$(date +%T) DIRECT SAMP $ln /" | tee -a "$LOG"
  return 0
}

# One synchronous sample, tagged with the phase it was taken in. The background series above gives
# the trajectory; these give the three numbers a reader compares across arms (at load, after a decode
# arm, after a 32k prefill) without aligning timestamps by hand.
sampler_once() {  # <phase>
  local out
  out=$(timeout 30 ssh -o BatchMode=yes "$GX" "bash /tmp/qm_gx_sampler.sh 0 1 '$CT' /dev/stdout" \
          2>/dev/null | tr -d '\r' | grep -a '^SAMP ' | head -1)
  say "PHASESAMP $1 ${out:-unavailable}"
}
stop_server() {
  local out avail swap avail2 i hc
  timeout 30 ssh -o BatchMode=yes "$GX" "docker exec $CT pkill -f '^python3 -u qmesh/worker_seam_server.py' 2>/dev/null; true" >/dev/null 2>&1
  # 118400 stays: it is not an arbitrary cushion over the server's own preflight (116220 pack +
  # SEAM_LOAD_MARGIN_MIB 2048 = 118268), it is what the one load we can measure needed. On
  # 2026-11-02 11:03 `SEAM_PACK_LOAD need_mib=116220 avail_mib=119630` ran to
  # `SEAM_SERVER_SLAB … avail_mib=867 swapped_mib=948` -- it ate its whole 2 GiB margin and
  # 948 MiB of swap, so a load started at 118268 would have begun 1.4 GiB tighter than a run that
  # already ended in swap. (117760 was the older, too-low value: load1 died SEAM_NO_MEMORY at
  # 118082 on 09:11.) Two readings 6 s apart, because the stopped server's pages come back in steps.
  hc=$(pair_ticket_hardcap 2>/dev/null || true)
  case "${hc:-}" in ''|*[!0-9]*) hc=0 ;; esac
  for i in $(seq 1 90); do
    out=$(gx_mem); read -r avail swap <<< "$out"
    # Never sit in this loop past the ticket's own cap: the queue KILLs the group 30 s after it, and
    # a KILL skips the trap that would otherwise record where the box was left. 11:16 was exactly
    # that, with a 113 GiB load stranded inside a container.
    if [ "$hc" -gt 0 ] && [ "$(date +%s)" -gt $((hc - 150)) ]; then
      say "gx GIVE UP at the ticket's own cap (${hc}) with avail_mib=${avail:-0}: $(gx_holders)"
      return 1
    fi
    if [ -n "${avail:-}" ] && [ "$avail" -ge "${CX_MIN_AVAIL:-118400}" ]; then
      sleep 6
      out=$(gx_mem); read -r avail2 swap <<< "$out"
      if [ -n "${avail2:-}" ] && [ "$avail2" -ge "${CX_MIN_AVAIL:-118400}" ]; then
        return 0
      fi
      say "gx not settled avail_mib=($avail->${avail2:-0}) swap_free_mib=${swap:-?} t=$((i * 4 + 6))s"
    fi
    if [ $((i % 10)) = 0 ]; then
      # Two tickets started side by side at 11:24 and one of them could only ever have waited here.
      # Naming the holders turns the next "STILL BUSY" into a diagnosis instead of a shrug.
      say "gx waiting avail_mib=${avail:-0} swap_free_mib=${swap:-?} t=$((i * 4))s holders=[$(gx_holders)]"
    fi
    sleep 4
  done
  say "gx STILL BUSY avail_mib=${avail:-0} after 360s holders=[$(gx_holders)]"
  return 1
}
det_env() {  # -> 1 when the *loaded* server process carries CUDA_EXL3_DETERMINISTIC=1
  # The switch is read by the module at import, so "I exported it" is not evidence: this reads the
  # environment of the process that is actually serving, from /proc, on the GX.
  timeout 20 ssh -o BatchMode=yes "$GX" \
    "docker exec $CT sh -c 'for p in \$(pgrep -f qmesh/worker_seam_server.py); do tr \"\\0\" \"\\n\" < /proc/\$p/environ 2>/dev/null | grep -qx CUDA_EXL3_DETERMINISTIC=1 && echo 1 && break; done'" \
    2>/dev/null | tr -d '\r' | tail -1
}

# The shipping client shape, unchanged from pair_arm.sh: the two halves must be the same pair.
CLIENT_ENV=(
  -e "CUDA_VISIBLE_DEVICES=$GPU" -e "REMOTE_TRANSPORT=seam"
  -e "SEAM_DEV=$SEAM_DEV" -e "SEAM_GID=$SEAM_GID" -e "SEAM_PEER=$SEAM_PEER" -e "SEAM_PORT=$PORT"
  -e "SEAM_BUCKETS=$BUCKETS"
  -e "CHECKPOINT=$CKPT" -e "DRAFT_CHECKPOINT=$CKPT"
  -e "DSPARK=1" -e "NUM_DRAFT=5" -e "KV_BITS=4" -e "EXL3_WHOLE_GRAPH=0" -e "MAX_CHUNK=8192"
  -e "EXL3_MOE_SPLIT_FUSED=0" -e "EXL3_SEAM_MARK=1" -e "EXL3_SEAM_MARK_LAYERS=8"
  -e "EXL3_SEAM_PROF=1" -e "EXL3_SEAM_PROF_ROWS=2560"
  -e "PREFILL_ADAPTIVE=1" -e "PREFILL_ADAPTIVE_LOG=1" -e "ARM_MAX_S=1200"
  -e "SEAM_ROWS=8192" -e "SEAM_NS=1" -e "SEAM_OVERLAP=1" -e "SEAM_COPY_STREAM=1"
  -e "PROMPTS=/artifacts/2x-rtx-tp2/dtrain/prompts_held.json" -e "CACHE_TOKENS=1048576"
)
# Determinism. The default MoE path sums the down-gemm into `acc` with fp32 atomics whenever the
# kernel can write acc directly (qmesh/exl3_moe_cuda.py: HAS_ACC_OUT and not DETERMINISTIC), so the
# order of those adds varies run to run and the last bit of every logit with it. That is the
# standing explanation for two loads of one table producing 67158aeb and 3a05b6c3, and it is
# testable: with CUDA_EXL3_DETERMINISTIC=1 the module calls exl3_moe_combine instead, the order is
# fixed, and two loads must then produce one digest. Set on both halves, because both run the same
# module: the question here is whether the ids *can* be made reproducible, not which half drifted.
if [ "${CX_DETERMINISTIC:-0}" = 1 ]; then
  SEAM_ENV_DIRECT+=" CUDA_EXL3_DETERMINISTIC=1"
  CLIENT_ENV+=(-e "CUDA_EXL3_DETERMINISTIC=1")
fi
# One table per pair, named on both halves. The client container mounts the worktree at
# /src/exllamav3 and /models rw, so any path the server also holds the bytes of will do -- the
# Hello carries the file's sha, not its path.
[ -z "$CLIENT_PLACEMENT" ] || CLIENT_ENV+=(-e "PLACEMENT=$CLIENT_PLACEMENT")
CLIENT_ENV+=(-e "ARM_IDS_GAP=${CX_IDS_GAP:-1}")

# What the *server* was actually started with, from its own line, not from what this arm exported.
# The workspace knob has two spellings (the pair's WS_ROWS pin and this arm's SEAM_WS_ROWS) and a
# start path that goes through the pair's own environment overrides the second one silently: on
# 2026-11-02 that is how a "256-row" load can be a 1024-row load with a 256-row label, which would
# make the whole census mislabelled rather than wrong. `rows_max` is the largest bucket granted, so
# with SEAM_WS_ROWS_AUTO=0 the server serves one request in pieces of min(ws_rows, rows_max): 1024
# rows against an 8192 bucket is 8 pieces, 256 is 32, and that is a different prefill shape, not
# only a different workspace - print it beside the number instead of letting a reader infer it.
shape_assert() {  # <tag>
  local tag=$1 line wr rm wm sl
  line=$(grep -a '^SEAM_SERVER_SLAB ' "$E/server_$tag.log" 2>/dev/null | tail -1)
  wr=$(sed -n 's/.* ws_rows=\([0-9]*\).*/\1/p' <<< "$line")
  rm=$(sed -n 's/.* rows_max=\([0-9]*\).*/\1/p' <<< "$line")
  wm=$(sed -n 's/.* ws_mib=\([0-9.]*\).*/\1/p' <<< "$line")
  sl=$(sed -n 's/.* slab_mib=\([0-9.]*\).*/\1/p' <<< "$line")
  [ -n "$wr" ] || { say "SHAPE $tag server_ws_rows=? requested=${CX_WS_ROWS:-1024} (no SLAB line; not asserting)"; return 0; }
  say "SHAPE $tag server_ws_rows=$wr requested=${CX_WS_ROWS:-1024} rows_max=${rm:-?} ws_mib=${wm:-?} slab_mib=${sl:-?} pieces_per_${rm:-?}=$(( (${rm:-1} + wr - 1) / wr ))"
  [ "$wr" = "${CX_WS_ROWS:-1024}" ] ||
    { say "DIRECT FAIL $tag the server was started at ws_rows=$wr, not ${CX_WS_ROWS:-1024}: every number from this load would carry the wrong label"; exit 25; }
}
server_log() {  # <tag> -> archive the container log; print the SEAM_ summary lines and the counts
  local tag=$1 host=$E/server_$1.log
  timeout 60 ssh -o BatchMode=yes "$GX" "docker cp $CT:/tmp/seam_server.log /tmp/srv_$tag.log" >/dev/null 2>&1
  scp -q -o BatchMode=yes "$GX:/tmp/srv_$tag.log" "$host" 2>/dev/null || true
  [ -s "$host" ] || { say "SERVER_LOG missing tag=$tag"; return 1; }
  grep -aE '^SEAM_(PACK_KERNEL|MOE_DIRECT|MOE_DIRECT_FAIL|SERVER_LOADED|SERVER_DEVMEM|SERVER_SLAB|SERVER_SWAP_WARN|MOE_NOFIT|PLACEMENT)' "$host" |
    sed "s/^/DIRECT $tag   /" | tee -a "$LOG"
  say "VERDICT tag=$tag cudaexl3_40=$(grep -ac 'cudaexl3_layers=40' "$host")" \
      "direct=$(grep -ac '^SEAM_MOE_DIRECT layer=' "$host")" \
      "direct_fail=$(grep -ac '^SEAM_MOE_DIRECT_FAIL' "$host")" \
      "nofit=$(grep -ac '^SEAM_MOE_NOFIT' "$host")" \
      "mixedk=$(grep -ao 'mixedk_layers=[0-9]*' "$host" | tail -1)"
}

ids_arm() {  # <tag> -> decode arm with --ids; the ARM_IDS sha256 is the load's ids digest
  local tag=$1 logf=$E/ids_$1.log out=$E/ids_$1.json ctr=$ARM_IDS_DIR/ids_$1.json
  : > "$logf"
  rm -f "$out" "$ctr"
  timeout 420 docker exec "${CLIENT_ENV[@]}" \
    -e "MAX_PROMPTS=1" -e "MAX_NEW=${CX_IDS_MAX_NEW:-8}" -e "WARMUP=${CX_IDS_WARMUP:-0}" -e "WARMUP_NEW=${CX_IDS_WARMUP_NEW:-0}" \
    -e "ARM_IDS_OUT=$ctr" \
    "$P7CT" bash -lc "cd /src/exllamav3 && nice -n 10 python3 -u ${CX_ID_DRIVER:-/artifacts/gpu0-gx1/scripts/pair_arm_ids_gap.py}" \
    2>&1 | tee -a "$logf" >> "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ -s "$ctr" ]; then cp -f "$ctr" "$out"; else say "IDS_NOTE no json at $ctr (rc=$rc)"; fi
  say "IDS tag=$tag rc=$rc $(grep -a '^ARM_IDS ' "$logf" | head -1 | cut -c1-160) gx=[$(gx_mem)]"
  # Decode cost, from the arm's own per-prompt row (e2e_prompts.py: decode_ms_per_tok). Raised
  # CX_IDS_MAX_NEW turns the digest arm into the decode-timing arm as well, so a ws_rows width A/B
  # takes its decode number from the same load and the same session as its prefill number.
  local row dms dnt dtt dar
  row=$(grep -a '^{"i":' "$logf" 2>/dev/null | tail -1)
  dms=$(printf '%s' "$row" | sed -n 's/.*"decode_ms_per_tok": \([0-9.]*\).*/\1/p')
  dnt=$(printf '%s' "$row" | sed -n 's/.*"new_tokens": \([0-9]*\).*/\1/p')
  dtt=$(printf '%s' "$row" | sed -n 's/.*"ttft_ms": \([0-9.]*\).*/\1/p')
  dar=$(printf '%s' "$row" | sed -n 's/.*"accept_rate": \([0-9.]*\).*/\1/p')
  [ -n "$dms" ] && say "DECODE tag=$tag decode_ms_per_tok=$dms decode_tok_per_s=$(awk -v m="$dms" 'BEGIN{ printf "%.1f", 1000/m }') new_tokens=${dnt:-?} ttft_ms=${dtt:-?} accept_rate=${dar:-?} max_new=${CX_IDS_MAX_NEW:-8}"
  # The margin behind each chosen token, for the ties that make two digests out of one computation.
  grep -a '^ARM_GAP ' "$logf" | sed "s/^/DIRECT $tag   /" | tee -a "$LOG"
  return $rc
}
ids_runs() {  # <tag prefix> -> IDS_RUNS client runs against the one loaded server
  local pre=$1 r rc=0
  for r in $(seq 1 "$IDS_RUNS"); do
    ids_arm "${pre}r$r" || rc=$?
  done
  return $rc
}

prefill_arm() {  # <tag> -> 32k prefill in 8192-row calls, the rate the references were taken at
  local tag=$1 logf=$E/pf32k_$1.log
  : > "$logf"
  # DEPTH_PROBES="" disables mem_budget.py's depth sweep (524288/1048570 tokens, ~10 min per arm): it
  # measures statics at deep context, not the 32k rate this arm is for, and the gate window is finite.
  timeout 1500 docker exec "${CLIENT_ENV[@]}" \
    -e "PREFILL_ARMS=32768:8192" -e "CORPUS=/artifacts/2x-rtx-tp2/corpus/prose_24k.txt" \
    -e "DEPTH_PROBES=" \
    -e "LOAD_ONLY=0" -e "MAX_PROMPTS=1" -e "MAX_NEW=8" -e "WARMUP=0" \
    "$P7CT" bash -lc 'cd /src/exllamav3 && nice -n 10 python3 -u qmesh/mem_budget.py' \
    2>&1 | tee -a "$logf" >> "$LOG"
  local rc=${PIPESTATUS[0]}
  say "PF32K tag=$tag rc=$rc gx=[$(gx_mem)]"
  grep -aE '^P7_LOAD_OK|^MEMB device (after_load|avail)|^PREFILL_ARM |^PREFILL_ARMS_DONE' "$logf" |
    cut -c1-220 | sed "s/^/DIRECT $tag   /" | tee -a "$LOG"
  return $rc
}


cleanup() {
  local rc=$?
  trap - EXIT INT TERM       # a signal must not run this twice, or resume after it
  stop_server || true
  # stop_server asks nicely, which a server inside a 113 GiB load ignores; this is the KILL that
  # keeps a cancelled ticket from leaving the box at 676 MiB (11:16). Anchored so it cannot match
  # the `sh -c` that spawned the server.
  timeout 12 ssh -o BatchMode=yes "$GX" \
    "docker exec $CT sh -c 'pkill -KILL -f \"^python3 -u qmesh/worker_seam_server.py\" || true'" \
    >/dev/null 2>&1
  if [ "${CX_GATE:-take}" = own ]; then
    # The caller (seam_direct_ab.sh) holds the gate and owns the pair's server for both arms:
    # two arms in one window is the whole point, so this script must not hand the box back
    # between them.
    say "DIRECT rc=$rc arm=${CX_ARM_TAG:-?} evidence=$E (gate left to the caller)"
    exit $rc
  fi
  pair_gx_gate_down          # pair_serve takes the gate itself, so mine has to be down first
  # If this ticket stopped the pair's server it is the one that reloads it, from the release, with
  # pair_serve (which takes its own gate, so mine is already down here). Leaving it down is what
  # turned one stop at 07:53 into two windows of churn; GxPort2's queued ServerUp no-ops when it
  # finds a worker up, so this cannot double a load.
  # Never start a fresh 113 GiB load on the way out of a signal: the queue KILLs the group 30 s
  # after the TERM, and a restore that gets cut in half is a worse mess than a pair with no server.
  if [ "$rc" -lt 128 ]; then
    cx_pair_server_restore "$E" "$STAMP" || true
  else
    say "DEV_SERVER restore skipped on rc=$rc: leaving the box clear for the next ticket"
  fi
  say "DEV_SERVER pair_stopped=${CX_PAIR_RESTORE:-0} left=$([ "${CX_PAIR_RESTORE:-0}" = 1 ] && echo restored || echo untouched) gx=[$(gx_mem)]"
  say "DIRECT rc=$rc arm=${CX_ARM_TAG:-?} evidence=$E"
  exit $rc
}
trap cleanup EXIT INT TERM

# A ticket that loads 116 GiB owns the box for its whole life, so the claim comes first: the stop of
# the pair's server below is only legitimate while this process holds it (cx_pair_server_stop
# refuses otherwise). Unknown arguments are refused -- `--dry` once meant nothing here and ran for real.
[ $# = 0 ] || { echo "usage: $0   (no arguments: every knob is an env var, see the header)" >&2; exit 2; }
if [ "${CX_GATE:-take}" = own ]; then
  say "gate held by the caller ($(pair_gx_gate_file)); arm=${CX_ARM_TAG:-?}"
else
  status 42 "taking the gx2 gate for two --fresh direct-pack loads"
  pair_gx_gate_take SeamRobust-direct-pack 45 20 || { say "DIRECT FAIL gate_unresolved"; exit 3; }
  say "gate file=$(pair_gx_gate_file) owner=SeamRobust-direct-pack"
  cx_pair_server_stop | sed "s/^/DIRECT pair   /" | tee -a "$LOG"
  stop_server || { say "DIRECT FAIL the box never came free before the run"; exit 3; }
fi
say "script=$0 sha=$(sha256sum "$0" | cut -c1-16) arm=${CX_ARM_TAG:-?} ranked=$RANKED server=$SERVER_PLACEMENT client=${CLIENT_PLACEMENT:-<manifest>}"
# The container's tree is a copy, not a share: commit on P7 and the GX keeps running yesterday's
# server. Prove by sha that the container sees this commit -- a stale worker_seam_server.py numbers
# its cold experts from a different table and nothing in the transport can tell the two numbering
# schemes apart. Proof, not a copy: since the release rollout the container's /src/exllamav3 is a
# symlink into <pair root>/current/src, and rsyncing a different commit into a release directory makes
# pair_check report PAIR_DRIFT, which makes the next release prepare refuse to run.
# WT has to exist before the status line reads it: under `set -u` the old order -- status, then
# WT=${CX_WT:-...} -- killed the arm with "WT: unbound variable" 9 s in, after the gate was taken and
# the tree had already been verified, with nothing loaded and no evidence produced.
WT=${CX_WT:-.}
status 44 "verifying the container's tree equals $WT"
SRC="$WT" GX="$GX" CX_CT="$CT" bash "$HERE/cx_verify_tree.sh" 2>&1 | tee -a "$E/verify_$STAMP.log" >> "$LOG"
stg=${PIPESTATUS[0]}
say "VERIFY rc=$stg $(grep -ac '^VERIFY ' "$E/verify_$STAMP.log") lines from cx_verify_tree.sh"
[ "$stg" = 0 ] || { say "DIRECT FAIL the container is not running this commit; nothing it prints would be evidence"; exit 13; }
# The census reads the host kernel's slab through the container, so the helper has to be there
# before the first sample; a failed ship is reported, not fatal, because the memory numbers the arm
# already takes (gx_mem, the server's own lines) are still evidence.
if scp -q -o BatchMode=yes "$HC_SH" "$GX:/tmp/qm_host_census.sh" 2>/dev/null; then
  say "SLAB_SHIP ok sha=$(ssh -o BatchMode=yes "$GX" 'sha256sum /tmp/qm_host_census.sh' 2>/dev/null | cut -c1-16)"
else
  say "SLAB_SHIP failed src=$HC_SH: every SLAB line will say unavailable"
fi
# The sampler ships the same way: a missing helper costs the trajectory, not the arm.
if scp -q -o BatchMode=yes "$SAM_SH" "$GX:/tmp/qm_gx_sampler.sh" 2>/dev/null; then
  say "SAMPLE_SHIP ok sha=$(ssh -o BatchMode=yes "$GX" 'sha256sum /tmp/qm_gx_sampler.sh' 2>/dev/null | cut -c1-16)"
else
  say "SAMPLE_SHIP failed src=$SAM_SH: no SAMPLE_START/SAMPLES line will appear"
fi
avail=$(gx_mem | cut -d' ' -f1)
# A client on the other cold-expert table must be refused at the Hello and the server must survive
# it. This is the only place the two tables are deliberately mixed: a --stub server in this same
# container (seconds, no pack, no CUDA) plus the shipping refusal check pointed at the other file.
# It runs on the free box, after the pair's own server is stopped, so nobody pays for it.
if [ "${CX_REFUSAL:-1}" = 1 ]; then
  status 41 "refusal: a client on the other table is refused without killing the server"
  # Against whichever table this ticket is *not* on; an explicit two-sided run can override it.
  refcli=${CX_REFUSAL_CLIENT_PLACEMENT:-}
  [ -n "$refcli" ] || case "$SERVER_PLACEMENT" in
    *ranked*) refcli=/cold/placement.exl3.json ;;
    *)        refcli=/cold/placement.exl3.ranked.json ;;
  esac
  CX_CT="$CT" GX="$GX" CX_SERVER_PLACEMENT="$SERVER_PLACEMENT" \
    CX_CLIENT_PLACEMENT="$refcli" \
    bash "$WT/qmesh/tests/refusal_stub.sh" 2>&1 | tee -a "$LOG"
  rrc=${PIPESTATUS[0]}
  [ "$rrc" = 0 ] || { say "DIRECT FAIL a client on the other table was not refused cleanly (rc=$rrc)"; exit 12; }
fi

# The oracle first: two real layers built both ways, byte for byte. It holds no pack and finishes in
# ~2 min, so it rides in the same gate instead of costing its own window; a mismatch stops the loads
# here, before a 116 GiB pack is spent on a path the oracle just contradicted.
if [ "${CX_ORACLE:-1}" = 1 ]; then
  status 40 "from_pack byte oracle layers ${CX_ORACLE_LAYERS:-5,39} (no pack)"
  ol=$E/oracle_$STAMP.log
  timeout 900 ssh -o BatchMode=yes "$GX" \
    "docker exec -e PYTHONPATH=/src/exllamav3 -e OMP_NUM_THREADS=8 -e SEAM_PLACEMENT=$SERVER_PLACEMENT -e SEAM_MODEL=${SEAM_MODEL_DIR:-/cold} $CT bash -lc 'cd /src/exllamav3 && python3 -u qmesh/tests/cx_direct_check.py --layers ${CX_ORACLE_LAYERS:-5,39} --margin-mib 7000'" \
    2>&1 | tee -a "$ol" >> "$LOG"
  orc=${PIPESTATUS[0]}
  grep -aE '^CXD_(OK|MISMATCH|NO_|DONE)' "$ol" | sed "s/^/DIRECT oracle   /" | tee -a "$LOG"
  okc=$(grep -ac '^CXD_OK' "$ol")
  bad=$(grep -ac '^CXD_MISMATCH' "$ol")
  want=$(printf '%s\n' "${CX_ORACLE_LAYERS:-5,39}" | tr ',' ' ' | wc -w)
  say "ORACLE rc=$orc ok=$okc mismatch=$bad want=$want gx=[$(gx_mem)]"
  { [ "$orc" = 0 ] && [ "$bad" = 0 ] && [ "$okc" = "$want" ]; } ||
    { say "DIRECT FAIL the oracle does not reproduce the module layout; the loads are not evidence"; exit 10; }
fi
for n in $(seq 1 "$LOADS"); do
  # The oracle's shard reads, and the previous iteration's unload, both leave a few GiB the box
  # needs a moment to give back (08:02, load1 refused five seconds after the oracle exited;
  # 09:11, the server's own preflight refused at 118082 for want of 118268).
  stop_server || { say "DIRECT FAIL box not clean before load$n"; exit 3; }
  host_census "before_load$n"
  sampler_start "load$n"
  status $(( 36 - n * 12 )) "load $n/$LOADS --fresh, SEAM_MOE_DIRECT=1"
  for attempt in 1 2; do
    PAIR_SCRIPTS="$PAIR_SCRIPTS" SEAM_ENV="$SEAM_ENV_DIRECT" GX="$GX" CT="$CT" SEAM_GATE_CT="$SEAM_GATE_CT" \
    GX_DEV=${CX_GX_DEV:-rocep1s0f0} GX_GID="$GX_GID" SEAM_PLACEMENT="$SERVER_PLACEMENT" \
    SEAM_LOG=/tmp/seam_server.log SERVER_BUCKETS="$BUCKETS" SEAM_SLAB_SLOTS=1 \
      bash -c '. "$PAIR_SCRIPTS/seam_server_ensure.sh"; seam_server_ensure --fresh --buckets '"$BUCKETS" \
      > "/tmp/direct_load$n.ensure.log" 2>&1
    rc=$?
    sed "s/^/DIRECT load$n   /" "/tmp/direct_load$n.ensure.log" | tail -6 | tee -a "$LOG"
    [ "$rc" = 0 ] && break
    # Only a preflight that refused for want of memory is worth a second try: a missing pack, a
    # dead container or a bad argument would fail the same way, and waiting 6 min for it wastes
    # the ticket. The preflight runs before the load, so nothing has been committed by it.
    grep -q 'SEAM_NO_MEMORY' "/tmp/direct_load$n.ensure.log" || break
    say "load$n preflight refused for want of memory (attempt $attempt); waiting for the box"
    stop_server || break
  done
  [ "$rc" = 0 ] || { say "DIRECT FAIL load$n ensure_rc=$rc"; exit 5; }
  server_log "load$n" || exit 6
  # Before any number is attributed to this load: was the server started at the shape this arm
  # asked for? A 256-row label on a 1024-row server is silent everywhere else.
  shape_assert "load$n"
  host_census "after_load$n"
  sampler_once "after_load$n"
  say "DET load$n server_deterministic=$(det_env | tr -d '\n' | sed 's/^$/0/') client=${CX_DETERMINISTIC:-0} ws_rows=${CX_WS_ROWS:-1024}"
  status $(( 34 - n * 12 )) "load $n/$LOADS ids arm x$IDS_RUNS (rows=1 new=8)"
  ids_runs "load${n}r"
  host_census "after_decode$n"
  sampler_once "after_decode$n"
  # The 32k prefill is the rate acceptance number, but it is also a 3 GiB transient that has to page
  # in against the resident pack, so an arm that measures *determinism* alone must be able to skip
  # it: a digest comparison whose loads differ only by the prefill having run is a weaker test for
  # half again the time. CX_PREFILL=0 skips both the pass and its completeness check; the SLACK
  # line then says prefill=0, so a reader can not mistake the un-paged headroom for the post-prefill
  # one that the 256 MiB floor is applied to.
  if [ "${CX_PREFILL:-1}" = 1 ]; then
    status $(( 31 - n * 12 )) "load $n/$LOADS 32k prefill, 8192-row chunks"
    prefill_arm "load$n"
    # The rate the references were taken at is an acceptance number, so a missing measurement is a
    # failure, not a footnote: the arm is bounded (DEPTH_PROBES="" above, timeout 1500) and prints
    # PREFILL_ARMS_DONE as soon as the 32k pass is in.
    grep -aq '^PREFILL_ARMS_DONE' "$E/pf32k_load$n.log" &&
      grep -aq '^PREFILL_ARM depth=32768 ' "$E/pf32k_load$n.log" ||
      { say "DIRECT FAIL load$n the 32k prefill did not complete; see $E/pf32k_load$n.log"; exit 14; }
    host_census "after_prefill$n"
    sampler_once "after_prefill$n"
  fi
  sampler_stop "load$n" "load$n"
  # Headroom with the pack resident, which is the number the workspace geometry is bought with:
  # MemAvailable on the GX while its server is up, and whether that same load paged.
  read -r av sw < <(gx_mem)
  wsw=$(grep -ac '^SEAM_SERVER_SWAP_WARN' "$E/server_load$n.log")
  say "SLACK load$n avail_mib=${av:-?} swap_free_mib=${sw:-?} ws_rows=${CX_WS_ROWS:-1024} prefill=${CX_PREFILL:-1} swap_warn=$wsw"
  status $(( 28 - n * 12 )) "load $n/$LOADS done, stopping my server"
  stop_server || { say "DIRECT FAIL server did not release the box after load$n"; exit 7; }
done

status 2 "verdict"
RUNS=(); DIG=()
for n in $(seq 1 "$LOADS"); do
  for r in $(seq 1 "$IDS_RUNS"); do
    RUNS+=("load${n}r${r}")
    DIG+=("$(grep -ao 'sha256=[0-9a-f]*' "$E/ids_load${n}r${r}.log" 2>/dev/null | head -1 | cut -d= -f2)")
  done
done
say "DIGESTS$(for i in "${!RUNS[@]}"; do printf ' %s=%s' "${RUNS[$i]}" "${DIG[$i]:-none}"; done) want_prefix=$WANT_DIGEST"
# Two client runs against one loaded server, differing at one position, is the whole reason this
# is a set and not a constant: report the position and both ids rather than a bare hash mismatch.
# The ids lists are in the json, so the comparison is over the tokens themselves.
if [ "${#RUNS[@]}" -ge 2 ]; then
  python3 - "$E" "${RUNS[@]}" <<'PY' 2>&1 | sed 's/^/DIRECT /' | tee -a "$LOG"
import json, os, sys
d, runs = sys.argv[1], sys.argv[2:]
ids = {}
for r in runs:
    p = os.path.join(d, f"ids_{r}.json")
    try:
        with open(p) as fh:
            a = json.load(fh).get("arms", {})
    except (OSError, ValueError):
        continue
    for v in a.values():
        if v:
            ids[r] = v
            break
shown = [r for r in runs if r in ids]
if len(shown) < 2:
    print(f"IDSET NOTE only {len(shown)} of {len(runs)} runs left a json to compare")
else:
    base = shown[0]
    for r in shown[1:]:
        a, b = ids[base], ids[r]
        at = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        if at is None:
            print(f"IDSET {base}=={r} n={len(a)}")
        else:
            print(f"IDSET {base}!={r} first_diff_pos={at} {base}={a[at]} {r}={b[at]} "
                  f"n={len(a)}/{len(b)}")
PY
fi
ndistinct=$(printf '%s\n' "${DIG[@]}" | sort -u | wc -l)
say "IDSET runs=${#RUNS[@]} distinct=$ndistinct"
[ "$ndistinct" = 1 ] || say "IDSET NOTE the decode is not bitwise reproducible in this window: $ndistinct outcomes"
for n in $(seq 1 "$LOADS"); do
  dl=$(grep -ac '^SEAM_MOE_DIRECT layer=' "$E/server_load$n.log")
  [ "${dl:-0}" -ge 40 ] || { say "DIRECT FAIL load$n direct_layers=$dl"; exit 8; }
  [ "$(grep -ac '^SEAM_MOE_DIRECT_FAIL' "$E/server_load$n.log")" = 0 ] || { say "DIRECT FAIL load$n fell back"; exit 8; }
done

if [ "$RANKED" = 1 ]; then
  # The ranked table is only worth anything if the server loaded every layer in *its* order: one
  # SEAM_PLACEMENT_GROUPS line per direct layer, the 4-bit group size equal to the file's
  # warm_4bpw_n, the 4-bit experts first, and the same numbering on every load.
  for n in $(seq 1 "$LOADS"); do
    g=$E/server_load$n.log
    grep -aq '^SEAM_PLACEMENT .*ranked=1' "$g" || { say "DIRECT FAIL load$n served an unranked table"; exit 11; }
    grp=$(grep -ac '^SEAM_PLACEMENT_GROUPS layer=.* ok=1$' "$g")
    [ "${grp:-0}" = 40 ] || { say "DIRECT FAIL load$n group_checks=$grp of 40"; exit 11; }
    grep -aq '^SEAM_PLACEMENT_GROUPS layer=.* MISMATCH' "$g" &&
      { say "DIRECT FAIL load$n the shard and the file disagree on a layer's 4-bit boundary"; exit 11; }
    read -r cw ck c4 < <(awk '/^SEAM_PLACEMENT_GROUPS layer=/ {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^n=/)           { split($i, a, "="); n  += a[2] }
          if ($i ~ /^warm_4bpw_n=/) { split($i, a, "="); k  += a[2] }
          if ($i ~ /^groups4=/)     { split($i, a, "="); g4 += a[2] } } }
        END { print n + 0, k + 0, g4 + 0 }' "$g")
    say "RANKED load$n cold_total=$cw warm_4bpw_total=$ck shard_4bpw_total=$c4"
    [ "$ck" = "$c4" ] || { say "DIRECT FAIL load$n 4-bit boundary total $ck != shard $c4"; exit 11; }
    grep -a '^SEAM_PLACEMENT_GROUPS layer=' "$g" |
      sed -E 's/.*layer=([0-9]+) n=([0-9]+) warm_4bpw_n=([0-9]+).*/\1 \2 \3/' > "/tmp/ranked_l${n}_$STAMP.txt"
  done
  if [ "$LOADS" -ge 2 ]; then
    cmp -s "/tmp/ranked_l1_$STAMP.txt" "/tmp/ranked_l2_$STAMP.txt" ||
      { say "DIRECT FAIL the loads did not number their cold experts identically"; exit 11; }
  fi
  # The client's half of the same table: the Hello's sha against the sha of the file it loaded.
  # The server advertises 12 hex and the client prints 16, and the client's own check is a prefix
  # match on purpose (it cannot know how much of the sha the server chose to send), so this
  # comparison has to be the same prefix rule -- an equality test here failed a correct run.
  for n in $(seq 1 "$LOADS"); do
    cl=$(grep -a '^SEAM_PLACEMENT server_sha=' "$E/ids_load${n}r1.log" | head -1)
    say "RANKED load$n client ${cl:-no SEAM_PLACEMENT line}"
    set -- $(printf '%s\n' "$cl" | sed -E 's/.*server_sha=([0-9a-f]*).*client_sha=([0-9a-f]*).*/\1 \2/')
    ss=${1:-}; cs=${2:-}
    case "$cs" in
      "$ss"*) [ -n "$ss" ] || { say "DIRECT FAIL load$n the client printed no sha"; exit 11; } ;;
      *) say "DIRECT FAIL load$n client and server disagree on the table (server=${ss:-unset} client=${cs:-unset})"; exit 11 ;;
    esac
  done
fi
for i in "${!RUNS[@]}"; do
  [ -n "${DIG[$i]:-}" ] || { say "DIRECT FAIL ${RUNS[$i]} left no digest to compare"; exit 9; }
done
# CX_WANT_DIGEST=any asks only that every run left *a* digest, and leaves the meaning to the caller.
# The determinism ticket needs it: CUDA_EXL3_DETERMINISTIC=1 routes the down-gemm through
# exl3_moe_combine instead of fp32 atomics, which may well land on a third digest that the default
# path never produced -- that is not a regression, and a membership test would have reported it as
# one before the cross-load comparison could run.
if [ "$WANT_DIGEST" != any ]; then
  for i in "${!RUNS[@]}"; do
    hit=""
    for w in $WANT_DIGEST; do
      case "${DIG[$i]:-}" in "$w"*) hit=$w ;; esac
    done
    [ -n "$hit" ] || { say "DIRECT FAIL ${RUNS[$i]} digest ${DIG[$i]:-none} is in neither of $WANT_DIGEST"; exit 9; }
  done
  case "$(printf '%s\n' "${DIG[@]}")" in
    *"$WANT_ANY"*) ;;
    *) say "DIRECT FAIL none of the ${#RUNS[@]} runs produced the canonical digest $WANT_ANY"; exit 9 ;;
  esac
  say "DIRECT PASS: $LOADS --fresh load(s), ${#RUNS[@]} client run(s), 40/40 on cudaexl3, from_pack every layer, ids in {$WANT_DIGEST} incl $WANT_ANY"
else
  say "DIRECT PASS: $LOADS --fresh load(s), ${#RUNS[@]} client run(s), 40/40 on cudaexl3, from_pack every layer, ids {$(printf '%s ' "${DIG[@]}")}"
fi
