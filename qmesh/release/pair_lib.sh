#!/usr/bin/env bash
# pair_lib.sh -- shared helpers for the pair release/update system.
# Sourced, never executed. Every script in qmesh/release sources it so that the pair layout, the ssh
# and docker quoting and the release identity live in exactly one place.
#
# Nothing here is allowed to guess: a missing pair, a missing manifest or an unreachable box is an
# error, never a default.

PAIR_HOME=${PAIR_HOME:-$HOME}
PAIR_REPO=${PAIR_REPO:-$PAIR_HOME/recipe-gpu0-gx1-v2}
PAIR_ART=${PAIR_ART:-$PAIR_HOME/recipe-artifacts/gpu0-gx1}
PAIR_SCRIPTS=${PAIR_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PAIR_STATE=${PAIR_STATE:-$PAIR_ART/pairs}
PAIR_QUEUE=${PAIR_QUEUE:-$PAIR_ART/queue}
PAIR_CONF=${PAIR_CONF:-$PAIR_SCRIPTS/pairs.conf}
PAIR_BRANCH=${PAIR_BRANCH:-tools/pair-release}
PAIR_SSH=${PAIR_SSH:-ssh}
PAIR_DRY=${PAIR_DRY:-0}
PAIR_DRY_SSH=${PAIR_DRY_SSH:-}
PAIR_MOUNT=${PAIR_MOUNT:-/pair}                 # container path of the pair root
PAIR_SRC_LINK=${PAIR_SRC_LINK:-/src/exllamav3}  # container symlink -> /pair/current/src
# EXLLAMAV3_TUNE_CACHE: where exllamav3's coop autotuner keeps its picks (exllamav3_ext/quant/
# coop_autotune.cu disk_cache_path(); unset, it resolves to $XDG_CACHE_HOME or $HOME/.cache/exllamav3/
# autotune *inside the container*). Nothing else in the pair sets it, so those picks land in the
# container's writable layer and die with it -- and the pair re-creates a container rather than never.
# So pin them inside /pair: that is the pair root bind-mounted into both containers (containers/*.json
# mount __PAIR_ROOT__ at /pair) and the one writable place a re-create and a release cut both leave
# alone. The value is the CONTAINER path because both carriers are `docker exec -e`: pair_arm.sh's
# client and pair_server_env() -> seam_server_ensure's worker. Per pair, not shared: /pair is *this*
# pair's root on each box, so the same string is a different file for prod and dev (host
# $PAIR_ROOT/tunecache/coop_autotune_v1.bin on each side). One file per pair is safe to keep across
# releases: the record key is the GEMM shape plus device, compute capability and max-SM count
# (exl3_gemm.cu gemm_autotune_hash) and it is salted with COOP_AUTOTUNE_VERSION (coop_autotune.cu
# salt_hash), so a different card or a bumped search space misses instead of misreading. Pass the file
# and not its directory: the loader appends coop_autotune_v1.bin only when the path already *is* a
# directory, so a path that does not exist yet would silently become the cache file itself. The writer
# creates the parent directory (append_disk_cache_locked -> create_directories), so no mkdir is needed.
# The vendor cold-expert kernel has nothing to pin here: cuda-exl3's MoE gemm keeps its picks in a
# process-local map (vendor/cuda-exl3/src/cuda_exl3/csrc/gemm.cu tune_cache) -- in memory only -- and
# the one persisted vendor tuner (mla_decode.cu, CUDA_EXL3_TUNE_CACHE) is MLA attention, which this
# pair never calls: the worker runs only the exl3_moe_* ops and P7's attention is Triton.
PAIR_TUNE_CACHE=${PAIR_TUNE_CACHE:-$PAIR_MOUNT/tunecache/coop_autotune_v1.bin}
# Triton's JIT cache is the *other* thing that dies with the client container, and on this pair it is
# the expensive one: the first prefill arm after a `docker run` spends 10-13 s compiling the attention
# kernels it is about to launch (measured 2026-11-03: four identical 32k arms in one container, arm 1
# 41.00 s / 799 tok/s with no /root/.triton, arms 2-4 30.1-30.6 s / 1071-1088 tok/s with it, and
# wiping the EXLLAMAV3_TUNE_CACHE file instead changed nothing at all). Triton 3.8 resolves its cache
# from $TRITON_CACHE_DIR (knobs.cache.dir, /root/.triton/cache by default), so one exec-level variable
# pins it; no container recreate is needed. The key is a content hash of the kernel source, the
# specialisation constants, the backend and the arch -- every cache directory's name is that hash and
# carries a json with it (verified on 81 directories of this pair's own cache) -- so a release whose
# kernels changed misses and recompiles instead of reading a stale cubin, and sharing one directory
# across releases is safe. Same shape as PAIR_TUNE_CACHE above: a container path, on the pair's root.
PAIR_TRITON_CACHE=${PAIR_TRITON_CACHE:-$PAIR_MOUNT/tritoncache}
# The third cache a fresh container loses, and the one that turned out to matter most: torch's NVRTC
# kernel cache. `torch.cuda.jiterator` compiles one cubin per elementwise kernel on first call and
# writes it to `$(dirname torch.hub.get_dir())/kernels` -- /root/.cache/torch/kernels by default --
# and until it is there the first prefill arm of a container pays for it. Ablated 2026-11-03: same
# ticket, same arms, Triton and the autotune file pinned in both; seeding only this cubin moved the
# first arm from 42.56 s / 769.9 tok/s to 34.32 s / 954.8 tok/s against a second arm at 28.3-29.6 s.
# TORCH_HOME moves the whole tree (verified in the pair's image: TORCH_HOME=/pair/torchhome gives
# hub=/pair/torchhome/hub and kernels=/pair/torchhome/kernels), so one variable pins it. Exec-level,
# like the two above; no container recreate.
PAIR_TORCH_HOME=${PAIR_TORCH_HOME:-$PAIR_MOUNT/torchhome}
PAIR_LIB=libqmesh_expert_seam                   # qmesh/native/bin/<this>.<arch>.so
PAIR_CPP_LIB=libqmesh_moe_cpp                    # the C moe hook: same bin/, same stamp rule
PAIR_SEAM_FILES=(qmesh/native/qmesh_expert_seam.cu qmesh/native/qmesh_expert_seam.h qmesh/native/qes.py)
PAIR_VERIFY_PATHS=(qmesh exllamav3)             # what "the tree is the commit" covers
PAIR_SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3)

pair_die() { echo "pair: $*" >&2; exit 2; }
pair_say() { echo "$(date '+%H:%M:%S') $*"; }
# pair_step: timestamped line, also appended to $PAIR_LOG when set (every job sets it).
pair_step() { say_line=$(printf '%s %s' "$(date '+%H:%M:%S')" "$*"); echo "$say_line"; [ -n "${PAIR_LOG:-}" ] && echo "$say_line" >> "$PAIR_LOG"; return 0; }

pair_load() {  # <prod|dev> -> PAIR P7CT GPU SEAM_DEV SEAM_GID GX_GID SEAM_PEER GX GXCT PAIR_ROOT PORT BUCKETS P7_ARCH GX_ARCH LOAD_MARGIN
  local p=${1:?usage: pair_load <pair>} conf=${2:-$PAIR_CONF} line
  [ -r "$conf" ] || pair_die "config not readable: $conf"
  line=$(awk -v p="$p" '!/^#/ && NF && $1==p {print; exit}' "$conf")
  [ -n "$line" ] || pair_die "pair '$p' is not in $conf"
  read -r PAIR P7CT GPU SEAM_DEV SEAM_GID GX_GID SEAM_PEER GX GXCT PAIR_ROOT PORT BUCKETS P7_ARCH GX_ARCH LOAD_MARGIN WS_ROWS <<<"$line"
  PAIR_ROOT=${PAIR_ROOT_OVERRIDE:-$PAIR_ROOT}
  # A relative root puts every release, log and stamp wherever the caller happened to stand: a
  # stubbed harness once grew four full source trees inside that repository way.
  case "$PAIR_ROOT" in
    /*) ;;
    *) pair_die "pair root must be absolute: '$PAIR_ROOT'" ;;
  esac
  # LOAD_MARGIN: what the pack load must leave behind, in MiB. Measured on both GX boxes, the load
  # consumes the pack plus ~1200 MiB and compute needs ~1600 MiB more, so the worker's own default
  # (2048) is a floor, not a promise -- the pair states its number here. GX2 has ~120585 MiB clean,
  # so 3072 passes by ~1150 MiB and the box still lands at the wall: the margin can only refuse a
  # dirtier box, it cannot make GX2 fit.
  LOAD_MARGIN=${LOAD_MARGIN:-2048}
  # WS_ROWS: the row piece a server is asked to take per call, pinned by the pair rather than left
  # to a per-image default. 1024 is the measured shape on both loaded GX boxes (512-row pieces cost
  # -7/-22% and a single 4096 costs 224-326 ms p50 with 3-5 s tails), and the picker that would
  # choose it adapts by itself only when a box has ~1 GiB of headroom, which neither has. "auto"
  # hands the decision back to whatever server binary the release ships.
  WS_ROWS=${WS_ROWS:-1024}
  if [ "$WS_ROWS" = auto ]; then
    WS_ENV=""
  else
    # AUTO=0 is the other half of the pin: without it a picker may widen the piece under memory
    # pressure on its own, and the pair would no longer run the shape its numbers were taken on.
    WS_ENV="SEAM_WS_ROWS=$WS_ROWS SEAM_WS_ROWS_AUTO=0"
  fi
  # MOE_ENV: the kernel the GX half serves the cold experts on. cuda-exl3 restacks a layer's cold
  # experts per bitrate group and is preloaded while the pack loads; without it (a release cut
  # without the vendored extension built -- its .so is untracked, so every cut needs
  # qmesh/build_cudaexl3.sh) the worker prints SEAM_MOE_NOEXT and every layer stays on the mixed-K
  # module, at a fraction of the measured rate. The smoke refuses to call such a release verified.
  # SEAM_CX_MARGIN_MIB is what must stay free after the load for the restack + workspace (the
  # worker's own default; the validated numbers were taken with it).
  # PAIR_SRV_EXTRA_ENVS: the server half's A/B hook, the counterpart of PAIR_ARM_EXTRA_ENVS. The
  # knobs that decide what the GX half *does* -- SEAM_MOE_CPP=1 (serve cold experts from the C hook
  # instead of the Python piece) is the one this was added for -- are read by worker_seam_server.py
  # and never by the P7 client, so an arm that sets them in the client's env measures nothing. Set
  # through the ensure's SEAM_ENV so it reaches the worker's own process, and it must stay EMPTY for
  # any number anyone quotes as the shipped shape: a value here is a different server.
  # SEAM_MOE_CPP ships ON: the C hook served all 40 layers (fallback=[]) and measured 113.98 ->
  # 77.19 ms/tok at rows=1 on one release, one box, back-to-back (04:59/05:14Z 2026-11-03, bca2373b,
  # ws_rows=1024, prefill 769.0 -> 855.1 tok/s), with identical ids (0c9bbe87, same=1 diff=0) and
  # identical text (58051d96) on both arms. It is fail-open: a missing .so or a failed registration
  # prints SEAM_MOE_CPP_OFF and the worker serves the Python path. MOE_CPP=0 puts the Python path back
  # for an A/B; PAIR_SRV_EXTRA_ENVS still appends afterwards, so it can override either way.
  # MOE_FUSED: SEAM_MOE_CPP_FUSED=1 ships ON from 2026-11-04. Three same-ticket base->fused pairs on
  # 3cd80749 (warm-up discarded, order reversed, k35, shipped table untouched) read 61.74 -> 64.74
  # tok/s (+4.9 % mean; +5.3 %, +7.4 %, +1.8 %), one w6 verify step 52.99 -> 50.25 ms (-5.1 %,
  # accepts 3.043 -> 3.110 draft tokens per step), and paired wiki2 quick8 2.0436 -> 2.0406 (-0.0030,
  # inside the +/-0.005 gate); every fused arm reported SEAM_MOE_CPP_FUSED_STATS fused=100.0 with an
  # empty fallback list. It is a single launch per projection instead of one per group, so it changes
  # *timing*, not arithmetic: the fallback is byte-identical and `=2` refuses a fallback instead of
  # taking it. MOE_FUSED=0 restores the per-group path for an A/B. The box drifts +-5 % load to load
  # (55.48/58.38/58.48 on one identical base read), so quote the means, never a single pair.
  # MOE_SIDE: SEAM_MOE_CPP_SIDE=1 ships ON from 2026-11-04 (GXProfile 5a786daf). The response D2H of
  # piece p is enqueued on a second stream behind a completion marker of the compute stream, so it
  # crosses the bus while piece p+1's copies and gemms run; the accumulator is double buffered
  # (slot piece&1, stride cap_rows so a short last piece cannot alias), side_done per slot orders the
  # third piece's first write after the first piece's copy, and ev_side_done joins the side stream
  # back onto the compute stream before the call's single sync -- so the slot is complete when the
  # call returns. Same bytes, same buffer, same slot; only the stream carrying the copy changes, and
  # the profile arm reported gen_ids MATCH. With SEAM_MOE_CPP_OUT=bf16|fp16 there is one out16 slot a
  # flying copy would be reading, so the code prints `SEAM_MOE_CPP_SIDE ignored: ...` and runs
  # single-stream -- it does not refuse the call (an earlier version of this comment said "refused").
  # This pair does not set SEAM_MOE_CPP_OUT. Device evidence: write 3.527 ->
  # 0.010 ms/layer, dev 117.176 -> 113.359 ms/layer (-3.26 %) at 1024-row pieces; the rate jump in
  # that ticket is not attributable (page-in on the first arm).
  # DEFAULT 0 from 2026-11-04, not 1: the merged side path stalled a 32k prefill on 0df75a38 at
  # `qes_issue layer=11 rows=8192 ... every slot is still unacknowledged after 30 s` (first prefill of
  # the same build passed, so it is a race in the side code and not a size limit). SIDE is parked until
  # that is understood, so every pair -- prod included -- runs the single-stream copy unless a ticket
  # asks for MOE_SIDE=1 explicitly. This is also why the shipped candidate is f9d57851, whose tree has
  # no SIDE code at all.
  MOE_SIDE=${MOE_SIDE:-0}
  MOE_CPP=${MOE_CPP:-1}
  MOE_FUSED=${MOE_FUSED:-1}
  # MOE_DET: the cold-expert epilogue the GX half runs. 0 -- the C hook's own default,
  # qmesh/native/qmesh_moe_cpp.cpp:997 -- folds the routing weight into the down gemm's fp32
  # atomicAdd, so "the answer depends on the order the blocks retire in (and on block_m), and it is
  # neither reproducible run to run" (same file, :18-21). Two rows of one batch are then two
  # different retirement orders, so identical prompts in a batch answer differently from each other
  # and from the same prompt served alone; that is the whole of the BATCH>1 split seen on 2026-11-05
  # (b8probe-20261105T050207Z, b2arms-20261105T061623Z), and it is why a run of the *same* code
  # either split or did not depending on how many near-ties the stream happened to cross.
  # 1 runs the deterministic epilogue: the down stage writes one unweighted row per routed row (no
  # two blocks touch an element) and exl3_moe_combine reads the top_k rows back in a fixed k order,
  # with one fp32 fold per group -- "no atomics, no scratch, deterministic" (hadamard.cu:218).
  #
  # INSTALLED 2026-11-05 07:52 (BatchFix): the worker's epilogue follows the batch width the *client*
  # is started with, so `BATCH=8` cannot reach a pair with the folded epilogue by omission.
  # `serve_pair.py` refuses BATCH>1 unless its own SEAM_MOE_CPP_DET=1, and the client half of that line
  # is pair_arm.sh / pair_server.sh, which emit SEAM_MOE_CPP_DET=1 whenever this variable is 1. One knob,
  # both halves. At BATCH<=1 the value stays 0 and the variable is not emitted, so the shipped
  # single-stream env is byte-identical to what every pack number was taken on.
  # The width is read from PAIR_SRV_EXTRA_ENVS only -- that is the string docker receives last and the
  # only one that reaches the client (BATCH is in neither shipping list). The ticket's own $BATCH is
  # deliberately NOT consulted: a stray `BATCH=8` left in an operator's shell would otherwise load the
  # worker on the deterministic epilogue while the client ran BATCH=1, i.e. move the shipped single
  # stream off the pack baseline silently.
  # The install condition is a conjunction, not the single `D1` line this comment used to name: det
  # alone (no admission window) splits -- `D1` b2=0/3 b8=0/2 -- and so does the window alone on the
  # folded epilogue (`A0`, b2=0/3 b8=0/2). Only both together are clean: `A1` b2=3/3 b8=2/2, and 5/5 reps
  # `same=all` with the window at 30, 60 and 100 ms, worker environ `det=1 want=1` throughout.
  # SEAM_COPY_STREAM is exonerated -- `D0s` (det off, window on, copies on the caller's stream) split
  # too -- so the seam ordering fix in remote_seam_host.py is not what the arms turned on; it is in the
  # shipping cut on the strength of its mechanism (§4.3 of notes/BatchFix.md), not of an arm.
  # Cost at width 8, pack 4460, ms/token/user, two reps per arm unless noted: A1 det=1 window on
  # 82.73-83.48 (eight reps over the three ceilings), A0 det=0 window on 83.43/85.29, D0s det=0 window
  # on with copies on the caller's stream 89.88/90.05, D1 det=1 window off 90.48/92.07, D0 det=0 window
  # off 87.64/90.94. Read: det costs nothing measurable with the window on (equal to or faster than det
  # off, inside the run-to-run spread) and about 2 ms with it off; the levers are the window and the
  # default copy stream, not the epilogue. The old "~5 % decode, DET refuses the fused stages
  # (:729-735)" pricing is stale: qmesh_moe_cpp.cpp:736-741 says DET keeps both fused stages, the down
  # stage writing unweighted routed rows for the pick-order combine, "deterministic by construction".
  # Determinism tickets can still force either value per arm.
  local _bw _kv
  # Per-pair serving default (pairs/<pair>.serve.env, Main 2026-11-08): `PAIR_SRV_EXTRA_ENVS=<line>` and optionally
  # `MOE_DET=<0|1>`. Applies only when the caller set neither, so A/B tickets keep full control. prod: BATCH=8 on R4's
  # layer path (MOE_DET=0; tournament adaptive-batch/tourney/20261108T142717Z-r4).
  local _se="$PAIR_STATE/$PAIR.serve.env"
  if [ -z "${PAIR_SRV_EXTRA_ENVS+x}" ] && [ -r "$_se" ]; then
    PAIR_SRV_EXTRA_ENVS=$(sed -n 's/^PAIR_SRV_EXTRA_ENVS=//p' "$_se" | tail -1)
    [ -z "${MOE_DET:-}" ] && MOE_DET=$(sed -n 's/^MOE_DET=//p' "$_se" | tail -1)
    export PAIR_SRV_EXTRA_ENVS
  fi
  _bw=1
  for _kv in ${PAIR_SRV_EXTRA_ENVS:-}; do
    case "$_kv" in BATCH=*) _bw=${_kv#BATCH=} ;; esac
  done
  case "$_bw" in ''|*[!0-9]*) _bw=1 ;; esac
  [ "$_bw" -ge 2 ] || _bw=1
  if [ -z "${MOE_DET:-}" ]; then
    if [ "$_bw" -gt 1 ]; then MOE_DET=1; else MOE_DET=0; fi
  fi
  SRV_ENV_EXTRA=${PAIR_SRV_EXTRA_ENVS:-}
  CX_MARGIN_MIB=${CX_MARGIN_MIB:-1536}
  # No $SRV_ENV_EXTRA here: that was a copy of PAIR_SRV_EXTRA_ENVS taken at source time, i.e. before any
  # ticket had a chance to export it, and it also carried the client's own knobs (BATCH, EXL3_MOE_*) onto
  # the GX. pair_server_env() reads the variable when it is called and forwards only SEAM_*.
  MOE_ENV="SEAM_MOE_KERNEL=cudaexl3 SEAM_MOE_PRELOAD=1 SEAM_CX_MARGIN_MIB=$CX_MARGIN_MIB SEAM_MOE_CPP=$MOE_CPP SEAM_MOE_CPP_FUSED=$MOE_FUSED SEAM_MOE_CPP_SIDE=$MOE_SIDE"
  [ "$MOE_DET" = 1 ] && MOE_ENV="$MOE_ENV SEAM_MOE_CPP_DET=1"
  export PAIR P7CT GPU SEAM_DEV SEAM_GID GX_GID SEAM_PEER GX GXCT PAIR_ROOT PORT BUCKETS P7_ARCH GX_ARCH LOAD_MARGIN WS_ROWS WS_ENV CX_MARGIN_MIB MOE_ENV MOE_CPP MOE_FUSED MOE_SIDE MOE_DET SRV_ENV_EXTRA PAIR_TUNE_CACHE
  # PAIR_STATE_OVERRIDE lets the stub harnesses (@see tests/dry_run.sh) keep every log, ref and
  # restore note out of the real pairs/ tree: a dry run must never touch what a real one wrote.
  PAIR_STATE=${PAIR_STATE_OVERRIDE:-$PAIR_STATE}
  PAIR_STATE_DIR=$PAIR_STATE/$PAIR
  PAIR_MANIFEST=$PAIR_STATE/$PAIR.json
  export PAIR_STATE_DIR PAIR_MANIFEST
  # Every server start truncates the container's log, and that log is the only per-load record of
  # which layers the cuda-exl3 restack refused; seam_server_ensure copies it here first. Without it
  # two ids digests can differ across loads with no way to tell whether the layer set differed too
  # (2026-11-02, canary 38/2 vs 39/1).
  SEAM_LOG_ARCHIVE=${SEAM_LOG_ARCHIVE:-$PAIR_STATE_DIR/logs}
  export SEAM_LOG_ARCHIVE
}

# Read current at launch, including after a flip or rollback; this is data, not shell code.
pair_release_env() {
  local file="$PAIR_ROOT/current/src/qmesh/release/shipping.env" line key value seen=' '
  [ -e "$file" ] || return 0
  [ -r "$file" ] || pair_die "release env unreadable: $file"
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    key=${line%%=*}; value=${line#*=}
    case "$key:$value" in
      SEAM_QES_OUT:bf16|SEAM_QES_OUT:f32|SEAM_MOE_CPP_LAYER:0|SEAM_MOE_CPP_LAYER:1) ;;
      *) pair_die "invalid release env line in $file: $line" ;;
    esac
    case "$seen" in *" $key "*) pair_die "duplicate release env key $key in $file" ;; esac
    seen="$seen$key "
    printf '%s\n' "$line"
  done < "$file"
}

# The release's P7 HTTP-server defaults (R5 on): current/src/qmesh/release/serving.env, KEY=VALUE per line, data
# only. pair_server.sh puts them after its base list and before PAIR_SRV_EXTRA_ENVS, so an A/B still overrides any
# key; CACHE_TOKENS is only the default for the caller's CACHE_TOKENS. A release without the file (R4 and older)
# contributes nothing and keeps pairs/<pair>.serve.env as its serving config.
pair_release_serving_env() {
  local file="$PAIR_ROOT/current/src/qmesh/release/serving.env" line key value seen=' '
  [ -e "$file" ] || return 0
  [ -r "$file" ] || pair_die "release serving env unreadable: $file"
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    key=${line%%=*}; value=${line#*=}
    [[ $key =~ ^[A-Z][A-Z0-9_]*$ && $value =~ ^[A-Za-z0-9_./:,-]+$ && $line == *=* ]] \
      || pair_die "invalid release serving env line in $file: $line"
    case "$key" in SEAM_*) pair_die "SEAM_* keys belong in shipping.env, not $file: $line" ;; esac
    case "$seen" in *" $key "*) pair_die "duplicate release serving env key $key in $file" ;; esac
    seen="$seen$key "
    printf '%s\n' "$line"
  done < "$file"
}

 
# The server half's environment as ONE string: the shipping config every number in the notes was
# taken on. Every starter (the smoke, the restore path, an operator) passes this and nothing else, so
# a server can not come up on a different kernel, piece size or tuner cache than the pair serves.
pair_server_env() {  # <build_id> -> "SEAM_BUILD_ID=… SEAM_LOAD_MARGIN_MIB=… EXLLAMAV3_TUNE_CACHE=… <MOE_ENV> <WS_ENV> <SEAM_* extras>"
  # The SEAM_* extras are read HERE and not through MOE_ENV: MOE_ENV froze $SRV_ENV_EXTRA at source
  # time, and every ticket that uses this door (b8_probe_ticket.sh:62, b2_arms.sh:71, b2_trace.sh:65,
  # b2_torder.sh:76) exports PAIR_SRV_EXTRA_ENVS *after* sourcing this file. So the worker was started
  # without SEAM_MOE_CPP_DET on 2026-11-05 while the ticket's own log said it was set, and
  # `grep SEAM_MOE_CPP_DET /proc/<worker>/environ` came back empty on a freshly ensured worker
  # (RM5-b2torder2, det=0 after the load). Only SEAM_* keys go to the worker: everything else in that
  # variable (BATCH, EXL3_MOE_*, NUM_DRAFT) is read by the P7 client and has no business on the GX.
  local extra="" kv defaults
  defaults=$(pair_release_env) || return
  defaults=${defaults//$'\n'/ }
  for kv in ${PAIR_SRV_EXTRA_ENVS:-${SRV_ENV_EXTRA:-}}; do
    case "${kv%%=*}" in SEAM_*) extra="$extra $kv" ;; esac
  done
  printf 'SEAM_BUILD_ID=%s SEAM_LOAD_MARGIN_MIB=%s EXLLAMAV3_TUNE_CACHE=%s %s %s %s%s' "$1" "$LOAD_MARGIN" "$PAIR_TUNE_CACHE" "$MOE_ENV" "$WS_ENV" "$defaults" "$extra"
}

pair_arch_for() {  # <sm_120|sm_121> -> x86_64|aarch64  (lib name per side)
  case "$1" in sm_120) echo x86_64 ;; sm_121) echo aarch64 ;; *) echo "unknown" ;; esac
}
pair_cuda_arch_for() {  # <sm_120|sm_121> -> torch arch list entry
  case "$1" in sm_120) echo 12.0 ;; sm_121) echo 12.1 ;; *) echo "?" ;; esac
}

# ---- remote access ------------------------------------------------------------------------------
# Every GX command goes through pair_gxsh so a dry run can swap the transport (PAIR_DRY_SSH) and so
# no script ever hand-rolls ssh flags. stdin passes through untouched.
# Paths the containers can see. The P7 container mounts the pair root at /pair and the artifacts
# tree at /artifacts (containers/*.json); a host path handed to `docker exec -e` must be
# translated or the file a driver writes "does not exist" from inside.
pair_ct_path() {  # <host path> -> path inside the pair containers
  local p=$1 art=$PAIR_HOME/recipe-artifacts
  case "$p" in
    "$PAIR_ROOT"/*) echo "/pair/${p#"$PAIR_ROOT"/}" ;;
    # the mount is the whole artifacts tree, not the pair's subdirectory: /artifacts/gpu0-gx1/...
    "$art"/*) echo "/artifacts/${p#"$art"/}" ;;
    *) echo "$p" ;;
  esac
}

# ---- the client checkpoint is a property of the cold pack, not of this file ---------------------
# A pair is ONE checkpoint split in two, and the split is the pack. Both halves carry a PACK_MANIFEST.json
# naming the placement they were built for (`placement_sha256`), and only the two halves with the SAME sha
# are one model: the old client with a new pack loads and then dies at the placement preflight, or, worse,
# if the tags happened to line up, serves a mixture of two builds. So the P7 checkpoint to load is read
# from the GX container's own /cold manifest and matched by that sha, never written down here -- the
# 4780p8 switch (2026-11-03) is a mount change, and no script in this tree should be edited for it.
#   pair_cold_src       <gxct>        -> host path behind that container's /cold
#   pair_cold_placement <gxct>        -> placement_sha256 that container's pack declares
#   pair_p7_pool        <p7ct>        -> host path any P7 container mounts at /models/dsv41
#   pair_p7_placements  <p7ct>        -> "<sha>\t<dir>" per hot pack on the P7 side
#   pair_ckpt_auto      <p7ct> <gxct> -> the one P7 hot half with the pack's sha ("" when not exactly one)
#   pair_ckpt_verified  <want|auto>   -> the same, after its placement file hashes to its manifest
#   pair_ckpt_resolve   <want|auto>   -> what to load, or a refusal on stderr
#   pair_ckpt_named     [explicit]    -> the hot half the caller named (arg/PAIR_PACK_P7/AB_CKPT), else ""
#   pair_ckpt_xcheck    <named>       -> PAIR_CKPT_XCHECK line; `auto` logged, never the decision
#   pair_ckpt_require   [explicit]    -> the named half, or PAIR_CKPT_UNNAMED; what consumers call
pair_cold_src() {  # <gxct> -> host path behind /cold ("" when the container has none)
  pair_gxsh "docker inspect -f '{{range .Mounts}}{{if eq .Destination \"/cold\"}}{{.Source}}{{end}}{{end}}' ${1:-$GXCT}" 2>/dev/null | tail -1 | tr -d '\r'
}
pair_cold_placement() {  # <gxct> -> placement_sha256 the pack behind /cold declares
  local gx=${1:-$GXCT} src out
  # The manifest is a plain file on the GX host behind the mount, and `docker inspect` answers for a
  # stopped container, so the answer does not need the container to be up -- nor to be the one whose
  # /cold still matches any later mount change. Read the host path first, the container's view second.
  src=$(pair_cold_src "$gx")
  if [ -n "$src" ]; then
    out=$(pair_gxsh "python3 -c 'import json;print(json.load(open(\"$src/PACK_MANIFEST.json\")).get(\"placement_sha256\",\"\"))'" 2>/dev/null | tail -1 | tr -d '\r')
    [ -n "$out" ] && { echo "$out"; return 0; }
  fi
  pair_gxsh "docker exec $gx python3 -c 'import json;print(json.load(open(\"/cold/PACK_MANIFEST.json\")).get(\"placement_sha256\",\"\"))'" 2>/dev/null | tail -1 | tr -d '\r'
}
pair_p7_pool() {  # [p7ct] -> host path any P7 container mounts at /models/dsv41 ("" when none does)
  local ct src
  for ct in "${1:-$P7CT}" $(docker ps -a --format '{{.Names}}' 2>/dev/null); do
    src=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/models/dsv41"}}{{.Source}}{{end}}{{end}}' "$ct" 2>/dev/null)
    [ -n "$src" ] && { echo "$src"; return 0; }
  done
  return 1
}
pair_p7_placements() {  # <p7ct> -> "<placement_sha256>\t<container path>" for every hot pack that declares one
  local p7=${1:-$P7CT} pool
  # Host first: the pool is a bind mount, so /models/dsv41/exl3-p7hot-*/PACK_MANIFEST.json is readable from
  # this workstation and reading it starts nothing inside the pair's container (a `docker exec` there is not
  # free: the pair's own drift check compares extension state, and a resolver must not be a reason to rebuild).
  # The container's view is the fallback for specs whose /models/dsv41 lives only inside the container.
  pool=$(pair_p7_pool "$p7") || {
    docker exec "$p7" python3 -c 'import glob, json, os
for d in sorted(glob.glob("/models/dsv41/exl3-p7hot-*") + glob.glob("/models/dsv41/*/exl3-p7hot-*")):
    try: m = json.load(open(os.path.join(d, "PACK_MANIFEST.json")))
    except Exception: continue
    sha = m.get("placement_sha256") or ""
    if sha: print(sha, d, sep="\t")' 2>/dev/null | tr -d '\r'
    return 0
  }
  python3 - "$pool" <<'PY2' 2>/dev/null
import glob, json, os, sys
root = sys.argv[1]
for d in sorted(glob.glob(os.path.join(root, "exl3-p7hot-*")) + glob.glob(os.path.join(root, "*", "exl3-p7hot-*"))):
    try:
        m = json.load(open(os.path.join(d, "PACK_MANIFEST.json")))
    except Exception:
        continue
    sha = m.get("placement_sha256") or ""
    if sha:
        print(sha, "/models/dsv41/" + os.path.relpath(d, root), sep="\t")
PY2
}
pair_ckpt_auto() {  # <p7ct> <gxct> -> P7 hot half with the pack's placement_sha256; empty if 0 or >1 match
  local p7=${1:-$P7CT} gx=${2:-$GXCT} sha hits n
  sha=$(pair_cold_placement "$gx")
  [ -n "$sha" ] || return 0
  hits=$(pair_p7_placements "$p7" | awk -F'\t' -v s="$sha" '$1 == s { print $2 }')
  n=$(printf '%s\n' "$hits" | grep -c .)
  # RM10 2026-11-05 (Main): two hot halves on one sha is a refusal, and a loud one -- 08:32-08:40 a second
  # exl3-p7hot-* declaring pack N's sha voided a dev switch, a Coop pass and a ColdSteal window.
  [ "$n" -gt 1 ] && echo "PAIR_CKPT_AMBIGUOUS sha=${sha:0:16} matches=$n dirs=$(printf '%s' "$hits" | tr '\n' ',') -- name the hot half (PAIR_PACK_P7 / AB_CKPT / CHECKPOINT)" >&2
  [ "$n" = 1 ] && printf '%s\n' "$hits"
}
pair_p7_verify() {  # <dir in the container> [p7ct] -> "ok <sha>" | "mismatch <declared> <actual>" | "unverified <why>"
  local name=${1##*/} pool=${PAIR_P7_POOL:-}
  [ -n "$pool" ] || pool=$(pair_p7_pool "${2:-$P7CT}") || { echo "unverified no-pool"; return 0; }
  [ -d "$pool" ] || { echo "unverified no-pool"; return 0; }
  # RM2 2026-11-05 addon: a half that is not a direct child of the pool (the staged
  # exl3-p7hot-<pack>-h6 lives one level down) used to be joined pool/basename, read nothing and
  # report "unverified manifest" -- which pair_ckpt_verified passes through as state=ok, so a
  # named nested half was never actually verified while the run said it was. The pool is a bind
  # mount at the same path the container sees, so an absolute argument is used as given when it
  # holds a manifest here.
  python3 - "$pool" "$name" "$1" <<'PY2' 2>/dev/null || echo "unverified reader"
import hashlib, json, os, sys
pool, name, arg = sys.argv[1], sys.argv[2], sys.argv[3]
d = arg if os.path.isabs(arg) and os.path.isfile(os.path.join(arg, "PACK_MANIFEST.json")) else os.path.join(pool, name)
try:
    m = json.load(open(os.path.join(d, "PACK_MANIFEST.json")))
except Exception:
    print("unverified manifest"); raise SystemExit
sha, pl = m.get("placement_sha256") or "", m.get("placement") or ""
if not sha or not pl:
    print("unverified no-placement"); raise SystemExit
path = pl if os.path.isabs(pl) else os.path.join(d, pl)
try:
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
except Exception:
    print("unverified placement"); raise SystemExit
print("ok " + sha if got == sha else "mismatch " + sha + " " + got)
PY2
}
pair_ckpt_verified() {  # <ckpt|auto> [p7ct gxct] -> prints PAIR_CKPT_VERIFIED …, 1 when the pack is inconsistent
  local want=${1:-auto} p7=${2:-${P7CT:-}} gx=${3:-${GXCT:-}} ckpt state
  ckpt=$(pair_ckpt_resolve "$want" "$p7" "$gx") || return 1
  # A pack is written before it is announced: a half-finished one has a manifest whose placement file
  # does not hash to the sha the manifest declares, and loading it costs a whole arm to find out. The
  # manifest names the placement file itself, so this needs no container -- just the pool on this box.
  state=$(pair_p7_verify "$ckpt" "$p7")
  case "$state" in
    ok*)       echo "PAIR_CKPT_VERIFIED ckpt=$ckpt sha=${state#ok } state=ok"; return 0 ;;
    mismatch*) echo "PAIR_CKPT_VERIFIED ckpt=$ckpt declared=${state#mismatch } state=mismatch" >&2; return 1 ;;
    *)         echo "PAIR_CKPT_VERIFIED ckpt=$ckpt state=${state#unverified }"; return 0 ;;
  esac
}
pair_ckpt_resolve() {  # <want|auto|""> [p7ct gxct] -> checkpoint on stdout; 1 when it cannot be named
  local want=${1:-auto} p7=${2:-${P7CT:-}} gx=${3:-${GXCT:-}} hits n sha src
  case "$want" in
    ""|auto)
      hits=$(pair_ckpt_auto "$p7" "$gx"); n=$(printf '%s\n' "$hits" | grep -c .)
      if [ "$n" = 1 ]; then echo "$hits"; return 0; fi
      sha=$(pair_cold_placement "$gx"); src=$(pair_cold_src "$gx")
      if [ "$PAIR_DRY" = 1 ]; then  # dry run: print the shape that would be used rather than refuse
        echo "/models/dsv41/exl3-p7hot-<pack-placement-${sha:0:8}>"; return 0
      fi
      echo "pair_ckpt_resolve: cannot name the P7 client half for $gx:/cold='${src##*/}'" \
           "placement_sha256='${sha}' matches=$n; the pair is one checkpoint split in two, so the" \
           "client half must declare that sha -- pass CHECKPOINT= / PAIR_PACK_P7= to name it (auto is" \
           "a cross-check since 2026-11-05, not the decision)" >&2
      return 1
      ;;
    *) echo "$want"; return 0 ;;
  esac
}

# ---- the hot half is NAMED by the caller, never derived (RM10 2026-11-05, Main) --------------------
# 08:32-08:40 a second `exl3-p7hot-*` dir declared a live pack's placement sha; `auto` refusing was right,
# and three runs died because every consumer had asked it to decide. The half is named by the caller now --
# its own flag first, then PAIR_PACK_P7 (what the gate and the smoke export), then AB_CKPT (what
# placement_ab_arm.sh and rm_abwindow.sh pass). `auto` still runs, but only as a cross-check line: it never
# decides, and an ambiguity is printed loud by pair_ckpt_auto at the point it is found.
#   pair_ckpt_named   [explicit]          -> the named half, "" when nobody named one
#   pair_ckpt_xcheck  <named> [p7ct gxct] -> one PAIR_CKPT_XCHECK line on stderr; never changes the answer
#   pair_ckpt_require [explicit] [p7ct gxct] -> the named half + that line; 1 (PAIR_CKPT_UNNAMED) when unnamed
pair_ckpt_named() {  # [explicit] -> the hot half the caller named, "" when it named none
  local want=${1:-}
  [ -n "$want" ] || want=${PAIR_PACK_P7:-}
  [ -n "$want" ] || want=${AB_CKPT:-}
  [ -n "$want" ] && printf '%s\n' "$want"
  return 0
}
pair_ckpt_xcheck() {  # <named> [p7ct gxct] -> PAIR_CKPT_XCHECK on stderr; the named half stays the answer
  local named=$1 p7=${2:-${P7CT:-}} gx=${3:-${GXCT:-}} auto
  auto=$(pair_ckpt_resolve auto "$p7" "$gx") || auto=none   # its own refusal/ambiguity line stays on stderr
  if [ "${auto:-none}" = "${named:-none}" ]; then
    printf 'PAIR_CKPT_XCHECK named=%s auto=%s agree=1\n' "$named" "${auto:-none}" >&2
  else
    printf 'PAIR_CKPT_XCHECK named=%s auto=%s agree=0 (cross-check only; %s is what is loaded)\n' \
      "$named" "${auto:-none}" "$named" >&2
  fi
  return 0
}
pair_ckpt_require() {  # [explicit] [p7ct gxct] -> the named hot half on stdout; 1 when nobody named one
  local want p7=${2:-${P7CT:-}} gx=${3:-${GXCT:-}}
  want=$(pair_ckpt_named "${1:-}")
  if [ -z "$want" ]; then
    echo "PAIR_CKPT_UNNAMED: name the P7 client half (PAIR_PACK_P7= / AB_CKPT= / CHECKPOINT=);" \
         "pair_ckpt_resolve auto is a cross-check now and never decides" >&2
    return 1
  fi
  pair_ckpt_xcheck "$want" "$p7" "$gx"
  printf '%s\n' "$want"
}

pair_gxsh() {
  if [ "$PAIR_DRY" = 1 ] && [ -n "$PAIR_DRY_SSH" ]; then
    "$PAIR_DRY_SSH" "${PAIR_SSH_OPTS[@]}" "$GX" "$1"
  else
    "$PAIR_SSH" "${PAIR_SSH_OPTS[@]}" "$GX" "$1"
  fi
}
pair_gxstdin() {
  if [ "$PAIR_DRY" = 1 ] && [ -n "$PAIR_DRY_SSH" ]; then
    "$PAIR_DRY_SSH" "${PAIR_SSH_OPTS[@]}" "$GX" bash -s
  else
    "$PAIR_SSH" "${PAIR_SSH_OPTS[@]}" "$GX" bash -s
  fi
}

# A reading of the GX box and the seam worker in it, one ssh round-trip so the parts cannot interleave
# with a phase change: "avail_mib=<n> box_swap_used_mib=<n> pid=<n>|- vm_swap_mib=<n>|- rss_mib=<n>|-".
# The worker is picked by comm, not by argv, so an absolute-path worker (/pair/current/src/qmesh/...)
# is seen too, and the `bash -lc` wrapper that also carries the name is skipped (it is 288 kB of swap
# and reads as a happy worker). Same probe as pair_smoke.sh's gx_ws, lifted here because the arm
# scripts need it too: prefill at a wider MAX_CHUNK is exactly where the worker's own VmSwap walks,
# and that is not visible from the client's allocator peak.
pair_gx_ws() {
  local script out
  script=$(sed "s/@CT@/${GXCT:?pair_gx_ws: no GX container known}/g" <<'EOS'
awk '/^MemAvailable:/{a=int($2/1024)} /^SwapTotal:/{t=int($2/1024)} /^SwapFree:/{f=int($2/1024)} /^SUnreclaim:/{u=int($2/1024)} END{printf "avail_mib=%d box_swap_used_mib=%d sunreclaim_mib=%d ", a, t-f, u}' /proc/meminfo
echo "sessions=$(loginctl list-sessions --no-legend 2>/dev/null | wc -l)"
docker exec @CT@ sh -c '
best=""; bestr=0
for p in $(pgrep -x python3 -a | grep "worker_seam_server[.]py" | cut -d" " -f1); do
  r=$(awk "/^VmRSS:/{print \$2}" /proc/$p/status 2>/dev/null); r=${r:-0}
  [ "$r" -gt "$bestr" ] && { bestr=$r; best=$p; }
done
if [ -n "$best" ]; then awk "/^VmSwap:/{s=\$2} /^VmRSS:/{r=\$2} END{printf \"pid=%d vm_swap_mib=%d rss_mib=%d\n\", $best, s/1024, r/1024}" /proc/$best/status; else echo "pid=- vm_swap_mib=- rss_mib=-"; fi'
EOS
)
  out=$(printf '%s\n' "$script" | pair_gxstdin 2>/dev/null | tr '\n' ' ')
  printf '%s\n' "${out% }"
}

# ---- GX box gate ---------------------------------------------------------------------------------
# A GX box carries one cold pack and one server, and two kinds of job want it: a release build (or a
# release smoke) and an experiment that has the pack loaded. They must not overlap -- a compile next to
# a loaded pack moves avail_mib by hundreds of MiB, which both OOM-kills builds and blurs a measurement
# that is about avail_mib. So the box is claimed the way the GPU is: a file, advisory and self-expiring,
# "<owner> pid=… until=<epoch>". A missing file, or a past `until`, means free, so a crashed job cannot
# wedge a box, and a reader never has to trust that the writer cleaned up.
pair_gx_gate_file() {  # -> <queue dir>/gx{N}.gate, N from the GX container name (seam-gx2-dev -> 2)
  local n=${GXCT:-}; n=${n#seam-gx}
  # An unset GXCT used to resolve to "gx.gate", which never exists, so the reader answered "free" for a
  # box that was claimed: on 2026-11-02 03:1x a hand-run check reported GX2 free while a smoke held it.
  # A gate reader must never guess, so an unusable GXCT is an error, not "free".
  if [ -z "$n" ] || [ "$n" = "${GXCT:-}" ]; then
    echo "pair_gx_gate_file: GXCT='${GXCT:-<unset>}' is not seam-gx<N>[<suffix>]: refusing to guess which" \
         "gate to read (an unresolvable path reads as 'free' for a claimed box)" >&2
    return 2
  fi
  printf '%s\n' "${PAIR_GATE_DIR:-$PAIR_HOME/recipe-artifacts/gpu0-gx1/queue}/gx${n%%-*}.gate"
}
pair_p7_gpu_users() {  # [gpu index...] -> "gpu<N> pid=<pid> <name>" per compute process; empty = idle
  # A container create re-initialises the host's single nvidia-uvm instance even though the create
  # names one card: on 2026-11-02 02:22 the dev recreate made a live CUDA process in someone's
  # container die ("Can't initialize NVML: Unknown Error") while the host was healthy, and it did so
  # with no module load and no Xid in kern.log -- the reset is the runtime's device hook, not a
  # reload. The queue serialises this pair's own tickets but cannot see a container started by hand,
  # so anything that creates a container asks the GPUs first, all of them, because the damage is not
  # confined to the card it names.
  local want=" ${*:-0 1 2} " idx uuid pid name
  local -A ix=()
  while IFS=', ' read -r idx uuid; do
    [ -n "${uuid:-}" ] && ix[$uuid]=$idx
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null \
  | while IFS=',' read -r uuid pid name; do
      uuid=${uuid// /}; pid=${pid// /}; name=${name# }
      idx=${ix[$uuid]:-?}
      case "$want" in *" $idx "*) echo "gpu$idx pid=$pid ${name:-?}" ;; esac
    done
}

pair_nvml_ok() {  # <p7|gx> <ct> -> 0 when the container can still reach the driver
  if [ "$1" = gx ]; then pair_gxsh "docker exec ${2:?ct} nvidia-smi -L" >/dev/null 2>&1
  else docker exec "$2" nvidia-smi -L >/dev/null 2>&1; fi
}

pair_ensure_nvml() {  # <p7|gx> <ct> -> restarts the container once when its device permissions were lost
  # A systemd daemon reload (2026-11-02 02:18, triggered by an nmcli MTU change) re-derives the device
  # cgroup of every *running* container and drops its GPU access: NVML then answers "Unknown Error"
  # and CUDA "No CUDA GPUs are available", while the host is healthy. A restart gives the container a
  # fresh cgroup. This must run where the box's gate is held, because the restart kills whatever the
  # container was doing -- the smoke calls it after pair_gx_gate_take, never before.
  local i
  pair_nvml_ok "$1" "$2" && return 0
  if [ "$1" = gx ]; then pair_gxsh "docker restart $2" >/dev/null 2>&1
  else docker restart "$2" >/dev/null 2>&1; fi
  for i in $(seq 1 15); do
    if pair_nvml_ok "$1" "$2"; then echo "PAIR_NVML_RESTART $1 ct=$2 after=$((i * 2))s"; return 0; fi
    sleep 2
  done
  echo "PAIR_NVML_FAIL $1 ct=$2 (device permissions lost by a systemd/toolkit reload; restart did not restore NVML)" >&2
  return 1
}

pair_pid_in_container() {  # <pid> <container> -> 0 when that pid runs inside the named container
  local cid
  cid=$(docker inspect -f '{{.Id}}' "${2:?container}" 2>/dev/null) || return 1
  [ -n "$cid" ] && grep -qs "$cid" "/proc/${1:?pid}/cgroup"
}

pair_gpu_foreign_users() {  # <gpu index...> <own container> -> users on those GPUs that are not its own
  # The pair's own container may be executing something (a client from the pair's previous ticket);
  # everything else on GPU0/1 is someone's job and a create would take their context with it.
  local own=${!#} line p out=""
  set -- "${@:1:$#-1}"
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    p=${line#*pid=}; p=${p%% *}
    pair_pid_in_container "$p" "$own" && continue
    out="$out$line;"
  done <<<"$(pair_p7_gpu_users "$@")"
  printf '%s' "${out%;}"
}

pair_gx_gate_state() {  # -> "free" or "<owner> until=<epoch>"; a past until or a dead local holder is free
  local f o p u h
  f=$(pair_gx_gate_file) || { echo gate_unresolved; return 0; }
  [ -f "$f" ] || { echo free; return 0; }
  o=$(sed -n 's/^owner=\([^ ]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  p=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  u=$(sed -n 's/.*until=\([0-9][0-9]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  h=$(sed -n 's/.*host=\([^ ]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  case "${u:-}" in ''|*[!0-9]*) echo "foreign_bad_gate ${o:-?}"; return 0 ;; esac
  if [ "$u" -lt "$(date +%s)" ]; then echo free; return 0; fi
  # A killed job leaves its claim behind, and every claim this pair writes comes from this host, so a
  # pid that is gone means nobody is holding the box (without this, a killed job wedged the GX for its
  # whole until). Only on this host, though: a claim written on another box, or from inside a container
  # whose /proc is a different namespace, carries a pid this shell cannot see, and `kill -0` on it does
  # not mean the holder is dead. On 2026-11-02 02:26:47 a runner read a live claim, could not see the
  # holder's pid from where it ran, called it dead and took the box while a load was in flight. A claim
  # without a host line predates the field and is this host's.
  if [ -n "${p:-}" ] && [ "${h:-$(hostname -s)}" = "$(hostname -s)" ] && ! kill -0 "$p" 2>/dev/null
  then echo free; return 0; fi
  echo "${o:-?} until=$u"
}
pair_ticket_hardcap() {  # -> epoch the calling queue ticket must be off this box by; fails outside one
  # A claim is a promise about the box's future ("I will be gone by until"). Only a queue ticket can
  # make that promise: it runs under a hard cap, and the pair queue is the one place that bounds a
  # job's life. A claim taken outside one -- an agent's ad hoc bench, a script run by hand -- has no
  # owner that will end it: GxPort2 held gx2.gate from 07:47 to 08:13Z on 2026-11-02 with a claim
  # minted outside any ticket, and the Integrator ticket that came due behind it failed.
  local q t pid i v w
  if [ -n "${GPU_TICKET_HARDCAP:-}" ]; then printf '%s' "$GPU_TICKET_HARDCAP"; return 0; fi
  q=$PAIR_QUEUE/$PAIR/gpu_queue
  pid=$$
  for i in $(seq 1 20); do
    case "${pid:-}" in ''|*[!0-9]*) break ;; esac
    [ "$pid" -gt 1 ] || break
    # Tickets are named <rank>-<ns>-<pid>-<Agent>; the wrapper that owns this process wrote one.
    t=$(ls "$q"/[0-9][0-9]-*-"$pid"-* 2>/dev/null | head -1)
    if [ -n "$t" ]; then
      v=$(sed -n 's/^hardcap=\([0-9][0-9]*\)$/\1/p' "$t" 2>/dev/null | head -1)
      if [ -n "$v" ]; then printf '%s' "$v"; return 0; fi
      # A ticket started by a wrapper that predates the hardcap line: the queue's ceiling is 100 min
      # from the start of the ticket, and a claim may never be longer than that, so use it as the cap
      # (the ticket's real hard cap can only be smaller, never larger).
      w=$(stat -c %W "$t" 2>/dev/null) || w=$(stat -c %Y "$t" 2>/dev/null) || w=$(date +%s)
      printf '%s' "$(( w + 100 * 60 ))"; return 0
    fi
    pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  done
  return 1
}

pair_gate_is_ours() {  # 0 when the claim line names this shell or one of its ancestors
  # A server may only be stopped inside the claim that protects the box (pair_smoke's smoke_exit says so
  # in as many words). On 2026-11-02 07:53 a foreign prologue stopped dev's worker without holding the
  # gate, could not take it afterwards, and left the pair serving nothing until the next ticket. The
  # claim is written with the holder's pid, so "mine" is that pid being this process or one of its
  # parents; a claim from another host can not be ours, because its /proc is not here.
  local f p rest
  f=$(pair_gx_gate_file) || return 1
  p=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  case "${p:-}" in ''|*[!0-9]*) return 1 ;; esac
  while [ "${p:-0}" -gt 1 ] 2>/dev/null; do
    [ "$p" = "$$" ] && return 0
    rest=$(sed -n 's/.*) //p' "/proc/$p/stat" 2>/dev/null)
    p=$(printf '%s\n' "$rest" | awk '{print $2}')
  done
  return 1
}
pair_gx_gate_up() {  # <label> <minutes> -> atomic; never replaces a claim that is still held
  local f cap until now
  f=$(pair_gx_gate_file) || { echo "PAIR_GATE_REFUSED gate_unresolved label=${1:-?}" >&2; return 5; }
  [ "$PAIR_DRY" = 1 ] && return 0
  now=$(date +%s)
  if ! cap=$(pair_ticket_hardcap); then
    echo "PAIR_GATE_REFUSED outside_ticket label=${1:-?}: a box claim may only be taken inside a" \
         "queue ticket (queue/run_dev.sh|run_prod.sh); re-run the job through the queue" >&2
    return 5
  fi
  case "$cap" in ''|*[!0-9]*)
    echo "PAIR_GATE_REFUSED bad_hardcap label=${1:-?} cap=$cap" >&2; return 5 ;;
  esac
  until=$(( now + ${2:-60} * 60 ))
  if [ "$until" -gt "$cap" ]; then
    # The claim must end with the ticket that took it, never after it: a claim past the cap is an
    # orphan the next ticket waits on.
    if [ "$cap" -le "$now" ]; then
      echo "PAIR_GATE_REFUSED ticket_hardcap_passed label=${1:-?} cap=$cap now=$now" >&2; return 5
    fi
    echo "PAIR_GATE_CLAMP label=${1:-?} asked_min=${2:-60} until=$until cap=$cap" >&2
    until=$cap
  fi
  mkdir -p "$(dirname "$f")" || return 1
  (
    # The claim is a lock, so "is it free" and "write it" have to be one step: two jobs that both read
    # free and then write in turn both believe they own the box.
    command -v flock >/dev/null 2>&1 && exec 9>"$f.lock" && flock -w 20 9
    [ "$(pair_gx_gate_state)" = free ] || exit 3
    printf 'owner=%s pid=%s host=%s until=%s\n' "${1:?label}" "$$" "$(hostname -s)" \
      "$until" >"$f" || exit 4
  ) || return 1
}
pair_gx_gate_take() {  # <label> <minutes> <wait_minutes> -> 0 claimed, 1 still someone else's, 5 refused
  local waited=0 limit=$(( ${3:-0} * 6 )) rc
  while :; do
    pair_gx_gate_up "$1" "$2"; rc=$?
    [ "$rc" = 0 ] && return 0
    [ "$rc" = 5 ] && return 5
    [ "$waited" -lt "$limit" ] || return 1
    sleep 10; waited=$((waited + 1))
  done
}
pair_gx_gate_down() {  # drop the claim only if the line is still this process's, on this host
  local f h
  f=$(pair_gx_gate_file) || return 0
  [ "$PAIR_DRY" = 1 ] && return 0
  h=$(sed -n 's/.*host=\([^ ]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  [ "${h:-$(hostname -s)}" = "$(hostname -s)" ] || return 0
  [ "$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' "$f" 2>/dev/null | head -1)" = "$$" ] && rm -f "$f"
  return 0
}

# ---- release identity ---------------------------------------------------------------------------
pair_release_dir() {  # <sha|sha-prefix> -> release dir; a prefix resolves only when it is unambiguous
  local d=$PAIR_ROOT/releases/$1 m
  if [ ! -d "$d" ]; then
    # A 12-char sha from a log is what humans type; the directory is named by the full one.
    m=$(ls -d "$PAIR_ROOT"/releases/"$1"* 2>/dev/null | head -2)
    if [ -n "$m" ] && [ "$(printf '%s\n' "$m" | wc -l)" = 1 ] && [ -d "$m" ]; then d=$m; fi
  fi
  echo "$d"
}
pair_current_sha() {  # <root> [remote 1] -> sha the `current` symlink points at ("" when absent)
  local root=$1 remote=${2:-0} t
  if [ "$remote" = 1 ]; then
    t=$(pair_gxsh "readlink -f $root/current 2>/dev/null || true")
  else
    t=$(readlink -f "$root/current" 2>/dev/null || true)
  fi
  # readlink -f echoes a missing path unchanged: only a real releases/<sha> counts as current
  case "$t" in "$root"/releases/*) basename "$t" ;; *) return 0 ;; esac
}

# ---- what a ticket loaded, and how it puts it back ------------------------------------------------
# A ticket owns the pack it loaded and the client it started, and nothing else. Neither may be
# stopped by pattern: on 2026-11-02 a killed 23:32 ticket left `worker_seam_server.py` holding 116 GiB,
# and the 23:54 smoke refused with `pair_busy gx_procs=2` while a colleague waited on the box. The
# server is addressed by the pid the ticket recorded plus a cmdline check, so a reused pid can not be
# mistaken for it; the pair's own P7 container is the only place the client can be.
pair_server_pidfile() { echo "$PAIR_STATE_DIR/server-$PAIR.pid"; }
pair_server_record() {  # <build id> -- records the worker this ticket's ensure started
  # Anchored on the worker's own argv. The launcher is `docker exec -d … bash -lc 'cd … && stdbuf …
  # python3 -u qmesh/worker_seam_server.py …'`, and *two* processes can carry that string: the
  # shell/timeout/stdbuf wrapper before it execs, and the worker itself. An unanchored pgrep made
  # head -1 return the wrapper, so the stop below TERMed a shell and orphaned the process holding the
  # pack (2026-11-02, pid 3297 wrapper / 3305 worker, 116 GiB left behind).
  local pid
  pid=$(pair_gxsh "docker exec $GXCT pgrep -f '^python3 -u qmesh/worker_seam_server.py' | head -1" 2>/dev/null)
  case "${pid:-}" in ''|*[!0-9]*) return 1 ;; esac
  printf 'pair=%s build=%s pid=%s at=%s\n' "$PAIR" "$1" "$pid" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$(pair_server_pidfile)"
  return 0
}
pair_gx_serving_build() {  # build id the worker in our GX container carries, empty when it does not serve
  # The release a pair serves is a property of the *running worker*, not of the pidfile and not of the
  # `current` link: a restore ticket that queues behind an evicting one must be able to tell "already
  # up on the build I was asked for" from "up on something else", or it either dies noisily or reloads
  # 116 GiB over a healthy pair.
  [ "$PAIR_DRY" = 1 ] && return 0
  pair_gxsh "docker exec $GXCT sh -c 'p=\$(pgrep -f \"^python3 -u qmesh/worker_seam_server[.]py\" | head -1); [ -n \"\$p\" ] && tr \"\\0\" \"\\n\" < /proc/\$p/environ 2>/dev/null | sed -n \"s/^SEAM_BUILD_ID=//p\" | head -1'" 2>/dev/null | tail -1
}
pair_gx_worker_det() {  # SEAM_MOE_CPP_DET the running worker was started with; empty when unset
  # Presence is not the question. A worker started with =0 has the variable on its command env and
  # still folds the routing weight into the down gemm's fp32 atomics (qmesh_moe_cpp.cpp:997, :18-21),
  # so a reuse check that only greps for the *name* keeps a nondeterministic server through an arm
  # that just changed its value -- and a ticket that asserts on a byte comparison then measures an
  # epilogue it did not ask for. Unset prints nothing: that is the C hook's own default (0) and the
  # state every run on 2026-11-05 was in while its log said SEAM_MOE_CPP_DET=1 (see pair_server_env).
  [ "$PAIR_DRY" = 1 ] && return 0
  pair_gxsh "docker exec $GXCT sh -c 'p=\$(pgrep -f \"^python3 -u qmesh/worker_seam_server[.]py\" | head -1); [ -n \"\$p\" ] && tr \"\\0\" \"\\n\" < /proc/\$p/environ 2>/dev/null | sed -n \"s/^SEAM_MOE_CPP_DET=//p\" | head -1'" 2>/dev/null | tr -dc '0-9'
}
pair_torch_locks() {  # <ct> -- JIT build locks a dead process left in the container
  pair_gxsh "docker exec $1 sh -c 'for f in /root/.cache/torch_extensions/*/*/lock /tmp/torch_extensions/*/*/lock; do [ -e \"\$f\" ] && echo \"\$f\"; done'" 2>/dev/null
}
pair_torch_build_live() {  # <ct> -- how many compilers or servers are alive there right now
  pair_gxsh "docker exec $1 sh -c 'pgrep -fc \"(^|/)(nvcc|cicc|ninja)( |\$)|worker_seam_server[.]py\" || true'" 2>/dev/null | tr -dc 0-9
}
pair_gx_foreign_server() {  # "server_in=<ct>:<pid>" for a pack load in a container that is not this pair's
  # One pack load per box is the rule, and the load is what costs 116 GiB and 3 minutes. Our own
  # container is excluded by name; a worker in anybody else's is somebody's work, and a ticket that
  # flipped a release before noticing it is a ticket that mutates a busy pair (2026-11-02 02:41).
  [ "$PAIR_DRY" = 1 ] && return 0
  # A --stub worker is skipped: it returns before the pack and before CUDA (SeamServer.__init__), so it
  # costs the box nothing and must not be read as "the box is busy" (SeamRobust's ctl test starts one).
  pair_gxsh "for c in \$(docker ps --format '{{.Names}}'); do [ \"\$c\" = '$GXCT' ] && continue; p=\$(docker exec \"\$c\" sh -c 'for pid in \$(pgrep -f \"python3 -u qmesh/worker_seam_server[.]py\" 2>/dev/null); do tr \"\\0\" \" \" < /proc/\$pid/cmdline 2>/dev/null | grep -q -- --stub || { echo \$pid; break; }; done' 2>/dev/null | head -1); [ -n \"\$p\" ] && echo \"server_in=\$c:\$p\"; done" 2>/dev/null
}
pair_server_stop() {  # stops the recorded worker if it is still the one that was recorded
  local f pid build live out
  f=$(pair_server_pidfile)
  [ "$PAIR_DRY" = 1 ] && { echo "SERVER_STOP_DRY $(cat "$f" 2>/dev/null)"; return 0; }
  pid=$(sed -n 's/.* pid=\([0-9][0-9]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  build=$(sed -n 's/.* build=\([^ ]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  [ -n "$pid" ] || return 0
  live=$(pair_gxsh "docker exec $GXCT sh -c 'tr \"\\0\" \" \" < /proc/$pid/cmdline 2>/dev/null'" 2>/dev/null)
  # The cmdline has to *be* the worker's, not merely mention the script: a wrapper pid that a
  # pre-anchoring record could have left in the pidfile must not be TERMed again.
  case "$live" in
    python3\ -u\ qmesh/worker_seam_server.py*) ;;
    *) if [ -n "$live" ]; then
         printf 'SERVER_STOP_SKIP pid=%s (not the worker: %s)\n' "$pid" "$(printf '%s' "${live:0:60}")"
       else
         printf 'SERVER_STOP_SKIP pid=%s (gone or not the worker)\n' "$pid"
       fi
       rm -f "$f"; return 0 ;;
  esac
  # Stopping is a lifecycle event on the box: it belongs inside the claim that protects it. A job that
  # does not hold the gate can stop a server it can not then reload -- on 2026-11-02 07:53 that is how
  # dev served nothing between two tickets. Free box or our own claim: proceed as before.
  if [ "${PAIR_STOP_FORCE:-0}" != 1 ]; then
    case "$(pair_gx_gate_state)" in
      free|gate_unresolved) ;;
      *) pair_gate_is_ours || {
           printf 'SERVER_STOP_REFUSED pid=%s build=%s gate="%s" -- another job owns %s, so this stop could not reload it (PAIR_STOP_FORCE=1 overrides)\n' \
             "$pid" "${build:-?}" "$(pair_gx_gate_state)" "$GXCT"
           return 1
         } ;;
    esac
  fi
  out=$(pair_gxsh "docker exec $GXCT kill -TERM $pid && echo sent" 2>/dev/null)
  if [ "${out:-}" = sent ]; then
    printf 'SERVER_STOP pid=%s build=%s\n' "$pid" "${build:-?}"
  else
    printf 'SERVER_STOP_FAILED pid=%s build=%s\n' "$pid" "${build:-?}"
  fi
  rm -f "$f"
  return 0
}
pair_client_stop() {  # the arm's client lives in the pair's own P7 container; nobody else's does
  [ "$PAIR_DRY" = 1 ] && return 0
  docker exec "$P7CT" pkill -f 'qmesh/mem_budget.py' >/dev/null 2>&1 &&
    echo "CLIENT_STOP ct=$P7CT"
  return 0
}

# Seam identity: the three files that define the wire. A release that changes any byte of them is a
# different protocol, whatever a constant says, so the fingerprint is the sha256 over their names and
# contents (same on P7 and GX for the same release).
pair_seam_fingerprint() {  # <tree>
  local tree=$1 f
  for f in "${PAIR_SEAM_FILES[@]}"; do
    [ -r "$tree/$f" ] || { echo missing; return 1; }
    printf '%s\0' "$f"; cat "$tree/$f"
  done | sha256sum | cut -c1-16
}
pair_seam_version() {  # <tree> -> PROTO_VERSION / QES_PROTO_VERSION when the tree carries one, else "-"
  local tree=$1 v=""
  v=$(sed -n 's/^PROTO_VERSION *= *\([0-9][0-9]*\).*/\1/p' "$tree/qmesh/native/qes.py" 2>/dev/null | head -1)
  [ -z "$v" ] && v=$(sed -n 's/^#define QES_PROTO_VERSION *\([0-9][0-9]*\).*/\1/p' "$tree/qmesh/native/qmesh_expert_seam.h" 2>/dev/null | head -1)
  echo "${v:--}"
}
# pair_canary_gate <sha> <evidence-file> -- prints "ok" when that sha passed its dev canary, else the
# reasons it did not. ONE definition, because three places must agree on what "dev passed" means:
# pair_update.sh (the fast refusal), pair_promote.sh (the only way in) and pair_update_run.sh (the
# script that actually flips, so queueing it directly is not a way around the gate).
pair_canary_gate() {
  python3 - "${2:?evidence file}" "${1:?sha}" <<'PY'
import json, sys
path, sha = sys.argv[1:3]
try:
    d = json.load(open(path))
except Exception as exc:                                                # noqa: BLE001
    print(f"unreadable {exc}")
    raise SystemExit(0)
bad = []
if d.get("release_sha") != sha:
    bad.append(f"sha={str(d.get('release_sha'))[:12]}")
if d.get("pair") != "dev":
    bad.append(f"pair={d.get('pair')}")
if not d.get("ok"):
    bad.append(f"ok=false at {d.get('failed_at_phase') or '?'} ({d.get('why') or 'no reason'})")
smoke = d.get("smoke") or {}
if not smoke.get("row"):
    bad.append("no_smoke_row")
try:
    n = int((smoke.get("ids") or {}).get("n") or 0)
except (TypeError, ValueError):
    n = 0
if n <= 0:
    bad.append("no_decode_ids")
print(" ".join(bad) if bad else "ok")
PY
}

# ---- manifests ----------------------------------------------------------------------------------
pair_manifest_get() {  # <field.path...> -> value on stdout ("" when absent); reads $PAIR_MANIFEST
  python3 - "$PAIR_MANIFEST" "$@" <<'PY'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception:
    sys.exit(0)
for key in sys.argv[2:]:
    if not isinstance(d, dict):
        d = None
        break
    d = d.get(key)
    if d is None:
        break
if isinstance(d, (dict, list)):
    print(json.dumps(d))
elif d is not None:
    print(d)
PY
}

# ---- queue --------------------------------------------------------------------------------------
pair_run_script() { echo "$PAIR_QUEUE/run_$PAIR.sh"; }
# Enqueue one job on the pair's queue with the given agent name. The queue ranks agents by
# gpu_priority.txt in that pair's QBASE; PairUpdate is ranked so an update lands right after the
# job that is running now, ahead of every other waiter.
pair_enqueue() {  # <agent> <max_minutes> <cmd...>
  local agent=${1:?agent} mins=${2:?minutes}; shift 2
  # A ticket with no command would hold the GPU for zero seconds and report success; the queue
  # would look like it ran the update. Refuse instead -- and the caller must be the one that dies.
  [ "$#" -gt 0 ] || pair_die "pair_enqueue $agent: no command given (nothing would run)"
  local run q logdir log pid t i before after
  run=$(pair_run_script)
  [ -x "$run" ] || pair_die "queue runner missing: $run"
  q=$PAIR_QUEUE/$PAIR/gpu_queue
  logdir=$PAIR_STATE_DIR/logs
  mkdir -p "$logdir" "$q"
  log=$logdir/enqueue-$PAIR-$agent-$(date -u +%Y%m%dT%H%M%SZ).log
  # What is already queued, so the new ticket is found by difference and not by guessing its name.
  before=$(ls "$q" 2>/dev/null)
  # setsid: the ticket wrapper and everything it starts form their own process group, so cancelling
  # the job kills the whole group (the queue does kill -- -<pgid>) and nothing outlives its window.
  setsid nohup "$run" "$agent" "$mins" "$@" >"$log" 2>&1 </dev/null &
  pid=$!
  # Wait for the ticket so the caller reports the one it actually got. Match on the agent suffix and
  # on "not there before", never on $pid: the runner names the ticket with its own $$ and setsid(1)
  # may fork, so the name's pid is not $! -- matching it reported ENQUEUE_FAILED for tickets that
  # were queued and did run (2026-11-01, a probe ran while the caller was told nothing had).
  t=""
  for i in $(seq 1 40); do
    after=$(ls "$q" 2>/dev/null)
    t=$(comm -13 <(printf '%s\n' "$before" | sort) <(printf '%s\n' "$after" | sort) 2>/dev/null \
          | awk -v s="-$agent" 'length($0) > length(s) && substr($0, length($0) - length(s) + 1) == s' \
          | head -1)
    [ -n "$t" ] && break
    sleep 0.25
  done
  if [ -z "$t" ]; then
    echo "ENQUEUE_FAILED $PAIR agent=$agent pid=$pid log=$log"
    tail -5 "$log" 2>/dev/null
    return 1
  fi
  echo "ENQUEUED $PAIR agent=$agent pid=$pid ticket=$t log=$log"
}
# PairUpdate must land right after the job that is running now, ahead of every other waiter: the
# queue ranks agents by their line in gpu_priority.txt, so line 1 is enough. Tickets already taken
# keep the rank they were created with, so existing waiters are not reordered.
pair_ensure_priority() {  # <agent>
  local agent=${1:?agent} prio=$PAIR_QUEUE/$PAIR/gpu_priority.txt tmp
  mkdir -p "$(dirname "$prio")"
  tmp=$(mktemp "${prio}.XXXXXX") || pair_die "cannot write $prio"
  { echo "$agent"; [ -r "$prio" ] && grep -vx "$agent" "$prio" | grep -v '^$'; } > "$tmp"
  mv -f "$tmp" "$prio"
}
