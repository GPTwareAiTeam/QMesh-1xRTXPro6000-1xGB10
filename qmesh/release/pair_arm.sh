#!/usr/bin/env bash
# pair_arm.sh -- run a client measurement through a pair's P7 container, on the release it serves.
#
#   pair_arm.sh <pair> [--rows N] [--max-new N] [--warmup N] [--cache-tokens N]
#                      [--prompts PATH] [--checkpoint PATH] [--timeout S] [--out FILE]
#                      [--prefill DEPTH] [--ids | --wide [--wide-manifest PATH]]
#
# Decode (the default): prints whatever qmesh/e2e_prompts.py prints (one JSON object per prompt on
# stdout, diagnostics on stderr) and returns the client's exit code, or 124 on timeout. The timeout is
# not politeness: a client against a mismatched or dead seam server blocks forever instead of failing,
# and a smoke step that can hang is worse than one that fails.
#
# `--prefill DEPTH` runs qmesh/mem_budget.py instead, with PREFILL_ARMS=DEPTH:8192 -- one arm at that
# depth, host-timed prefill and engine tok/s, which is the instrument the prefill numbers were taken
# with and what the smoke gates dev on.
#
# `--ids` runs the same driver through pair_arm_ids.py, which additionally prints one
# `ARM_IDS arm=… n=… sha256=… first=…` line per arm (the ids the engine generated) and writes them
# to $ARM_IDS_OUT. Everything else is unchanged, so a row recorded with `--ids` compares equal.
#
# `--wide` runs the same driver through pair_arm_wide.py over the 22-prompt WideTrace decode corpus
# (--wide-manifest, default the artifact-tree corpus.json) instead of the 8 held prompts: same
# release, same driver, same env list, only load_prompts() is answered from the manifest, so the rows
# are the shipped instrument on a wider sample. The manifest lists one entry per prompt with its own
# kind (whole text file, or pre-rendered token ids) so --prompts/--corpus are not involved; --rows
# still caps the count.
#
# The environment below is the shipping client config and is not inherited from the caller: the
# smoke, a manual arm and a promotion probe all have to run the pair's real shape, or their numbers
# describe something nobody serves.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_arm.sh <pair> [--rows N] ...}
shift
ROWS=1; MAX_NEW=8; WARMUP=0; CACHE=1048576; ARM_TIMEOUT=300; OUT=""; IDS=0; WIDE=0
WIDE_MANIFEST=${PAIR_ARM_WIDE_MANIFEST:-$HOME/recipe-artifacts/routetrace/widetrace-corpus/corpus.json}
PROMPTS=${PAIR_ARM_PROMPTS:-/artifacts/2x-rtx-tp2/dtrain/prompts_held.json}
# The client half is the one whose PACK_MANIFEST.json declares the placement_sha256 the GX box's cold pack
# carries: a pair is one checkpoint split in two, so after the 4780p8 pack switch a hard-coded 4460 client
# would be paired with a 4780p8 cold half. RM10 2026-11-05 (Main): it is NAMED, never derived --
# --checkpoint / PAIR_ARM_CHECKPOINT, else PAIR_PACK_P7 (what pair_prepare/pair_smoke export) or AB_CKPT.
# `pair_ckpt_resolve auto` cross-checks the named half on stderr and never decides; naming none is a refusal.
CHECKPOINT=${PAIR_ARM_CHECKPOINT:-}
CORPUS=${PAIR_ARM_CORPUS:-/artifacts/2x-rtx-tp2/corpus/prose_24k.txt}
# PREFILL_ARMS_LIST is mem_budget's own "depth:chunk,depth:chunk" form: several arms in ONE model load,
# which is the only way to compare chunk widths without paying a 113 GiB reload between them (and the
# only way to keep the comparison inside one page-in state). --prefill is the single-arm shorthand.
PREFILL_DEPTH=""; PREFILL_ARMS_LIST=""; SRC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rows) ROWS=$2; shift ;;
    --max-new) MAX_NEW=$2; shift ;;
    --warmup) WARMUP=$2; shift ;;
    --cache-tokens) CACHE=$2; shift ;;
    --prompts) PROMPTS=$2; shift ;;
    --checkpoint) CHECKPOINT=$2; shift ;;
    --timeout) ARM_TIMEOUT=$2; shift ;;
    --out) OUT=$2; shift ;;
    --prefill) PREFILL_DEPTH=$2; shift ;;
    --prefill-arms) PREFILL_ARMS_LIST=$2; shift ;;
    --ids) IDS=1 ;;
    --wide) WIDE=1 ;;
    --wide-manifest) WIDE_MANIFEST=$2; shift ;;
    --src) SRC=$2; shift ;;
    *) pair_die "unknown flag $1" ;;
  esac
  shift
done
pair_load "$PAIR_ARG"
CHECKPOINT=$(pair_ckpt_require "$CHECKPOINT") \
  || pair_die "CHECKPOINT: name the P7 client half (--checkpoint / PAIR_ARM_CHECKPOINT / PAIR_PACK_P7 / AB_CKPT)"
pair_step "PAIR_ARM_CKPT checkpoint=$CHECKPOINT cold=$(pair_cold_src) sha=$(pair_cold_placement)"
if ! ckpt_line=$(pair_ckpt_verified "$CHECKPOINT" 2>&1); then
  pair_die "CHECKPOINT: $CHECKPOINT is not the pack its manifest describes ($ckpt_line)"
fi
pair_step "$ckpt_line"
# The client half must name the release it is running, or the server's session handshake accepts a
# mixed pair (`SEAM_SESSION_START ... build=unknown proto=3`, 2026-11-01). A released tree has no
# .git, so qes.build_id() falls back to <tree>/qmesh/BUILD_ID (written by pair_cut.sh); exporting it
# here makes the env the first answer on both halves, which is the pair-level check that matters.
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' \
  "$PAIR_ROOT/current/RELEASE.json" 2>/dev/null) || BUILD_ID=""


# Which tree the client imports, and therefore which extension cache it may write. torch's version
# hash covers the *include path spelling*, not the content, so a client that imports a second
# checkout of the same sources rebuilds the extension in place -- in the cache it was pointed at. On
# 2026-11-03 20:59:31 a client running /artifacts/dspark-tpr/src with the pair's own
# TORCH_EXTENSIONS_DIR=/pair/extcache rebuilt exllamav3_ext.so in the pair's release cache (5780205b
# replacing the stamped 04a61140), and `pair_serve.sh` then refused the pair as DRIFT for an hour
# while every later client import rebuilt or hung on it. So the release cache
# (/pair/extcache, the release's own ext/) is for the release tree only; any other --src gets a
# cache of its own, under the pair's state dir, named by the tree so two agents cannot collide.
SRC=${SRC:-/src/exllamav3}
# `/pair/current/src` is the same tree as `/src/exllamav3` but not the same *spelling*, and torch's
# version hash covers the spelling, so importing through it against the release cache rebuilds
# exllamav3_ext.so in place and leaves BUILD_STAMP.json naming the artifact it replaced. On
# 2026-11-03 a client that did exactly that left the dev cache at ext_so=c7908310 want=e74d4951
# (build.ninja `-I/pair/current/src/exllamav3`) and every dev ticket read DRIFT for hours. So
# canonicalise the alias instead of giving it the release cache: one tree, one spelling, one cache.
case "$SRC" in
  /pair/current/src|/pair/current/src/)
    echo "ARM_SRC_CANON src=$SRC -> /src/exllamav3 (the alias is not an include spelling)"
    SRC=/src/exllamav3
    ;;
esac
case "$SRC" in
  /src/exllamav3|/src/exllamav3/) EXT_DIR=/pair/extcache ;;
  *)
    ext_tag=$(printf '%s' "$SRC" | sed 's#^/*##; s#/*$##; s#[^A-Za-z0-9._-]#_#g')
    mkdir -p "$PAIR_STATE_DIR/extcache/$ext_tag"
    EXT_DIR=$(pair_ct_path "$PAIR_STATE_DIR/extcache/$ext_tag")
    echo "ARM_SRC_ISOLATED src=$SRC extcache=$EXT_DIR (not the release cache)"
    ;;
esac
envs=(
  # Stated, not inherited: the container's env is the release's, and a --src above must not be able
  # to reach the release cache through an inherited variable. For the release tree these are the two
  # values the container already carries, so the shipped exec is unchanged.
  -e "PYTHONPATH=$SRC" -e "TORCH_EXTENSIONS_DIR=$EXT_DIR"
  # Same pin the worker gets from pair_server_env, for the client's half of the autotuner: unset,
  # EXLLAMAV3_TUNE_CACHE resolves under $HOME *inside* the container, so the picks die with it (see
  # the constant in pair_lib.sh). A container path, like the two above, not a host one.
  -e "EXLLAMAV3_TUNE_CACHE=$PAIR_TUNE_CACHE"
  # The client's Triton JIT cache, pinned the same way and for the same reason: unset, it resolves
  # under $HOME inside the container and every `docker run` pays 10-13 s of ptxas+nvcc in the first
  # prefill (measured, see the constant in pair_lib.sh). The server half is aarch64 and runs no
  # Triton kernels, so this is the client's alone; it costs the worker nothing to carry it.
  -e "TRITON_CACHE_DIR=$PAIR_TRITON_CACHE"
  # torch's NVRTC kernel cache, same carrier rule again (see PAIR_TORCH_HOME in pair_lib.sh): unset,
  # the cubin is compiled into the container's own /root/.cache/torch and thrown away with it.
  # The server half runs no torch, so this is the client's.
  -e "TORCH_HOME=$PAIR_TORCH_HOME"
  # pairs.conf's `gpu` is a *host* index: the container spec spends it on --gpus device=<gpu>, so
  # inside the container that one card is device 0 again. Passing the host index here asked dev's
  # container (which had every GPU) for GPU 1 and asked a scoped one for a GPU it cannot see.
  -e "CUDA_VISIBLE_DEVICES=0"
  -e "REMOTE_TRANSPORT=seam"
  -e "SEAM_DEV=$SEAM_DEV" -e "SEAM_GID=$SEAM_GID" -e "SEAM_PEER=$SEAM_PEER"
  # SEAM_BUCKETS is a *request*: the server grants the intersection of this list and its own
  # (--buckets, from pairs.conf). Leaving it unset used the client's fallback default
  # (1,6,64,256,512,1024,2048,4096), which does not contain the shipping prefill width 8192: the
  # intersection was 1,6,64, so the 23:32 dev arm negotiated chunk_rows=64 and split every 8192-row
  # prefill chunk into 128 calls of 64 rows (accepted=1,6,64, 1.6 s/token decode). Ask for the pair's
  # own list and the grant is exact.
  -e "SEAM_BUCKETS=$BUCKETS"
  -e "MAX_PROMPTS=$ROWS" -e "MAX_NEW=$MAX_NEW" -e "WARMUP=$WARMUP" -e "WARMUP_NEW=$MAX_NEW"
  # PROMPTS and CHECKPOINT are host paths as often as not (the default prompts file is a container path, so
  # an untranslated PROMPTS went unnoticed until a ticket passed one from the host and died inside the
  # container on e2e_prompts.py:120 FileNotFoundError -- Ports2x, 2026-11-04). pair_ct_path passes container
  # paths through unchanged, so translating is safe for both callers.
  -e "CACHE_TOKENS=$CACHE" -e "PROMPTS=$(pair_ct_path "$PROMPTS")" \
  -e "CHECKPOINT=$(pair_ct_path "$CHECKPOINT")"
)
[ -n "$BUILD_ID" ] && envs+=(-e "SEAM_BUILD_ID=$BUILD_ID")
# The shipping client config, one list. Every knob here was in the run the notes quote (4460 packs,
# drafter on, 1M FP4 KV, 8192-row chunks with adaptive chunking, seam buckets up to the grant); a
# release that needs a different value changes this list, so the smoke checks what ships.
envs+=(
  -e "DRAFT_CHECKPOINT=$CHECKPOINT" -e "DSPARK=1" -e "NUM_DRAFT=5"
  # Graphs on from 2026-11-04 (GraphsShip): the untraced A/B on this release with the fused GX kernel,
  # arms interleaved eager/graphs/eager/graphs on one server load, 8 prompts x 128, gave ms/token
  # 15.731 / 14.902 / 15.982 / 14.596 -- pairs -5.3 % and -8.7 %, mean -7.0 % -- and ms/round 49.58 /
  # 46.38 / 48.78 / 46.71 (mean -5.4 %). Lossless: SEAM_MOE_CPP_DET=1 at 4x64, all four arms
  # bit-identical (IDENTITY_OK sha=1c1655d579f2e7e...). The switch is read at import in the *client*, so this
  # line is what an arm actually runs with -- pair_server.sh's copy only serves -- and it is an env default, not
  # a code change: PAIR_WHOLE_GRAPH=0 (or PAIR_ARM_EXTRA_ENVS=EXL3_WHOLE_GRAPH=0) is the escape. An A/B
  # reference arm therefore has to override it in the client env (final_block_ticket.sh logs the effective value
  # as wg= on every DECODE_ARM line, because an un-overridden reference would measure graphs twice and read as
  # "no effect"). EXL3_WHOLE_GRAPH_MAXSEQ stays at its default 8 -- a wider verify falls back to eager by
  # itself. Do not add EXL3_DSV4_B1_BATCH here: 2.3 ms/token regression (dsv4.py:31-34), and it cannot
  # express B>1 at S>1.
  # Chunk width is the same shipping default as pair_server.sh:39 (16384 since 2026-11-04, ChunkAB's
  # same-load A/B: 32k prefill +7.3 %, 128k +9.7 %, decode decide_us unchanged); MAX_CHUNK=8192 is the
  # escape on either half, and an A/B that wants the old width sets it here, not in the code.
  -e "KV_BITS=4" -e "EXL3_WHOLE_GRAPH=${PAIR_WHOLE_GRAPH:-1}" -e "MAX_CHUNK=${MAX_CHUNK:-16384}"
  -e "EXL3_MOE_SPLIT_FUSED=0"
  # Off from 2026-11-04: NSight's trace counted 55.7k cudaEvent create/destroy plus 57.5k record per
  # measured window on this pair (~0.7 ms per round, 1.2 % of a 60 ms round) for instrumentation that
  # writes a log line nobody reads in production. Lossless by construction -- no bytes, no ordering, no
  # arithmetic on any tensor -- so it ships on that arithmetic rather than on an A/B (a 1.2 % effect is
  # inside this pair's arm-to-arm spread).
  -e "EXL3_SEAM_PROF=0"
  -e "PREFILL_ADAPTIVE=1" -e "PREFILL_ADAPTIVE_LOG=1"
  -e "SEAM_ROWS=8192"
  -e "SEAM_NS=1" -e "SEAM_OVERLAP=1" -e "SEAM_COPY_STREAM=1"
)
# The client half of pair_lib.sh's MOE_DET, and the same knob: serve_pair.py refuses BATCH>1 unless the
# *client* was started with SEAM_MOE_CPP_DET=1 (the hello carries no such field, qmesh/native/qes.py
# PROTO_VERSION=4, so the client can only trust its own value and the deploy sets both halves alike).
# One variable on both halves is what keeps a diagnostic arm honest: MOE_DET=0 pair_arm.sh ... starts
# the worker on the folded epilogue and this client on it too, which is the pre-switch shape. Appended
# before PAIR_ARM_EXTRA_ENVS so an arm can still override it last. Installed behaviour of
# pair_lib.sh since 2026-11-05 07:21 (md5 e51d8096): MOE_DET follows the client's BATCH, so a
# `BATCH>1` arm cannot start on the folded epilogue by omission (see that file).
if [ "$MOE_DET" = 1 ]; then envs+=(-e "SEAM_MOE_CPP_DET=1"); fi

# PAIR_ARM_EXTRA_ENVS: space-separated K=V, each appended as `-e K=V` after the list above (docker
# keeps the last one, so this can also override a shipping knob). It exists for measurement-only
# instrumentation -- e.g. EXL3_SEAM_MARK_MIN_ROWS lowered so a 6-row decode call is not skipped by the
# seam timeline's 512-row floor -- and must stay empty for anything whose numbers are quoted as the
# shipped shape. Unset, the exec is byte-for-byte the one it was before.
for kv in ${PAIR_ARM_EXTRA_ENVS:-}; do envs+=(-e "$kv"); done
if [ -n "$PREFILL_ARMS_LIST" ]; then
  envs+=(-e "PREFILL_ARMS=$PREFILL_ARMS_LIST" -e "CORPUS=$(pair_ct_path "$CORPUS")"
         -e "ARM_MAX_S=${PAIR_ARM_ARM_MAX_S:-1200}" -e "LOAD_ONLY=0")
  drv=qmesh/mem_budget.py
elif [ -n "$PREFILL_DEPTH" ]; then
  envs+=(-e "PREFILL_ARMS=$PREFILL_DEPTH:8192" -e "CORPUS=$(pair_ct_path "$CORPUS")"
         -e "ARM_MAX_S=${PAIR_ARM_ARM_MAX_S:-1200}" -e "LOAD_ONLY=0")
  drv=qmesh/mem_budget.py
fi
[ -n "$OUT" ] && envs+=(-e "OUT=$(pair_ct_path "$OUT")")
# The driver has to be a real file, not stdin: the client's CPU-host module starts its worker with
# multiprocessing "spawn", which re-executes the parent's main file by path -- with `python3 -` that
# path is "<stdin>", the child dies on FileNotFoundError and the parent sees only EOFError. The
# artifact tree is mounted in the container, so the driver is written there and run by path.
if [ "$IDS" = 1 ] && [ "$WIDE" = 1 ]; then
  pair_die "--ids and --wide both replace the driver; pick one"
fi
if [ "$IDS" = 1 ]; then
  drv_host=$PAIR_STATE_DIR/arm/pair_arm_ids.py
  mkdir -p "$(dirname "$drv_host")"
  cp "$HERE/pair_arm_ids.py" "$drv_host"
  drv=$(pair_ct_path "$drv_host")
  [ -n "${ARM_IDS_OUT:-}" ] && envs+=(-e "ARM_IDS_OUT=$(pair_ct_path "$ARM_IDS_OUT")")
elif [ "$WIDE" = 1 ]; then
  drv_host=$PAIR_STATE_DIR/arm/pair_arm_wide.py
  mkdir -p "$(dirname "$drv_host")"
  cp "$HERE/pair_arm_wide.py" "$drv_host"
  drv=$(pair_ct_path "$drv_host")
  envs+=(-e "WIDE_MANIFEST=$(pair_ct_path "$WIDE_MANIFEST")")
fi

if [ "$PAIR_DRY" = 1 ]; then
  echo "DRY docker exec ${envs[*]} $P7CT bash -lc 'cd $SRC && timeout $ARM_TIMEOUT python3 -u ${drv:-qmesh/e2e_prompts.py}'"
  exit 0
fi
CACHE_PROBE="e=$EXT_DIR/exllamav3_ext; stat -c '%Y %s' \"\$e/build.ninja\" \"\$e/exllamav3_ext.so\" 2>/dev/null | tr '\\n' '/'"
ext_fp_before=$(docker exec "$P7CT" sh -c "$CACHE_PROBE" 2>/dev/null)

timeout "$ARM_TIMEOUT" docker exec "${envs[@]}" "$P7CT" \
  bash -lc "cd $SRC && python3 -u ${drv:-qmesh/e2e_prompts.py}"

# A client must never build into the cache it was pointed at, and the release cache least of all: on
# 2026-11-03 18:50:56 some client inside seam-p7-dev regenerated build.ninja in the release cache
# (/pair/extcache/exllamav3_ext) and recompiled 200 units for 5 minutes, which replaced the stamped
# exllamav3_ext.so and left `pair_check dev` reading DRIFT ext_so=3ef6ed68 want=3e2d504a for 35 minutes
# -- every serve on the pair refused, and nobody knew which client did it. torch decides to rebuild
# quietly, so the arm has to look: fingerprint the cache before and after the client, and refuse the
# numbers when it moved. A hit costs two stats; a miss is the one event this whole script exists to
# name (and an arm that rebuilt 5 minutes of kernels cannot have measured the release's warm path
# anyway). The tag printed here is the only handle back to the invocation.
client_rc=$?
ext_fp_after=$(docker exec "$P7CT" sh -c "$CACHE_PROBE" 2>/dev/null)
if [ -n "$ext_fp_before" ] && [ "$ext_fp_before" != "$ext_fp_after" ]; then
  echo "ARM_EXT_WRITE_REFUSED src=$SRC extcache=$EXT_DIR before=$ext_fp_before after=$ext_fp_after"
  echo "ARM_EXT_WRITE_REFUSED numbers_void=1 why=the client rebuilt its own extension cache inside the arm"
  [ "$EXT_DIR" = /pair/extcache ] && echo "ARM_EXT_WRITE_REFUSED release_cache=1 next=scripts/pair_check.sh $PAIR then scripts/pair_prepare.sh $PAIR <sha> --side p7"
  exit 13
fi
exit "$client_rc"
