#!/usr/bin/env bash
# pair_server.sh -- start/stop/status the pair's OpenAI-compatible API server.
#
#   pair_server.sh <prod|dev> start [-e KEY=VALUE ...]
#   pair_server.sh <prod|dev> stop
#   pair_server.sh <prod|dev> status
#
# The pair serves on two boxes: the GX worker (qmesh/worker_seam_server.py, the cold experts, owned
# by pair_serve.sh and recorded in $PAIR_STATE_DIR/server-$PAIR.pid) and this HTTP server on the P7
# card (qmesh/serve_pair.py), the only process that loads the hot pack. start ensures the worker
# first -- a P7 server dialling a box without a pack spends the load and then fails its seam
# handshake -- and then launches the API server against it.
#
# stop stops ONLY this HTTP server; the worker is not this script's to stop. That is deliberate:
# the worker's pack load is the expensive half, and an API restart (a new client build, a knob A/B)
# must not cost it. Stop the worker with pair_lib.sh's pair_server_stop, or the pid in
# $(pair_server_pidfile) if the box has to come down; nothing here reads that pidfile.
#
# The server serves the release `current` names, from the same container path pair_arm.sh runs
# clients from, after the same IN_SYNC check pair_serve.sh gates the worker on: an API in front of
# a tree whose artifact came from somewhere else is exactly the delivery this system exists to end.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_server.sh <prod|dev> start [-e K=V ...]|stop|status}
COMMAND=${2:-start}
if [ $# -ge 2 ]; then shift 2; else shift 1; fi
pair_load "$PAIR_ARG"

# HTTP endpoint. The P7 container is created with --network host (containers/*.json), so 127.0.0.1
# inside the container *is* the host's loopback and the host's smoke reaches the same socket.
SRV_HOST=${SRV_HOST:-127.0.0.1}
# RM-2 2026-11-05 (BatchTrace2 finding): every P7 container is --network host, so one shared default
# port let dev's readiness probe pass on PROD's server (dev requests then went to prod). Each pair has
# its own default: prod 8000, dev 8001, ptest 8002; SRV_PORT still overrides.
case "$PAIR" in prod) _defport=8000 ;; dev) _defport=8001 ;; ptest) _defport=8002 ;; *) _defport=8009 ;; esac
SRV_PORT=${SRV_PORT:-$_defport}
# Server-side cache and chunk: the hot pack plus one 64k-token FP4 KV pool fits the card, and the
# chunk the load may take is the shipping 16384-row width unless the caller says otherwise (ChunkAB's
# same-load A/B on f9d57851: 32k prefill 1131.8 -> 1214.2 tok/s, 128k 1100.8 -> 1207.1; the driver's
# PrefillChunkLimiter still shrinks a chunk to the room it has, and MAX_CHUNK=8192 is the escape).
CACHE_TOKENS_CALLER=${CACHE_TOKENS+set}   # a caller's CACHE_TOKENS beats the release's serving.env default
CACHE_TOKENS=${CACHE_TOKENS:-65536}
MAX_CHUNK=${MAX_CHUNK:-16384}
# The client half is the one whose PACK_MANIFEST.json declares the placement_sha256 the GX box's cold pack
# carries; a hard-coded one here would serve the old pack's client into the new pack after the 4780p8 switch
# and be refused by the Hello, or worse, load. It is NAMED, never derived (RM10 2026-11-05, Main):
# CHECKPOINT= / PAIR_ARM_CHECKPOINT / PAIR_PACK_P7 / AB_CKPT. `pair_ckpt_resolve auto` cross-checks the named
# half on stderr and never decides; naming none is a refusal (PAIR_CKPT_UNNAMED).
CHECKPOINT=${CHECKPOINT:-${PAIR_ARM_CHECKPOINT:-}}
if [ "$COMMAND" = start ]; then   # stop/status touch no pack, so they name none and stay usable
  CHECKPOINT=$(pair_ckpt_require "$CHECKPOINT") \
    || pair_die "CHECKPOINT: name the P7 client half (CHECKPOINT= / PAIR_ARM_CHECKPOINT / PAIR_PACK_P7 / AB_CKPT)"
  ckpt_line=$(pair_ckpt_verified "$CHECKPOINT" 2>&1) \
    || pair_die "CHECKPOINT: $CHECKPOINT is not the pack its manifest describes ($ckpt_line)"
  pair_step "$ckpt_line"
  pair_step "PAIR_SRV_CKPT checkpoint=$CHECKPOINT cold=$(pair_cold_src) sha=$(pair_cold_placement)"
fi
PAIR_SRV_READY_S=${PAIR_SRV_READY_S:-1800}   # s: weight load + warm-up before SERVER_READY
PAIR_SRV_STOP_S=${PAIR_SRV_STOP_S:-60}

# Which tree the server imports, and therefore which extension cache it may write. Same rule as
# pair_arm.sh:76-88: torch's version hash covers the include path spelling, so a second checkout
# pointed at /pair/extcache rebuilds exllamav3_ext.so in place and pair_check.sh then calls the
# pair DRIFT. Only the release tree may use the release cache.
SRC=${SRC:-/src/exllamav3}
case "$SRC" in
  /src/exllamav3|/src/exllamav3/|/pair/current/src) EXT_DIR=/pair/extcache ;;
  *)
    ext_tag=$(printf '%s' "$SRC" | sed 's#^/*##; s#/*$##; s#[^A-Za-z0-9._-]#_#g')
    mkdir -p "$PAIR_STATE_DIR/extcache/$ext_tag"
    EXT_DIR=$(pair_ct_path "$PAIR_STATE_DIR/extcache/$ext_tag")
    pair_step "PAIR_SRV_SRC_ISOLATED src=$SRC extcache=$EXT_DIR (not the release cache)"
    ;;
esac

SDIR=$PAIR_STATE_DIR/server
PIDFILE=$SDIR/pair-server.pid
LOG=$SDIR/pair-server.log
CT_SDIR=$(pair_ct_path "$SDIR")
CT_PIDFILE=$(pair_ct_path "$PIDFILE")
CT_LOG=$(pair_ct_path "$LOG")

# The pidfile carries the pid *and* what it was started from, so "is the pair already serving this
# build?" is answerable without reading a container's environment: pair=… pid=… build=… sha=…
# port=… at=…  (the pid is the container's, because docker exec's `exec` keeps the shell's pid).
pidfile_get() {  # <field> -> value, empty when the pidfile or the field is absent
  awk -v k="$1=" '{for (i = 1; i <= NF; i++) if (index($i, k) == 1) {print substr($i, length(k) + 1); exit}}' \
    "$PIDFILE" 2>/dev/null
}
server_pid() { pidfile_get pid; }
# Every liveness question goes through the container's own shell: `docker exec ct kill -0 pid` needs a
# kill binary on the exec path, and a missing one would report every server as dead and let a start
# overwrite a live pidfile.
ct_alive() { docker exec "$P7CT" sh -c 'kill -0 "$1"' sh "$1" >/dev/null 2>&1; }
server_alive() {
  local pid; pid=$(server_pid)
  [ -n "$pid" ] && ct_alive "$pid"
}
current_build() {
  [ -f "$PAIR_ROOT/current/RELEASE.json" ] || return 0
  python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("build_id",""))
except Exception: print("")' "$PAIR_ROOT/current/RELEASE.json" 2>/dev/null
}

# Readiness is the server's own /v1/models, asked from inside the container: 200 with a model list
# means serve_pair.py finished setup() and is accepting; 401 means it is up and enforcing API_KEY.
# The socket must belong to the server this script started: a listener owned by any other pid (another
# pair's server on the same host network) is not readiness.
socket_owner_ok() {
  # Asked inside this pair's own container: ss there shows the owning pid only for its own processes,
  # and the pidfile pid is container-side, so the two compare directly.
  local want=$1
  docker exec "$P7CT" sh -c "ss -Hltnp 'sport = :$SRV_PORT' 2>/dev/null | grep -q 'pid=$want,'" && return 0
  pair_step "PAIR_SRV_PORT_NOT_OURS pair=$PAIR port=$SRV_PORT want_pid=$want ($(ss -Hltn "sport = :$SRV_PORT" 2>/dev/null | head -1 | tr -s ' ' | cut -c1-60))"
  return 1
}

ready_probe() {
  docker exec "$P7CT" python3 -c '
import json, sys, urllib.error, urllib.request
url = "http://%s:%s/v1/models" % (sys.argv[1], sys.argv[2])
try:
    with urllib.request.urlopen(url, timeout=3) as response:
        ok = bool(json.load(response)["data"])
    sys.exit(0 if ok else 1)
except urllib.error.HTTPError as exc:
    sys.exit(0 if exc.code == 401 else 1)
except Exception:
    sys.exit(1)
' "$SRV_HOST" "$SRV_PORT" >/dev/null 2>&1
}

log_tail() {
  docker exec "$P7CT" tail -n 40 "$CT_LOG" 2>/dev/null | sed 's/^/  /'
}

start() {
  local extra=()
  while [ $# -gt 0 ]; do
    case "$1" in
      -e|--env) [ $# -ge 2 ] || pair_die "missing value for $1"
                extra+=(-e "$2"); shift 2 ;;
      --env=*) extra+=(-e "${1#--env=}"); shift ;;
      *) pair_die "usage: pair_server.sh $PAIR start [-e K=V ...]" ;;
    esac
  done

  local SHA REL BUILD_ID
  SHA=$(pair_current_sha "$PAIR_ROOT")
  [ -n "$SHA" ] || pair_die "PAIR_SRV_NO_RELEASE pair=$PAIR (no $PAIR_ROOT/current)"
  REL=$(pair_release_dir "$SHA")
  [ -f "$REL/RELEASE.json" ] || pair_die "PAIR_SRV_NO_RELEASE pair=$PAIR sha=${SHA:0:12} (no RELEASE.json)"
  BUILD_ID=$(python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("build_id",""))
except Exception: print("")' "$REL/RELEASE.json")

  # One model per card: a second instance cannot exist, so an already-serving pair is either this
  # release (report it, exit 0 -- the caller wanted the steady state and it is there) or somebody
  # else's load, which only a person may evict.
  if server_alive; then
    local have_pid have_build have_sha have_port
    have_pid=$(server_pid); have_build=$(pidfile_get build)
    have_sha=$(pidfile_get sha); have_port=$(pidfile_get port)
    if [ -n "$BUILD_ID" ] && [ "$have_build" = "$BUILD_ID" ] && [ "$have_sha" = "$SHA" ] \
       && [ "$have_port" = "$SRV_PORT" ]; then
      pair_step "PAIR_SRV_ALREADY_UP pair=$PAIR build=$BUILD_ID sha=${SHA:0:12} port=$have_port pid=$have_pid ct=$P7CT"
      exit 0
    fi
    pair_die "PAIR_SRV_ALREADY pair=$PAIR pid=$have_pid build=${have_build:-?} sha=${have_sha:0:12} port=${have_port:-?} (started for another release: pair_server.sh $PAIR stop)"
  elif [ -e "$PIDFILE" ]; then
    rm -f "$PIDFILE"        # recorded pid is gone: a stale pidfile, not a running server
    pair_step "PAIR_SRV_STALE_PID removed=$PIDFILE (recorded pid is gone)"
  fi

  local envs=(
    # Stated, not inherited (pair_arm.sh:88-92): a SRC above must not reach the release's extension
    # cache through a variable the container happens to carry.
    -e "PYTHONPATH=$SRC" -e "TORCH_EXTENSIONS_DIR=$EXT_DIR"
    # Same pin the client and the worker get, for the server's half of the autotuner: unset,
    # EXLLAMAV3_TUNE_CACHE resolves under $HOME *inside* the container and the picks die with it.
    -e "EXLLAMAV3_TUNE_CACHE=$PAIR_TUNE_CACHE"
    # The two caches the server's own prefill fills, same pins as the client and the worker
    # (pair_lib.sh:54-64, pair_arm.sh:116-120): unset, Triton's kernels and torch's NVRTC cubin are
    # compiled into the container's own /root and die with it, so the first request after a restart
    # pays for them again (measured 2026-11-03: 12.9 s on the first prefill arm, ~6 s of it still open).
    -e "TRITON_CACHE_DIR=$PAIR_TRITON_CACHE"
    -e "TORCH_HOME=$PAIR_TORCH_HOME"
    # The startup warm-up itself (serve_pair.py:62-72,143-168). On by default because the cost it
    # hides is real and is paid again on every restart; `pair_server.sh dev start -e WARM_PREFILL=0`
    # turns it off, and WARM_PREFILL_ROWS overrides the big chunk (default: the load-time row count).
    -e "WARM_PREFILL=${WARM_PREFILL:-1}"
    -e "WARM_PREFILL_ROWS=${WARM_PREFILL_ROWS:-0}"
    # pairs.conf's `gpu` is a host index (spent on --gpus device=N at container creation), so inside
    # the container that one card is device 0 again -- the index serve_pair.py loads on.
    -e "CUDA_VISIBLE_DEVICES=0"
    -e "REMOTE_TRANSPORT=seam"
    -e "SEAM_DEV=$SEAM_DEV" -e "SEAM_GID=$SEAM_GID"
    -e "SEAM_PEER=$SEAM_PEER" -e "SEAM_PORT=$PORT"
    # The pair's own bucket list, not the client fallback: it does not contain the shipping 8192-row
    # prefill width, and the grant would then be the intersection's 1,6,64 (pair_arm.sh:100-106).
    -e "SEAM_BUCKETS=$BUCKETS"
    -e "HOST=$SRV_HOST" -e "PORT=$SRV_PORT"
    -e "CHECKPOINT=$CHECKPOINT" -e "DRAFT_CHECKPOINT=$CHECKPOINT"
    -e "CACHE_TOKENS=$CACHE_TOKENS"
    # The shipping client config, one list, copied from pair_arm.sh:110-117 so the API serves the
    # shape the numbers were taken on; a release that needs a different value changes both. (The
    # launcher's DSPARK/NUM_DRAFT/MAX_CHUNK are the same ones serve_pair.py would default to.)
    -e "DSPARK=1" -e "NUM_DRAFT=5"
    # Graphs on from 2026-11-04, same default and same escape as pair_arm.sh:152 (the A/B, the identity
    # check and why the switch is env and not code are documented there). The API path is new to graphs
    # -- that is what pair_server_smoke.sh and the final block's WG=0 reference arm exist to cover.
    -e "KV_BITS=4" -e "EXL3_WHOLE_GRAPH=${PAIR_WHOLE_GRAPH:-1}" -e "MAX_CHUNK=$MAX_CHUNK"
    -e "EXL3_MOE_SPLIT_FUSED=0"
    # Off from 2026-11-04 (see pair_arm.sh): ~0.7 ms/round of cudaEvent create/record, 1.2 % of a round.
    -e "EXL3_SEAM_PROF=0"
    -e "PREFILL_ADAPTIVE=1" -e "PREFILL_ADAPTIVE_LOG=1"
    -e "SEAM_ROWS=8192"
    -e "SEAM_NS=1" -e "SEAM_OVERLAP=1" -e "SEAM_COPY_STREAM=1"
    # The drafter's gate: the k35 expected-value policy that shipped (MORNING_REPORT, "Adaptive K
    # {3,5} (EV gate) -- WIN"): stop at K=3 where acceptance saturates, K=5 otherwise. Client-side
    # knobs only -- they price the verify rows this card pays for.
    -e "EXL3_DSPARK_GATE=ev" -e "EXL3_DSPARK_CONF=0" -e "EXL3_DSPARK_EV_KMIN=3"
    -e "EXL3_DSPARK_EV_COST=1:1e9,2:1e9,3:1e9,4:75.69,5:1e9,6:82.95"
  )
  # The client half of pair_lib.sh's MOE_DET -- same knob, same reason as pair_arm.sh: BATCH>1 is
  # refused at startup unless this process carries SEAM_MOE_CPP_DET=1, and the worker must be on the
  # same epilogue for the refusal to mean anything. Installed 2026-11-05 07:21 with pair_lib.sh
  # e51d8096 (this file f55ce81e), so `PAIR_SRV_EXTRA_ENVS="BATCH=8"` alone reaches batch mode.
  if [ "$MOE_DET" = 1 ]; then envs+=(-e "SEAM_MOE_CPP_DET=1"); fi
  local release_env
  release_env=$(pair_release_env) || exit 1
  for kv in $release_env; do envs+=(-e "$kv"); done
  # The release's serving defaults (qmesh/release/serving.env, R5 on): after the base list so they replace its
  # values (docker keeps the last -e), before PAIR_SRV_EXTRA_ENVS so an A/B still overrides them.
  local serving_env
  serving_env=$(pair_release_serving_env) || exit 1
  for kv in $serving_env; do
    case "$kv" in CACHE_TOKENS=*) [ -n "$CACHE_TOKENS_CALLER" ] && continue; CACHE_TOKENS=${kv#*=} ;; esac
    envs+=(-e "$kv")
  done
  [ -n "$serving_env" ] && pair_step "PAIR_SRV_RELEASE_ENV $(printf '%s ' $serving_env)cache_tokens=$CACHE_TOKENS"
  # Identity and authorization are not shape, so they are forwarded from the caller's environment
  # when set: `MODEL_ID=… API_KEY=… pair_server.sh dev start` serves under that name and key (the
  # readiness probe counts a 401 as "up" for the same reason). Every other knob is a shape knob and
  # goes through PAIR_SRV_EXTRA_ENVS, where an A/B shows up as one line rather than an exported var.
  [ -n "${MODEL_ID:-}" ] && envs+=(-e "MODEL_ID=$MODEL_ID")
  [ -n "${API_KEY:-}" ] && envs+=(-e "API_KEY=$API_KEY")
  [ -n "$BUILD_ID" ] && envs+=(-e "SEAM_BUILD_ID=$BUILD_ID")
  # PAIR_SRV_EXTRA_ENVS: the pair's server-half A/B door (pair_lib.sh:106), the counterpart of
  # PAIR_ARM_EXTRA_ENVS. pair_lib forwards it to the GX worker through MOE_ENV; it is appended here
  # too, because the P7 side now runs a server of its own. Docker keeps the last -e, so it can
  # override a shipping knob -- and it must stay empty for anything quoted as the shipped shape.
  local kv
  for kv in ${SRV_ENV_EXTRA:-}; do envs+=(-e "$kv"); done

  if [ "$PAIR_DRY" = 1 ]; then
    echo "DRY ensure $PAIR_SCRIPTS/pair_serve.sh $PAIR"
    echo "DRY docker exec -d -w $SRC ${envs[*]} ${extra[*]} $P7CT bash -lc 'cd $SRC && python3 -u qmesh/serve_pair.py > $CT_LOG 2>&1' (pidfile $CT_PIDFILE port $SRV_PORT)"
    return 0
  fi

  # IN_SYNC first, exactly as pair_serve.sh does for the worker: the sync check is the only thing
  # that knows whether the on-disk artifact is the one the stamp names, and an API in front of a
  # mismatched extension serves a different model than the one this build is stamped as.
  local chk
  chk=$("$PAIR_SCRIPTS/pair_check.sh" "$PAIR" 2>&1)
  printf '%s\n' "$chk" | sed 's/^/  /'
  case "$chk" in
    *DRIFT*|*MISSING*) pair_die "PAIR_SRV_DRIFT pair=$PAIR (build it first: scripts/pair_build_ext.sh $PAIR --side p7|gx)" ;;
  esac
  case "$chk" in
    *"RESULT $PAIR P7 IN_SYNC"*"RESULT $PAIR GX IN_SYNC"*) ;;
    *) pair_die "PAIR_SRV_NOT_IN_SYNC pair=$PAIR (pair_check did not report both halves IN_SYNC)" ;;
  esac

  # The worker first, and through its own script: it owns the box claim, the pack load and the
  # pidfile a later ticket stops, and it is a no-op when this build already serves (PAIR_SERVE_
  # ALREADY_UP, still rc 0). A worker serving another build is refused there, not silently reused.
  local gxrc gxout
  gxout=$("$PAIR_SCRIPTS/pair_serve.sh" "$PAIR" 2>&1); gxrc=$?
  printf '%s\n' "$gxout" | grep -E '^(PAIR_SERVE|SEAM_|RESULT|STALE_)' | sed 's/^/  /'
  # pair_die's own words ("pair: PAIR_SERVE_GX_BUSY ...") carry the one line that says why the worker did
  # not come up -- the gate already held, nvml gone, the pack not covered by avail -- and the filter above
  # drops them, which is how an 11:52 run reported only "rc=2". Echo them on the failure path.
  if [ "$gxrc" != 0 ]; then
    printf '%s\n' "$gxout" | grep -E '^pair: ' | sed 's/^/  /'
    pair_die "PAIR_SRV_GX_FAILED pair=$PAIR rc=$gxrc (the worker did not come up; see pair_serve.sh $PAIR)"
  fi

  mkdir -p "$SDIR"
  : > "$LOG"
  pair_step "PAIR_SRV_START pair=$PAIR sha=${SHA:0:12} build=${BUILD_ID:-unknown} port=$SRV_PORT ct=$P7CT"
  # `exec` matters twice: the pid the pidfile records becomes the python process's own (so one
  # SIGTERM reaches the server, not a wrapper shell), and `-d` leaves it running after docker exec
  # returns. The log is truncated per start, like the worker's: this file is the record of the load
  # now running, not of every load the box ever had. `cd` is explicit as well as `-w`, because a
  # login shell's profile is free to change the directory and the script is found relative to it.
  docker exec -d -w "$SRC" "${envs[@]}" "${extra[@]}" "$P7CT" bash -lc \
    'mkdir -p "$1" && cd "$8" && printf "pair=%s pid=%s build=%s sha=%s port=%s at=%s\n" "$4" $$ "$5" "$6" "$7" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$3" && exec python3 -u qmesh/serve_pair.py > "$2" 2>&1' \
    sh "$CT_SDIR" "$CT_LOG" "$CT_PIDFILE" "$PAIR" "${BUILD_ID:-unknown}" "$SHA" "$SRV_PORT" "$SRC" \
    || pair_die "PAIR_SRV_LAUNCH pair=$PAIR ct=$P7CT (docker exec failed)"

  local waited=0 pid
  while [ "$waited" -lt "$PAIR_SRV_READY_S" ]; do
    pid=$(server_pid)
    if [ -n "$pid" ] && ! ct_alive "$pid"; then
      pair_step "PAIR_SRV_DIED pair=$PAIR pid=$pid after ${waited}s"
      log_tail
      rm -f "$PIDFILE"
      exit 1
    fi
    if [ -n "$pid" ] && socket_owner_ok "$pid" && ready_probe; then
      pair_step "PAIR_SRV_UP pair=$PAIR sha=${SHA:0:12} build=${BUILD_ID:-unknown} port=$SRV_PORT pid=$pid http://$SRV_HOST:$SRV_PORT log=$LOG"
      # `log=` is the load's own log: PREFIX_CACHE, DSPARK_GATE and MAX_CHUNK_EFFECTIVE are printed by the
      # *load* and land there, not in this launcher's stdout (srv_start.log holds only PAIR_* lines). Anything
      # that reads a pool size or the resolved gate back must take the path from this line and follow it, or it
      # greps empty and reports a pool that was never measured (Ports2x, 2026-11-04).
      return 0
    fi
    sleep 5; waited=$((waited + 5))
  done
  # Still loading after the deadline: stop what this start began rather than leave 60+ GiB half-in.
  pid=$(server_pid)
  pair_step "PAIR_SRV_TIMEOUT pair=$PAIR pid=${pid:-?} waited=${waited}s"
  log_tail
  if [ -n "$pid" ]; then
    docker exec "$P7CT" sh -c 'kill -TERM "$1"' sh "$pid" >/dev/null 2>&1
    local stop_waited=0
    while [ "$stop_waited" -lt 30 ] && ct_alive "$pid"; do
      sleep 1; stop_waited=$((stop_waited + 1))
    done
  fi
  rm -f "$PIDFILE"
  exit 1
}

stop() {
  local pid build
  pid=$(server_pid); build=$(pidfile_get build)
  if [ -z "$pid" ]; then
    pair_step "PAIR_SRV_STOP_NONE pair=$PAIR (no pidfile $PIDFILE)"
    rm -f "$PIDFILE"
    return 0
  fi
  if [ "$PAIR_DRY" = 1 ]; then
    echo "DRY docker exec $P7CT kill -TERM $pid"
    return 0
  fi
  # Anchored on the process's own argv: gone, or the pid now belongs to something else (a container
  # restart reuses pids), and killing by a number read from an old file is how an unrelated process
  # dies. -Fqz because /proc/<pid>/cmdline is NUL-separated, so each argv element is one record.
  if ! docker exec "$P7CT" sh -c 'grep -Fqz "qmesh/serve_pair.py" "/proc/$1/cmdline"' sh "$pid" 2>/dev/null; then
    pair_step "PAIR_SRV_STOP_SKIP pid=$pid (gone or not this server; pidfile removed)"
    rm -f "$PIDFILE"
    return 0
  fi
  docker exec "$P7CT" sh -c 'kill -TERM "$1"' sh "$pid" >/dev/null 2>&1 \
    || { pair_step "PAIR_SRV_STOP_FAILED kill -TERM $pid"; return 1; }
  local waited=0
  while [ "$waited" -lt "$PAIR_SRV_STOP_S" ]; do
    ct_alive "$pid" || break
    sleep 1; waited=$((waited + 1))
  done
  if ct_alive "$pid"; then
    pair_step "PAIR_SRV_STOP_TIMEOUT pid=$pid waited=${waited}s (SIGTERM ignored; pidfile kept, SIGKILL is a person's call)"
    return 1
  fi
  rm -f "$PIDFILE"
  pair_step "PAIR_SRV_STOP pair=$PAIR pid=$pid build=${build:-?} (GX worker untouched: $(pair_server_pidfile))"
}

status() {
  local pid build sha port alive=no
  pid=$(server_pid); build=$(pidfile_get build); sha=$(pidfile_get sha); port=$(pidfile_get port)
  [ -n "$build" ] || build=$(current_build)
  if [ "$PAIR_DRY" = 1 ]; then
    echo "DRY docker exec $P7CT kill -0 ${pid:-?}"
    return 0
  fi
  if [ -n "$pid" ] && ct_alive "$pid"; then
    alive=yes
  fi
  printf 'PAIR_SRV_STATUS pair=%s pid=%s alive=%s build=%s sha=%s port=%s ct=%s log=%s\n' \
    "$PAIR" "${pid:--}" "$alive" "${build:-unknown}" "${sha:0:12}" "${port:--}" "$P7CT" "$LOG"
  case "$alive" in
    yes) return 0 ;;
    *) return 1 ;;
  esac
}

case "$COMMAND" in
  start) start "$@" ;;
  stop) stop ;;
  status) status ;;
  *) pair_die "usage: pair_server.sh <prod|dev> start [-e K=V ...]|stop|status" ;;
esac
