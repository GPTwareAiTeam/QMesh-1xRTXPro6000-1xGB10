#!/usr/bin/env bash
# Versioned copy of the operational script: install as
#   $HOME/recipe-artifacts/gpu0-gx1/scripts/seam_server_ensure.sh
# Hardened on branch gxport2/next: seam_preflight() before every start (memory, other containers'
# servers, builds), no wait loops; a refusal returns 3 and prints PREFLIGHT_FAIL.
# Shared "ensure the cold-expert seam server is running" helper for the bench scripts.
#
# The seam server now holds its session model forever: it loads the cold pack once and serves
# any number of client lifetimes through the control port. A script therefore no longer needs
# a fresh server per arm -- reusing a running one keeps every arm on the same build/placement
# and costs no reload. Restart only when you explicitly want a clean server (--fresh).
#
#   seam_server_ensure --stub --buckets 1,2,3 --wait    # protocol work, no pack
#   seam_server_ensure --buckets auto                   # size the buckets to the box
# Both return the real status: 0 running, 1 the server died, 2 usage, 3 preflight refused (the
# printed PREFLIGHT_FAIL says which: gx-busy, gx-mem, gate_*, or a failed build).
#
# Knobs (env): GX (required ssh host), CT (container, default seam-gx2),
#   GX_DEV (rocep1s0f0), GX_GID (3), PORT (47311), SEAM_LOG (/tmp/seam_server.log),
#   SERVER_BUCKETS (1,6,64,256,2048; auto = seam_auto_buckets), SEAM_SERVER_TIMEOUT (900 s),
#   SEAM_SLAB_SLOTS (1; payload slots per handle, 1..8: NS>1 lets a client pipeline prefill
#   micro-batches so slot k+1's wire transfer overlaps slot k's GX compute),
#   SEAM_MODEL_DIR (/cold), SEAM_PLACEMENT (/cold/placement.exl3.json), SEAM_EXTRA (extra argv),
#   SEAM_ENV (space-separated K=V exported into the server container; e.g. SEAM_ENV="QES_TRACE=1"
#   turns on the transport's write/geometry trace, which lands in SEAM_LOG next to the server log).
#   auto sizing: SEAM_PACK_MIB (116000), SEAM_SLAB_FLOOR_MIB (1024), SEAM_MAX_ROWS (8192).
#
# Readiness = "^SEAM_SERVER_READY" in the log, which the C server prints only after
# worker_seam_server.py finished loading the pack (SEAM_SERVER_LOADED) and opened the control
# port, so READY implies the cold experts are live. A reused server additionally has to serve
# every bucket this job will request; a missing bucket is a hard error here, not a mismatch
# deep inside a run.
#
# The kill/start calls use a container-wide pkill, so a container must host only this pair's seam
# server (true of seam-gx1b/seam-gx1v2/seam-gx2). A second server there would be collateral.
set -uo pipefail

GX=${GX:?GX must name the remote host}
CT=${CT:-seam-gx2}
GX_DEV=${GX_DEV:-rocep1s0f0}
GX_GID=${GX_GID:-3}
PORT=${PORT:-47311}
SEAM_LOG=${SEAM_LOG:-/tmp/seam_server.log}
# Exported, not just set: the preflight assembles its probe in a child shell, so an unexported SEAM_LOG
# reaches that child as an unbound `${SEAM_LOG}` under `set -u` and the probe dies before it looks at the
# box -- printed as `SEAM_LOG: unbound variable`, or as a bare failed grep when stderr is redirected
# (2026-11-03, PairRelease: a caller that passed SEAM_LOG as a prefix assignment to `.` could not probe
# a log at all; the ticket reported `probe_marker_not_reached` for a probe that was never run).
export SEAM_LOG
# Where the previous server log is copied before a start truncates it. Empty disables the copy; the
# pair scripts set it to <pair state>/logs (pair_lib.sh), so every load -- smoke, update, serve,
# restore -- leaves behind the record its successor would otherwise destroy.
SEAM_LOG_ARCHIVE=${SEAM_LOG_ARCHIVE:-}
SERVER_BUCKETS=${SERVER_BUCKETS:-1,6,64,256,2048}
SEAM_SERVER_TIMEOUT=${SEAM_SERVER_TIMEOUT:-900}
SEAM_LEGACY_GRACE=${SEAM_LEGACY_GRACE:-60}    # s to wait for SEAM_SERVER_READY after LOADED
SEAM_MODEL_DIR=${SEAM_MODEL_DIR:-/cold}
SEAM_PLACEMENT=${SEAM_PLACEMENT:-/cold/placement.exl3.json}
SEAM_SLAB_SLOTS=${SEAM_SLAB_SLOTS:-1}
SEAM_EXTRA=${SEAM_EXTRA:-}
# SEAM_SERVER_WRAP prefixes the worker's python3 with a launcher, for a profiler and nothing else:
#   SEAM_SERVER_WRAP="nsys profile -t cuda -o /tmp/gx --force-overwrite true"
# It is measurement-only state, so it stays empty for every arm whose number is quoted in the notes,
# and it must not carry a single quote or a `$` (it is pasted inside a bash -lc payload through ssh).
SEAM_SERVER_WRAP=${SEAM_SERVER_WRAP:-}
SEAM_ENV=${SEAM_ENV:-}
# Pre-flight before a server is started: one 113 GiB pack load on a 121 GiB box dies (or wedges the box)
# when something else is already running there. Every start therefore requires, with no waiting:
#   - the GX gate (see below) held by this process or one of its ancestors, so two agents can not both
#     decide the box is free and load on top of each other,
#   - no worker_seam_server.py in any container on the box (this pair's own container excluded, its
#     server is killed just before the check); the refusal names the container and pid,
#   - for a real start (--stub excluded: nothing is loaded) MemAvailable >= GX_PREFLIGHT_AVAIL_GIB
#     (default 115), and a short fall names the container that is holding the memory,
#   - no ninja / nvcc / cicc process on the box (an extension build).
# A failed check prints PREFLIGHT_FAIL and returns 3; nothing is started or waited for, and the
# command exits with the real status (a wrapper that mapped every failure to 2 hid the reason from
# pair_smoke.sh, which classifies on rc 3 plus the line).
GX_PREFLIGHT=${GX_PREFLIGHT:-1}
GX_PREFLIGHT_AVAIL_GIB=${GX_PREFLIGHT_AVAIL_GIB:-115}
#
# The gate is the same one-line file pair_lib.sh uses, <queue dir>/gx<N>.gate:
#   owner=<label> pid=<pid> until=<epoch seconds>
# pair_smoke.sh takes it around a whole smoke; this helper only *checks* it, never takes or drops it:
#   SEAM_GATE_REQUIRED=1 (default) refuses to start or reuse a server when the file is missing,
#   expired, stale (its pid is gone) or held by a live process that is not this one or an ancestor.
# That makes the gate binding for every launcher that goes through this script: a bench that skips
# the take can no longer start a server while a smoke holds the box, and a smoke can not be handed a
# box another job claimed. SEAM_GATE_REQUIRED=0 is for stub harnesses only.
SEAM_GATE_DIR=${SEAM_GATE_DIR:-${PAIR_QUEUE_ROOT:-${PAIR_ART:-$HOME/recipe-artifacts/gpu0-gx1}/queue}}
# The gate belongs to the *pair's* container, not to whichever container this job runs in: CT may be a
# private container that shares the box (a worktree of the same pack), and the gate that protects that
# box is still gx<N>.gate. Default unchanged, so every existing caller keeps CT's gate.
SEAM_GATE_CT=${SEAM_GATE_CT:-$CT}
SEAM_GATE_REQUIRED=${SEAM_GATE_REQUIRED:-1}

seam_gxsh() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "$GX" "$1"; }
# The payload goes through two shells (ssh's, then `docker exec … bash -lc`), so a single quote inside it used to
# end the wrapper's argument and hand bash -lc a truncated command: `grep -aq '^SEAM_SERVER_READY' $SEAM_LOG`
# arrived as two words and the probe answered "not ready" forever, and `seam_kill_pattern` sent no signal at all
# (2026-11-03, VisionTest found it by shlex-splitting this exact string). Escape each inner quote for the outer
# shell; nothing else is touched, so callers still write $VAR and \$(...) deliberately.
seam_gxct() { local q=${1//\'/\'\\\'\'}; ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "$GX" "docker exec $CT bash -lc '$q'"; }
# ---- server log archive ----------------------------------------------------------------------
# Every start below ends in `rm -f $SEAM_LOG`, and that log carries the only per-load record of
# which layers the cuda-exl3 restack refused (`SEAM_MOE_NOFIT layer=N`). Those lines decide whether
# two ids digests are comparable at all: on 2026-11-02 the canary drew 38/2 on one load and 39/1 on
# the next, produced two different digests, and the 38/2 log was already gone, so the difference
# could not be attributed. Copy is best effort by design -- a pair must never stay loaded because an
# archive write failed -- and it never overwrites: the destination names the load and its time.
seam_archive_log() {
  local dir=${SEAM_LOG_ARCHIVE:-} bytes build ts dest
  [ -n "$dir" ] || return 0
  bytes=$(seam_gxct "wc -c < $SEAM_LOG 2>/dev/null" 2>/dev/null | tr -dc '0-9')
  [ "${bytes:-0}" -gt 0 ] || return 0
  build=$(seam_gxct "grep -o \"build=[A-Za-z0-9._-]*\" $SEAM_LOG 2>/dev/null | head -1 | cut -d= -f2" 2>/dev/null |
          tr -dc 'A-Za-z0-9._-')
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "$dir" 2>/dev/null || return 0
  dest=$dir/server-${build:-unknown}-$ts.log
  seam_gxsh "docker exec $CT bash -lc 'cat $SEAM_LOG'" > "$dest" 2>/dev/null || { rm -f "$dest"; return 0; }
  [ -s "$dest" ] || { rm -f "$dest"; return 0; }
  echo "SEAM_LOG_ARCHIVED build=${build:-unknown} bytes=$(wc -c < "$dest") dest=$dest"
}
# The worker is `python3 ... worker_seam_server.py` whatever path it was started with: the ensure path
# starts it relative (`qmesh/worker_seam_server.py`), the release serve path absolute
# (`/pair/current/src/qmesh/worker_seam_server.py`), and an argv-anchored match saw only the first --
# so a resident pack was invisible to this file and to the box scan, the smoke loaded a second one
# beside it and failed box_busy with 1005 MiB free (2026-11-03). Match the interpreter's comm (pgrep -x,
# so the `bash -lc` wrapper that carries the script name in its argv cannot be counted) and the script
# name anywhere in its arguments (bracketed, so no grep in this file matches its own line).
seam_pids()  { seam_gxct "pgrep -x python3 -a" 2>/dev/null | grep -c 'worker_seam_server[.]py'; }
# Every payload above is quote-free for the same reason: one string through ssh's shell and then bash -lc.
# A `'` there is escaped by seam_gxct now, but a payload that needs no quotes cannot be broken by a future
# wrapper change either -- and tests/sweep_rules.py refuses one.
# The same selection as a kill list: never a pattern kill on a name that a wrapper also carries. Quote-free on
# purpose -- it is pasted into seam_gxct and an inner quote there ends the wrapper's argument -- and the match is
# read from /proc, so neither the `bash -lc … stdbuf … worker_seam_server.py` launcher nor ssh's own command line
# can be selected. `[.]` anywhere in this file would also make a grep match itself; /proc cannot.
seam_kill_pattern() { printf '%s' "for p in \$(pgrep -x python3); do grep -qa worker_seam_server /proc/\$p/cmdline && kill ${1:+$1 }\$p; done"; }
# ---- the GX gate ------------------------------------------------------------------------------
# The file is pair_lib.sh's: <queue dir>/gx<N>.gate, one line "owner=<label> pid=<pid> until=<epoch>".
# A claim whose until is past, whose pid is gone, or that is malformed does not protect anything, so
# it is refused by name instead of being quietly replaced: the caller takes the gate, and taking it
# is a deliberate act with a new until. This script never writes the file -- pair_gx_gate_take and
# pair_gx_gate_down own its lifecycle -- it only answers "may I touch this box".
seam_gate_file() {  # seam-gx2-dev -> <queue dir>/gx2.gate, the same mapping pair_lib.sh uses
  local n=${SEAM_GATE_CT#seam-gx}
  printf '%s/gx%s.gate\n' "$SEAM_GATE_DIR" "${n%%-*}"
}
seam_gate_is_ancestor() {  # <pid> -> 0 when it is this process or one of its ancestors
  local want=$1 cur=$$
  [ -n "$want" ] || return 1
  while [ -n "$cur" ] && [ "$cur" -gt 0 ]; do
    [ "$cur" = "$want" ] && return 0
    [ "$cur" = 1 ] && break
    cur=$(ps -o ppid= -p "$cur" 2>/dev/null | tr -dc 0-9)
  done
  return 1
}
seam_gate_check() {  # -> 0 allowed; prints PREFLIGHT_FAIL gate_* and returns 3 otherwise
  [ "$SEAM_GATE_REQUIRED" = 1 ] || return 0
  local f o p u h now
  f=$(seam_gate_file)
  if [ ! -f "$f" ]; then
    echo "PREFLIGHT_FAIL gate_free owner=none file=$f host=$GX (take the gate before loading: gx${SEAM_GATE_CT#seam-gx} is shared)"
    return 3
  fi
  o=$(sed -n 's/^owner=\([^ ]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  p=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  u=$(sed -n 's/.*until=\([0-9][0-9]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  h=$(sed -n 's/.*host=\([^ ]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  now=$(date +%s)
  case "${u:-}" in
    ''|*[!0-9]*)
      echo "PREFLIGHT_FAIL gate_bad owner=${o:-?} file=$f (no until=; take the gate again)"
      return 3 ;;
  esac
  if [ "$u" -lt "$now" ]; then
    echo "PREFLIGHT_FAIL gate_expired owner=${o:-?} until=$u now=$now (take the gate again)"
    return 3
  fi
  if seam_gate_is_ancestor "${p:-}"; then
    return 0
  fi
  # The pid is only evidence on the host that wrote the claim: a claim from another box (or from a
  # container whose /proc is a different namespace) carries a pid this shell cannot see, and calling
  # that stale is how a live claim got overwritten on 2026-11-02 02:26:47.
  if [ -z "${p:-}" ] || { [ "${h:-$(hostname -s)}" = "$(hostname -s)" ] && ! kill -0 "$p" 2>/dev/null; }; then
    echo "PREFLIGHT_FAIL gate_stale owner=${o:-?} pid=${p:-none} (that process is gone; take the gate again)"
    return 3
  fi
  echo "PREFLIGHT_FAIL gate_foreign owner=${o:-?} pid=$p until=$u file=$f (${CT} is someone else's until then)"
  return 3
}
# A READY line is only true while the process that wrote it is alive: a server killed from outside
# (kill -9, OOM, another job's pkill) leaves its log behind, and a stale READY made seam_server_ensure
# answer "reused" on a box that had no server at all - the client then failed on a dead control port.
seam_ready() {
  local n
  n=$(seam_pids)
  [ "${n:-0}" -gt 0 ] || return 1
  seam_gxct "grep -aq ^SEAM_SERVER_READY $SEAM_LOG" >/dev/null 2>&1
}
seam_ready_line() { seam_gxct "grep -a ^SEAM_SERVER_READY $SEAM_LOG | tail -1"; }
seam_served() {  # buckets=... from the READY line, e.g. "1,2,3,4,5,6,64"
  seam_ready_line 2>/dev/null | sed -n 's/.*buckets=\([0-9,]*\).*/\1/p'
}
seam_slots() {  # slots=... from the READY line (protocol v3); empty on an older build
  seam_ready_line 2>/dev/null | sed -n 's/.*slots=\([0-9]*\).*/\1/p'
}
seam_slack() {  # avail_mib=/swapped_mib=/ws_mib= from the server's own memory gate
  seam_gxct "grep -a ^SEAM_SERVER_SLAB $SEAM_LOG | tail -1"
}
# A server can pass `fits=1` and still be unusable: on 2026-11-01 09:13 GX2 started with
# avail_mib=1226 swapped_mib=3630 (a 113.54 GiB pack on a 124.5 GiB box) and its 47th call, a
# one-row decode, died with "CUDA error: unspecified launch failure". `fits` only asks whether the
# slab+workspace allocation succeeds. What compute needs is slack *after* the workspace: every
# forward allocates transient device buffers on top of ws_mib, and on this box device memory is
# host memory, so a thin avail is a crash budgeted minutes later. The number is therefore reported
# on every start (SEAM_SLACK) and a thin one warns; SEAM_SLACK_GATE=1 turns that into a refusal for
# runs whose numbers would be a lie.
SEAM_SLACK_MIN_MIB=${SEAM_SLACK_MIN_MIB:-256}
SEAM_SLACK_SWAP_MIB=${SEAM_SLACK_SWAP_MIB:-512}
SEAM_SLACK_GATE=${SEAM_SLACK_GATE:-0}
seam_slack_gate() {
  local line avail swap ws slack
  line=$(seam_slack)
  avail=$(printf '%s\n' "$line" | sed -n 's/.*avail_mib=\([0-9-]*\).*/\1/p')
  swap=$(printf '%s\n' "$line" | sed -n 's/.*swapped_mib=\([0-9]*\).*/\1/p')
  ws=$(printf '%s\n' "$line" | sed -n 's/.*ws_mib=\([0-9.]*\).*/\1/p')
  [ -n "$avail" ] || { echo "SEAM_SLACK unavailable (no SEAM_SERVER_SLAB line)" >&2; return 0; }
  ws=${ws%%.*}; ws=${ws:-0}
  slack=$(( avail - ws ))
  echo "SEAM_SLACK avail_mib=$avail ws_mib=$ws swapped_mib=${swap:-?} slack_mib=$slack min_mib=$SEAM_SLACK_MIN_MIB gate=$SEAM_SLACK_GATE"
  if [ "${swap:-0}" -gt "$SEAM_SLACK_SWAP_MIB" ]; then
    echo "SEAM_SLACK_WARN swapped_mib=$swap: the pack is paging; real forwards will see multi-second outliers or launch failures (--ws-rows, a narrower --buckets, or a smaller cold pack)" >&2
  fi
  if [ "$slack" -lt "$SEAM_SLACK_MIN_MIB" ]; then
    echo "SEAM_SLACK_WARN slack_mib=$slack below min_mib=$SEAM_SLACK_MIN_MIB: a forward needs transient device buffers beyond ws_mib" >&2
    [ "$SEAM_SLACK_GATE" = 1 ] && return 4
  fi
  return 0
}
# `--slab-slots` and the `slots=` READY field are protocol v3: only the newest seam branch has them.
# Every tree this helper is pointed at today (v2 worktree `e9fedb62`, staging `d763aa13`) is v2 and
# aborts on the flag with `unrecognized arguments: --slab-slots`, so probe the file that will run and
# only pass it when the build knows it. Cached: one ssh per script run.
SEAM_SLOTS_OK=""
seam_slots_supported() {
  if [ -z "$SEAM_SLOTS_OK" ]; then
    if seam_gxct "grep -qs -- --slab-slots /src/exllamav3/qmesh/worker_seam_server.py" >/dev/null 2>&1; then
      SEAM_SLOTS_OK=1
    else
      SEAM_SLOTS_OK=0
    fi
  fi
  [ "$SEAM_SLOTS_OK" = 1 ]
}

# A pre-multi-session build (GX1 until it is synced) never prints SEAM_SERVER_READY: its log stops at
# SEAM_SERVER_LOADED. Accept that as "started" so a converted script still works there, but only in the
# start path and only after the new build has had 20 s to print READY; reuse never accepts it, because
# such a server dies with its one client.
seam_loaded() { seam_gxct "grep -aq ^SEAM_SERVER_LOADED $SEAM_LOG" >/dev/null 2>&1; }
seam_loaded_line() { seam_gxct "grep -a ^SEAM_SERVER_LOADED $SEAM_LOG | tail -1"; }
seam_kind() {  # 1 = stub (no experts loaded, every forward answered with zeros), 0 = real pack
  seam_loaded_line 2>/dev/null | sed -n 's/.*stub=\([01]\).*/\1/p'
}

# Runs a script on the GX host through stdin, so remote loops need no local quoting.
seam_gxsh_stdin() { ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "$GX" bash -s; }

# Pinned-slab cost of a bucket list, MiB per side, mirroring geometry() in
# qmesh/native/qmesh_expert_seam.cu protocol v3: per rows bucket `slots` payload slots of
# 3 x 64 B flag/header lines + align64(req) + align64(resp), req = rows*(HIDDEN*2 + TOPK*4 +
# TOPK*2), resp = rows*HIDDEN*4. That is ~30.2 KiB per row per slot, so the whole standard bucket
# set costs tens of MiB instead of the 1.18 MiB per row the per-layer layout needed.
seam_slab_mib() {
  awk -v list="${1:-}" -v slots="${2:-1}" 'BEGIN {
    split(list, b, ","); tot = 0
    for (i = 1; i <= length(b); i++) {
      n = b[i] + 0
      req = n * (5120 * 2 + 6 * 4 + 6 * 2); resp = n * 5120 * 4
      tot += (192 + int((req + 63) / 64) * 64 + int((resp + 63) / 64) * 64) * slots
    }
    printf "%d", tot / 1048576
  }'
}

# Bucket list sized to what this GX box can afford: 1,6,64 for decode/verify plus powers of two
# from 256 up to the largest that still fits. The slabs and the transient workspace are allocated
# *after* the 113.65 GiB pack load, so the budget is the box's whole unified pool (MemTotal) minus
# the pack minus a floor -- not the free memory now, which is the pool before the pack is allocated
# and would make every candidate look affordable. The workspace is the dominant term (measured 147
# KiB per row of the largest bucket: 301 MiB at 2048, 602 at 4096), so it is counted per candidate
# along with the slabs, and the default cap of 2048 rows is the throughput optimum: one GX layer
# call costs ~40 ms fixed plus ~4 us/row up to 2048 rows, and 210 us/row above it.
# Prints the list on stdout, the decision on stderr; fails when /proc/meminfo is unreadable.
# Call with no server running (or expect the running server's pack to be counted as used).
seam_auto_buckets() {
  local pool_mib pack_mib floor_mib max_rows budget list=1,6,64 b cum
  pool_mib=$(seam_gxsh "awk '/^MemTotal:/{print int(\$2/1024)}' /proc/meminfo" 2>/dev/null | tr -dc 0-9)
  [ -n "$pool_mib" ] || return 1
  pack_mib=${SEAM_PACK_MIB:-116000}
  floor_mib=${SEAM_SLAB_FLOOR_MIB:-1024}
  max_rows=${SEAM_MAX_ROWS:-2048}
  budget=$(( pool_mib - pack_mib - floor_mib ))
  for b in 256 512 1024 2048 4096 8192; do
    [ "$b" -gt "$max_rows" ] && break
    cum=$(( $(seam_slab_mib "$list,$b" "$SEAM_SLAB_SLOTS") + b * 147 / 1024 ))
    [ "$cum" -gt "$budget" ] && break
    list="$list,$b"
  done
  echo "SEAM_BUCKETS_AUTO host=$GX pool_mib=$pool_mib pack_mib=$pack_mib floor_mib=$floor_mib" \
       "budget_mib=$budget list=$list slots=$SEAM_SLAB_SLOTS" \
       "slab_mib=$(seam_slab_mib "$list" "$SEAM_SLAB_SLOTS")" \
       "ws_mib=$(( ${list##*,} * 147 / 1024 )) max_rows=${list##*,}" >&2
  echo "$list"
}

seam_preflight() {  # $1 = stub (1: a stub loads no pack, so only the exclusivity checks apply)
  local stub=${1:-0}
  [ "$GX_PREFLIGHT" = 1 ] || return 0
  local out avail_kib other build uverbs top
  out=$(seam_gxsh_stdin <<EOF 2>/dev/null
set -u
if [ -r /proc/meminfo ]; then
  awk '/^MemAvailable:/{print "mem_avail_kib="\$2}' /proc/meminfo
else
  echo "mem_avail_kib="
fi
self=$CT
for c in \$(docker ps --format '{{.Names}}' 2>/dev/null); do
  [ "\$c" = "\$self" ] && continue
  p=\$(docker exec "\$c" sh -c "pgrep -x python3 -a" 2>/dev/null | grep 'worker_seam_server[.]py' | cut -d' ' -f1 | head -1)
  [ -n "\$p" ] && echo "server_in=\$c:\$p"
done
# Who holds the memory when it is short. cgroup v2 memory.current is the container's whole charge
# (anonymous + page cache), which is what a 116 GiB pack looks like from the host; a box without
# cgroup v2 falls through to 0 and the message says top_mem=none.
for c in \$(docker ps --format '{{.Names}}' 2>/dev/null); do
  [ "\$c" = "\$self" ] && continue
  id=\$(docker inspect -f '{{.Id}}' "\$c" 2>/dev/null)
  m=\$(cat "/sys/fs/cgroup/system.slice/docker-\$id.scope/memory.current" 2>/dev/null)
  case "\$m" in ''|*[!0-9]*) m=0 ;; esac
  [ "\$m" -gt 0 ] && echo "mem_in=\$c \$((m / 1048576))"
done
# A container created without --device /dev/infiniband still loads the whole 113 GiB pack and only
# then fails in qes_server_open ("RDMA device ... not found"): four minutes and 100 GiB of churn for
# nothing, and the client sees a refused control port. Check the nodes before the load.
n=\$(docker exec "\$self" ls /dev/infiniband 2>/dev/null | grep -c uverbs 2>/dev/null)
echo "rdma_uverbs=\${n:-0}"
for p in ninja nvcc cicc; do
  # "cd /src && ninja -v" and "/usr/local/cuda/bin/nvcc ..." are both builds: allow a space too.
  n=\$(pgrep -fc "(^|[ /])\${p}( |\$)" 2>/dev/null)
  [ "\${n:-0}" -ge 1 ] && echo "build=\$p"
done
exit 0
EOF
)
  if [ -z "$out" ]; then
    echo "PREFLIGHT_FAIL gx-unreachable host=$GX"; return 3
  fi
  avail_kib=$(printf '%s\n' "$out" | sed -n 's/^mem_avail_kib=//p' | tr -dc 0-9)
  other=$(printf '%s\n' "$out" | sed -n 's/^server_in=//p' | tr '\n' ';')
  build=$(printf '%s\n' "$out" | sed -n 's/^build=//p' | tr '\n' ';')
  uverbs=$(printf '%s\n' "$out" | sed -n 's/^rdma_uverbs=//p' | tr -dc 0-9)
  top=$(printf '%s\n' "$out" | sed -n 's/^mem_in=//p' | sort -k2 -rn | head -1 | tr ' ' ':')
  [ -n "$top" ] || top=none
  # A server in another container is the first thing to say: it explains the memory too, and on
  # 2026-11-02 the memory line fired first, so the ticket said "avail=0GiB" about a box whose real
  # problem was a foreign server (seam-gx2-4460, pid 1083950) holding the pack and the port.
  if [ -n "$other" ]; then
    echo "PREFLIGHT_FAIL gx-busy server_in=${other%;} host=$GX (one pack load per box: that server holds the pack)"
    return 3
  fi
  if [ -z "$avail_kib" ]; then
    echo "PREFLIGHT_FAIL gx-mem-unknown host=$GX"; return 3
  fi
  if [ "$stub" != 1 ] && [ "$avail_kib" -lt $(( GX_PREFLIGHT_AVAIL_GIB * 1048576 )) ]; then
    echo "PREFLIGHT_FAIL gx-mem avail=$(( avail_kib / 1048576 ))GiB need=${GX_PREFLIGHT_AVAIL_GIB}GiB host=$GX held_by=${other:-none} top_mem=$top"
    return 3
  fi
  if [ "${uverbs:-0}" -lt 1 ]; then
    echo "PREFLIGHT_FAIL no-rdma-device container=$CT host=$GX (recreate it with --device /dev/infiniband)"; return 3
  fi

  if [ -n "$build" ]; then
    echo "PREFLIGHT_FAIL build=${build%;} host=$GX (no builds while a server runs)"
    return 3
  fi
  return 0
}

seam_server_ensure() {
  local fresh=0 stub=0 wait=0 any=0 req=$SERVER_BUCKETS
  while [ $# -gt 0 ]; do
    case "$1" in
      --fresh) fresh=1 ;;
      --stub) stub=1 ;;
      --wait) wait=1 ;;
      --any-kind) any=1 ;;   # reuse a server of the other kind too (a probe that never forwards)
      --buckets) req=$2; shift ;;
      --buckets=*) req=${1#*=} ;;
      *) echo "seam_server_ensure: unknown argument $1" >&2; return 2 ;;
    esac
    shift
  done
  # The gate comes first, before anything on the box is touched: a launcher that does not hold it must
  # not kill, load, or reuse a server here (2026-11-02: a bench started at 01:38 without a claim, the
  # dev smoke took the gate at 01:56, found the port taken and failed 900 s later with a message that
  # named neither the container nor the pid).
  local gate
  if ! gate=$(seam_gate_check); then
    echo "${gate:-PREFLIGHT_FAIL gate_unknown host=$GX}" >&2
    return 3
  fi
  local pids; pids=$(seam_pids)
  local kind; kind=$(seam_kind)
  if [ "$fresh" = 0 ] && [ "${pids:-0}" -ge 1 ] && seam_ready; then
    local served; served=$(seam_served)
    # auto = "whatever this box should serve": a live server already made that decision, so adopt
    # its list instead of measuring again (its own pack is resident, which would shrink the budget).
    [ "$req" = auto ] && req=$served
    local missing="" b
    for b in ${req//,/ }; do
      case ",$served," in *,${b},*) ;; *) missing="$missing $b" ;; esac
    done
    if [ -n "$missing" ]; then
      echo "SEAM_ENSURE state=reused-but-missing-buckets missing=$missing served=$served request=$req" >&2
      return 2
    fi
    # A live server with fewer payload slots than this job asks for would silently run the client
    # at NS=1: the client caps its in-flight micro-batches at what the server advertises.
    local served_slots; served_slots=$(seam_slots)
    if [ -n "$served_slots" ] && [ "$served_slots" != "$SEAM_SLAB_SLOTS" ]; then
      echo "SEAM_ENSURE state=reused-but-wrong-slots served_slots=$served_slots wanted=$SEAM_SLAB_SLOTS" >&2
      return 2
    fi
    if [ "$any" = 0 ] && [ "$kind" != "$stub" ]; then
      echo "SEAM_ENSURE state=restarting (kind: server stub=${kind:-?} wanted stub=$stub)"
    else
      echo "SEAM_ENSURE state=reused pid=$pids stub=${kind:-?} buckets=$served $(seam_ready_line)"
      seam_slack_gate || return 4
      return 0
    fi
  fi
  [ "$fresh" = 1 ] && echo "SEAM_ENSURE state=restarting (--fresh)"
  # By comm, like seam_pids: the launcher's shell carries the script name in its cmdline until it
  # execs, and a pattern stop TERMed the shell and left the worker holding the pack.
  seam_gxct "$(seam_kill_pattern)" >/dev/null 2>&1
  sleep 3
  if [ "$(seam_pids)" -ge 1 ]; then
    echo "SEAM_ENSURE state=error old server still alive" >&2; return 2
  fi
  # The box now has no server of this pair; anything else that still occupies it must be reported
  # instead of loaded on top of. Runs after the kill so a previously loaded own server does not
  # trip the memory check.
  local pf
  if ! pf=$(seam_preflight "$stub"); then
    echo "${pf:-PREFLIGHT_FAIL unknown}" >&2
    return 3
  fi
  [ -n "$pf" ] && echo "$pf"
  # auto is measured here, after the kill: the pack is gone, so MemAvailable is the whole box again.
  if [ "$req" = auto ]; then
    req=$(seam_auto_buckets) || {
      echo "SEAM_ENSURE state=error auto-buckets-unreadable host=$GX" >&2; return 2; }
    [ -n "$req" ] || { echo "SEAM_ENSURE state=error auto-buckets-empty" >&2; return 2; }
  fi
  # The bytes about to be deleted are the previous load's only record of its restack (and of its
  # SEAM_PACK_KERNEL line); keep them next to the pair's logs before the fresh worker starts.
  seam_archive_log
  seam_gxct "rm -f $SEAM_LOG"
  # torch's JIT build holds <torch_extensions>/<pyver>/<name>/lock and removes it on a normal exit.
  # A worker killed mid-build -- our own pkill above, or a launch that died because the container had
  # lost NVML -- leaves the lock behind, and every later import then blocks forever in
  # file_baton.wait at ~512 MB RSS with a completely empty log: indistinguishable from a hung pack
  # load. It cost a whole window on 2026-11-02 (the 07:27 build started with no CUDA, died, and its
  # lock then held the 07:30 and 07:37 loads at 512 MB, rchar frozen, for 4 min each). seam_pids is 0
  # on every path that reaches this line and the preflight above has already refused a live build, so
  # a lock that still exists belongs to a process that is gone: drop it, say so, let the import
  # rebuild what it must.
  local stale
  stale=$(seam_gxct "for f in /root/.cache/torch_extensions/*/*/lock /tmp/torch_extensions/*/*/lock; do
                       [ -e \"\$f\" ] && echo \"\$f\"; done" 2>/dev/null)
  if [ -n "$stale" ]; then
    echo "SEAM_ENSURE stale torch build lock(s): $(echo $stale)"
    # The token names each path, not just the count: a reader (GxPort2's A/B driver, a post-mortem
    # grep) has to be able to tell *which* import was blocked, and `$(echo $stale)` below flattens
    # the list onto one line. The removal itself is the same rm -f that follows.
    for _f in $stale; do echo "STALE_LOCK_REMOVED path=$_f"; done
    seam_gxct "for f in /root/.cache/torch_extensions/*/*/lock /tmp/torch_extensions/*/*/lock; do
                 [ -e \"\$f\" ] && rm -f \"\$f\"; done" >/dev/null 2>&1
  fi
  local slots=""
  seam_slots_supported && slots="--slab-slots $SEAM_SLAB_SLOTS"
  [ -n "$slots" ] || echo "SEAM_SLOTS_UNSUPPORTED (v2 build: one payload slot per handle, flag omitted)" >&2
  local args="--model-dir $SEAM_MODEL_DIR --placement $SEAM_PLACEMENT --dev $GX_DEV --gid $GX_GID --port $PORT --buckets $req $slots"
  [ "$stub" = 1 ] && args="--stub $args"
  # docker stops parsing options at the container name, so the -e pairs have to come *before* $CT:
  # "docker exec -d <ct> -e K=V cmd" runs a command literally called "-e".
  # Only this script may start a server: worker_seam_server.py refuses to start unless SEAM_GATE_OK=1,
  # and nothing else sets it. A script that launches the worker directly bypasses the gate check and
  # the preflight (2026-11-02: three such scripts started workers around a held gate), so the marker
  # makes that failure immediate and on the box instead of a pack loaded onto somebody else's claim.
  # The worker's own extension cache must be *stated*, never inherited. The containers' default became
  # /pair/scratch-ext on 2026-11-04 (containers/*.json), which is where an experiment tree's JIT
  # belongs; the shipped tree must not fall into it. Measured the first time it did: the 02:03Z dev
  # smoke started a release worker with no TORCH_EXTENSIONS_DIR here, it JIT'ed exllamav3 into
  # /pair/scratch-ext, that nvcc run took MemAvailable from 120052 to 116596 MiB, and the load's own
  # check then refused -- SEAM_NO_MEMORY need=116220+2048 avail=116596. Serving the release tree gets
  # the release cache; any other tree keeps the container default on purpose (scratch is for that).
  local wext=""
  case "$(seam_gxsh "docker exec $CT readlink -f /src/exllamav3" 2>/dev/null | tr -d '\r')" in
    /pair/releases/*/src|/pair/current/src) wext="-e TORCH_EXTENSIONS_DIR=/pair/extcache" ;;
  esac
  # The container name is joined to $denv with a space, never by concatenation: with an empty $wext the
  # old `$denv$CT` worked by accident (the value ended in a space), and the moment the pin above made
  # $wext non-empty docker read the glued token as the -e value and the *next* word as the container
  # name -- "Error response from daemon: No such container: bash", rc 2 in 7 s, twice at 02:43Z.
  local denv="-e SEAM_GATE_OK=1 $wext"
  for kv in $SEAM_ENV; do denv="$denv -e $kv"; done
  seam_gxsh "docker exec -d $denv $CT bash -lc 'cd /src/exllamav3 && stdbuf -oL -e0 $SEAM_SERVER_WRAP python3 -u qmesh/worker_seam_server.py $args $SEAM_EXTRA > $SEAM_LOG 2>&1'" || return 2
  local i
  for i in $(seq 1 $((SEAM_SERVER_TIMEOUT / 2))); do
    if seam_ready; then
      echo "SEAM_ENSURE state=started after=$((i * 2))s $(seam_ready_line)"
      seam_slack_gate || return 4
      return 0
    fi
    if [ "$i" -ge 10 ] && seam_loaded; then
      # The LOADED line lands when the pack is in memory, ~10 s before qes.Server() has allocated the
      # pinned slabs and printed SEAM_SERVER_READY with the bucket map. Answering "started-legacy" the
      # instant LOADED shows made every caller that then reads seam_served see an empty list (and one
      # measured 1585 MiB of slabs get killed mid-allocation). Give a proto-2 server a grace window to
      # print READY; only a pre-multi-session build (GX1 until it is synced) needs the legacy answer.
      local j
      for j in $(seq 1 $((SEAM_LEGACY_GRACE / 2))); do
        if seam_ready; then
          echo "SEAM_ENSURE state=started after=$((i * 2 + j * 2))s $(seam_ready_line)"
          seam_slack_gate || return 4
          return 0
        fi
        sleep 2
      done
      echo "SEAM_ENSURE state=started-legacy after=$((i * 2))s $(seam_loaded_line)"
      return 0
    fi
    if [ $((i % 15)) -eq 0 ] && [ "$(seam_pids)" = "0" ]; then
      echo "SEAM_ENSURE state=died after=$((i * 2))s" >&2
      seam_gxct "tail -30 $SEAM_LOG" >&2
      return 2
    fi
    sleep 2
  done
  echo "SEAM_ENSURE state=timeout after=${SEAM_SERVER_TIMEOUT}s" >&2
  return 2
}

seam_server_stop() {
  # Same selection as above: stop the worker, not the shell that spawned it.
  seam_gxct "$(seam_kill_pattern)" >/dev/null 2>&1
  local i
  for i in $(seq 1 30); do [ "$(seam_pids)" = "0" ] && break; sleep 1; done
  if [ "$(seam_pids)" != "0" ]; then
    seam_gxct "$(seam_kill_pattern -9)" >/dev/null 2>&1; sleep 2
  fi
  # The log outlives the process, so a dead server's READY line would make seam_ready (and every
  # caller that trusts it without checking the pid) believe the cold experts are still up.
  [ "$(seam_pids)" = "0" ] && { seam_archive_log; seam_gxct "rm -f $SEAM_LOG" >/dev/null 2>&1; }
  [ "$(seam_pids)" = "0" ]
}

seam_server_status() {  # idle|busy|down, straight from the server
  local out
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "$GX" \
        "docker exec $CT bash -lc 'cd /src/exllamav3 && python3 qmesh/seam_status.py 127.0.0.1 $PORT 2>/dev/null'" 2>/dev/null)
  case "$out" in
    *state=idle*) echo idle ;;
    *state=busy*) echo busy ;;
    *) echo down ;;
  esac
}

# Direct use as a command: prints the SEAM_ENSURE line and exports nothing. The status is passed
# through unchanged: pair_smoke.sh classifies on rc 3 (preflight refused) versus 2 (server died or
# timed out), and mapping every failure to 2 made a box-busy refusal read as a broken release.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  seam_server_ensure "$@"
  exit $?
fi
