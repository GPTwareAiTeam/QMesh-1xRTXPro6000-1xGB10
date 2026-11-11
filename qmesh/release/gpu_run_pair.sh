#!/usr/bin/env bash
# Priority GPU queue for one P7 GPU (QGPUS) and its GX node. Usage:
#   gpu_run.sh <agent> <max_minutes> <command...>
# Priority comes from gpu_priority.txt (one agent name per line, top = first). Unlisted agents go last.
# Same priority: first come first served. The GPU is released as soon as the command exits.
#
# Deadline handling (soft, not a guillotine):
#   The command gets $GPU_RUN_STATUS (a file path). Write progress into it at any time, e.g.
#       echo "eta_min=6 doing=nsys profile arm 3/4" > "$GPU_RUN_STATUS"
#   At <max_minutes>, the job is extended (no kill) if its latest eta_min <= EXTEND_MAX_MIN (default 10),
#   as long as the total stays under HARD_CAP_MIN (default max_minutes + 30). Extensions repeat while
#   the ETA keeps qualifying. The supervisor can also grant time by writing minutes into
#   gpu_queue/extend-<agent> (consumed on read). With no qualifying ETA or grant, the script waits
#   GRACE_SEC (default 90) for one to appear, then kills the job.
#
# Release guard (qmesh/release/pair_check.sh): the GPU is free but the pair may still be serving a
# different release on its P7 and GX halves -- that is exactly how a pair hangs with `peer layout
# mismatch` or dies with `remote invalid request`. So before a job starts, the pair is checked
# against its manifest, and a job is refused (exit 4 drift, 5 check error) instead of poisoning the
# queue.
#
# Who this applies to: a ticket that names one of the pair's own containers (`seam-p7-dev`,
# `seam-p7-prod`, `seam-gx2-dev`, `seam-gx1-prod`) on its command line, and the update jobs
# themselves. A ticket that names no pair container -- an experiment in its own container, a
# different GPU, no `/pair` mount -- cannot be affected by that drift and is not held up by it: on
# 2026-11-02 the whole queue stalled behind a repair because every ticket was checked. `PAIR_CHECK_SKIP=1`
# still turns the check off for a command that reaches a pair indirectly, and `PAIR_CHECK_FORCE=1`
# turns it on for one that does not name the container it will use.
set -u
AGENT=${1:?agent}; MAXMIN=${2:?minutes}; shift 2
BASE=${QBASE:?QBASE}
Q=$BASE/gpu_queue; LOCK=$BASE/gpu.lock; PRIO=$BASE/gpu_priority.txt; LOG=$BASE/gpu_queue.log
# $BASE/running names the ticket whose command is alive right now. A ticket file in $Q is not evidence
# of that: the wrapper removes it on exit, but between phases -- a GX-only bench, a JIT build, the
# prepare before the claim -- the P7 GPU is idle, so gpu_busy() is false and the ticket's own file is
# the only other trace. A later rank-01 entry sorts *below* the running ticket's name and becomes
# head, which is how three tickets ran side by side on 2026-11-02 (PairUpdate 11:20:43 and CanaryB84
# 11:22:07 beside G6Bound; SeamRobust 11:24:09 beside CanaryB84). Rank orders the WAITERS and must
# never preempt: a head ticket starts only when no other ticket's wrapper is alive.
RUNFILE=$BASE/running
EXTEND_MAX_MIN=${EXTEND_MAX_MIN:-10}; HARD_CAP_MIN=${HARD_CAP_MIN:-$((MAXMIN + 30))}; GRACE_SEC=${GRACE_SEC:-90}
mkdir -p "$Q"
log() { echo "$(date +%T) $*" >> "$LOG"; }
# A ticket that queues another ticket deadlocks the queue: the outer ticket is still head while its
# own command waits for the inner one, and neither can run. On 2026-11-02 10:23-10:32 four waiters sat
# behind exactly that (an IDTrace ticket whose body called run_dev.sh). GPU_TICKET_HARDCAP is exported
# by the wrapper into its command and nowhere else, so its presence here means we are inside a ticket.
if [ -n "${GPU_TICKET_HARDCAP:-}" ]; then
  echo "TICKET_NESTED: a ticket cannot queue another ticket -- the outer entry is the head and would" >&2
  echo "wait on the inner one forever. Queue the work as one ticket, or run the inner command directly." >&2
  log "TICKET_NESTED refused $AGENT: queued from inside an existing ticket"
  exit 2
fi
SCRIPTS=""
for _c in "$BASE/../../scripts" "$BASE/../scripts"; do
  [ -x "$_c/pair_check.sh" ] && { SCRIPTS=$(cd "$_c" && pwd); break; }
done
# check_release: 0 run, 4 drift (refuse), 5 check error (refuse). Transient ssh failures are retried
# three times 10 s apart: a 10 s network blip must not silently run a job on a stale pair.
# The pair's containers, as pairs.conf names them: a ticket that names none of these is not
# touching the pair's release.
PAIR_CONTAINERS="seam-p7-dev seam-p7-prod seam-gx2-dev seam-gx1-prod"
# Named on the command line, not reached through some script we cannot read: that is the line
# between "this ticket can hurt the pair" and "this ticket cannot".
names_pair() {
  local a c
  for a in "$@"; do
    for c in $PAIR_CONTAINERS; do
      case "$a" in *"$c"*) return 0 ;; esac
    done
  done
  return 1
}
check_release() {
  [ -n "${PAIR_CHECK_SKIP:-}" ] && return 0
  if [ -z "${PAIR_CHECK_FORCE:-}" ] && ! names_pair "$@" \
     && [ "$AGENT" != PairUpdate ] && [ "$AGENT" != PairUpdateRun ]; then
    log "PAIR_CHECK_SKIP release guard not applicable: $AGENT names no pair container"
    return 0
  fi
  [ "$AGENT" = PairUpdate ] && return 0
  [ "$AGENT" = PairUpdateRun ] && return 0
  local name; name=$(basename "$BASE")
  if [ -z "$SCRIPTS" ]; then
    log "PAIR_DRIFT $name no_scripts (pair_check.sh not found above $BASE) refuse=$AGENT"
    return 4
  fi
  local out rc i
  for i in 1 2 3; do
    out=$("$SCRIPTS/pair_check.sh" "$name" 2>&1); rc=$?
    case $rc in
      0) return 0 ;;
      4) log "PAIR_DRIFT $name refuse=$AGENT :: $(printf '%s' "$out" | grep -E 'DRIFT' | sed 's/^ *//' | tr '\n' ';')"
         return 4 ;;
      *) sleep 10 ;;
    esac
  done
  log "PAIR_CHECK_ERROR $name rc=$rc refuse=$AGENT :: $(printf '%s' "$out" | tail -3 | tr '\n' ';')"
  return 5
}
# ---- what "busy" means -------------------------------------------------------------------------
# One queue serves one pair (one P7 GPU + its GX box). Up to 2026-11-03 only the P7 side was looked at,
# and that is blind to the half where the model actually lives: a serve whose client has exited keeps
# its worker resident on the GX GPU with nothing at all on the P7 side, so the queue read "free" and
# started the next load on top of it (10:17Z: a ticket killed with an orphan client plus a live
# neighbour, two 92 GiB workers contending for one box). The box is busy when any of four things says
# so: a compute app on the ticket's own P7 GPU, a live claim in the pair's gx gate file, a compute app
# on the pair's GX box, or (handled by the caller) a ticket whose wrapper is alive.
#
# The GX box and its gate name are read from pairs.conf -- the one file that says which box, container
# and address belong to which pair -- joined on the `gpu` column so run_dev.sh/run_prod.sh keep
# carrying nothing but QGPUS. No file, no matching row, or an unreachable box leaves the probe silent
# instead of wedging the queue; the log says which, once.
QPAIR_ROW=""
for _pc in "$BASE/../../scripts/pairs.conf" "$BASE/../scripts/pairs.conf"; do
  [ -r "$_pc" ] && { QPAIR_ROW=$(awk -v g="${QGPUS:-}" '!/^#/ && NF >= 9 && $3 == g { print; exit }' "$_pc"); break; }
done
# columns: pair p7ct gpu seam_dev gid gxgid peer_ip gx_mgmt gxct release_root … (see pairs.conf header)
QGX_HOST=$(printf '%s' "$QPAIR_ROW" | awk '{print $8}')
QGX_CT=$(printf '%s' "$QPAIR_ROW" | awk '{print $9}')
# seam-gx2-dev -> 2: the gate file is named after the GX container, like pair_gx_gate_file() does.
_qn=${QGX_CT#seam-gx}; _qn=${_qn%%[!0-9]*}
QGATE=${_qn:+$(dirname "$BASE")/gx$_qn.gate}
QGX_PROBE_TTL=${QGX_PROBE_TTL:-20}
# "Any compute app on the GX box" is not the predicate: that box also runs a desktops' remote-access
# helper (rustdesk holds a 228 MiB context there, always, up 9h47m when this was written), so the rule
# would read the box as busy forever and wedge every ticket. What must not be started beside is the
# pack or a comparable job, so: a `worker_seam_server` process, or an app holding more than
# QGX_MIN_MIB. Both are one ssh round trip.
QGX_MIN_MIB=${QGX_MIN_MIB:-1024}
_gx_state=init; _gx_ts=0
gx_busy() {  # 0 when a seam worker (or a >QGX_MIN_MIB app) is resident on the pair's GX box
  local now out apps procs rc mib seen=""
  [ -n "$QGX_HOST" ] || { [ "$_gx_state" = init ] && { log "gx_guard off: no pairs.conf row for QGPUS=${QGPUS:-} (GX side unguarded)"; _gx_state=noconf; }; return 1; }
  now=$(date +%s)
  if [ "$_gx_state" != init ] && [ "$_gx_state" != noconf ] && [ $((now - _gx_ts)) -lt "$QGX_PROBE_TTL" ]; then
    [ "$_gx_state" = busy ]
    return
  fi
  # `pgrep` exits 1 when it finds nothing, and ssh hands that back as "unreachable" -- so the remote
  # side always ends with exit 0 and only a transport failure sets rc != 0.
  out=$(timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=6 "$QGX_HOST" \
          'nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null;
           echo "---SEAM---"; pgrep -f "[w]orker_seam_server" 2>/dev/null; exit 0' 2>/dev/null); rc=$?
  if [ "$rc" != 0 ]; then
    [ "$_gx_state" = init ] && log "gx_guard blind: ssh $QGX_HOST rc=$rc (GX side unguarded this wait)"
    _gx_state=unreach; _gx_ts=$now; return 1
  fi
  apps=${out%%---SEAM---*}; procs=${out#*---SEAM---}
  if [ -n "$(printf '%s' "$procs" | tr -d '[:space:]')" ]; then
    seen="worker pid(s) $(printf '%s' "$procs" | tr -s '[:space:]' ' ')"
  else
    while read -r mib; do
      [ -n "$mib" ] && [ "$mib" -ge "$QGX_MIN_MIB" ] && seen="$seen app ${mib}MiB"
    done <<<"$(printf '%s\n' "$apps" | sed -n 's/^[0-9]*, *\([0-9][0-9]*\) MiB.*/\1/p')"
  fi
  if [ -n "$seen" ]; then
    [ "$_gx_state" = init ] && log "gx_busy $QGX_HOST:$seen"
    _gx_state=busy; _gx_ts=$now; return 0
  fi
  [ "$_gx_state" = busy ] && log "gx_idle $QGX_HOST"
  _gx_state=idle; _gx_ts=$now; return 1
}
gate_busy() {  # 0 when the pair's gate file holds an unexpired claim that is not this ticket
  local f=${QGATE:-} line pid until
  [ -n "$f" ] && [ -r "$f" ] || return 1
  line=$(cat "$f" 2>/dev/null)
  pid=$(printf '%s' "$line" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)
  until=$(printf '%s' "$line" | sed -n 's/.*until=\([0-9][0-9]*\).*/\1/p' | head -1)
  [ -n "$until" ] && [ "$until" -le "$(date +%s)" ] && return 1
  [ -n "$pid" ] && [ "$pid" != "$$" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  return 0
}
gpu_busy() {  # 0 when any half of the pair is in use
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i ${QGPUS:?QGPUS})" ] && return 0
  gate_busy && return 0
  gx_busy && return 0
  return 1
}
# running_owner: the live ticket that holds the box, empty when $RUNFILE is absent or nothing that
# belongs to it is alive. The marker names two pids: the wrapper (`pid=`) and the command it launched
# (`cmd_pid=`, appended right after the launch). The box is held while *either* lives, because on
# 2026-11-02 11:43 the wrapper was killed by the harness that owned its terminal while its
# `setsid`'d command tree (pair_update_run → pair_smoke → pair_arm) kept running: the marker went
# stale, the box looked free, and the next rank-01 ticket would have run beside a live release smoke.
# A marker whose two pids are both gone is removed here, so a killed ticket costs one read, not a
# manual rm.
running_owner() {
  local line pid cpid
  [ -r "$RUNFILE" ] || return 0
  line=$(cat "$RUNFILE" 2>/dev/null)
  pid=$(printf '%s' "$line" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)
  cpid=$(printf '%s' "$line" | sed -n 's/.*cmd_pid=\([0-9][0-9]*\).*/\1/p' | head -1)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then printf '%s' "$line"; return 0; fi
  if [ -n "$cpid" ] && kill -0 "$cpid" 2>/dev/null; then printf '%s' "$line"; return 0; fi
  rm -f "$RUNFILE"
  return 0
}
release_running() {  # only the wrapper that wrote the marker may clear it
  local pid
  pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' "$RUNFILE" 2>/dev/null | head -1)
  [ "$pid" = "$$" ] && rm -f "$RUNFILE"
  return 0
}
# The wrapper is a supervisor: a hangup on its terminal must not take the queue's bookkeeping with
# it. The command runs in its own session (setsid), so it does not receive the hangup either, and the
# EXIT trap below is what stops it if the wrapper does go away for another reason.
trap '' HUP
trap 'if [ -n "${cpid:-}" ] && kill -0 "$cpid" 2>/dev/null; then kill -TERM -- -"$cpid" 2>/dev/null; fi
      rm -f "$ticket" "$GPU_RUN_STATUS"; release_running' EXIT
rank=$(grep -nx "$AGENT" "$PRIO" 2>/dev/null | cut -d: -f1); rank=${rank:-99}
ticket="$Q/$(printf %02d "$rank")-$(date +%s%N)-$$-$AGENT"
export GPU_RUN_STATUS="$Q/status-$AGENT-$$"
GRANT="$Q/extend-$AGENT"
touch "$ticket"; rm -f "$GPU_RUN_STATUS"
log "queued $AGENT rank=$rank pid=$$"
while :; do
  # drop tickets whose owner process is gone
  for t in "$Q"/[0-9]*; do
    [ -e "$t" ] || continue
    p=$(basename "$t" | cut -d- -f3)
    kill -0 "$p" 2>/dev/null || rm -f "$t"
  done
  # Head = lowest ticket name (rank prefix, then enqueue time). Every waiter must use this same rule:
  # waiters running different script versions must agree on the head or the queue deadlocks.
  head=$(ls "$Q" | grep -E '^[0-9]' | sort | head -n1)
  if [ "$Q/$head" = "$ticket" ]; then
    # Being head is not permission to run: the ticket ahead of us may still be executing a command that
    # keeps no P7 process (a GX bench, a JIT build, the prepare before it takes the claim), and then
    # gpu_busy() is false while the box is very much in use. Its wrapper is alive, and that is what
    # $RUNFILE records. rank orders the waiters; it never preempts a running ticket.
    owner=$(running_owner)
    if [ -n "$owner" ]; then
      if [ "$owner" != "${last_owner:-}" ]; then
        log "wait $AGENT head=rank$rank running=$(printf '%s' "$owner" | cut -d' ' -f1-2)"
        last_owner=$owner
      fi
      sleep 5; continue
    fi
    last_owner=""
    # The release guard forks ssh and docker and can spend 90 s doing it. It must not run with the
    # queue lock held: `exec 9>lock` gives every child of the holder the same open description, and a
    # flock survives the holder -- on 2026-11-02 at 09:03 a holder died inside it and the queue stayed
    # write-locked with no live owner at all, so three waiters (my ServerUp, SeamRobust's acceptance,
    # GxPort2's G6) sat in `flock 9` until their wrappers were killed by hand. So: guard first, lock
    # afterwards for nothing but the head re-read and the busy check, then release and close 9 --
    # a ticket command that inherits the description must never be able to keep a lock alive.
    gpu_busy && { sleep 5; continue; }
    check_release "$@"; grc=$?
    if [ "$grc" != 0 ]; then
      log "refused $AGENT release_check_rc=$grc (not started)"
      exit "$grc"
    fi
    exec 9>"$LOCK"
    flock 9
    head=$(ls "$Q" | grep -E '^[0-9]' | sort | head -n1)
    # Two heads read the same marker: whoever writes it first runs, the other goes back to waiting.
    # Writing it inside the lock is what makes "the box is free" and "I am the one running" one step.
    if [ "$Q/$head" = "$ticket" ] && ! gpu_busy && [ -z "$(running_owner)" ]; then
      printf 'agent=%s pid=%s start=%s ticket=%s\n' "$AGENT" "$$" "$(date +%s)" \
        "$(basename "$ticket")" > "$RUNFILE.tmp" && mv -f "$RUNFILE.tmp" "$RUNFILE"
      flock -u 9
      exec 9>&-
      break
    fi
    flock -u 9
    exec 9>&-
  fi
  sleep 5
done
start=$(date +%s); deadline=$((start + MAXMIN * 60)); hard=$((start + HARD_CAP_MIN * 60))
# The hard cap is part of the ticket's contract, not just this loop's business: pair_gx_gate_take
# refuses a claim taken outside a ticket and never lets one run past the ticket's cap, so hand the
# number to the command and write it into the ticket file (a claim taken by a child reads it from
# either). A claim past the cap is exactly the orphan that made the Integrator ticket fail on
# 2026-11-02 after GxPort2 held gx2.gate 26 minutes from outside the queue.
export GPU_TICKET_HARDCAP=$hard
printf 'hardcap=%s\n' "$hard" >>"$ticket" 2>/dev/null
# Every ticket leaves a log, whether or not the agent wraps its command in tee. On 2026-11-02 a
# canary ended rc=2 after 175 s and *nothing* recorded why: the ticket carried no PAIR_LOG, its
# stdout was the launching terminal's, and the process that owned the terminal exited in the same
# second. The wrapper knows the start time and the agent name; it does not know what the command
# will print, so it captures all of it, keeps the terminal copy, and names the file in the `start`
# line. `GPU_TICKET_LOG` is exported for scripts that write their own lines into the same file.
#
#   setsid bash -c 'exec "$@" > >(tee -a --output-error=warn "$0") 2>&1' <log> <command...>
#
# The `exec` keeps the pid, so `kill -- -$cpid` still hits the command's own process group (tee sits
# in that group too), and the exit status is the command's, not tee's. `--output-error=warn` is what
# makes the terminal copy safe to keep: without it tee exits on the first broken pipe, the command
# then dies of SIGPIPE on its next write, and a ticket would die with the terminal of whoever queued
# it -- the same loss, one step earlier. A tee that does not know the option (not GNU coreutils 8.24+,
# not uutils) is not used at all; the command writes straight into the file then, and the log is the
# only copy. Either way the log is written by the file side, never by the terminal.
LOGDIR=$BASE/logs; mkdir -p "$LOGDIR"
TICKET_LOG=$LOGDIR/$(printf '%s' "$AGENT" | tr -c 'A-Za-z0-9._-' '_')-$(date -u +%Y%m%dT%H%M%SZ)-$$.log
: >"$TICKET_LOG"
export GPU_TICKET_LOG=$TICKET_LOG
printf 'log=%s\n' "$TICKET_LOG" >>"$ticket" 2>/dev/null
log "start $AGENT max=${MAXMIN}m hardcap=${HARD_CAP_MIN}m log=$TICKET_LOG load=$(cut -d" " -f1 /proc/loadavg)"
if tee --output-error=warn </dev/null >/dev/null 2>&1; then
  _ticket_run() { setsid bash -c 'exec "$@" > >(tee -a --output-error=warn "$0") 2>&1' "$TICKET_LOG" "$@"; }
else
  _ticket_run() { setsid bash -c 'exec "$@" >>"$0" 2>&1' "$TICKET_LOG" "$@"; }
fi
# A mis-typed ticket must not read as a failed build. `setsid "$@"` execs argv[0] as a filename, so a
# command passed as ONE quoted string ("scripts/x.sh a b") becomes a filename with spaces and dies as
# rc=127 in 0 s with setsid's "No such file or directory" -- which two of my own tickets ate on
# 2026-11-02 (one relative, one absolute) before anyone saw the cause. Refuse it before the GPU is
# touched, and say what to do instead.
case "$1" in
  *' '*|'') log "TICKET_BAD_CMD argv0=\"$1\" hint=pass the command as separate words, not one quoted string"
            printf 'TICKET_BAD_CMD argv0=%s\n' "$1" >&2; exit 2 ;;
esac
# argv[0] may be a path or a PATH name: `bash $0`, `python3 …`, `env FOO=1 …` are all legitimate
# tickets. `command -v` resolves both; an `[ -x "$1" ]` here would reject every PATH name.
command -v -- "$1" >/dev/null 2>&1 || {
  log "TICKET_BAD_CMD argv0=\"$1\" hint=argv[0] must be an executable file or a command on PATH (cwd is $PWD)"
  printf 'TICKET_BAD_CMD argv0=%s not found (cwd=%s)\n' "$1" "$PWD" >&2; exit 2
}
_ticket_run "$@" & cpid=$!
# The marker must outlive this wrapper: a wrapper killed by the harness that owns its terminal used
# to leave a `pid=` that died with it while the command kept running (2026-11-02 11:43, a live smoke).
# `cmd_pid` is what the queue tests after `pid`, so the box stays held by whoever is actually on it.
{ [ -w "$RUNFILE" ] && printf 'cmd_pid=%s\n' "$cpid" >>"$RUNFILE"; } 2>/dev/null || true
grace_until=0
while kill -0 "$cpid" 2>/dev/null; do
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    eta=$(grep -o 'eta_min=[0-9]*' "$GPU_RUN_STATUS" 2>/dev/null | tail -n1 | cut -d= -f2)
    doing=$(grep -o 'doing=.*' "$GPU_RUN_STATUS" 2>/dev/null | tail -n1)
    grant=$(cat "$GRANT" 2>/dev/null); rm -f "$GRANT"
    if [ -n "$grant" ] && [ "$now" -lt "$hard" ]; then
      deadline=$((now + grant * 60)); [ "$deadline" -gt "$hard" ] && deadline=$hard; grace_until=0
      log "extend $AGENT +${grant}m (supervisor grant) $doing"
    elif [ -n "$eta" ] && [ "$eta" -le "$EXTEND_MAX_MIN" ] && [ "$now" -lt "$hard" ]; then
      deadline=$((now + (eta + 2) * 60)); [ "$deadline" -gt "$hard" ] && deadline=$hard; grace_until=0
      log "extend $AGENT +$((eta + 2))m (eta=${eta}m) $doing"
    elif [ "$now" -ge "$hard" ]; then
      log "kill $AGENT (hard cap ${HARD_CAP_MIN}m) eta=${eta:-none} $doing"; break
    elif [ "$grace_until" -eq 0 ]; then
      grace_until=$((now + GRACE_SEC))
      log "deadline $AGENT eta=${eta:-none} $doing — waiting ${GRACE_SEC}s for eta/grant"
    elif [ "$now" -ge "$grace_until" ]; then
      log "kill $AGENT (no qualifying eta/grant) eta=${eta:-none} $doing"; break
    fi
  fi
  sleep 5
done
if kill -0 "$cpid" 2>/dev/null; then
  kill -TERM -- -"$cpid" 2>/dev/null; sleep 30; kill -KILL -- -"$cpid" 2>/dev/null
fi
wait "$cpid" 2>/dev/null; rc=$?
# container-side runners can outlive the host command: wait for the GPU to drain before releasing
for _ in $(seq 1 60); do gpu_busy || break; sleep 5; done
log "end $AGENT rc=$rc elapsed=$((($(date +%s) - start) / 60))m gpu_clear=$(gpu_busy && echo NO || echo yes) load=$(cut -d" " -f1 /proc/loadavg)"
exit $rc
