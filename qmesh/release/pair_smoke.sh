#!/usr/bin/env bash
# pair_smoke.sh -- prove a pair can serve one decode row AND one 32k prefill on the release it is
# wired to, on the kernel and piece size it ships.
#
#   pair_smoke.sh <pair> [sha]
#
# What it does, in order, with a bound on every wait:
#   1. records the GX box's memory before the load (MemAvailable, swap),
#   2. starts the seam server from the release (seam_server_ensure --fresh) with the pair's shipping
#      server env (pair_lib.sh: cuda-exl3 preload, load margin, pinned piece size) and SEAM_BUILD_ID
#      set, so the SEAM_SERVER_LOADED line names the release that is actually running,
#   3. refuses a server that came up without the cuda-exl3 kernel -- SEAM_MOE_NOEXT means the vendored
#      extension was not built into this release and every layer runs the mixed-K module at a
#      fraction of the rate -- and one where too few layers could be restacked,
#   4. records the box's memory *after* the pack is loaded and the slack the ensure script reports --
#      on GX2 a 116 GiB pack leaves ~1.4 GiB, and a server that dies of that has to be caught here,
#      not in the middle of someone's arm -- and from then on prints, at every phase, the box's
#      avail_mib and swap, and the worker's own pid, VmSwap and RSS beside the load's swapped_mib,
#   5. runs one discarded decode through the P7 client half -- the same arm, so the first-touch
#      allocation of the draft/verify buffers lands before the window rather than in it -- then takes
#      the swap baseline and runs the real decode row, checks it produced tokens and ids, that the
#      wire did not answer `peer layout mismatch` / `remote invalid request`, and that the worker's
#      own swap did not grow inside that window (the growth gate; see SMOKE_WSWAP_GROW_MAX),
#   6. runs one 32k prefill arm and gates its rate and aborts (dev: the owner's canary bar; prod: what
#      the same release measured on prod's own hardware),
#   7. prints SMOKE_OK or SMOKE_FAIL <reason> and exits 0/1.
#
# A server that never prints READY is reported as `server_timeout` (never waited on indefinitely):
# the bound comes from SMOKE_SERVER_TIMEOUT and is enforced by seam_server_ensure itself.
set -uo pipefail
# bash reads a script lazily, and this file sits in the installed directory that pair_install.sh replaces
# in place: a replace or a hand edit while a smoke is running makes the rest of the file parse as garbage.
# (2026-11-03: a canary smoke died at line 580 with "syntax error near unexpected token `('" -- the prefill
# gate had already been measured and the arm was thrown away.) So run from a private copy of this file; the
# copy takes HERE from PAIR_SELF_HERE, because its own $0 is in /tmp.
if [ "${PAIR_SMOKE_SELF:-0}" != 1 ]; then
  PAIR_SELF_COPY=$(mktemp "${TMPDIR:-/tmp}/pair_smoke.XXXXXX") || exit 2
  PAIR_SELF_HERE=$(cd "$(dirname "$0")" && pwd)
  if cp "$0" "$PAIR_SELF_COPY"; then
    PAIR_SMOKE_SELF=1 PAIR_SELF_HERE=$PAIR_SELF_HERE PAIR_SELF_COPY=$PAIR_SELF_COPY \
      exec bash "$PAIR_SELF_COPY" "${@}"
  fi
  echo "SMOKE_FAIL no_self_copy $0"; exit 2
fi
HERE=${PAIR_SELF_HERE:-$(cd "$(dirname "$0")" && pwd)}
# Until smoke_exit (which owns the server and the box claim) is installed, a refusal in between must
# still drop the private copy.
trap 'rm -f "${PAIR_SELF_COPY:-}"' EXIT
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_smoke.sh <pair> [sha]}
pair_load "$PAIR_ARG"
SHA=${2:-$(pair_current_sha "$PAIR_ROOT")}
[ -n "$SHA" ] || { echo "SMOKE_FAIL no_release"; exit 1; }
REL=$(pair_release_dir "$SHA")
[ -d "$REL/src" ] || { echo "SMOKE_FAIL release_absent $REL"; exit 1; }
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json" 2>/dev/null)
[ -n "$BUILD_ID" ] || { echo "SMOKE_FAIL no_build_id"; exit 1; }

SMOKE_SERVER_TIMEOUT=${SMOKE_SERVER_TIMEOUT:-900}
SMOKE_ARM_TIMEOUT=${SMOKE_ARM_TIMEOUT:-600}
SMOKE_PREFILL_TIMEOUT=${SMOKE_PREFILL_TIMEOUT:-1800}
SMOKE_NEW=${SMOKE_NEW:-8}
# The shipping cache: 1M tokens of FP4 KV. A smoke that runs with a smaller cache tests a different
# memory shape than the pair serves.
SMOKE_CACHE_TOKENS=${SMOKE_CACHE_TOKENS:-1048576}
SMOKE_SLACK_GATE=${SMOKE_SLACK_GATE:-0}          # 1 = a thin slack refuses the release (see notes)
# Payload slots per handle. One is what the validated run used: a second slot is only needed to
# pipeline a split prefill, and on GX2 (~1.4 GiB free after the load) the extra slab is what a paging
# wall costs. seam_server_ensure.sh only passes --slab-slots to a worker that knows the flag.
SMOKE_SLAB_SLOTS=${SMOKE_SLAB_SLOTS:-1}
# The swap gate (owner decision 2026-11-03, "Option B"). GX2 puts 1.2-1.7 GiB into swap merely by
# mapping a 116 GiB pack on a ~121 GiB machine, with nothing of ours running, so an absolute level on
# the worker refuses every honest fresh load. What is judged is GROWTH inside a measured window:
# `before_arm` -> the reading right after the decode rows, refused above SMOKE_WSWAP_GROW_MAX. Prefill
# growth is expected (356 MiB at ws 256, 734 at ws 1024) and is printed and warned, never refused.
# One absolute level survives -- a second tenant's pack -- at SMOKE_WSWAP_MAX, and a reading whose
# avail_mib says it cannot be post-residency (> SMOKE_WSWAP_JUDGE_MIN) is printed but not judged.
# Option B is about the *window*, not the number: the decode gate refuses on growth inside the
# measured decode window, with the baseline taken in the same run immediately before the arm, and
# every reading prints the starting level, the box's swap, avail_mib, pid and the load's swapped_mib
# beside it as a WARNING so a refusal is reproducible from the log alone.
# The number stayed at 512 (supervisor, 2026-11-04 12:2xZ, on the owner's "keep it moving"): a 128 MiB
# bar produced false failures on the shipped config itself -- 165 MiB on f9d57851's own canary, and
# 110-264 MiB is routine movement on GX2 for an honest load -- and the prod block would trip it the
# same way. Do not tighten it again without a reading that shows growth is ours. Growth is warned
# here and refused only past the bar; the prefill window (356 MiB expected at ws 256, 734 at ws 1024
# after a 32k prefill) is printed and warned, never refused, and SMOKE_WSWAP_MAX=2048 still refuses
# another tenant's pack outright.
SMOKE_WSWAP_MAX=${SMOKE_WSWAP_MAX:-2048}
SMOKE_WSWAP_GROW_MAX=${SMOKE_WSWAP_GROW_MAX:-512}
SMOKE_WSWAP_PREFILL_GROW_MAX=${SMOKE_WSWAP_PREFILL_GROW_MAX:-512}
SMOKE_WSWAP_JUDGE_MIN=${SMOKE_WSWAP_JUDGE_MIN:-2000}
# The grant the smoke server is given: the pair's own list (pairs.conf). Workspace for a bucket nobody
# asks for still comes out of the box (242 MiB slab at 8192 rows), so the smoke grants exactly what
# the pair grants -- otherwise it would pass a release on a config nobody serves.
#
# The prefill gate. dev is the canary and carries the owner's bar: 32k tokens in one arm at >= 800
# tok/s with no abort (the validated release did 837.6). prod's own hardware measured 796.2 tok/s on
# the same release, so its bar is that number minus a smoke's worth of noise; a prod smoke that lands
# near dev's is the promotion confirming the copy, not a different instrument.
SMOKE_BUCKETS=${SMOKE_BUCKETS:-$BUCKETS}
SMOKE_PREFILL_DEPTH=${SMOKE_PREFILL_DEPTH:-32768}
if [ "$PAIR" = dev ]; then
  SMOKE_PREFILL_MIN_TPS=${SMOKE_PREFILL_MIN_TPS:-800}
else
  SMOKE_PREFILL_MIN_TPS=${SMOKE_PREFILL_MIN_TPS:-760}
fi
# The prefill rate is only a statement about the release when the box could serve the pack out of
# memory. On 2026-11-03 the rel-2 canary read 698.9 tok/s against a bar of 800 while the GX worker sat
# at avail_mib 788 with 1567 MiB of its own load already swapped out: that arm measured swap I/O, not
# the seam. So when the box's available memory is below this floor at either end of the prefill window
# the rate is printed, warned and NOT judged -- the same way the swap gate prints a prefill growth
# instead of refusing it. It never skips the abort/compute checks below, which do not depend on speed.
SMOKE_PREFILL_MIN_AVAIL_MIB=${SMOKE_PREFILL_MIN_AVAIL_MIB:-4000}
# The release warms one prefill call per seam bucket before SERVER_READY (serve_pair.py, WARM_PREFILL=1),
# which is aimed at the 31 % first-arm penalty (955 -> 1156 rows/s once the caches were seeded). Its two
# observables are here: SMOKE_WARM_ROWS `WARM_PREFILL_OK` lines in the worker's log before the first
# request (four shipped: 1, 6, 64 and the load-time row count), and a second identical 32k arm whose
# rate must land within SMOKE_PREFILL_RATIO_MAX of the first. SMOKE_WARM_OPTIONAL=1 / SMOKE_PREFILL_ARMS=1
# relax them for a release that deliberately ships the warm-up off.
SMOKE_WARM_ROWS=${SMOKE_WARM_ROWS:-4}
SMOKE_PREFILL_ARMS=${SMOKE_PREFILL_ARMS:-2}
SMOKE_PREFILL_RATIO_MAX=${SMOKE_PREFILL_RATIO_MAX:-1.05}
ENSURE=${SEAM_ENSURE:-}
if [ -z "$ENSURE" ]; then
  for c in "$PAIR_SCRIPTS/seam_server_ensure.sh" "$HERE/seam_server_ensure.sh"; do
    [ -x "$c" ] && { ENSURE=$c; break; }
  done
fi
[ "$PAIR_DRY" = 1 ] || [ -x "$ENSURE" ] || { echo "SMOKE_FAIL no_seam_server_ensure ${ENSURE:-<not found>}"; exit 1; }

fail() { echo "SMOKE_FAIL $*"; exit 1; }

gx_mem() {  # prints "avail_mib=<n> swap_free_mib=<n> swap_total_mib=<n>"
  pair_gxsh "awk '/^MemAvailable:/{a=int(\$2/1024)} /^SwapFree:/{f=int(\$2/1024)} /^SwapTotal:/{t=int(\$2/1024)} END{printf \"avail_mib=%d swap_free_mib=%d swap_total_mib=%d\n\", a, f, t}' /proc/meminfo" \
    | tail -1
}

# The worker's OWN swap, not the box's: the box sits 1.3 GiB into swap merely from mapping the pack
# (measured 2026-11-03 with zero workers) while the worker itself can be clean, and the reverse -- a
# load that starts clean and then walks its own memory out during a 32k+512k prefill -- is the state
# that makes a later decode number describe paging instead of the release. So a reading is a set of
# numbers that names itself: the host's avail_mib and box swap (SwapTotal-SwapFree), and the worker's
# pid, VmSwap and RSS. One ssh round-trip, so the parts of a reading cannot interleave with a phase
# change. Prints "avail_mib=<n> box_swap_used_mib=<n> pid=<n>|- vm_swap_mib=<n>|- rss_mib=<n>|-".
gx_ws() {
  local script out
  # Sent on stdin, not as one ssh argument: the probe is a shell program inside a `docker exec sh -c`
  # inside an ssh command, and three levels of quoting is how the first version silently read an empty
  # pid and reported 0 MiB. The container name is substituted, everything else travels literally.
  script=$(sed "s/@CT@/$GXCT/g" <<'EOS'
awk '/^MemAvailable:/{a=int($2/1024)} /^SwapTotal:/{t=int($2/1024)} /^SwapFree:/{f=int($2/1024)} END{printf "avail_mib=%d box_swap_used_mib=%d\n", a, t-f}' /proc/meminfo
docker exec @CT@ sh -c '
best=""; bestr=0
for p in $(pgrep -x python3 -a | grep "worker_seam_server[.]py" | cut -d" " -f1); do
  r=$(awk "/^VmRSS:/{print \$2}" /proc/$p/status 2>/dev/null); r=${r:-0}
  [ "$r" -gt "$bestr" ] && { bestr=$r; best=$p; }
done
if [ -n "$best" ]; then awk "/^VmSwap:/{s=\$2} /^VmRSS:/{r=\$2} END{printf \"pid=%d vm_swap_mib=%d rss_mib=%d\n\", $best, s/1024, r/1024}" /proc/$best/status; else echo "pid=- vm_swap_mib=- rss_mib=-"; fi'
EOS
)
  # The worker is the python3 process, not the `bash -lc` wrapper that also carries its name; the
  # wrapper is 288 kB of swap and 364 kB of rss, which reads as a happy worker. Selecting by comm,
  # not by argv, is what makes an absolute-path worker (/pair/current/src/qmesh/...) visible too.
  out=$(printf '%s\n' "$script" | pair_gxstdin 2>/dev/null | tr '\n' ' ')
  case "$out" in
    *avail_mib=*pid=*) ;;
    *) out="avail_mib=- box_swap_used_mib=- pid=- vm_swap_mib=- rss_mib=-" ;;
  esac
  echo "$out"
}

if [ "$PAIR_DRY" = 1 ]; then
  echo "DRY ensure: ${ENSURE:-<not found>} GX=$GX CT=$GXCT GX_GID=$GX_GID PORT=$PORT SEAM_SERVER_TIMEOUT=$SMOKE_SERVER_TIMEOUT SEAM_ENV='$(pair_server_env "$BUILD_ID")' $ENSURE --fresh --buckets $SMOKE_BUCKETS"
  echo "DRY arm: $HERE/pair_arm.sh $PAIR --rows 1 --max-new $SMOKE_NEW --cache-tokens $SMOKE_CACHE_TOKENS"
  echo "DRY prefill: $HERE/pair_arm.sh $PAIR --prefill $SMOKE_PREFILL_DEPTH --timeout $SMOKE_PREFILL_TIMEOUT"
  echo "SMOKE_ROW i=0 new_tokens=$SMOKE_NEW text_sha256=dry"
  echo "SMOKE_PREFILL depth=$SMOKE_PREFILL_DEPTH tok_per_s=dry aborted=0"
  echo "SMOKE_OK"
  exit 0
fi

# The pack check runs before the GX gate (and only for a real smoke: DRY above never reads a
# filesystem). It costs a second and it reads the spec and the pack tree, so a pack that is not on the
# serving box's own disk is refused without a server start or a GPU second. PAIR_PACK_DISK=0 skips it
# (tests whose stub ssh has no filesystem to resolve).
if [ "${PAIR_PACK_DISK:-1}" = 1 ]; then
  # The hot half this check validates must be the client of the pack the GX box serves -- a promoted
  # cold pack checked against the previous hot half's placement refs passes on a hash that belongs to a
  # different pair of halves. RM10 2026-11-05 (Main): it is NAMED (PAIR_PACK_P7=, else AB_CKPT), never
  # derived -- two hot dirs on one sha voided three runs this morning. `pair_ckpt_resolve auto` is printed
  # as one cross-check line and never decides; naming none refuses the smoke: there is no default hot pack.
  PAIR_PACK_P7=$(pair_ckpt_require "${PAIR_PACK_P7:-}") \
    || fail "PAIR_PACK_P7: name the client half of $(pair_cold_src) sha=$(pair_cold_placement) (PAIR_PACK_P7=)"
  export PAIR_PACK_P7
  echo "  SMOKE_PACK p7=${PAIR_PACK_P7} cold=$(pair_cold_src) sha=$(pair_cold_placement)"
  pack_out=$(python3 "$HERE/pair_pack_disk.py" "$PAIR" 2>&1); pack_rc=$?
  printf '%s\n' "$pack_out" | sed 's/^/  /'
  [ "$pack_rc" = 0 ] || fail "pack_disk rc=$pack_rc (a pack is not on the serving box's own disk)"
fi

# The smoke owns the GX box until this gate expires: it loads the pack and measures avail_mib, and an
# experiment that loads or unloads a pack next to it turns every number below into a measurement of two
# jobs at once. A foreign claim is waited out (SMOKE_GATE_WAIT_MIN, then the smoke gives up rather than
# measure on a shared box); ours is self-expiring, so a killed smoke frees the box.
if ! pair_gx_gate_take "smoke-$PAIR" $(( (SMOKE_SERVER_TIMEOUT + SMOKE_ARM_TIMEOUT + SMOKE_PREFILL_TIMEOUT) / 60 + 15 )) "${SMOKE_GATE_WAIT_MIN:-20}"; then
  echo "SMOKE_GATE_BUSY $(pair_gx_gate_state)"
  fail "gx_busy $(pair_gx_gate_state)"
fi
# The claim and the server this smoke loaded are both its own, and both have to be released when it
# ends -- pass or fail. It used to keep the server on a passing verdict as "the update's result", and a
# prod smoke from 01:17 was still serving an hour later with nobody attached: a server outliving the
# ticket that started it is a leak whether or not it measured well. Stop first, then drop the claim:
# a stop is a lifecycle event on the box, so it belongs inside the claim, not after it.
smoke_exit() {
  pair_server_stop
  pair_gx_gate_down
  [ -n "${PAIR_SELF_COPY:-}" ] && rm -f "$PAIR_SELF_COPY"
}
trap smoke_exit EXIT
echo "SMOKE_GATE owner=smoke-$PAIR until=$(sed -n 's/.*until=\([0-9][0-9]*\).*/\1/p' "$(pair_gx_gate_file)")"

# A container can be alive and still be unable to reach the GPU (its device cgroup was re-derived by a
# systemd reload). Check both ends now that the box is ours; a restart is the fix and it costs seconds,
# while discovering it inside the worker's init costs the whole ticket.
pair_ensure_nvml p7 "$P7CT" || fail "nvml p7=$P7CT"
pair_ensure_nvml gx "$GXCT" || fail "nvml gx=$GXCT"
# A JIT build killed mid-link leaves torch's ext lock behind and every import then blocks in
# file_baton.wait at 512 MB with an empty log (2026-11-02). The ensure clears it too, but the smoke
# says so in its own log, and it only touches a lock when nothing alive could own it.
locks=$(pair_torch_locks "$GXCT")
if [ -n "$locks" ]; then
  if [ "$(pair_torch_build_live "$GXCT")" = 0 ]; then
    pair_gxsh "docker exec $GXCT sh -c 'rm -f /root/.cache/torch_extensions/*/*/lock /tmp/torch_extensions/*/*/lock'" >/dev/null 2>&1
    echo "SMOKE_TORCH_LOCK cleared=$(printf '%s' "$locks" | tr '\n' ',')"
  else
    echo "SMOKE_TORCH_LOCK kept (a build or server is alive beside it): $(printf '%s' "$locks" | tr '\n' ' ')"
  fi
fi

echo "SMOKE_MEM_BEFORE $(gx_mem)"

# A previous server's memory is not returned the instant its process exits: on 2026-11-02 the smoke
# started 40 s after an experiment had stopped, read avail 116726 MiB, and the worker refused a load
# of 116220+3072; minutes later the same box read 120254. Nothing about the release was wrong, and the
# ticket failed on a transient. So before asking the worker, wait for the box to come back: poll until
# MemAvailable covers the pack plus the load margin, up to SMOKE_MEM_WAIT_S. The pack size is read from
# the pack the server will actually load (the worker's own need is the same sum of the same files).
PACK_MIB=$(pair_gxsh "docker exec $GXCT sh -c 'du -bc /cold/cold-L*.safetensors 2>/dev/null | tail -1'" 2>/dev/null | cut -f1)
case "${PACK_MIB:-}" in ''|*[!0-9]*) PACK_MIB=0 ;; esac
if [ "$PACK_MIB" -gt 0 ]; then
  PACK_MIB=$(( PACK_MIB / 1048576 ))
  want=$(( PACK_MIB + LOAD_MARGIN ))
  waited=0
  while :; do
    mem=$(gx_mem); avail=$(printf '%s' "$mem" | sed -n 's/.*avail_mib=\([0-9]*\).*/\1/p')
    [ "${avail:-0}" -ge "$want" ] && break
    [ "$waited" -ge "${SMOKE_MEM_WAIT_S:-120}" ] && break
    echo "SMOKE_MEM_WAIT waited=${waited}s avail_mib=$avail want_mib=$want $(printf '%s' "$mem" | sed 's/^avail_mib=[0-9]* //')"
    sleep 10; waited=$(( waited + 10 ))
  done
  echo "SMOKE_MEM_READY avail_mib=${avail:-0} want_mib=$want pack_mib=$PACK_MIB margin_mib=$LOAD_MARGIN waited=${waited}s"
  # The wait is a courtesy for a server that just exited; it is not permission to load a box someone
  # else is holding. On 2026-11-03 dev read avail 325 MiB for the whole 120 s while prod read 119805
  # on the same pack, the load still went through (page cache is evictable), and the 32k prefill came
  # out 757.2 tok/s against prod's 973.0 -- a number about the box, attributed to the release. A load
  # that cannot get its pack plus margin in MemAvailable is refused with the numbers.
  if [ "${avail:-0}" -lt "$want" ] && [ "${SMOKE_MEM_STRICT:-1}" = 1 ]; then
    fail "memory box_busy avail_mib=${avail:-0} want_mib=$want pack_mib=$PACK_MIB margin_mib=$LOAD_MARGIN waited=${waited}s (another tenant holds GX RAM; a load here is served from disk -- SMOKE_MEM_STRICT=0 measures it anyway)"
  fi
fi
# A stamp proves the artifact was the one built; it does not prove a client's next import is a hit.
# torch's version hash covers the include-path spelling and the flags, so a cache written under a
# different spelling has a full build plan waiting for whoever imports first -- and inside a smoke
# that is the arm, under load. On 2026-11-02 a 196-unit rebuild there evicted the 113.65 GiB cold pack
# from page cache and the prefill fell from 790 to 351 rows/s on its first chunk (565.2 tok/s). Probe
# both halves with the client's own import, short: a hit costs a second, a miss fails the ticket before
# it loads a pack. The build that follows the failure is out of band (scripts/pair_build_ext.sh).
# The probe has to be the client's import, not merely an import. Container env already carries
# PYTHONPATH=/src/exllamav3, TORCH_EXTENSIONS_DIR=/pair/extcache and the right arch list per box
# (12.0 P7, 12.1 GX) -- overriding the arch here made every GX probe a miss by construction. And the
# cwd is /tmp, not /src/exllamav3: `python3 -c` resolves '' with getcwd(), which returns the release
# src dir with symlinks resolved, so from there the probe builds the /pair spelling of the include
# path while the arm and the server, both run as files, use the /src one.
ext_probe() {  # <side> <container>
  local side=$1 ct=$2 cmd
  # The cache is stated, not inherited. Until 2026-11-04 the container spec carried
  # TORCH_EXTENSIONS_DIR=/pair/extcache as its default, so every ad-hoc `import exllamav3` inside
  # seam-p7-dev/gx2-dev silently targeted whichever release the extcache link happened to name -- and
  # because torch's rebuild decision is textual in build.ninja, a stray import from another release's
  # src rewrote that release's cache and voided two A/B arms (20:12) plus a flip smoke. The spec now
  # defaults to /pair/scratch-ext; anything that means the *release's* cache has to say so.
  cmd='cd /tmp && timeout -k 5 ${SMOKE_EXT_PROBE_S:-60} \
python3 -c "import exllamav3.ext as e; print(\"SMOKE_EXT_HIT\", e.exllamav3_ext.__file__)"'
  if [ "$side" = p7 ]; then
    docker exec -e TORCH_EXTENSIONS_DIR=/pair/extcache "$ct" sh -c "$cmd" 2>&1
  else
    pair_gxsh "docker exec -e TORCH_EXTENSIONS_DIR=/pair/extcache $ct sh -c '$cmd'" 2>&1
  fi
}
for _s in p7 gx; do
  if [ "$_s" = p7 ]; then _ct=$P7CT; else _ct=$GXCT; fi
  _out=$(ext_probe "$_s" "$_ct"); _rc=$?
  case "$_out" in
    *SMOKE_EXT_HIT*)
      echo "SMOKE_EXT_HIT $_s ct=$_ct $(printf '%s' "$_out" | grep -m1 SMOKE_EXT_HIT | sed 's/^SMOKE_EXT_HIT //')" ;;
    *)
      fail "ext_client_miss $_s ct=$_ct rc=$_rc cache=/pair/extcache (a client import does not reuse the stamped artifact; run scripts/pair_build_ext.sh $PAIR --side $_s out of band)" ;;
  esac
done
pair_step "SMOKE_SERVER_START pair=$PAIR ct=$GXCT build=$BUILD_ID buckets=$SMOKE_BUCKETS timeout=${SMOKE_SERVER_TIMEOUT}s"
srv_out=$(GX=$GX CT=$GXCT GX_DEV=${GX_DEV:-rocep1s0f0} GX_GID=$GX_GID PORT=$PORT \
          SEAM_GATE_DIR="$(dirname "$(pair_gx_gate_file)")" \
          SERVER_BUCKETS=$SMOKE_BUCKETS SEAM_SERVER_TIMEOUT=$SMOKE_SERVER_TIMEOUT \
          SEAM_SLACK_GATE=$SMOKE_SLACK_GATE SEAM_SLAB_SLOTS=$SMOKE_SLAB_SLOTS \
          SEAM_ENV="$(pair_server_env "$BUILD_ID")" \
          "$ENSURE" --fresh --buckets "$SMOKE_BUCKETS" 2>&1)
srv_rc=$?
[ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$srv_out" >> "$PAIR_LOG"
printf '%s\n' "$srv_out" | grep -E '^(SEAM_ENSURE|SEAM_SLACK|SEAM_NO_MEMORY|SEAM_NO_PACK)' | sed 's/^/SMOKE_SERVER /'
# The worker refuses to load a pack that does not fit in MemAvailable; that refusal happens inside the
# container and reaches us as the worker's own line, so the verdict is read from the text as well as
# from the status. The ensure passes its real status through (it used to map every failure to 2), so a
# refusal before the load arrives as rc 3 and the line names the container and pid that said no.
case "$srv_out" in
  *SEAM_NO_MEMORY*)
    fail "memory server_no_memory $(printf '%s' "$srv_out" | grep -m1 SEAM_NO_MEMORY)" ;;
  *SEAM_NO_PACK*)
    fail "no_pack $(printf '%s' "$srv_out" | grep -m1 SEAM_NO_PACK)" ;;
esac
case "$srv_rc" in
  0) ;;
  3) line=$(printf '%s' "$srv_out" | grep -m1 PREFLIGHT_FAIL)
     case "$line" in
       # the box, not the release, said no: the ticket keeps the code and says so (memory_fail)
       *gx-mem*)  fail "memory server_preflight $line" ;;
       # another container's server holds the pack and the port: roll back, free the pair, and name it
       # (2026-11-02: "server_timeout_or_died" hid seam-gx2-4460 for the whole 900 s timeout).
       *gx-busy*) fail "gx_busy server_preflight $line" ;;
       *gate_*)   fail "gx_gate server_preflight $line" ;;
       *)         fail "server_preflight_refused $line" ;;
     esac
     ;;
  4) fail "memory server_slack $(printf '%s' "$srv_out" | grep -m1 SEAM_SLACK)"
     ;;
  2) fail "server_timeout_or_died after<${SMOKE_SERVER_TIMEOUT}s $(printf '%s' "$srv_out" | grep -m1 SEAM_ENSURE)"
     ;;
  *) fail "server_start rc=$srv_rc $(printf '%s' "$srv_out" | tail -1)"
     ;;
esac
case "$srv_out" in
  *state=started*) ;;
  *) fail "server_not_ready $(printf '%s' "$srv_out" | grep -m1 SEAM_ENSURE)" ;;
esac
# This smoke started that worker, so it owns it: the pid goes to state so an interrupted ticket can
# stop exactly this process instead of whatever a later ticket loaded (see pair_server_stop). Without
# it a ticket killed at 23:32 left the pack loaded and the 23:54 smoke refused `pair_busy gx_procs=2`.
if pair_server_record "$BUILD_ID"; then
  echo "SMOKE_SERVER_PID $(sed -n 's/.* pid=\([0-9]*\).*/\1/p' "$(pair_server_pidfile)")"
else
  echo "SMOKE_SERVER_PID unrecorded"
fi

# The decode arm runs FIRST, on the freshly loaded worker, and the prefill runs last -- a prefill is
# what walks the worker's own memory out (0 -> 1068 MiB on a 32k+512k run, 2026-11-02), so a decode
# measured after one is a decode measured on a paging server.
#
# The worker names the release it loaded, and that name is the release this smoke is measuring: a
# build_id mismatch is the box serving something else, and every number below would belong to it.
loaded=$(pair_gxsh "docker exec $GXCT sh -c \"grep -m1 '^SEAM_SERVER_LOADED' /tmp/seam_server.log\"" 2>/dev/null | tail -1)
echo "SMOKE_SERVER_LOADED $loaded"
case "$loaded" in
  *"build=$BUILD_ID"*) ;;
  *) fail "server_build_mismatch loaded=${loaded:-none} want=$BUILD_ID" ;;
esac
# Which cold-expert path the worker installed, and what the pair asked of it. SEAM_MOE_CPP is read by
# worker_seam_server.py, not by the client, so an arm that sets it in the client's environment is the
# same arm as its control -- 2026-11-03 04:16/04:31 ran exactly that and produced two identical rows
# (115.53 vs 114.64 ms/tok, same ids). The server knob is PAIR_SRV_EXTRA_ENVS; print both sides of it
# here so a hook that failed to install (SEAM_MOE_CPP_OFF err=...) can never be reported as a win.
cpp_line=$(pair_gxsh "docker exec $GXCT sh -c \"grep -m1 -E '^SEAM_MOE_CPP_(ON|OFF)' /tmp/seam_server.log\"" 2>/dev/null | tail -1)
echo "SMOKE_SERVER_CPP asked='${SRV_ENV_EXTRA:-none}' worker='${cpp_line:-python (no SEAM_MOE_CPP line)}'"
echo "SMOKE_MEM_SERVER $(gx_mem)"
#
# The load's own box pressure travels with every reading: a worker at 344 MiB of swap whose load found
# avail_mib=1132 / swapped_mib=1765 is this box reclaiming under a 116 GiB mapping, not a release that
# pages; the same worker at 344 MiB on a load that found 119205 / 0 is a different story. Both numbers
# are printed beside every phase and neither refuses anything.
SRV_LOAD_SWAPPED=$(printf '%s' "$loaded" | sed -n 's/.* swapped_mib=\([0-9][0-9]*\).*/\1/p')
SRV_LOAD_AVAIL=$(printf '%s' "$loaded" | sed -n 's/.* avail_mib=\([0-9][0-9]*\).*/\1/p')
ws_field() {  # <line> <key> -> "<n>", or "" (a "-" reading names no worker and is unjudgeable)
  printf '%s' "$1" | sed -n "s/.*[[:space:]]$2=\([0-9][0-9]*\).*/\1/p"
}
ws_before_arm=
ws_after_decode=
ws_after_prefill=
ws_check() {  # <phase> -> print the reading; refuse only a level a second pack explains
  local phase=$1 line avail mib
  line=$(gx_ws)
  printf 'SMOKE_WSWAP %s %s load_swapped_mib=%s load_avail_mib=%s\n' "$phase" "$line" \
    "${SRV_LOAD_SWAPPED:--}" "${SRV_LOAD_AVAIL:--}"
  case "$phase" in
    before_arm) ws_before_arm=$line ;;
    after_decode) ws_after_decode=$line ;;
    after_prefill) ws_after_prefill=$line ;;
  esac
  avail=$(ws_field "$line" avail_mib); mib=$(ws_field "$line" vm_swap_mib)
  [ -n "$mib" ] || { echo "SMOKE_WSWAP_UNJUDGED phase=$phase why=no_worker"; return 0; }
  if [ -n "$avail" ] && [ "$avail" -gt "$SMOKE_WSWAP_JUDGE_MIN" ]; then
    echo "SMOKE_WSWAP_UNJUDGED phase=$phase avail_mib=$avail min=$SMOKE_WSWAP_JUDGE_MIN why=not_post_residency"
    return 0
  fi
  if [ "$mib" -le "$SMOKE_WSWAP_MAX" ]; then
    # The level is printed as a warning, never as a refusal (owner, Option B): only another pack's depth
    # is refused, and only the growth inside the decode window is judged.
    echo "SMOKE_WSWAP_LEVEL phase=$phase vm_swap_mib=$mib max=$SMOKE_WSWAP_MAX verdict=warn-only"
    return 0
  fi
  fail "paging_level phase=$phase vm_swap_mib=$mib max=$SMOKE_WSWAP_MAX $(printf '%s' "$line" | sed 's/^vm_swap_mib=[0-9]* //') -- the worker is this deep in swap with avail_mib=${avail:-?}: that is a second pack on the box, not this release; free it (stop the other container's pack) and re-run"
}

# The growth gate over one measured window: <label> <before-line> <after-line> <refuse 0|1>. Only a
# window that starts and ends post-residency can be judged, and only growth is a fault -- a worker that
# reclaims less than it swapped in (delta < 0) is not one.
ws_growth_check() {  # <label> <before> <after> <refuse>
  local label=$1 before=$2 after=$3 refuse=$4 bv av av_a d lim
  # The window's own bar: the decode window is the owner's 128, a prefill window only labels itself.
  lim=$SMOKE_WSWAP_GROW_MAX
  [ "$refuse" = 1 ] || lim=$SMOKE_WSWAP_PREFILL_GROW_MAX
  bv=$(ws_field "$before" vm_swap_mib); av=$(ws_field "$after" vm_swap_mib)
  av_a=$(ws_field "$after" avail_mib)
  if [ -z "$bv" ] || [ -z "$av" ]; then
    echo "SMOKE_WSWAP_UNJUDGED window=$label why=no_worker"; return 0
  fi
  if [ -n "$av_a" ] && [ "$av_a" -gt "$SMOKE_WSWAP_JUDGE_MIN" ]; then
    echo "SMOKE_WSWAP_UNJUDGED window=$label avail_mib=$av_a min=$SMOKE_WSWAP_JUDGE_MIN why=not_post_residency"
    return 0
  fi
  d=$(( av - bv ))
  if [ "$d" -le "$lim" ]; then
    echo "SMOKE_WSWAP_GROW window=$label before_mib=$bv after_mib=$av delta_mib=$d max=$lim verdict=ok"
  elif [ "$refuse" = 1 ]; then
    fail "paging_growth window=$label before_mib=$bv after_mib=$av delta_mib=$d max=$lim -- the worker walked $d MiB of its own memory out inside this window, so these numbers describe the disk; free the box and re-run"
  else
    echo "SMOKE_WSWAP_GROW window=$label before_mib=$bv after_mib=$av delta_mib=$d max=$lim verdict=warn (expected after a prefill: this is printed, not refused)"
  fi
}
ws_check after_load

# The kernel the server actually serves on. cuda-exl3 is imported from the release's vendored tree
# (qmesh/exl3_moe_cuda.py), whose .so is untracked and therefore has to be built per cut: a release
# whose build stage skipped that imports nothing, prints SEAM_MOE_NOEXT and serves every layer on the
# mixed-K module -- correct output at a fraction of the rate, i.e. exactly the kind of release that
# must not reach a pair. On a memory-tight box a few layers may legitimately refuse the restack
# (SEAM_MOE_NOFIT); the validated run had 2 of 40, so allow up to a tenth and no more. The NOFIT
# lines are read and printed because they name *which* layers went to the mixed-K module, and on
# this box the same release draws 38/2 on one load and 39/1 on the next -- an ids digest is only
# comparable with another smoke whose list matches (2026-11-02, canary: 38/2 -> 0c9bbe87, 39/1 ->
# f96fc065). The log itself survives in pairs/<pair>/logs/server-<build>-<ts>.log.
kernel=$(pair_gxsh "docker exec $GXCT sh -c \"grep -E '^(SEAM_MOE_NOEXT|SEAM_MOE_NOFIT|SEAM_PACK_KERNEL)' /tmp/seam_server.log | tail -12\"" 2>/dev/null)
printf '%s\n' "$kernel" | grep -v '^$' | sed 's/^/SMOKE_KERNEL /'
case "$kernel" in
  *SEAM_MOE_NOEXT*) fail "moe_kernel_missing $(printf '%s' "$kernel" | grep -m1 SEAM_MOE_NOEXT | cut -c1-240)" ;;
esac
kv=$(printf '%s\n' "$kernel" | sed -n 's/^SEAM_PACK_KERNEL .*cudaexl3_layers=\([0-9]*\) mixedk_layers=\([0-9]*\) of=\([0-9]*\).*/\1 \2 \3/p' | tail -1)
read -r cx_n mx_n nl_n <<<"$kv"
if [ -z "${nl_n:-}" ]; then
  fail "moe_kernel_unknown $(printf '%s' "$kernel" | tail -1 | cut -c1-240)"
fi
if [ $((cx_n * 10)) -lt $((nl_n * 9)) ]; then
  fail "moe_kernel_mostly_mixedk cudaexl3_layers=$cx_n mixedk_layers=$mx_n of=$nl_n (the release's cuda-exl3 fits fewer layers than it did when it was validated)"
fi

# Which layers this load refused, in one line the ticket can carry: cx/mixedk counts from the
# PACK_KERNEL summary and the layer indices from the NOFIT lines. Two smokes with different
# nofit_layers were served by different kernels, so their ids are not comparable whatever the
# release says.
mixed_layers=$(printf '%s\n' "$kernel" | sed -n 's/^SEAM_MOE_NOFIT[^0-9]*\([0-9][0-9]*\).*/\1/p' | paste -sd, -)
echo "SMOKE_KERNEL_MIX cx_layers=$cx_n mixedk_layers=$mx_n nofit_layers=${mixed_layers:-none}"

# The box's own verdict on the load, on every smoke: avail after the pack, the swap the load pushed
# to, and the slack the server reserved. A pair that passes here with 1.4 GiB and 3.7 GiB of swap is a
# pair whose next failure will look like a CUDA abort (2026-11-01 16:44, GX1).
printf '%s\n' "$srv_out" | grep -E '^SEAM_SERIAL|^SEAM_SLACK|^SEAM_SERVER_SLAB' | sed 's/^/SMOKE_SERVER /'

# The arm's own log: a client that hangs (a silent JIT rebuild, a stale lock baton, a seam handshake
# that never answers) produces no output at all by the time timeout(1) kills it, and "rc=124" then
# says nothing about why. Keep the bytes, and on failure the server's log tail next to them.
ARM_LOG=${SMOKE_ARM_LOG:-$PAIR_STATE_DIR/logs/smoke-arm-$PAIR-$(date -u +%Y%m%dT%H%M%SZ).log}
mkdir -p "$(dirname "$ARM_LOG")" 2>/dev/null || true
ARM_IDS_OUT=${ARM_IDS_OUT:-${ARM_LOG%.log}.ids.json}
#
# One arm's worth of decode, thrown away, before the window opens: the first forward in a client
# allocates the draft and verify buffers, and on a box whose avail is under a gigabyte the kernel
# answers that allocation by swapping ~330 MiB of the worker out. That spill belongs to the box, and
# if it lands inside the window the gate can only refuse (2026-11-03 04:05: 0 -> 333 MiB inside an
# 8-token decode, while the hook-off arm of the same pair had spilled its 336 before the arm and grew
# not at all -- same work, same box, different side of the baseline). Warming the same path first
# moves it before the baseline, so the window measures the arm and not the first touch; the baseline
# is still taken right before the arm and the gate is unchanged.
if [ "${SMOKE_ARM_WARM:-1}" = 1 ]; then
  pair_step "SMOKE_ARM_WARM pair=$PAIR new=$SMOKE_NEW"
  warm_out=$("$HERE/pair_arm.sh" "$PAIR" --rows 1 --max-new "$SMOKE_NEW" --cache-tokens "$SMOKE_CACHE_TOKENS" \
             --timeout "$SMOKE_ARM_TIMEOUT" 2>&1)
  warm_rc=$?
  printf '%s\n' "$warm_out" > "${ARM_LOG%.log}.warm.log" 2>/dev/null || true
  echo "SMOKE_ARM_WARM rc=$warm_rc $(printf '%s\n' "$warm_out" | grep -m1 'decode_ms_per_tok' | cut -c1-160)"
  [ "$warm_rc" = 0 ] || fail "arm_warm rc=$warm_rc $(printf '%s\n' "$warm_out" | grep -m1 -E 'Traceback|Error|error' | cut -c1-300)"
fi
ws_check before_arm
pair_step "SMOKE_ARM pair=$PAIR rows=1 new=$SMOKE_NEW"
arm_out=$(ARM_IDS_OUT="$ARM_IDS_OUT" \
          "$HERE/pair_arm.sh" "$PAIR" --rows 1 --max-new "$SMOKE_NEW" --cache-tokens "$SMOKE_CACHE_TOKENS" \
          --timeout "$SMOKE_ARM_TIMEOUT" --ids 2>&1)
arm_rc=$?
printf '%s\n' "$arm_out" > "$ARM_LOG" 2>/dev/null || true
[ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$arm_out" >> "$PAIR_LOG"
server_tail() {
  local log=${SEAM_LOG:-/tmp/seam_server.log}
  echo "SMOKE_SERVER_TAIL ct=$GXCT log=$log"
  pair_gxsh "docker exec $GXCT sh -c 'tail -40 $log 2>/dev/null; echo ---; grep -c . $log 2>/dev/null'" 2>&1 | tail -45
  echo "SMOKE_EXT_CACHE $(pair_gxsh "docker exec $GXCT sh -c 'ls -l /pair/extcache/exllamav3_ext/lock /pair/extcache/exllamav3_ext/*.so 2>/dev/null'" 2>&1 | head -3)"
  echo "SMOKE_P7_EXT_CACHE $(docker exec "$P7CT" sh -c 'ls -l /pair/extcache/exllamav3_ext/lock /pair/extcache/exllamav3_ext/*.so 2>/dev/null' 2>&1 | head -3)"
}
if [ "$arm_rc" != 0 ]; then
  echo "SMOKE_ARM_LOG $ARM_LOG ($(wc -c < "$ARM_LOG" 2>/dev/null || echo 0) bytes)"
  server_tail
  # A client killed for memory (the box swapped the pack out from under it) is a capacity failure:
  # classify it so the ticket keeps the release instead of rolling good code back.
  case "$(printf '%s\n' "$arm_out"; cat "$ARM_LOG" 2>/dev/null)" in
    *"out of memory"*|*"OutOfMemory"*|*"Cannot allocate memory"*|*"unspecified launch failure"*)
      fail "memory arm rc=$arm_rc $(printf '%s' "$arm_out" | grep -m1 -iE 'out of memory|OutOfMemory|Cannot allocate memory|unspecified launch failure' | cut -c1-200)" ;;
  esac
  fail "arm rc=$arm_rc $(printf '%s' "$arm_out" | grep -m1 -E 'peer layout mismatch|remote invalid request|Traceback|Error|error' | cut -c1-300)"
fi
case "$arm_out" in
  *"peer layout mismatch"*|*"remote invalid request"*)
    fail "wire_mismatch $(printf '%s' "$arm_out" | grep -m1 -E 'peer layout mismatch|remote invalid request' | cut -c1-300)" ;;
esac

# The decode window closes here, before any parsing or any prefill: baseline `before_arm` -> this
# reading, refused above SMOKE_WSWAP_GROW_MAX. A decode that walked the worker's own memory out is a
# decode of the disk, whatever its tok/s says, so it must not become a number anybody quotes.
ws_check after_decode
ws_growth_check decode "$ws_before_arm" "$ws_after_decode" 1

# The grant, not the request, is the shape the pair runs: the server advertises its --buckets and the
# client adopts the intersection, so a client asking for less than the server grants silently caps
# every prefill chunk at its own largest bucket. That is what happened on 23:32 (server 1,6,64,8192,
# client default topped out at 4096 and contained neither 8192 nor 256, grant 1,6,64, chunk_rows=64,
# 1.6 s/token): the smoke measured a pair that was not the shipped one. A smoke that cannot see the
# shipping width has nothing to say about the release.
grant=$(printf '%s\n' "$arm_out" | sed -n 's/.*accepted=\([0-9][0-9,]*\).*/\1/p' | head -1)
grant_line=$(python3 "$HERE/pair_bucket_grant.py" "$grant" "$SMOKE_BUCKETS"); grant_rc=$?
case "$grant_rc" in
  0) echo "SMOKE_BUCKETS_GRANT $grant_line" ;;
  2) # No negotiated line: a client that never reached the server. The arm's own rc/rows checks below
     # already fail the smoke; inventing a grant verdict here would report the wrong cause.
     echo "SMOKE_BUCKETS_GRANT unknown want=$SMOKE_BUCKETS" ;;
  *) fail "bucket_grant $grant_line (a prefill chunk splits at the largest granted bucket; the arm must ask for the pair's list)" ;;
esac

# The rows are read from the arm's log, by path: `python3 - <<'PY'` owns stdin with its own source, so a
# piped arm_out read as empty and this check reported "arm_no_tokens NONE" for an arm whose log held a
# full 8-token row (23:29:19 on dev). The parser is pair_arm_rows.py so that it can be tested.
row=$(python3 "$HERE/pair_arm_rows.py" "$ARM_LOG")
printf '%s\n' "$row" | sed -n 's/^i=/SMOKE_ROW /p'
new_tokens=$(printf '%s\n' "$row" | sed -n 's/^NEW //p')
[ "${new_tokens:-0}" -gt 0 ] || fail "arm_no_tokens $(printf '%s\n' "$row" | head -1)"
# The ids are the promotion's proof that a decode really ran: a wrong build can still emit fluent
# text, but not the same ids. Missing ids fail the smoke unless SMOKE_IDS_OPTIONAL=1 (a release
# whose driver cannot report them), because prod may not be reached without them.
ids_lines=$(printf '%s\n' "$arm_out" | sed -n 's/^ARM_IDS /SMOKE_ARM_IDS /p')
ids_n=$(printf '%s\n' "$ids_lines" |
        awk '{for (i = 1; i <= NF; i++) if ($i ~ /^n=/) s += substr($i, 3) + 0} END {print s + 0}')
ids_n=${ids_n:-0}
if [ -n "$ids_lines" ]; then printf '%s\n' "$ids_lines"; else echo "SMOKE_ARM_IDS none"; fi
echo "SMOKE_ARM_LOG $ARM_LOG"
echo "SMOKE_ARM_IDS_OUT $ARM_IDS_OUT"
# One line the ticket can record without guessing which arm came first: the total across arms and a
# commitment over every ARM_IDS line (arm, count, per-arm hash), so two smokes of one release can be
# told apart even when a single arm's hash coincides.
if [ "$ids_n" -gt 0 ]; then
  ids_commit=$(printf '%s\n' "$ids_lines" | LC_ALL=C sort | sha256sum | cut -d' ' -f1)
  echo "SMOKE_ARM_IDS_TOTAL n=$ids_n sha256=$ids_commit"
fi
# The layer set travels with the ids: the ids json is what a later prod smoke cross-checks against,
# and a digest change caused by a restack has to be distinguishable from one caused by the release.
echo "SMOKE_ARM_IDS_KERNEL cx_layers=$cx_n mixedk_layers=$mx_n nofit_layers=${mixed_layers:-none}"
case "$ARM_IDS_OUT" in
  *.json) printf 'cx_layers=%s mixedk_layers=%s nofit_layers=%s\n' \
                 "$cx_n" "$mx_n" "${mixed_layers:-none}" > "${ARM_IDS_OUT%.json}.kernel.txt" 2>/dev/null || true ;;
esac
# Any single arm that generated zero ids fails as well: its digest is the constant hash of [], which
# would let two dead arms (or a dead dev and a dead prod) compare "identical" in the crosscheck.
ids_zero=$(printf '%s\n' "$ids_lines" | awk '{for (i = 1; i <= NF; i++) if ($i == "n=0") z++} END {print z + 0}')
if { [ "$ids_n" -le 0 ] || [ "${ids_zero:-0}" -gt 0 ]; } && [ "${SMOKE_IDS_OPTIONAL:-0}" != 1 ]; then
  fail "arm_no_ids (total n=$ids_n, arms with n=0: ${ids_zero:-0}; a release that cannot report ids cannot be promoted, SMOKE_IDS_OPTIONAL=1 skips this)"
fi
# The prefill arm: the number the owner's canary bar is stated in. One load, one 32k prompt, one
# chunked prefill through the release's seam pieces, engine tok/s as reported by the same driver the
# 837.6/908.0 were taken with. Tokens without a rate (an aborted arm) can not pass: the abort is the
# failure mode this exists to catch, so both are read from the same line.
pair_step "SMOKE_PREFILL pair=$PAIR depth=$SMOKE_PREFILL_DEPTH min_tps=$SMOKE_PREFILL_MIN_TPS timeout=${SMOKE_PREFILL_TIMEOUT}s"
pf_mem_before=$(gx_mem)
echo "SMOKE_MEM_PREFILL_BEFORE $pf_mem_before"
pf_log=$PAIR_STATE_DIR/logs/smoke-prefill-$PAIR-$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$(dirname "$pf_log")" 2>/dev/null || true
pf_out=$(ARM_IDS_OUT="" "$HERE/pair_arm.sh" "$PAIR" --prefill "$SMOKE_PREFILL_DEPTH" \
         --timeout "$SMOKE_PREFILL_TIMEOUT" 2>&1)
pf_rc=$?
printf '%s\n' "$pf_out" > "$pf_log" 2>/dev/null || true
[ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$pf_out" >> "$PAIR_LOG"
pf_line=$(printf '%s\n' "$pf_out" | sed -n "s/^PREFILL_ARM depth=$SMOKE_PREFILL_DEPTH .*/&/p" | tail -1)
if [ "$pf_rc" != 0 ] || [ -z "$pf_line" ]; then
  echo "SMOKE_PREFILL_LOG $pf_log"
  server_tail
  case "$(printf '%s\n' "$pf_out")" in
    *"out of memory"*|*"OutOfMemory"*|*"Cannot allocate memory"*|*"unspecified launch failure"*)
      fail "memory prefill rc=$pf_rc $(printf '%s' "$pf_out" | grep -m1 -iE 'out of memory|OutOfMemory|Cannot allocate memory|unspecified launch failure' | cut -c1-200)" ;;
  esac
  fail "prefill rc=$pf_rc $(printf '%s\n' "$pf_out" | grep -m1 -E 'Traceback|Error|error|MEM_ABORT' | cut -c1-300)"
fi
pf_kv=$(printf '%s\n' "$pf_line" | tr ' ' '\n' | sed -n 's/^\(tok_per_s\|aborted\|prefill_s\|shrunk\)=//p' | tr '\n' ' ')
pf_tps=$(printf '%s\n' "$pf_line" | sed -n 's/.*tok_per_s=\([0-9.]*\).*/\1/p')
pf_ab=$(printf '%s\n' "$pf_line" | sed -n 's/.*aborted=\([0-9]*\).*/\1/p')
echo "SMOKE_PREFILL_LOG $pf_log"
echo "SMOKE_PREFILL depth=$SMOKE_PREFILL_DEPTH ${pf_kv% } line_rc=0"
pf_mem_after=$(gx_mem)
echo "SMOKE_MEM_PREFILL_AFTER $pf_mem_after"
# The prefill window is judged for the absolute level only (a second pack would show there too): growth
# here is expected -- 356 MiB at ws 256, 734 at ws 1024 after a 32k prefill -- so it is printed and
# warned, never refused.
ws_check after_prefill
ws_growth_check prefill "${ws_after_decode:-$ws_before_arm}" "$ws_after_prefill" 0
[ "${pf_ab:-1}" = 0 ] || fail "prefill_aborted depth=$SMOKE_PREFILL_DEPTH aborted=$pf_ab line=$pf_line"
case "$(printf '%s\n' "$pf_out")" in
  *"SEAM_COMPUTE_FAIL"*)
    fail "prefill_compute_fail $(printf '%s' "$pf_out" | grep -m1 SEAM_COMPUTE_FAIL | cut -c1-240)" ;;
esac
[ -n "${pf_tps:-}" ] || fail "prefill_no_rate depth=$SMOKE_PREFILL_DEPTH line=$pf_line"
# awk, not test: tok/s is fractional and the gate is a real threshold, not a string compare. The rate
# is only judged when the box had memory to serve the pack at both ends of the window (see
# SMOKE_PREFILL_MIN_AVAIL_MIB): a starved box measures swap I/O and would fail a healthy release.
pf_avail_min=$(printf '%s\n%s\n' "$pf_mem_before" "$pf_mem_after" \
  | sed -n 's/.*avail_mib=\([0-9]\{1,\}\).*/\1/p' | sort -n | sed -n '1p')
if [ -n "$pf_avail_min" ] && [ "$pf_avail_min" -lt "$SMOKE_PREFILL_MIN_AVAIL_MIB" ]; then
  echo "SMOKE_PREFILL_UNJUDGED depth=$SMOKE_PREFILL_DEPTH tok_per_s=$pf_tps min=$SMOKE_PREFILL_MIN_TPS avail_mib_min=$pf_avail_min min_avail=$SMOKE_PREFILL_MIN_AVAIL_MIB why=box_starved"
else
  awk -v got="$pf_tps" -v want="$SMOKE_PREFILL_MIN_TPS" 'BEGIN { exit !(got + 0 >= want + 0) }' \
    || fail "prefill_slow depth=$SMOKE_PREFILL_DEPTH tok_per_s=$pf_tps min=$SMOKE_PREFILL_MIN_TPS avail_mib_min=${pf_avail_min:-unknown}"
  echo "SMOKE_PREFILL_OK depth=$SMOKE_PREFILL_DEPTH tok_per_s=$pf_tps min=$SMOKE_PREFILL_MIN_TPS aborts=0"
fi
# The 31 % first-arm penalty was a cache-in-a-fresh-container effect (Triton, NVRTC cubin, autotune file)
# plus in-process warm-up, and the release's answer to the in-process half is the prefill warm-up above.
# The gate for it is a second identical arm right here, on the same loaded worker: arm 1 is the shape a
# first request meets, arm 2 is what a warmed process gives, and the ratio is the whole claim. It runs
# after the rate gate so a slow release still fails on the number the owner's bar is stated in; a ratio
# above SMOKE_PREFILL_RATIO_MAX means the warm-up is not reaching the same shape (the log names the row
# counts it used). Judged only when the warm-up lines were actually present, because a release that
# ships WARM_PREFILL=0 must not be failed for the thing it turned off.
if [ "${SMOKE_PREFILL_ARMS:-2}" -ge 2 ]; then
  pair_step "SMOKE_PREFILL_ARM2 pair=$PAIR depth=$SMOKE_PREFILL_DEPTH timeout=${SMOKE_PREFILL_TIMEOUT}s"
  pf2_out=$(ARM_IDS_OUT="" "$HERE/pair_arm.sh" "$PAIR" --prefill "$SMOKE_PREFILL_DEPTH" \
            --timeout "$SMOKE_PREFILL_TIMEOUT" 2>&1)
  pf2_rc=$?
  printf '%s\n' "$pf2_out" > "${pf_log%.log}-arm2.log" 2>/dev/null || true
  [ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$pf2_out" >> "$PAIR_LOG"
  pf2_line=$(printf '%s\n' "$pf2_out" | sed -n "s/^PREFILL_ARM depth=$SMOKE_PREFILL_DEPTH .*/&/p" | tail -1)
  pf2_tps=$(printf '%s\n' "$pf2_line" | sed -n 's/.*tok_per_s=\([0-9.]*\).*/\1/p')
  pf2_ab=$(printf '%s\n' "$pf2_line" | sed -n 's/.*aborted=\([0-9]*\).*/\1/p')
  if [ "$pf2_rc" != 0 ] || [ -z "$pf2_line" ] || [ "${pf2_ab:-1}" != 0 ]; then
    fail "prefill_arm2 rc=$pf2_rc aborted=${pf2_ab:-unknown} $(printf '%s\n' "$pf2_out" | grep -m1 -E 'Traceback|Error|MEM_ABORT' | cut -c1-240)"
  fi
  if [ -n "${pf_tps:-}" ] && [ -n "${pf2_tps:-}" ]; then
    # Measured here (both arms on this smoke's worker, back to back); judged after the server block,
    # because the warm-up that is supposed to remove this penalty runs in the server process and only
    # that run can show the lines that justify the gate.
    pf_ratio=$(awk -v a="$pf_tps" -v b="$pf2_tps" 'BEGIN { if (b + 0 <= 0) print "nan"; else printf "%.3f", a / b }')
    echo "SMOKE_PREFILL_ARMS arm1_tok_per_s=$pf_tps arm2_tok_per_s=$pf2_tps ratio=$pf_ratio max=${SMOKE_PREFILL_RATIO_MAX:-1.05}"
  else
    pf_ratio=
    echo "SMOKE_PREFILL_ARMS unmeasured arm1_tok_per_s=${pf_tps:-none} arm2_tok_per_s=${pf2_tps:-none} why=no_rate"
  fi
fi
# The OpenAI-compatible server is on no critical path -- nothing in the release calls it unless an
# operator does -- so the default smoke does not pay for a second load. PAIR_SMOKE_SERVER=1 adds it:
# pair_server_smoke.sh starts the pair's server, exercises it (models, completion usage, chat
# streaming identity, a repeated ~4k prefix that must report cached_tokens, a positive tool call) and
# stops only the process it started. Opt-in because the prefill arm above has just paid for the load
# this needs, and because a pair that fails here should fail the smoke rather than a user's client.
if [ "${PAIR_SMOKE_SERVER:-0}" = 1 ]; then
  # The seam worker this smoke loaded holds the pack and the GX card, and the HTTP server brings its own
  # (pair_server.sh start): two workers on one pack is `gx_busy`, not a second measurement. Every arm
  # above has already been taken from this smoke's worker, so it goes now; the EXIT trap's stop then
  # finds no pidfile and is a no-op.
  # The claim goes with it: pair_serve.sh takes the gate as `serve-$PAIR` and dies `PAIR_SERVE_GX_BUSY`
  # if anyone holds it, so a smoke that kept `smoke-$PAIR` would fail the server stage for its own
  # claim. The server re-claims immediately (it is the only process that loads the pack from here on)
  # and drops the gate when its worker is up.
  pair_server_stop >/dev/null 2>&1 || true
  pair_gx_gate_down >/dev/null 2>&1 || true
  pair_step "SMOKE_HTTP_SERVER pair=$PAIR ct=$P7CT port=${PAIR_SRV_PORT:-${SRV_PORT:-8000}}"
  # No port argument: pair_server_smoke.sh resolves PAIR_SRV_PORT/SRV_PORT/8000 itself, the same way
  # pair_server.sh picked the port this pair serves on.
  srv_out=$("$HERE/pair_server_smoke.sh" "$PAIR" 2>&1)
  srv_rc=$?
  printf '%s\n' "$srv_out" | tail -40
  [ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$srv_out" >> "$PAIR_LOG"
  # pair_server_smoke.sh reports through pair_step, which prefixes HH:MM:SS, so the anchor must tolerate
  # the stamp: '^SMOKE_OK ' alone matched 0 lines and failed the HTTP stage on a serving-clean run.
  srv_ok=$(printf '%s\n' "$srv_out" | grep -cE '(^| )SMOKE_OK ')
  if [ "$srv_rc" != 0 ] || [ "$srv_ok" -lt 1 ]; then
    fail "http_server rc=$srv_rc $(printf '%s\n' "$srv_out" | grep -m1 -E '^SMOKE_FAIL|^PAIR_SRV' | cut -c1-300)"
  fi
  echo "SMOKE_HTTP_OK $(printf '%s\n' "$srv_out" | grep -m1 '^SMOKE_OK ')"
  # The release warms one prefill call per seam bucket before SERVER_READY (`serve_pair.py`,
  # `WARM_PREFILL=1`). That is a property of the *server* process, so the only run that can prove it is
  # this one -- and the lines land in the server's own log on P7 (`serve_pair.py` prints
  # `WARM_PREFILL_OK`, the client-side seam host in the same process prints `SEAM_NEGOTIATED`), never in
  # the GX worker's `/tmp/seam_server.log`. `tail -1` on the grant because the file is truncated per
  # start but a restart mid-run leaves an older line above the current one. SMOKE_WARM_OPTIONAL=1 is for
  # a release that deliberately ships the warm-up off; SMOKE_WARM_ROWS=0 disables the count check.
  srv_log=$(pair_ct_path "$PAIR_STATE_DIR/server/pair-server.log")
  warm_log=$(docker exec "$P7CT" grep -E '^(WARM_PREFILL_OK|SEAM_NEGOTIATED)' "$srv_log" 2>/dev/null)
  warm_lines=$(printf '%s\n' "$warm_log" | grep -c '^WARM_PREFILL_OK ')
  warm_neg=$(printf '%s\n' "$warm_log" | grep '^SEAM_NEGOTIATED ' | tail -1)
  echo "SMOKE_WARM_PREFILL lines=$warm_lines want=$SMOKE_WARM_ROWS ${warm_neg:-negotiated=none} log=$srv_log"
  if [ "$warm_lines" -lt "$SMOKE_WARM_ROWS" ] && [ "${SMOKE_WARM_OPTIONAL:-0}" != 1 ]; then
    fail "warm_prefill_missing lines=$warm_lines want=$SMOKE_WARM_ROWS log=$srv_log last=$(printf '%s\n' "$warm_log" | tail -1 | cut -c1-200)"
  fi
fi
# The first-arm ratio is judged here, where `warm_lines` is known. Only a run that actually started the
# release's own server can say whether the warm-up fired, so a smoke without one prints the ratio and
# says why it is not judging it -- a red gate for a scenario that was never run is how a canary stops
# being read.
if [ "${SMOKE_PREFILL_ARMS:-2}" -ge 2 ] && [ -n "${pf_ratio:-}" ]; then
  if [ "${warm_lines:-0}" -gt 0 ]; then
    awk -v r="$pf_ratio" -v m="${SMOKE_PREFILL_RATIO_MAX:-1.05}" 'BEGIN { exit !(r + 0 > m + 0) }' \
      && fail "prefill_first_arm_penalty arm1=$pf_tps arm2=$pf2_tps ratio=$pf_ratio max=${SMOKE_PREFILL_RATIO_MAX:-1.05} warm_lines=$warm_lines"
    echo "SMOKE_PREFILL_RATIO_OK ratio=$pf_ratio max=${SMOKE_PREFILL_RATIO_MAX:-1.05} warm_lines=$warm_lines"
  else
    echo "SMOKE_PREFILL_RATIO_UNJUDGED ratio=$pf_ratio why=no_warm_up (PAIR_SMOKE_SERVER=1 is the run that exercises it)"
  fi
fi
echo "SMOKE_OK $PAIR $SHA $BUILD_ID"
# The pidfile stays: the EXIT trap uses it to stop the server it started. Deleting it on the passing
# path is what turned a measured update into a server nobody could attribute an hour later.
