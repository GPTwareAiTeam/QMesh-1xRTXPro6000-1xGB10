#!/usr/bin/env bash
# Memory trajectory of a GX while the pack loads and while it serves a prefill.
#
#   gx_sampler.sh <interval_s> <max_samples> <container> [outfile]
#
# Prints one SAMP line per interval until <max_samples> or until /tmp/qm_sampler.stop appears, so the
# caller can end it without a signal: `touch /tmp/qm_sampler.stop`. Two halves, because "what grows
# during prefill" is answered by which half moves:
#
#   host   -- MemAvailable, MemFree, Cached, Slab/SReclaimable/SUnreclaim, Mapped, Shmem, SwapFree.
#             A rise in Cached with flat process memory is page cache (the pack being read); a rise in
#             SUnreclaim is kernel slab nobody can give back; Mapped+Shmem covers shared staging.
#   worker -- the seam server's own VmRSS, VmSwap and Pss (smaps_rollup). The unified pool is host
#             DRAM on GB10, so a CUDA allocation shows up as VmRSS growth and paging as VmSwap.
#             `nvidia-smi --query-gpu=memory.used` answers [N/A] on this box (unified, no dedicated
#             framebuffer), so there is no separate GPU-side number to read.
#
# torch's memory_reserved/allocated cannot be read from outside the process; the load line's
# SEAM_SERVER_DEVMEM stays the only allocator-internal point, and this samples the process instead.
#
# Cost per sample: one /proc/meminfo read and one docker exec.
set -uo pipefail
S=${1:-10}
MAX=${2:-400}
CT=${3:-seam-gx2-dev}
OUT=${4:-/dev/stdout}
STOP=/tmp/qm_sampler.stop
rm -f "$STOP"
mib() { echo $(( ${1:-0} / 1024 )); }

host_mem() {
  awk '/^MemAvailable:/{a=$2} /^MemFree:/{f=$2} /^Cached:/{c=$2} /^SUnreclaim:/{u=$2} \
       /^SReclaimable:/{r=$2} /^Mapped:/{m=$2} /^Shmem:/{h=$2} /^SwapFree:/{s=$2} \
       END{print a+0, f+0, c+0, u+0, r+0, m+0, h+0, s+0}' /proc/meminfo
}
# One exec, not five, and comm=python3 rather than pgrep's first hit: the serving process is started
# as `bash -lc 'cd /src/exllamav3 && stdbuf … python3 -u qmesh/worker_seam_server.py …'`, so the
# first match (pid 3260 on GX2) is a 1 MiB wrapper whose VmRSS would hide the whole question.
worker_mem() {
  docker exec "$CT" sh -c '
    p=0
    for c in $(pgrep -f "worker_seam_server[.]py"); do
      [ "$(cat /proc/$c/comm 2>/dev/null)" = python3 ] && { p=$c; break; }
    done
    [ "$p" != 0 ] || { echo "0 0 0 0"; exit 0; }
    r=0; s=0; ps=0
    while read -r k v _; do
      case "$k" in VmRSS:) r=$v ;; VmSwap:) s=$v ;; esac
    done < /proc/$p/status
    while read -r k v _; do
      case "$k" in Pss:) ps=$v ;; esac
    done < /proc/$p/smaps_rollup 2>/dev/null
    echo "$p $r $s $ps"' 2>/dev/null | tail -1
}

i=0
while [ "$i" -lt "$MAX" ] && [ ! -e "$STOP" ]; do
  i=$((i + 1))
  read -r av fr ca un rc mp sh sw < <(host_mem)
  pid=0; rss=0; vsw=0; pss=0
  read -r pid rss vsw pss < <(worker_mem) || true
  printf 'SAMP i=%d t=%s avail_mib=%d free_mib=%d cached_mib=%d sunreclaim_mib=%d sreclaimable_mib=%d mapped_mib=%d shmem_mib=%d swapfree_mib=%d pid=%s vm_rss_mib=%d vm_swap_mib=%d pss_mib=%d\n' \
    "$i" "$(date +%H:%M:%S)" "$(mib "$av")" "$(mib "$fr")" "$(mib "$ca")" "$(mib "$un")" \
    "$(mib "$rc")" "$(mib "$mp")" "$(mib "$sh")" "$(mib "$sw")" \
    "${pid:-0}" "$(mib "${rss:-0}")" "$(mib "${vsw:-0}")" "$(mib "${pss:-0}")" >> "$OUT"
  sleep "$S"
done
rm -f "$STOP"
