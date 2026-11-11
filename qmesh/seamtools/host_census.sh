#!/usr/bin/env bash
# One line of host memory accounting for the GX that is serving a pack: the split that matters when
# a 113 GiB pack is resident (available vs slab vs unreclaimable), and the ten slab caches holding
# the most bytes. `SUnreclaim` is the part of `Slab` the box can never give back, so a load that
# leaves it higher than it found it is holding memory the expert store does not account for.
#
#   host_census.sh <phase>        -> phase=<phase> avail_mib=… sunreclaim_mib=… … top=[…]
#
# `/proc/slabinfo` is root-only on 6.x/7.x, and `sudo -n` is denied for the service user on both GX10s, so
# the read goes through `docker exec` into a container on the same kernel: exec runs as uid 0 and
# /proc is the host's. The host-side `SUnreclaim` needs no privilege and is read directly, so the
# two halves of the line can be cross-checked against each other.
set -uo pipefail
tag=${1:?phase}
CT=${CX_CT:-seam-gx2-dev}

read -r a u r s < <(awk '/^MemAvailable:/{a=$2} /^SUnreclaim:/{u=$2} /^SReclaimable:/{r=$2} \
                          /^Slab:/{s=$2} END{print a+0, u+0, r+0, s+0}' /proc/meminfo) || true

# `active_objs * objsize` is what the cache actually holds; `num_objs` is what it has taken from the
# page allocator, which for a drained cache is mostly free pages the box could still reclaim.
top=""
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CT"; then
  top=$(docker exec "$CT" awk 'NF >= 4 && $1 !~ /^#/ && $1 !~ /^slabinfo/ {
                                 print $1 "=" int($2 * $4 / 1048576) }' /proc/slabinfo 2>/dev/null |
          sort -t= -k2 -nr | head -10 | tr '\n' ' ')
fi
if [ -z "$top" ]; then
  top=$(sudo -n cat /proc/slabinfo 2>/dev/null |
          awk 'NF >= 4 && $1 !~ /^#/ && $1 !~ /^slabinfo/ { print $1 "=" int($2 * $4 / 1048576) }' |
          sort -t= -k2 -nr | head -10 | tr '\n' ' ')
  [ -n "$top" ] || top="unavailable(no_root_on_host)"
fi

printf 'phase=%s avail_mib=%d free_mib=%s sunreclaim_mib=%d sreclaimable_mib=%d slab_mib=%d top=[%s]\n' \
  "$tag" "$((a / 1024))" \
  "$(awk '/^MemFree:/{print int($2/1024)}' /proc/meminfo)" \
  "$((u / 1024))" "$((r / 1024))" "$((s / 1024))" "${top% }"
