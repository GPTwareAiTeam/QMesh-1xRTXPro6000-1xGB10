#!/usr/bin/env bash
# Prove, once, that a client on the *other* cold-expert table is refused at the Hello and that the
# refusal costs a session and not the server.
#
# Runs entirely inside one GX container: a --stub server (no pack, no CUDA, exempt from the gate,
# a few hundred MiB) on one table, and the shipping refusal client pointed at the other. Nothing
# here touches the GPU, the pack or the pair's real port.
#
#   CX_CT=<container> GX=<ssh-host> bash qmesh/tests/refusal_stub.sh
#   -> REFUSAL PASS server_table=<sha> client_table=<sha> port=47411
#
# Exit: 0 refused as designed and the server survived, 1 the client accepted a foreign table,
#       2 the stub never came up, 3 the server did not survive the refusal, 4 both sides resolved
#       the same file (which proves nothing).
set -uo pipefail
CT=${CX_CT:?set CX_CT to the server container}
GX=${GX:?set GX to the server SSH host}
PORT=${CX_REFUSAL_PORT:-47411}
SRV=${CX_SERVER_PLACEMENT:-/cold/placement.exl3.ranked.json}   # container path
CLI=${CX_CLIENT_PLACEMENT:-/cold/placement.exl3.json}           # container path
DEV=${CX_GX_DEV:-rocep1s0f0}
GID=${CX_GX_GID:-3}

say() { echo "REFUSAL $*"; }

out=$(timeout 300 ssh -o BatchMode=yes "$GX" "docker exec \
  -e SRV='$SRV' -e CLI='$CLI' -e PORT='$PORT' -e DEV='$DEV' -e GID='$GID' \
  $CT bash -lc '\
cd /src/exllamav3 || exit 9
for f in \"\$SRV\" \"\$CLI\"; do [ -f \"\$f\" ] || { echo \"STUB missing_table \$f\"; exit 2; }; done
echo \"STUB tables server=\$(sha256sum \"\$SRV\" | cut -c1-12) client=\$(sha256sum \"\$CLI\" | cut -c1-12)\"
python3 -u qmesh/worker_seam_server.py --stub --model-dir /cold --placement \"\$SRV\" \
  --dev \"\$DEV\" --gid \"\$GID\" --port \"\$PORT\" --buckets 1 --slab-slots 1 \
  > /tmp/seam_stub.log 2>&1 &
pid=\$!
up=0
for i in \$(seq 120); do
  grep -q \"SEAM_SERVER_SLAB\" /tmp/seam_stub.log && { up=1; break; }
  kill -0 \$pid 2>/dev/null || break
  sleep 1
done
echo \"STUB up=\$up\"
if [ \"\$up\" = 0 ]; then
  tail -5 /tmp/seam_stub.log
  kill \$pid 2>/dev/null
  exit 2
fi
grep -a \"^SEAM_LISTEN \" /tmp/seam_stub.log | tail -1 | sed \"s/^/STUB hello /\"
PLACEMENT=\"\$CLI\" SEAM_HOST=127.0.0.1 SEAM_PORT=\"\$PORT\" \
  python3 -u qmesh/tests/placement_sha_refusal.py
echo \"STUB client_rc=\$?\"
kill -0 \$pid 2>/dev/null && echo \"STUB alive_after=1\" || echo \"STUB alive_after=0\"
if grep -aq \"Traceback\" /tmp/seam_stub.log; then echo \"STUB traceback=1\"; grep -a -A3 Traceback /tmp/seam_stub.log | tail -6; fi
kill \$pid 2>/dev/null; wait \$pid 2>/dev/null
exit 0'" 2>&1)
echo "$out" | sed 's/^/REFUSAL /'

okc=$(printf '%s\n' "$out" | grep -c '^PLACEMENT_REFUSAL ok=1 kind=mismatch')
same=$(printf '%s\n' "$out" | grep -c 'kind=same_table')
alive=$(printf '%s\n' "$out" | grep -c '^STUB alive_after=1')
up=$(printf '%s\n' "$out" | grep -c '^STUB up=1')
rc=$(printf '%s\n' "$out" | grep -o '^STUB client_rc=[0-9]*' | head -1 | cut -d= -f2)
stubtb=$(printf '%s\n' "$out" | grep -c '^STUB traceback=1')
[ "$same" = 0 ] || { say "FAIL the two sides resolved the same file; this proves nothing"; exit 4; }
[ "$up" = 1 ] || { say "FAIL the stub server never came up"; exit 2; }
[ "$stubtb" = 0 ] || { say "FAIL the stub server logged a traceback"; exit 3; }
[ "$alive" = 1 ] || { say "FAIL the server did not survive the refusal"; exit 3; }
[ "$okc" = 1 ] || { say "FAIL the client did not refuse the foreign table"; exit 1; }
[ "${rc:-1}" = 0 ] || { say "FAIL the refusal client exit=$rc"; exit 1; }
say "PASS $(printf '%s\n' "$out" | grep -o '^STUB tables server=[0-9a-f]* client=[0-9a-f]*' | sed 's/^STUB tables //')" \
    "hello=$(printf '%s\n' "$out" | grep -o 'placement_sha=[0-9a-f]*' | head -1 | cut -d= -f2)" \
    "port=$PORT refused_before_first_request=1 server_alive=1"
