#!/usr/bin/env python3
"""A client on another cold-expert table must be refused at the Hello, and the server must survive it.

The seam's per-layer cold-local ids are positions in a placement file. When the two halves of a pair
read different files -- a ranked table on the server, an unranked one on the client (or the reverse)
-- every id past the first differing group means a different expert, and the transport has no way to
tell: the ids are in range, so the server would answer from the wrong weights. The Hello carries the
sha of the server's file and the client compares it with the sha of the file it loaded before the
first request is sent (RemoteSeamHost._check_placement).

This test drives that path against a real server process and checks three things:

  1. the shas really differ (a test where they match proves nothing),
  2. the shipping check raises, with the mismatch named, instead of the client going on to send,
  3. the server is still accepting sessions afterwards -- a refusal is not a crash.

It needs a running server and no pack: run it against `worker_seam_server.py --stub`, one free port.
The refusal test is the one place a *deliberately* wrong table is loaded, so it must never point at
the pair's real port.

  PLACEMENT=<other placement> SEAM_PORT=47399 python3 -u qmesh/tests/placement_sha_refusal.py
  # expect: PLACEMENT_REFUSAL ok=1 ...  and exit 0
"""
import os
import sys

from qmesh.native import qes
from qmesh import placement as seam_placement
from qmesh.remote_seam_host import RemoteSeamHost


def open_control(host, port):
    return qes.Control(host, port, buckets=(1,))


def main():
    host = os.environ.get("SEAM_HOST", "127.0.0.1")
    port = int(os.environ.get("SEAM_PORT", "47399"))
    # Load the client's table the way a real client does (PLACEMENT first, then the pack manifest):
    # current_sha() is only set by load(), and the shipping check compares against it.
    _placement, path, mine = seam_placement.load()
    mine = (mine or "").strip()

    ctl = open_control(host, port)
    srv = (getattr(ctl, "placement_sha", "") or "").strip()
    print(f"PLACEMENT_REFUSAL hello server_sha={srv[:16] or 'unset'} "
          f"server_build={ctl.build} buckets={ctl.buckets}", flush=True)
    if not srv or not mine:
        print("PLACEMENT_REFUSAL ok=0 kind=unset_sha: one side did not advertise a table at all, "
              "so this run cannot show a refusal", flush=True)
        return 2
    if srv.startswith(mine[:16]) or mine.startswith(srv[:16]):
        print("PLACEMENT_REFUSAL ok=0 kind=same_table: both sides resolved the same file, so this "
              "run does not exercise the refusal (point PLACEMENT at the other table)", flush=True)
        return 2

    try:
        RemoteSeamHost._check_placement(ctl)
    except RuntimeError as e:
        text = str(e).splitlines()[0]
        print(f"PLACEMENT_REFUSAL ok=1 kind=mismatch refused_before_first_request=1 "
              f"server={srv[:16]} client={mine[:16]}", flush=True)
        print(f"PLACEMENT_REFUSAL message={text[:300]}", flush=True)
    else:
        print("PLACEMENT_REFUSAL ok=0 kind=not_refused: the sha differs and the client accepted the "
              "session anyway; every cold-local id would mean another expert", flush=True)
        ctl.close()
        return 1
    ctl.close()

    # Same server, a fresh session: a refused peer must cost the session, not the process.
    ctl2 = open_control(host, port)
    again = (getattr(ctl2, "placement_sha", "") or "").strip()
    ctl2.close()
    print(f"PLACEMENT_REFUSAL server_alive={int(bool(again))} same_sha={int(again[:16] == srv[:16])}",
          flush=True)
    return 0 if again else 3


if __name__ == "__main__":
    sys.exit(main())
