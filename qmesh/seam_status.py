#!/usr/bin/env python3
"""Query a running seam server without disturbing its session.

  python3 qmesh/seam_status.py <host> [port] [--require-idle]

Prints one line:
  SEAM_STATUS host=… port=… state=idle|busy version=… build=… placement_sha=… layers=… buckets=… message=…
Exit 0 when the server answered (and was idle, with --require-idle), 1 when it did not answer
or was busy under --require-idle.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmesh.native import qes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("port", nargs = "?", type = int, default = 47311)
    ap.add_argument("--require-idle", action = "store_true",
                    help = "exit 1 when the server is serving a session")
    args = ap.parse_args()

    try:
        n = qes.status(args.host, args.port)
    except RuntimeError as e:
        print(f"SEAM_STATUS host={args.host} port={args.port} state=down error={e}", flush = True)
        return 1
    state = "idle" if n.text == "idle" else "busy"
    print(f"SEAM_STATUS host={args.host} port={args.port} state={state} version={n.version} "
          f"build={n.build_id} placement_sha={n.placement} layers={n.layers} "
          f"buckets={','.join(map(str, n.bucket_list))} slots={n.slots} message={n.text}", flush = True)
    if args.require_idle and state != "idle":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
