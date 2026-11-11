#!/usr/bin/env python3
"""pair_bucket_grant.py -- did the seam grant the row buckets the pair ships?

  pair_bucket_grant.py <accepted> <want>

The seam client adopts the *intersection* of the buckets it requests and the buckets the server was
started with, and reports it as `accepted=...` in SEAM_NEGOTIATED. The pair's numbers were taken on
one shape: 1,6,64,8192 from pairs.conf. A client that asks for less (the release's own fallback
default listed 1,6,64,256,512,1024,2048,4096 -- no 8192) is granted 1,6,64, and every 8192-row
prefill chunk then costs 128 calls of 64 rows; that is what the 23:32 dev arm measured (1.6 s/token,
0 of 15 draft tokens accepted) while the server itself was healthy.

Prints one line: `OK accepted=<list> want=<list>` when the grant covers the pair's list, or
`MISS accepted=<list> want=<list> missing=<buckets>`. Exit 0 on OK, 1 on MISS, 2 when the accepted
list could not be read (a client that never negotiated, so the caller decides -- not a grant fault).
"""
import sys


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[0])
        return 2
    accepted, want = argv[1].strip(), argv[2].strip()
    try:
        got = {int(b) for b in accepted.split(",") if b.strip()}
        need = [int(b) for b in want.split(",") if b.strip()]
    except ValueError:
        print(f"MISS accepted={accepted or 'unknown'} want={want} unparsable")
        return 1
    if not got:
        print(f"MISS accepted=unknown want={want}")
        return 2
    missing = [b for b in need if b not in got]
    if missing:
        print(f"MISS accepted={','.join(map(str, sorted(got)))} want={want} "
              f"missing={','.join(map(str, missing))}")
        return 1
    print(f"OK accepted={','.join(map(str, sorted(got)))} want={want}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
