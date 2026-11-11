"""Check a c895ed1a DSpark transition server trace after SERVER_READY.

Requires a consecutive speculative/target-only/speculative sequence at active
width >=2 and validates every observed per-job state increment. Numerical token
identity and memory headroom are separate gates.
Run: python qmesh/tests/adaptive_dspark_trace_gate.py SERVER_LOG
"""
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
if "SERVER_READY" not in text:
    raise SystemExit("server never became ready")
text = text.split("SERVER_READY", 1)[1]
transition = re.compile(r"SERVE_DIAG_DRAFT_TRANSITION step=(\d+) window=(\d+)")
width = re.compile(r"SERVE_DECODE_WIDTH active_rows=(\d+) draft=(\d+) verify_rows=(\d+)")
state = re.compile(r"SERVE_DIAG_DRAFT_STATE serial=(\d+) position=(\d+) accepted=(\d+)")
rounds = []
pending = None
positions = {}
per_job = {}
current = None
state_count = 0
for line in text.splitlines():
    if match := transition.search(line):
        pending = tuple(map(int, match.groups()))
    elif match := width.search(line):
        active, draft, verify = map(int, match.groups())
        if pending is not None:
            if verify != active * (1 + draft):
                raise SystemExit("inconsistent verification row count")
            rounds.append((*pending, active, draft))
            current = rounds[-1]
            pending = None
    elif match := state.search(line):
        serial, pos, accepted = map(int, match.groups())
        if serial in positions and pos != positions[serial] + accepted:
            raise SystemExit(f"serial {serial}: position {positions[serial]} -> {pos}, accepted {accepted}")
        positions[serial] = pos
        if current is not None:
            per_job.setdefault(serial, []).append(current)
        state_count += 1
for serial, entries in per_job.items():
    for a, b, c in zip(entries, entries[1:], entries[2:]):
        if (all(r[2] >= 2 for r in (a, b, c))
                and a[3] > 0 and b[3] == 0 and c[3] > 0):
            print(f"PASS: serial {serial} draft/target-only/draft steps {a[0]},{b[0]},{c[0]}, "
                  f"active={a[2]},{b[2]},{c[2]}, state observations={state_count}")
            raise SystemExit(0)
raise SystemExit("no per-job multi-row draft/target-only/draft sequence")
