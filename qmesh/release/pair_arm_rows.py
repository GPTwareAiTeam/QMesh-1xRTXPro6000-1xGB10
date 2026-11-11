#!/usr/bin/env python3
"""The smoke's reading of an arm log: one summary line per run of rows, and the new-token total.

Why a file of its own rather than an inline heredoc in pair_smoke.sh: the first version was
`printf '%s\\n' "$arm_out" | python3 - <<'PY'`, and `python3 -` takes its *program* from stdin -- the heredoc
was consumed as source, the parser saw an empty stream, and the dev smoke of 9fc3ca8a printed
"arm_no_tokens NONE" for an arm whose log held a complete row with eight ids. Inline python that reads
stdin cannot be piped into, and it cannot be tested; this can (tests/arm_rows.sh).

Input: a path to the arm's log. Output, on stdout:
  i=<first index> rows=<count> new_tokens=<sum> ttft_ms=.. decode_ms_per_tok=.. text_sha256=<hash>
  NEW <sum>
or the single line NONE when the log holds no row. The hash covers "i|text" for every row, so a release
that answers differently on the same prompt is visible in one number.
"""
import hashlib
import json
import sys


def main(argv):
    if len(argv) != 2:
        print("usage: pair_arm_rows.py <arm log>", file=sys.stderr)
        return 2
    rows = []
    try:
        fh = open(argv[1], errors="replace")
    except OSError:
        print("NONE")
        return 0
    with fh:
        for line in fh:
            line = line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "text" in d:
                rows.append(d)
    if not rows:
        print("NONE")
        return 0
    rows.sort(key=lambda d: d.get("i", 0))
    blob = "\n".join(f"{d.get('i', 0)}|{d['text']}" for d in rows)
    new = sum(int(d.get("new_tokens") or 0) for d in rows)
    # accept_rate/tokens_per_round are diagnostics, not gates: a single 8-token decode has too few
    # rounds to judge a drafter, but a run whose acceptance is 0 while the drafter is on is visible in
    # the ticket instead of only in the arm's log (the 23:32 dev arm: 0 of 15, 1.14 tokens/round).
    print(f"i={rows[0].get('i', 0)} rows={len(rows)} new_tokens={new} "
          f"ttft_ms={rows[0].get('ttft_ms', '?')} decode_ms_per_tok={rows[0].get('decode_ms_per_tok', '?')} "
          f"accept_rate={rows[0].get('accept_rate', '?')} tokens_per_round={rows[0].get('tokens_per_round', '?')} "
          f"text_sha256={hashlib.sha256(blob.encode()).hexdigest()}")
    print(f"NEW {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
