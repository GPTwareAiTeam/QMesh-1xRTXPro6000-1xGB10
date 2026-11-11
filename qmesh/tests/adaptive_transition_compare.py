"""Compare two adaptive_batch_gate JSON reports (control and forced transitions).

Reports each arm's singleton-vs-batch differences and cross-arm singleton
and batch differences. Nonempty HTTP-200 responses are mandatory. This only
compares tokens; log coverage and resource safety require separate gates.
Run: python qmesh/tests/adaptive_transition_compare.py CONTROL.json FORCED.json
"""
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    if report["width"] != 8:
        raise ValueError(f"{path}: expected eight clients")
    for group in ("same", "different", "different_single"):
        if len(report[group]) != 8:
            raise ValueError(f"{path}: incomplete {group}")
    for item in [report["same_single"], *report["same"], *report["different"], *report["different_single"]]:
        if item["status"] != 200 or len(tokens(item)) < 16:
            raise ValueError(f"{path}: unsuccessful or too-short response")
    return report


def tokens(item):
    return item["result"]["choices"][0]["token_ids"]


def first_diff(a, b):
    a, b = tokens(a), tokens(b)
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)) if len(a) != len(b) else None)


def compare(label, left, right):
    diffs = [(i, n) for i, (a, b) in enumerate(zip(left, right))
             if (n := first_diff(a, b)) is not None]
    print(f"{label}: {len(diffs)}/{len(left)} differ; first indices={diffs}")


control, forced = map(load, sys.argv[1:3])
if (control["same_prompt"], control["different_prompts"]) != (forced["same_prompt"], forced["different_prompts"]):
    raise SystemExit("prompt mismatch")
for label, report in (("control", control), ("forced", forced)):
    compare(label + " same vs singleton", report["same"], [report["same_single"]] * 8)
    compare(label + " mixed vs singleton", report["different"], report["different_single"])
compare("cross-arm same singleton", [control["same_single"]], [forced["same_single"]])
compare("cross-arm mixed singletons", control["different_single"], forced["different_single"])
compare("cross-arm same batch", control["same"], forced["same"])
compare("cross-arm mixed batch", control["different"], forced["different"])
