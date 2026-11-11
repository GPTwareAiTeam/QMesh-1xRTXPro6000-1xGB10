#!/usr/bin/env python3
"""Record the token ids a decode arm produced, plus the margin that produced each of them.

Same as `pair_arm_ids.py` -- same `ARM_IDS` line, same `$ARM_IDS_OUT` json, so a run recorded with
this file compares against one recorded without it -- with one addition: when `ARM_IDS_GAP=1`,
every generator result that carries logits is annotated with the top-2 logits behind the chosen
token:

    ARM_GAP i=7 chosen=24 top1_id=24 top2_id=736 gap=0.00042

Written for the 11:07 question: two --fresh loads of the same table produced ids that agree for six
tokens and then differ at position seven (736 vs 24), with the same disagreement appearing on both
the plain and the ranked table. A digest differs for one of two reasons -- the model computed
something else, or the model computed the same thing and the argmax sat on a tie -- and the gap is
what tells them apart. A gap of ~1e-4 makes the flip a tie-break, and any reordering that leaves
the logits where they were is then proven transparent; a gap of 0.5 would make it a real change and
the table the prime suspect.
"""
import hashlib
import json
import os
import sys

import qmesh.e2e_prompts as E

CAPTURED = {}
GAP = os.environ.get("ARM_IDS_GAP") == "1"
_GAP_N = {}


def _ids_of(result):
    ids = result.get("token_ids")
    if ids is None:
        return None
    try:
        if hasattr(ids, "reshape"):
            ids = ids.reshape(-1).tolist()
        else:
            ids = [int(x) for x in ids]
    except Exception:                                                   # noqa: BLE001
        return None
    return [int(x) for x in ids]


def _gap_of(result, spec, n):
    """Top-2 logits for the token just produced, if the result carries them."""
    logits = result.get("logits")
    if logits is None:
        return
    try:
        import torch
        row = logits[0, 0] if getattr(logits, "dim", lambda: 0)() == 3 else logits.reshape(-1)
        top = torch.topk(row.float(), 2)
        chosen = _ids_of(result)
        print(f"ARM_GAP spec={spec} i={n} chosen={chosen[0] if chosen else -1} "
              f"top1_id={int(top.indices[0])} top2_id={int(top.indices[1])} "
              f"top1={float(top.values[0]):.5f} top2={float(top.values[1]):.5f} "
              f"gap={float(top.values[0] - top.values[1]):.6f}", flush=True)
    except Exception as exc:                                            # noqa: BLE001
        print(f"ARM_GAP_NOTE spec={spec} i={n} cannot read logits: {exc}", file=sys.stderr, flush=True)


class _Proxy:
    """Forwards everything to the real generator, except iterate(): each result is inspected."""

    __slots__ = ("_g", "_spec")

    def __init__(self, gen, spec):
        object.__setattr__(self, "_g", gen)
        object.__setattr__(self, "_spec", spec)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_g"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_g"), name, value)

    def iterate(self, *a, **k):
        spec = object.__getattribute__(self, "_spec")
        got = CAPTURED.setdefault(spec, [])
        for result in object.__getattribute__(self, "_g").iterate(*a, **k):
            if isinstance(result, dict):
                ids = _ids_of(result)
                if ids:
                    if GAP:
                        _gap_of(result, spec, len(got))
                    got.extend(ids)
            yield result


_orig_run_arm = E.run_arm


def run_arm(spec, generator, *a, **k):
    CAPTURED.setdefault(spec, [])
    out = _orig_run_arm(spec, _Proxy(generator, spec), *a, **k)
    got = CAPTURED.get(spec) or []
    print(f"ARM_IDS arm={spec} n={len(got)} sha256={hashlib.sha256(json.dumps(got).encode()).hexdigest()} "
          f"first={','.join(str(i) for i in got[:8])}", flush=True)
    return out


E.run_arm = run_arm


def main():
    rc = 0
    try:
        E.main()
    except SystemExit as exc:
        rc = int(exc.code or 0)
    finally:
        out_path = os.environ.get("ARM_IDS_OUT")
        if out_path:
            try:
                with open(out_path, "w") as fh:
                    json.dump({"format": 1, "arms": {k: v for k, v in CAPTURED.items()},
                               "n": sum(len(v) for v in CAPTURED.values())}, fh, sort_keys=True)
                    fh.write("\n")
            except OSError as exc:
                print(f"ARM_IDS_NOTE cannot write {out_path}: {exc}", file=sys.stderr, flush=True)
    return rc


# The client's CPU-host module starts its worker with multiprocessing "spawn", which re-executes
# this file under __mp_main__ in the child. The wrapper therefore does nothing at module level that
# the child must not repeat: main() only runs in the parent.
if __name__ == "__main__":
    sys.exit(main())
