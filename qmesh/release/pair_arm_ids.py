#!/usr/bin/env python3
"""Record the token ids a decode arm produced, on top of the release's own e2e_prompts.py.

Fed to `python3 -u -` inside the P7 container by `pair_arm.sh --ids`, from `/src/exllamav3`, with
the same environment a plain arm gets. It imports the release's driver unchanged and wraps
`run_arm` so the generator's results pass through a proxy that copies `token_ids` out; the rows,
the SUMMARY line and everything else are exactly what the unwrapped driver prints, so a
reference row recorded with `--ids` still compares against one recorded without it.

Why ids and not the decoded text: the text is the tripwire for "same release, same answer", but a
promotion has to prove that a *decode happened*, with something a wrong build cannot fake by
producing fluent-looking prose. The ids are printed as `ARM_IDS arm=<spec> n=<count>
sha256=<hash of the id list> first=<first 8 ids>`, and written to $ARM_IDS_OUT when set.
"""
import hashlib
import json
import os
import sys

import qmesh.e2e_prompts as E

CAPTURED = {}


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
        got = CAPTURED.setdefault(object.__getattribute__(self, "_spec"), [])
        for result in object.__getattribute__(self, "_g").iterate(*a, **k):
            if isinstance(result, dict):
                ids = _ids_of(result)
                if ids:
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
