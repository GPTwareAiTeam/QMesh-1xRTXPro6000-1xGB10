#!/bin/bash
# arm_ids_stub.sh -- pair_arm_ids.py captures ids from the release's own driver without a GPU.
#
#   ./tests/arm_ids_stub.sh
#
# The ids are the promotion gate's proof that a decode ran, so the wrapper that collects them has to
# be right without a model in the loop: this feeds it a fake qmesh.e2e_prompts whose generator yields
# both shapes the engine has used for `token_ids` (a tensor-like object and a plain list), and checks
# that the rows and the exit code pass through untouched while ARM_IDS and $ARM_IDS_OUT appear.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
export PAIR_CONF="$HERE/pairs.conf" PAIR_COLD_PACK=/home/test/cold-pack PAIR_ART_MOUNT=/home/test/artifacts
WORK=$(mktemp -d /tmp/pairids.XXXXXX)
mkdir -p "$WORK/pkg/qmesh"; : > "$WORK/pkg/qmesh/__init__.py"
cat > "$WORK/pkg/qmesh/e2e_prompts.py" <<'PY'
import os, sys


class T:                       # stands in for a tensor: token_ids arrives as one from the engine
    def __init__(self, v): self.v = v
    def reshape(self, *a): return self
    def tolist(self): return list(self.v)


class G:
    def __init__(self):
        self.num_draft_tokens = 5
        self._jobs = 1
    def num_remaining_jobs(self): return self._jobs
    def iterate(self):
        self._jobs = 0
        yield {"stage": "started"}
        yield {"text": "hel", "token_ids": T([11, 12, 13])}
        yield {"text": "lo", "token_ids": [14]}
        yield {"eos": True, "text": "", "new_tokens": 4}


def run_arm(spec, generator, draft_model, tokenizer, prompts, long_ids, fh):
    generator.num_draft_tokens = 5                       # a real arm writes to the generator
    while generator.num_remaining_jobs():
        for _r in generator.iterate():
            pass
    print("ROW " + spec + " text=hello")
    return {"arm": spec}


def main():
    run_arm("dspark1", G(), None, None, [], None, None)
    if os.environ.get("FAKE_EXIT"):
        sys.exit(int(os.environ["FAKE_EXIT"]))
PY
fails=0
run() {  # <label> <expect-rc> <env...>
  local label=$1 want_rc=$2; shift 2
  local out rc
  out=$(env PYTHONPATH="$WORK/pkg" ARM_IDS_OUT="$WORK/ids.json" "$@" python3 -u - < "$REL/pair_arm_ids.py" 2>&1)
  rc=$?
  if [ "$rc" = "$want_rc" ]; then echo "OK   $label rc=$rc"
  else echo "FAIL $label rc=$rc want=$want_rc"; printf '%s\n' "$out" | tail -4 | sed 's/^/     /'; fails=$((fails + 1)); fi
  if printf '%s' "$out" | grep -q "ARM_IDS arm=dspark1 n=4 sha256=[0-9a-f]\{64\} first=11,12,13,14"; then
    echo "OK   $label ids line"
  else
    echo "FAIL $label no ARM_IDS line"; printf '%s\n' "$out" | tail -4 | sed 's/^/     /'; fails=$((fails + 1))
  fi
  if printf '%s' "$out" | grep -q "^ROW dspark1 text=hello"; then :; else
    echo "FAIL $label driver output was altered"; fails=$((fails + 1)); fi
  if pythonjson=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["n"], d["arms"]["dspark1"])' "$WORK/ids.json" 2>&1); then
    [ "$pythonjson" = "4 [11, 12, 13, 14]" ] && echo "OK   $label ids file" \
      || { echo "FAIL $label ids file: $pythonjson"; fails=$((fails + 1)); }
  else
    echo "FAIL $label ids file unreadable: $pythonjson"; fails=$((fails + 1))
  fi
  rm -f "$WORK/ids.json"
}
run "passing arm" 0
run "arm that exits nonzero" 3 FAKE_EXIT=3

# The client starts its CPU-host worker with multiprocessing "spawn", which re-executes this file in
# the child under __mp_main__. Nothing may run there: a second main() would load the model twice,
# and a child that cannot find its main file at all is what turned tonight's smoke into EOFError.
out=$(env PYTHONPATH="$WORK/pkg" ARM_IDS_OUT="$WORK/child.json" python3 -c \
        'import runpy,sys; runpy.run_path(sys.argv[1], run_name="__mp_main__")' \
        "$REL/pair_arm_ids.py" 2>&1); rc=$?
[ "$rc" = 0 ] && echo "OK   re-executed as __mp_main__ exits 0" \
  || { echo "FAIL re-executed as __mp_main__ rc=$rc"; printf '%s\n' "$out" | tail -4 | sed 's/^/     /'; fails=$((fails + 1)); }
if printf '%s' "$out" | grep -q "^ARM_IDS \|^ROW "; then
  echo "FAIL the spawn child ran the arm again"; fails=$((fails + 1))
else echo "OK   the spawn child runs no arm"; fi
[ -e "$WORK/child.json" ] && { echo "FAIL the spawn child wrote the ids file"; fails=$((fails + 1)); } \
  || echo "OK   the spawn child writes nothing"

# ... and because the child re-executes *by path*, the arm has to be launched by path too: `python3 -`
# makes the child's main file "<stdin>", which does not exist.
out=$(env PYTHONPATH="$WORK/pkg" ARM_IDS_OUT="$WORK/path.json" python3 -u "$REL/pair_arm_ids.py" 2>&1); rc=$?
[ "$rc" = 0 ] && echo "OK   run by path exits 0" \
  || { echo "FAIL run by path rc=$rc"; printf '%s\n' "$out" | tail -4 | sed 's/^/     /'; fails=$((fails + 1)); }
if printf '%s' "$out" | grep -q "^ARM_IDS arm=dspark1 n=4 "; then echo "OK   run by path records ids"
else echo "FAIL run by path records no ids"; fails=$((fails + 1)); fi

# pair_arm.sh --ids must hand the container a path, and the artifacts mount, not a host path and not
# a pipe.
dry=$(PAIR_DRY=1 PAIR_ROOT_OVERRIDE=/tmp/pairdry/pair PAIR_STATE_OVERRIDE=/tmp/pairdry/pair/state \
      "$REL/pair_arm.sh" dev --ids 2>&1)
if printf '%s' "$dry" | grep -q "python3 -u /pair/state/dev/arm/pair_arm_ids.py"; then
  echo "OK   the driver is run by path inside the pair root"
else
  echo "FAIL dry arm does not run the driver by path"; printf '%s\n' "$dry" | sed 's/^/     /'; fails=$((fails + 1))
fi
if printf '%s' "$dry" | grep -q "python3 -u -"; then
  echo "FAIL the driver is still fed on stdin"; fails=$((fails + 1))
else echo "OK   no stdin driver"; fi

echo
if [ "$fails" = 0 ]; then echo "RESULT PASS fails=0"; else echo "RESULT FAIL fails=$fails"; fi
exit $((fails != 0))
