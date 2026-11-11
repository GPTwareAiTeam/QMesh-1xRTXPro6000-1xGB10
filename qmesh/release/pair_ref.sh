#!/usr/bin/env bash
# pair_ref.sh -- record (or compare against) the decode text a release produces.
#
#   pair_ref.sh <pair> [--rows N] [--max-new N] [--sha S]      # write <state>/refs-<sha>.json
#   pair_ref.sh <pair> --compare <refs.json>                   # re-run and compare
#
# Why the text and not token ids: this release's qmesh/e2e_prompts.py has no DUMP_IDS/TEXT lines, so
# the row's `text` is what is observable. It is a *tripwire*, not a numerics test: the same release on
# the same pair must produce the same text; a seam protocol change legitimately changes it, which is
# why the ref records the seam fingerprint and a compare across a different fingerprint is reported
# as stale instead of as a failure (that is exactly release 2's case).
#
# Exit: 0 equal (or written), 1 different, 3 stale (seam fingerprint changed, re-baseline), 2 error.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_ref.sh <pair> [--rows N] [--compare F]}
shift
ROWS=3; MAX_NEW=8; SHA=""; COMPARE=""; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rows) ROWS=$2; shift ;;
    --max-new) MAX_NEW=$2; shift ;;
    --sha) SHA=$2; shift ;;
    --compare) COMPARE=$2; shift ;;
    --out) OUT=$2; shift ;;
    *) pair_die "unknown flag $1" ;;
  esac
  shift
done
pair_load "$PAIR_ARG"
# RM10 2026-11-05 (Main): the P7 client half is named, never derived -- `pair_ckpt_resolve auto` is a
# cross-check now (a second exl3-p7hot-* dir declaring a live pack's sha voided three runs at 08:32).
PAIR_PACK_P7=$(pair_ckpt_require) || exit 1
export PAIR_PACK_P7
SHA=${SHA:-$(pair_current_sha "$PAIR_ROOT")}
[ -n "$SHA" ] || pair_die "pair $PAIR has no release"
REL=$(pair_release_dir "$SHA")
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json" 2>/dev/null)
SEAM=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["seam"]["fingerprint"])' "$REL/RELEASE.json" 2>/dev/null)
[ -n "$SEAM" ] || pair_die "release ${SHA:0:8} has no seam fingerprint"

arm_out=$("$HERE/pair_arm.sh" "$PAIR" --rows "$ROWS" --max-new "$MAX_NEW" 2>&1)
arm_rc=$?
printf '%s\n' "$arm_out" | sed -n 's/^{/ROW /p' | cut -c1-220 | head -"$ROWS"
[ "$arm_rc" = 0 ] || { echo "REF_FAIL arm rc=$arm_rc $(printf '%s' "$arm_out" | tail -2 | tr '\n' ' ')"; exit 2; }

ref=$OUT
if [ -z "$ref" ]; then
  case "$COMPARE" in
    "") ref=$PAIR_STATE_DIR/refs-$SHA.json ;;
    *) ref=$(mktemp) ;;
  esac
fi
# Data by path: `python3 - <<'PY'` already owns stdin with the script, so a piped arm_out read as empty
# here and every reference was written with rows=[] -- a reference that proves nothing (same fault as
# the arm-row check in pair_smoke.sh, found 2026-11-02).
ref_rows=$(mktemp "${TMPDIR:-/tmp}/pairref.XXXXXX")
printf '%s\n' "$arm_out" > "$ref_rows"
python3 - "$ref_rows" "$ref" "$SHA" "$BUILD_ID" "$SEAM" <<'PY'
import datetime, hashlib, json, sys
rows_path, out, sha, build_id, seam = sys.argv[1:6]
rows = []
for line in open(rows_path, errors="replace"):
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if "text" in d:
        rows.append({"i": d.get("i", len(rows)), "text_len": len(d["text"]),
                     "text_sha256": hashlib.sha256(d["text"].encode()).hexdigest()})
rows.sort(key=lambda r: r["i"])
doc = {"format": 1, "release_sha": sha, "build_id": build_id, "seam": seam, "rows": rows,
       "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
with open(out, "w") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"REF {doc['rows'][0]['i'] if rows else '-'} rows={len(rows)} sha={sha[:8]} seam={seam} file={out}")
PY
rm -f "$ref_rows"
# An empty reference is not a reference: promotion compares rows against it, and rows=[] would match a
# release that decoded nothing. Fail here, where the arm's own log is still around to say why.
[ -s "$ref" ] && grep -q '"rows": \[' "$ref" && ! grep -q '"rows": \[\]' "$ref" \
  || { echo "REF_FAIL no rows in the arm output (see the arm log for ARM_JOB_ERROR)"; exit 2; }

if [ -z "$COMPARE" ]; then
  exit 0
fi
python3 - "$COMPARE" "$ref" <<'PY'
import json, sys
want = json.load(open(sys.argv[1]))
got = json.load(open(sys.argv[2]))
if want.get("seam") != got.get("seam"):
    print(f"REF_STALE seam {got.get('seam')} != baseline {want.get('seam')} (re-baseline)")
    sys.exit(3)
if want.get("release_sha") != got.get("release_sha"):
    print(f"REF_STALE release {got.get('release_sha', '')[:12]} != baseline {want.get('release_sha', '')[:12]}")
    sys.exit(3)
w = {r["i"]: r["text_sha256"] for r in want.get("rows", [])}
g = {r["i"]: r["text_sha256"] for r in got.get("rows", [])}
diff = [i for i in sorted(set(w) & set(g)) if w[i] != g[i]]
missing = sorted(set(w) - set(g))
for i in diff:
    print(f"REF_DIFF i={i} now={g[i][:16]} baseline={w[i][:16]}")
for i in missing:
    print(f"REF_MISSING i={i}")
print(f"REF {'OK' if not diff and not missing else 'DIFF'} rows={len(g)} compared={len(set(w) & set(g))}")
sys.exit(0 if not diff and not missing else 1)
PY
