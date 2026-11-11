"""Opt-in per-round anatomy probe (EXL3_ROUND_PROF=1).

Records, for every speculative decode round, a mark per phase boundary: a perf_counter_ns timestamp
(host) and a CUDA event on the current stream (device). Consecutive marks give the host wall of the
segment between them and the GPU stream time of the same segment, so a segment where host >> GPU is
host/sync bound and one where GPU > host is device bound. Nothing is synchronized while generating;
the events are resolved once, on dump.

Counters (note) annotate a segment with host-visible facts (sync sites, collective calls).
Disabled by default: mark/note are no-ops, so instrumented and clean runs are one binary.
"""
import os
import time
import atexit
import statistics

ON = os.environ.get("EXL3_ROUND_PROF", "0") == "1"

if not ON:

    def new_round():
        pass

    def mark(name):
        pass

    def note(name, value = 1):
        pass

    def dump():
        pass

else:
    import torch

    _marks = []        # (round, name, host_ns, event)
    _notes = []        # (round, name, value)
    _round = 0

    def new_round():
        global _round
        _round += 1
        return _round

    def mark(name):
        ev = torch.cuda.Event(enable_timing = True)
        ev.record()
        _marks.append((_round, name, time.perf_counter_ns(), ev))

    def note(name, value = 1):
        _notes.append((_round, name, value))

    def dump():
        if not _marks:
            print("[RPROF] no marks", flush = True)
            return
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        # Decode rounds only: a round that never entered the draft dispatch is a prefill chunk (or a
        # teardown pass) and would otherwise dominate the counters
        rounds = sorted({m[0] for m in _marks if m[1] == "d0"})
        if not rounds:
            print("[RPROF] no decode rounds", flush = True)
            return
        # Per round, per segment: host delta and GPU delta between consecutive marks
        per_seg = {}
        totals = []   # first-to-last mark within each instrumented round
        for r in rounds:
            rs = sorted([m for m in _marks if m[0] == r], key = lambda m: m[2])
            if len(rs) > 1:
                totals.append((rs[-1][2] - rs[0][2]) / 1e6)
            for (_, n0, h0, e0), (_, n1, h1, e1) in zip(rs, rs[1:]):
                key = f"{n0}->{n1}"
                per_seg.setdefault(key, []).append(((h1 - h0) / 1e6, e0.elapsed_time(e1)))
        print(f"[RPROF] rounds={len(rounds)} segments={len(per_seg)}", flush = True)
        print(f"[RPROF] {'segment':38s} {'n':>4s} {'host_ms':>8s} {'gpu_ms':>8s} {'gap_ms':>8s}", flush = True)
        for key, vals in per_seg.items():
            hs = statistics.median(v[0] for v in vals)
            gs = statistics.median(v[1] for v in vals)
            print(f"[RPROF] {key:38s} {len(vals):4d} {hs:8.3f} {gs:8.3f} {hs - gs:8.3f}", flush = True)
        if totals:
            print(f"[RPROF] round_total_ms median={statistics.median(totals):.3f} "
                  f"min={min(totals):.3f} max={max(totals):.3f} n={len(totals)}", flush = True)
            # Slow rounds (> 2x median): where their time went, to separate stalls from steady per-round cost
            med = statistics.median(totals)
            slow = 0
            for r in rounds:
                rs = sorted([m for m in _marks if m[0] == r], key = lambda m: m[2])
                if len(rs) < 2 or (rs[-1][2] - rs[0][2]) / 1e6 <= 2 * med:
                    continue
                slow += 1
                segs = [(f"{a[1]}->{b[1]}", (b[2] - a[2]) / 1e6, a[3].elapsed_time(b[3])) for a, b in zip(rs, rs[1:])]
                worst = max(segs, key = lambda x: x[1])
                print(f"[RPROF] slow round={r} total_ms={(rs[-1][2] - rs[0][2]) / 1e6:.1f} "
                      f"worst={worst[0]} host_ms={worst[1]:.1f} gpu_ms={worst[2]:.1f}", flush = True)
            print(f"[RPROF] slow_rounds={slow} sum_total_ms={sum(totals):.1f}", flush = True)
        # Round cost as a function of the verify width: the numbers EXL3_DSPARK_EV_COST wants.
        rset = set(rounds)
        tot_by_round = {}
        for r in rounds:
            rs = sorted([m for m in _marks if m[0] == r], key = lambda m: m[2])
            if len(rs) > 1:
                tot_by_round[r] = (rs[-1][2] - rs[0][2]) / 1e6
        width_of = {r: int(v) for r, n, v in _notes if n == "width" and r in rset}
        if width_of:
            byw = {}
            for r, w in width_of.items():
                if r in tot_by_round:
                    byw.setdefault(w, []).append(tot_by_round[r])
            print(f"[RPROF] {'width':>5s} {'n':>5s} {'round_ms_med':>12s} {'round_ms_mean':>13s}", flush=True)
            for w in sorted(byw):
                vals = byw[w]
                print(f"[RPROF] {w:5d} {len(vals):5d} {statistics.median(vals):12.3f} "
                      f"{statistics.fmean(vals):13.3f}", flush=True)
            xs = [(float(w), tot_by_round[r]) for r, w in width_of.items() if r in tot_by_round]
            if len({w for w, _ in xs}) > 1:
                n = len(xs)
                sx = sum(w for w, _ in xs)
                sy = sum(t for _, t in xs)
                sxx = sum(w * w for w, _ in xs)
                sxy = sum(w * t for w, t in xs)
                den = n * sxx - sx * sx
                if den:
                    b = (n * sxy - sx * sy) / den
                    a = (sy - b * sx) / n
                    print(f"[RPROF] width_fit rounds={n} intercept_ms={a:.3f} "
                          f"slope_ms_per_row={b:.3f}", flush=True)
        # Counters
        agg = {}
        _rs = set(rounds)
        for r, n, v in _notes:
            if r not in _rs:
                continue
            agg[n] = agg.get(n, 0) + (v if isinstance(v, (int, float)) else 1)
        nr = max(len(rounds), 1)
        for n, v in sorted(agg.items()):
            print(f"[RPROF] count {n:34s} {v / nr:8.3f}/round", flush = True)

    atexit.register(dump)
