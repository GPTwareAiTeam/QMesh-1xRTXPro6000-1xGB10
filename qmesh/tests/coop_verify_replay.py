"""Test row independence of the coop entry point using a split pack's local expert weights.

Run only through the release manager's GPU queue. This isolates the routed expert
kernel; it does not load the router, shared expert, remote experts, or real model
activations, and is not an end-to-end correctness or isolation gate.
"""
import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'coop_fast'))
from bench_pair_prefill import DEV, TOPK, coop_case, k2_tables, load_layer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('layer', type=int)
    p.add_argument('--rows', default='1,6,12,18,24,48')
    p.add_argument('--tol', type=float, default=2e-3)
    p.add_argument('--capture')
    args = p.parse_args()
    widths = sorted({int(w) for w in args.rows.split(',')})
    if not widths or min(widths) < 1 or max(widths) > 48:
        p.error('row widths must be between 1 and 48')
    from exllamav3.ext import exllamav3_ext as ext

    ex, rates = load_layer(args.checkpoint, args.layer, expected_layout='coop_pair_v1')
    if len(ex) < TOPK:
        raise RuntimeError('not enough local experts for top-k routing')
    k2 = k2_tables(rates)
    h = ex[0]['g'][0].shape[0] * 16
    n = max(widths)
    gen = torch.Generator().manual_seed(1234)
    x = (torch.randn((n, h), generator=gen) * 0.05).half().to(DEV)
    sel = torch.stack([torch.randperm(len(ex), generator=gen)[:TOPK] for _ in range(n)]).to(DEV)
    rw = (torch.rand((n, TOPK), generator=gen) * 0.3 + 0.05).half().to(DEV)

    def run(xx, ss, ww):
        fn, out, _ = coop_case(ext, ex, rates, k2, len(xx), True, gen, (xx, ss, ww))
        fn()
        return out.clone()

    solo = torch.cat([run(x[i:i+1], sel[i:i+1], rw[i:i+1]) for i in range(n)])
    results = []
    failures = 0
    for w in widths:
        out = run(x[:w], sel[:w], rw[:w])
        diff = (out - solo[:w]).abs().amax(dim=1)
        peaks = solo[:w].abs().amax(dim=1).clamp_min(1e-12)
        rel = diff / peaks
        rows = [{'row': i, 'max_abs': diff[i].item(), 'rel_to_solo_peak': rel[i].item()}
                for i in range(w)]
        failures += sum(r['rel_to_solo_peak'] > args.tol for r in rows)
        record = {'width': w, 'rows': rows}
        results.append(record)
        print(json.dumps(record), flush=True)
        if w > 1:
            noisy_x = x[:w].clone()
            noisy_sel = sel[:w].clone()
            noisy_rw = rw[:w].clone()
            noisy_x[1:] = (torch.randn((w - 1, h), generator=gen) * 0.15).half().to(DEV)
            noisy_sel[1:] = torch.stack([torch.randperm(len(ex), generator=gen)[:TOPK]
                                         for _ in range(w - 1)]).to(DEV)
            noisy_rw[1:] = torch.rand((w - 1, TOPK), generator=gen).half().to(DEV)
            changed = run(noisy_x, noisy_sel, noisy_rw)
            same = torch.equal(out[0], changed[0])
            failures += not same
            print(json.dumps({'width': w, 'neighbor_isolation_bit_identical': same}), flush=True)
    result = {'pack': args.checkpoint, 'layer': args.layer, 'local_experts': len(ex),
              'input': 'seeded random; real paired local expert weights', 'results': results,
              'failures': failures}
    if args.capture:
        pathlib.Path(args.capture).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'REPLAY_DONE': True, 'failures': failures}), flush=True)
    return int(failures != 0)


if __name__ == '__main__':
    sys.exit(main())
