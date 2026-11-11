"""Compare decoded paged pool entries and raw recurrent buffers from rowtrace snapshots.

Usage: compare_rowtrace_state.py R1/state-reference.pt C1/state-mismatch.pt

Paged (logical): entry e of a layer's pool is produced exactly once by token window
[e*m, (e+1)*m), and the tracer covers every position-derived entry, so a traced entry is state a
consumer reads. The tracer decodes packed formats to fp16 with the runtime's own decoder, so this
compares values rather than words and a shared scale byte cannot mask a value difference.

Recurrent (physical): the checkpoint stash copies whole compressor/indexer/overlap buffers, including
rows no consumer reads (stale, overwritten or rollback storage). Raw differences are reported, never
scored, and never establish logical state, quality or isolation.
"""
import sys
import torch

LABELS = ('pool_idx', 'pool_c', 'pool_r')


def compare(a, b, path, stats):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            print(f'LAYOUT {path} keys={list(a)}/{list(b)}')
            stats['layout'] += 1
            return
        for k in sorted(a.keys() & b.keys(), key=str):
            compare(a[k], b[k], f'{path}/{k}', stats)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            print(f'LAYOUT {path} lengths={len(a)}/{len(b)}')
            stats['layout'] += 1
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}/{i}', stats)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            print(f'LAYOUT {path} shape={tuple(a.shape)}/{tuple(b.shape)} dtype={a.dtype}/{b.dtype}')
            stats['layout'] += 1
            return
        n = int((a != b).sum())
        if not n:
            stats['equal'] += 1
            return
        if a.is_floating_point():
            finite = torch.isfinite(a) & torch.isfinite(b)
            delta = (a[finite].double() - b[finite].double()).abs()
            scale = float(a[finite].double().abs().max()) if delta.numel() else float('nan')
            print(f'DIFF {path} dtype={a.dtype} count={n}/{a.numel()} '
                  f'max_abs={float(delta.max()):.9g} ref_abs={scale:.9g} '
                  f'nonfinite={int((~finite).sum())}')
        else:
            print(f'DIFF {path} dtype={a.dtype} count={n}/{a.numel()} values_not_decoded')
        stats['diff'] += 1
    elif type(a) is not type(b) or a != b:
        print(f'META {path} {a!r} != {b!r}')
        stats['layout'] += 1


def compare_layer(key, left, right, stats):
    if not isinstance(left, dict) or not isinstance(right, dict):
        print(f'LAYOUT paged/{key} not a traced layer snapshot')
        stats['layout'] += 1
        return
    for field in ('decoded', 'columns', 'compress_rate', 'epp', 'traced_entries'):
        if left.get(field) != right.get(field):
            print(f'LAYOUT paged/{key} {field}={left.get(field)}/{right.get(field)}')
            stats['layout'] += 1
            return
    if left['decoded'] != 'fp4_value_decode':
        print(f'LAYOUT paged/{key} not decoded to values: {left["decoded"]}')
        stats['layout'] += 1
        return
    m, epp = left['compress_rate'], left['epp']
    lb, rb = left['blocks'], right['blocks']
    if len(lb) != len(rb):
        print(f'LAYOUT paged/{key} blocks={len(lb)}/{len(rb)}')
        stats['layout'] += 1
        return
    for blk, (lp, rp) in enumerate(zip(lb, rb)):
        if not isinstance(lp, dict) or not isinstance(rp, dict) or lp.keys() != rp.keys():
            print(f'LAYOUT paged/{key} block{blk} fields {sorted(lp)}/{sorted(rp)}')
            stats['layout'] += 1
            continue
        entries = min(epp, left['traced_entries'] - blk * epp)
        for label in LABELS:
            if label not in lp:
                print(f'LAYOUT paged/{key} block{blk} missing {label}')
                stats['layout'] += 1
                continue
            # The leading axis is the entry index; entry e is consumed from token e*m onward.
            compare(lp[label][:entries], rp[label][:entries],
                    f'paged/{key}/{label}/tokens_from_{blk * epp * m}', stats)
    covered = sum(min(epp, left['traced_entries'] - i * epp) for i in range(len(lb)))
    if covered != left['traced_entries']:
        print(f'LAYOUT paged/{key} traced_entries={left["traced_entries"]} covered={covered} epp={epp}')
        stats['layout'] += 1


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = (torch.load(p, map_location='cpu', weights_only=False) for p in sys.argv[1:])
    if a['position'] != b['position'] or len(a['snapshots']) != len(b['snapshots']):
        raise SystemExit('INVALID position or row count')
    print(f"STATE_DIAGNOSTIC position={a['position']} rows={len(a['snapshots'])} "
          'paged=logical_decoded_entries recurrent=raw_checkpoint_not_semantic')
    for i, (x, y) in enumerate(zip(a['snapshots'], b['snapshots'])):
        stats = {'diff': 0, 'layout': 0, 'equal': 0}
        for key in sorted(x['paged'].keys() | y['paged'].keys(), key=str):
            compare_layer(key, x['paged'].get(key), y['paged'].get(key), stats)
        print(f'PAGED_ROW row={i} differing_tensors={stats["diff"]} equal_tensors={stats["equal"]} '
              f'layout_gaps={stats["layout"]}')
        compare(x['recurrent'], y['recurrent'], f'row{i}/recurrent_raw', stats)
    print('STATE_DIAGNOSTIC_DONE no quality, corruption or isolation verdict')


if __name__ == '__main__':
    main()
