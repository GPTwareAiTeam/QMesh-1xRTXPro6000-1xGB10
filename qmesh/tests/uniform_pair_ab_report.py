"""Fail-closed token and width reporting for the isolated uniform MoE pair A/B."""
import collections
import json
import pathlib
import re
import sys


def driver(path, count):
    waves, tokens = [], []
    for line in path.read_text().splitlines():
        if line.startswith('CMP_WAVE '):
            waves.append(json.loads(line[9:]))
        if line.startswith('CMP_TOKENS '):
            tokens.append(json.loads(line[11:]))
    if len(waves) != 1 or waves[0]['n'] != count or len(tokens) != count:
        raise ValueError(f'{path}: incomplete wave')
    tokens.sort(key=lambda x: x['i'])
    if [t['i'] for t in tokens] != list(range(count)) or any(not isinstance(t['token_ids'], list) or len(t['token_ids']) < 2 for t in tokens):
        raise ValueError(f'{path}: missing token IDs')
    return waves[0], tokens


def main():
    if sys.argv[1] == 'check':
        driver(pathlib.Path(sys.argv[2]), int(sys.argv[3]))
        return
    if sys.argv[1] == 'admission':
        path, offset, batch = pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        with path.open('rb') as f:
            f.seek(offset)
            text = f.read().decode('utf-8', errors='replace')
        widths = collections.Counter(int(x) for x in re.findall(r'SERVE_DECODE_WIDTH active_rows=(\d+) draft=\d+ verify_rows=\d+', text))
        print(json.dumps({'required_active_rows': batch, 'active_rows_histogram': dict(widths),
                          'established': widths[batch] > 0}))
        if not widths[batch]:
            raise ValueError('requested active row count not observed during this wave')
        return
    root = pathlib.Path(sys.argv[2])
    width_re = re.compile(r'SERVE_DECODE_WIDTH active_rows=(\d+) draft=(\d+) verify_rows=(\d+)')
    results = {}
    smoke = len(sys.argv) > 3 and sys.argv[3] == '1'
    for d in ((1,) if smoke else (0, 1)):
        for batch in ((1, 4) if smoke else (1, 2, 4, 8)):
            for run, rows in ((1, 8), (2, 48), (3, 48), (4, 8)):
                tag = f'd{d}-b{batch}-r{rows}-n{run}'
                hist = collections.Counter(tuple(map(int, m)) for m in width_re.findall((root / f'{tag}-server.log').read_text()))
                if not hist:
                    raise ValueError(f'{tag}: no width records')
                singles = {p: driver(root / f'{tag}-single-{p}.log', 1)[1][0] for p in ('A', 'B')}
                for kind in ('same', 'mixed'):
                    wave, toks = driver(root / f'{tag}-{kind}.log', batch)
                    seconds = wave['wall_s']
                    if seconds <= 0:
                        raise ValueError(f'{tag}: invalid wall time')
                    results[(d, batch, run, kind)] = toks
                    print(json.dumps({'tag': tag, 'kind': kind, 'wall_s': seconds,
                                      'throughput_tok_s': sum(wave['tok']) / seconds,
                                      'width_histogram': {str(k): v for k, v in hist.items()}}))
                    if kind == 'mixed':
                        for item in toks:
                            x, y = singles[item['prompt']]['token_ids'], item['token_ids']
                            first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v),
                                         min(len(x), len(y)) if len(x) != len(y) else None)
                            print(json.dumps({'tag': tag, 'comparison': 'singleton_vs_mixed',
                                              'row': item['i'], 'prompt': item['prompt'], 'first_flip': first,
                                              'positional_flips': sum(u != v for u, v in zip(x, y)),
                                              'singleton_tokens': len(x), 'mixed_tokens': len(y)}))
            for kind in ('same', 'mixed'):
                base = results[(d, batch, 1, kind)]
                for run in (2, 3, 4):
                    other = results[(d, batch, run, kind)]
                    for a, b in zip(base, other):
                        x, y = a['token_ids'], b['token_ids']
                        first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v),
                                     min(len(x), len(y)) if len(x) != len(y) else None)
                        print(json.dumps({'dspark': d, 'batch': batch, 'kind': kind, 'run': run,
                                          'row': a['i'], 'prompt': a['prompt'], 'first_flip': first,
                                          'positional_flips': sum(u != v for u, v in zip(x, y)),
                                          'baseline_tokens': len(x), 'arm_tokens': len(y)}))


if __name__ == '__main__':
    main()
