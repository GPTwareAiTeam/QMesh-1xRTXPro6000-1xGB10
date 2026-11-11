"""Compare aligned eager ABBA trace logits; report noise floor, not a quality verdict."""
import sys
from pathlib import Path
import torch

root = Path(sys.argv[1])
traces = [torch.load(root / tag / 'trace.pt', map_location='cpu', weights_only=False)
          for tag in ('R1', 'C1', 'C2', 'R2')]
ref = traces[0]['metadata']
for trace in traces[1:]:
    meta = trace['metadata']
    for key in ('batch', 'width', 'target_position', 'prefix', 'state_digest', 'modules', 'graph_scope', 'positions'):
        if meta[key] != ref[key]:
            raise SystemExit(f'INVALID alignment: {key}')
    if not torch.equal(meta['input_ids'], ref['input_ids']):
        raise SystemExit('INVALID alignment: input_ids')

def logits(trace):
    records = [r for r in trace['records'] if r['module'] == '__model_output__']
    if len(records) != 1:
        raise ValueError('missing model output')
    outputs = [x['tensor'] for x in records[0]['outputs'] if x['path'] in ('output', 'output[0]') and x['status'] == 'tensor_batch_axis_unverified']
    if len(outputs) != 1:
        raise ValueError('ambiguous model output')
    x = outputs[0].float()
    if x.ndim != 3 or tuple(x.shape[:2]) != (ref['batch'], ref['width']) or x.shape[-1] < 5 or not torch.isfinite(x).all():
        raise ValueError('invalid logits shape or values')
    return x

values = list(map(logits, traces))
print('ALIGNMENT OK: identical batch, IDs, positions, prefix, canonical state digest; eager only')
for name, i, j in (('reference_floor', 0, 3), ('candidate_repeat', 1, 2), ('reference_candidate_1', 0, 1), ('reference_candidate_2', 3, 2)):
    a, b = values[i], values[j]
    for row in range(ref['batch']):
        for pos in range(ref['width']):
            x, y = a[row, pos], b[row, pos]
            delta = (x - y).abs().max().item()
            peak = x.abs().max().item()
            xt, xi = x.topk(5)
            yt, yi = y.topk(5)
            print(f'{name} row={row} pos={pos} max_abs={delta:.9g} normalized={delta/max(peak, 1e-30):.9g} '
                  f'top1_flip={int(xi[0] != yi[0])} top5_set_flip={int(set(xi.tolist()) != set(yi.tolist()))} '
                  f'ref_margin={float(xt[0]-xt[1]):.9g} cand_margin={float(yt[0]-yt[1]):.9g}')
print('No automatic pass: compare candidate errors to repeat floor, inspect top-k margins and isolation separately.')
