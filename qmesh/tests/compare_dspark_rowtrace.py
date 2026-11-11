"""Compare trusted local eager rowtrace files; rc 1 on numerical/coverage differences."""
import sys
import torch

if len(sys.argv) != 3:
    raise SystemExit('usage: compare_dspark_rowtrace.py B1.pt BN.pt')
a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in sys.argv[1:]]
x, y = a['metadata'], b['metadata']


def require(ok, message):
    if not ok:
        raise SystemExit('INVALID ALIGNMENT: ' + message)


def equal(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                        b.contiguous().reshape(-1).view(torch.uint8)))


require(x['batch'] == 1 and y['batch'] in (2, 3, 4), 'wrong batch widths')
require(x['graph_scope'] == y['graph_scope'] == 'eager-only', 'wrong execution scope')
for key in ('width', 'target_position', 'prefix', 'state_digest', 'modules'):
    require(x[key] == y[key], f'{key} mismatch')
require(equal(x['input_ids'][0], y['input_ids'][0]), 'input IDs mismatch')
require(all(equal(y['input_ids'][0], row) for row in y['input_ids']), 'batched IDs differ')
require(all(p == x['target_position'] for p in x['positions'] + y['positions']), 'position mismatch')
print('ALIGNMENT OK: IDs, positions, prefix and canonical paged/recurrent state digest; eager only')

ra, rb = a['records'], b['records']
require([(r['module'], r.get('class')) for r in ra] == [(r['module'], r.get('class')) for r in rb],
        'module invocation plans differ')
missing = [idx for idx, _, _ in x['modules'] if idx not in {r['module'] for r in ra}]
differences = ambiguous = 0
first = None


def compare(left, right, key, is_output):
    global differences, ambiguous, first
    lo, ro = left.get(key, []), right.get(key, [])
    require([(z['path'], z['status']) for z in lo] == [(z['path'], z['status']) for z in ro], 'output structure differs')
    for l, r in zip(lo, ro):
        if l['status'] != 'tensor_batch_axis_unverified' or l['path'] not in ('output', 'output[0]', 'input', 'input[0]'):
            print('AMBIGUOUS', left['module'], l['path'], l['status'])
            ambiguous += 1
            continue
        lt, rt = l['tensor'], r['tensor']
        if lt.ndim < 2 or rt.ndim != lt.ndim or lt.shape[0] != 1 or rt.shape[0] != y['batch'] or lt.shape[1:] != rt.shape[1:]:
            print('AMBIGUOUS', left['module'], tuple(lt.shape), tuple(rt.shape))
            ambiguous += 1
            continue
        # Both arms trace whole module tensors, so the row to compare is simply row 0 of the
        # tensor: at B1 that is the only row, at B2.. it is the row that must match B1.
        exact = equal(lt[0], rt[0])
        if not is_output:
            # Inputs decide attribution: an exact left input means only this module can have moved the row.
            print('INPUT', 'EXACT' if exact else 'DIFFERENT', left['order'], left['module'])
            continue
        if exact:
            print('EXACT', left['order'], left['module'])
            continue
        delta = 'unavailable'
        if lt.dtype == rt.dtype and lt.is_floating_point():
            delta = (lt[0].double() - rt[0].double()).abs().max().item()
        print('DIFFERENT', left['order'], left['module'], 'max_abs=', delta)
        differences += 1
        if first is None:
            first = left['module']


for left, right in zip(ra, rb):
    if 'inputs' in left or 'inputs' in right:
        compare(left, right, 'inputs', False)
    compare(left, right, 'outputs', True)
print('first differing output:', first or None)
# Layouts only describe the traced call, never a row's value: a batched arm at 2 rows carries
# (2, ...) shapes for 2-row kernels, so they are reported and never compared.
print('row layouts:', x.get('layouts'), 'vs', y.get('layouts'))
print('differences:', differences, 'unobserved modules:', missing, 'ambiguous outputs:', ambiguous)
if differences or missing or ambiguous:
    raise SystemExit(1)
