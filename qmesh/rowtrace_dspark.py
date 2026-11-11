"""Dev-only eager verifier trace. Never enable on a production server.

Captures one forward after proving identical input IDs, prefix and canonical
pre-forward cache/recurrent state to the B1 reference. Instrumentation copies
tensors to CPU and changes execution timing; it is not a performance probe.
"""
import hashlib
import os
from pathlib import Path

import torch


def _candidate(params, ids, positions, jobs, expected, width, done):
    # iterate_gen marks speculative target verification with recurrent_history;
    # positions is optional RoPE offset data, not a verifier discriminator.
    return (not done and not params.get('prefill')
            and isinstance(ids, torch.Tensor) and isinstance(positions, torch.Tensor)
            and (width == 1 or params.get('recurrent_history') is True)
            and len(jobs) == expected and tuple(ids.shape) == (expected, width)
            and all(len(j.sequences) == 1 for j in jobs))


def _digest(value):
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(f'{t.dtype}:{tuple(t.shape)}:'.encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, (tuple, list)):
            h.update(f'{type(x).__name__}:{len(x)}:'.encode())
            for child in x:
                visit(child)
        elif isinstance(x, dict):
            h.update(f'dict:{len(x)}:'.encode())
            for key, child in sorted(x.items(), key=lambda kv: str(kv[0])):
                h.update(f'key:{key}:'.encode())
                visit(child)
        elif isinstance(x, (int, float, bool, str, type(None))):
            h.update(f'{type(x).__name__}:{x!r}:'.encode())
        else:
            raise TypeError(f'unsupported cache snapshot member {type(x)}')
    visit(value)
    return h.hexdigest()


_FP4_POOLS = ('pool_fp4', 'pool_fp4_scale', 'pool_idx_fp4', 'pool_idx_scale')


def _pools(layer, key):
    """The layer's fp4 pool planes by name, in CacheLayer_dsa.get_tensors() order."""
    tensors = layer.get_tensors()
    if layer.kv_format != 'fp4' or layer.mirrors or len(tensors) != len(_FP4_POOLS):
        raise RuntimeError(f'paged snapshot unsupported layer {key}: kv_format={layer.kv_format!r}, '
                           f'{len(tensors)} pools, {len(layer.mirrors)} mirrors')
    return dict(zip(_FP4_POOLS, tensors))


def _cache_snapshot(cache, seq, state, position, decode = False):
    """Pre-forward state of one row: every pool entry the position derives, plus the checkpoint stash.

    Pool entry e is produced once by token window [e*m, (e+1)*m) and lives at row e of its layer's
    pool planes, so the snapshot takes exactly entries [0, position // m) from this row's own pages.
    With decode the packed planes go to the pool's own dequantizer on device first (it takes 2D
    contiguous CUDA tensors, exactly as the runtime kernels call it) and the fp16 values are kept.
    """
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.cache.dsa import CacheLayer_dsa
    from exllamav3.ext import exllamav3_ext as ext
    if cache.model.loaded_tp:
        raise RuntimeError('TP snapshot unsupported')
    if position > 0 and state is None:
        raise RuntimeError('missing target recurrent state')
    layers = {}
    for key, layer in sorted(cache.layers.items(), key=lambda kv: str(kv[0])):
        if not isinstance(layer, CacheLayer_dsa):
            raise RuntimeError(f'unsupported paged layer {type(layer)}')
        pools = _pools(layer, key)
        blocks = []
        traced = 0
        for offset in range(0, position, PAGE_SIZE):
            length = min(PAGE_SIZE, position - offset) // layer.compress_rate
            if not length:
                continue
            page = seq.allocated_pages[offset // PAGE_SIZE].page_index
            slices = {name: t[page, :length].detach() for name, t in pools.items()}
            if decode:
                latent = ext.dsv4_fp4_dequant(slices['pool_fp4'], slices['pool_fp4_scale'], False)
                indexer = ext.dsv4_fp4_dequant(slices['pool_idx_fp4'], slices['pool_idx_scale'], True)
                if latent.shape[1] != layer.D_c + layer.D_r or indexer.shape[1] != layer.D_i:
                    raise RuntimeError(f'paged decode widths {latent.shape[1]}/{indexer.shape[1]} != '
                                       f'{layer.D_c + layer.D_r}/{layer.D_i} (layer {key})')
                slices = {'pool_idx': indexer, 'pool_c': latent[:, :layer.D_c],
                          'pool_r': latent[:, layer.D_c:]}
            blocks.append({name: t.cpu().clone() for name, t in slices.items()})
            traced += length
        if traced != position // layer.compress_rate:
            raise RuntimeError('paged snapshot does not cover every position-derived entry')
        entry = {'blocks': blocks, 'compress_rate': layer.compress_rate,
                 'epp': layer.epp, 'traced_entries': traced}
        if decode:
            entry['decoded'] = 'fp4_value_decode'
            entry['columns'] = {'pool_idx': 'indexer_rows', 'pool_c': 'latent_nope',
                                'pool_r': 'latent_rope'}
        layers[str(key)] = entry
    return {'paged': layers, 'recurrent': state.stash() if state is not None else None}


def _snapshot_digests(snapshot):
    """Raw-state digest per component, to locate divergence without calling it semantic."""
    return {'recurrent': _digest(snapshot['recurrent']),
            'paged': {key: _digest(value) for key, value in snapshot['paged'].items()}}


def _check_row_ownership(jobs, page_size, width):
    """Check writable pages and recurrent slots, not shared immutable prefixes."""
    pages, slots = set(), set()
    for job in jobs:
        for component in ('target', 'draft'):
            recurrent = getattr(job.recurrent_state, component, None)
            if recurrent is not None:
                key = (id(recurrent.cache), recurrent.slot)
                if key in slots:
                    raise RuntimeError(f'DSPARK_ROWTRACE_SHARED_SLOT component={component} slot={recurrent.slot}')
                slots.add(key)
        for seq in job.sequences:
            first, last = seq.kv_position // page_size, (seq.kv_position + width - 1) // page_size
            if last >= len(seq.allocated_pages):
                raise RuntimeError('DSPARK_ROWTRACE_MISSING_WRITE_PAGE')
            for page in seq.allocated_pages[first:last + 1]:
                key = page.page_index
                if key in pages or page.ref_count != 1:
                    raise RuntimeError(f'DSPARK_ROWTRACE_SHARED_WRITE_PAGE page={key} refs={page.ref_count}')
                pages.add(key)


def install(model, generator):
    output = Path(os.environ['DSPARK_ROWTRACE_OUT'])
    expected = int(os.environ['DSPARK_ROWTRACE_BATCH'])
    width = int(os.environ['DSPARK_ROWTRACE_WIDTH'])
    reference_path = os.environ.get('DSPARK_ROWTRACE_REFERENCE')
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)['metadata'] if reference_path else None
    target_position = int(os.environ.get('DSPARK_ROWTRACE_POSITION', '0'))
    if not 1 <= expected <= 8 or width < 1:
        raise ValueError('trace supports batch 1–8 and positive width')
    if reference is None and expected != 1 and os.environ.get('DSPARK_ROWTRACE_SAME_BATCH') != '1':
        raise ValueError('batched reference requires same-batch mode')
    if os.environ.get('EXL3_WHOLE_GRAPH') != '0' or os.environ.get('EXL3_DSV4_BATCH_GRAPH') != '0':
        raise RuntimeError('eager-only trace requires both graph switches off')
    if model.loaded_tp or output.exists():
        raise RuntimeError('TP or existing trace output is unsupported')
    output.mkdir(parents=True)
    state = {'active': False, 'done': False, 'records': []}
    modules = model.fwd_modules

    def capture(value, path, result):
        if isinstance(value, torch.Tensor):
            result.append({'path': path, 'status': 'tensor_batch_axis_unverified',
                           'shape': tuple(value.shape), 'dtype': str(value.dtype),
                           'tensor': value.detach().cpu().clone()})
        elif isinstance(value, (list, tuple)):
            for i, x in enumerate(value):
                capture(x, f'{path}[{i}]', result)
        elif isinstance(value, dict):
            for k, x in value.items():
                capture(x, f'{path}[{k!r}]', result)
        else:
            result.append({'path': path, 'status': 'non_tensor', 'type': type(value).__name__, 'repr': repr(value)})

    seen = set()
    for mod, instance, idx in modules:
        for obj, label in ((mod, str(idx)), (getattr(mod, 'attn', None), f'{idx}.attn'),
                           (getattr(mod, 'mlp', None), f'{idx}.mlp')):
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            original = obj.forward
            def wrapper(*args, _original=original, _label=label, _cls=type(obj), **kwargs):
                result = _original(*args, **kwargs)
                if state['active']:
                    entry = {'order': len(state['records']), 'module': _label,
                             'class': _cls.__module__ + '.' + _cls.__qualname__, 'outputs': []}
                    # The module's input, so a divergence can be attributed before it spreads
                    entry['inputs'] = []
                    capture(args[0] if args else kwargs.get('x'), 'input', entry['inputs'])
                    capture(result, 'output', entry['outputs'])
                    state['records'].append(entry)
                    # The active row layout of every traced call: a batch leak shows as a shape that
                    # carries other rows into a row's kernel, which the row comparison would otherwise
                    # only see as a value mismatch.
                    if isinstance(result, torch.Tensor):
                        x = args[0] if args else kwargs.get('x')
                        shape = tuple(getattr(x, 'shape', ())), tuple(result.shape)
                        if shape not in state.setdefault('layouts', []):
                            state['layouts'].append(shape)
                return result
            obj.forward = wrapper

    original_forward = model.forward
    def forward(*args, **kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        params = kwargs.get('params', args[1] if len(args) > 1 else None) or {}
        positions = params.get('cache_seqlens')
        jobs = [j for j in generator.active_jobs if j.is_prefill_done()]
        if not state['done'] and not params.get('prefill') and isinstance(ids, torch.Tensor):
            print(f'DSPARK_ROWTRACE_PROBE shape={tuple(ids.shape)} cache_type={type(positions).__name__} '
                  f'cache_shape={getattr(positions, "shape", None)} positions={params.get("positions") is not None} '
                  f'jobs={len(jobs)}', flush=True)
        if _candidate(params, ids, positions, jobs, expected, width, state['done']):
            from exllamav3.constants import PAGE_SIZE
            _check_row_ownership(jobs, PAGE_SIZE, width)
            ids_cpu = ids.detach().cpu().clone()
            pos = positions.detach().cpu().tolist()
            seqs = [j.sequences[0] for j in jobs]
            prefixes = [s.sequence_ids.torch().detach().cpu().flatten().tolist()[:p] for s, p in zip(seqs, pos)]
            aligned = (pos[0] >= target_position and all(p == pos[0] for p in pos)
                       and all(s.kv_position == pos[0] for s in seqs)
                       and all(prefix == prefixes[0] for prefix in prefixes)
                       and all(torch.equal(ids_cpu[i], ids_cpu[0]) for i in range(expected)))
            if reference is not None:
                aligned = (aligned and pos[0] == reference['target_position']
                           and width == reference['width'] and prefixes[0] == reference['prefix']
                           and torch.equal(ids_cpu[0], reference['input_ids'][0]))
            print(f'DSPARK_ROWTRACE_CANDIDATE pos={pos} width={width} aligned={aligned} '
                  f'ids={ids_cpu.tolist()}', flush=True)
            if aligned:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('CUDA graph capture is incompatible with trace')
                # Raw state decides alignment and carries the digest; a capture is only decoded
                # (on device, inside the snapshot) when its values are what gets compared
                snapshots = [_cache_snapshot(generator.cache, seq, job.recurrent_state.target, pos[0])
                             for seq, job in zip(seqs, jobs)]
                digests = [_digest(snapshot) for snapshot in snapshots]
                rows_differ = any(d != digests[0] for d in digests)
                if rows_differ or (reference is not None and digests[0] != reference['state_digest']):
                    if rows_differ:
                        print(f'DSPARK_ROWTRACE_STATE_MISMATCH positions={pos} digests={digests}', flush=True)
                    else:
                        print(f'DSPARK_ROWTRACE_REFERENCE_STATE_MISMATCH pos={pos[0]} '
                              f'ref={reference["state_digest"]} actual={digests[0]}', flush=True)
                    # The candidate rows' own values, decoded exactly as the reference capture was
                    torch.save({'position': pos[0], 'digests': digests,
                                'reference_digest': (reference['state_digest']
                                                     if reference is not None else None),
                                'snapshots': [_cache_snapshot(generator.cache, seq,
                                                              job.recurrent_state.target, pos[0],
                                                              decode = True)
                                              for seq, job in zip(seqs, jobs)]},
                               output / 'state-mismatch.pt')
                    reference_parts = reference.get('state_parts') if reference is not None else None
                    if reference_parts is not None:
                        actual_parts = _snapshot_digests(snapshots[0])
                        changed = [k for k in actual_parts['paged']
                                   if actual_parts['paged'][k] != reference_parts['paged'].get(k)]
                        print(f'DSPARK_ROWTRACE_STATE_COMPONENTS recurrent_changed='
                              f'{actual_parts["recurrent"] != reference_parts["recurrent"]} '
                              f'paged_layers_changed={changed}', flush=True)
                else:
                    meta = {'batch': expected, 'width': width, 'target_position': pos[0],
                            'input_ids': ids_cpu, 'positions': pos, 'prefix': prefixes[0],
                            'state_digest': digests[0], 'state_parts': _snapshot_digests(snapshots[0]),
                            'graph_scope': 'eager-only',
                            'modules': [(str(i), int(inst), type(m).__name__) for m, inst, i in modules]}
                    if reference is None:
                        torch.save({'position': pos[0],
                                    'snapshots': [_cache_snapshot(generator.cache, seq,
                                                                  job.recurrent_state.target, pos[0],
                                                                  decode = True)
                                                  for seq, job in zip(seqs, jobs)],
                                    'digests': digests},
                                   output / 'state-reference.pt')
                    state.setdefault('layouts', [])
                    state['active'] = True
                    try:
                        result = original_forward(*args, **kwargs)
                        entry = {'order': len(state['records']), 'module': '__model_output__', 'outputs': []}
                        capture(result, 'output', entry['outputs'])
                        state['records'].append(entry)
                        meta['layouts'] = state['layouts']
                        torch.save({'metadata': meta, 'records': state['records']}, output / 'trace.pt')
                        state['done'] = True
                        print(f'DSPARK_ROWTRACE_SAVED path={output / "trace.pt"} state={digests[0]}', flush=True)
                        return result
                    finally:
                        state['active'] = False
        return original_forward(*args, **kwargs)
    model.forward = forward
    print(f'DSPARK_ROWTRACE_ARMED path={output} batch={expected} width={width} reference={reference_path}', flush=True)
