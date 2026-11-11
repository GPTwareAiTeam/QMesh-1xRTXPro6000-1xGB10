#!/usr/bin/env python3
"""Concurrent users over four content types (structured, math, code, prose), cold prompts.

User i of a wave gets content type (i + rotation) % 5, so with 1 user each rep rotates through the types and the 2k /
24k rows are an average over content, not one type. Each prompt gets a unique first line, so no two requests share a
prefix (prefix cache is off anyway). 256 tokens, greedy. Decode tok/s per user = (tokens - 1) / (last - first token).
Total = sum over users of (tokens - 1) / (latest last token - earliest first token)."""
import json, subprocess, sys, threading, time, urllib.request
from pathlib import Path

base, output = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
corpus = Path(__import__('os').environ.get('R5_CORPUS', 'corpus'))  # dir holding <type>_2k / _24k .txt
TYPES = ('structured', 'math', 'code', 'prose')
TEXT = {(t, s): (corpus / f'{t}_{s}.txt').read_text() for t in TYPES for s in ('2k', '24k')}
NT = len(TYPES)
ASK = {'structured': 'Convert the key records above into a compact JSON summary.',
       'math': 'Work through the next problem in the same style as above.',
       'code': 'Review the code above and write the next function it needs, with a short explanation.',
       'prose': 'Write a detailed summary of the passage above.'}


def ask(i, ctype, size, results, tag):
    prompt = f'[{tag}-{i}] ' + TEXT[(ctype, size)] + '\n\n' + ASK[ctype]
    body = json.dumps({'model': 'deepseek-v4.1-flash-pair', 'messages': [{'role': 'user', 'content': prompt}],
                       'temperature': 0, 'max_tokens': 256, 'ignore_eos': True, 'stream': True,
                       'stream_options': {'include_usage': True}}).encode()
    req = urllib.request.Request(base + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    start = time.monotonic(); first = last = None; usage = {}; reason = None; ticks = []
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                obj = json.loads(data)
                if obj.get('usage'):
                    usage = obj['usage']
                for c in obj.get('choices') or []:
                    reason = c.get('finish_reason') or reason
                    if (c.get('delta') or {}).get('content'):
                        last = time.monotonic(); first = first or last; ticks.append(last)
        results[i] = {'index': i, 'type': ctype, 'start': start, 'first': first, 'last': last,
                      'prompt_tokens': usage.get('prompt_tokens'), 'tokens': usage.get('completion_tokens'),
                      'draft': usage.get('draft'), 'reason': reason, 'ticks': ticks}
    except Exception as e:
        results[i] = {'index': i, 'type': ctype, 'error': repr(e)}


def wave(size, n, rep):
    results = [None] * n; samples = []; stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                samples.append(int(subprocess.check_output(['nvidia-smi', '-i', __import__('os').environ.get('NV_GPU', '0'), '--query-gpu=memory.used',
                                                            '--format=csv,noheader,nounits'], text=True, timeout=5)))
            except (ValueError, subprocess.SubprocessError):
                pass
            stop.wait(1)
    th = threading.Thread(target=sample); th.start()
    ws = [threading.Thread(target=ask, args=(i, TYPES[(i + rep) % NT], size, results, f'{size}-{n}-{rep}')) for i in range(n)]
    for w in ws: w.start()
    for w in ws: w.join()
    stop.set(); th.join()
    bad = [r for r in results if 'error' in r or not r.get('tokens') or r.get('first') is None or r.get('last') is None]
    row = {'size': size, 'users': n, 'rep': rep, 'types': [r['type'] for r in results], 'requests': results,
           'peak_p7_mib': max(samples) if samples else None, 'errors': len(bad)}
    if not bad:
        row['last_ttft_s'] = max(r['first'] - r['start'] for r in results)
        win = max(r['last'] for r in results) - min(r['first'] for r in results)
        row['total_decode_tok_s'] = sum(max(r['tokens'] - 1, 0) for r in results) / win if win > 0 else None
        row['mean_user_decode_tok_s'] = sum((r['tokens'] - 1) / (r['last'] - r['first'])
                                            for r in results if r['last'] > r['first']) / n
        # Steady state: only the interval in which every user is decoding (latest first token .. earliest last token).
        # Streamed pieces carry several tokens each (accepted drafts), so tokens are apportioned per piece by the
        # request's own tokens/pieces ratio.
        a = max(r['first'] for r in results); b = min(r['last'] for r in results)
        row['all_decoding_s'] = max(0.0, b - a)
        if b > a:
            tok = 0.0
            for r in results:
                per_piece = r['tokens'] / max(1, len(r['ticks']))
                tok += per_piece * sum(1 for t in r['ticks'] if a < t <= b)
            row['steady_total_tok_s'] = tok / (b - a)
    return row


USERS = tuple(int(x) for x in __import__('os').environ.get('USERS', '1,2,4,8').split(','))
cells = [(s, n) for s in ('2k', '24k') for n in USERS]
with open(output, 'w') as f:
    for rep in range(REPS):
        for s, n in (cells if rep % 2 == 0 else cells[::-1]):
            row = wave(s, n, rep)
            f.write(json.dumps(row) + '\n'); f.flush()
            print(json.dumps({k: v for k, v in row.items() if k != 'requests'}), flush=True)
