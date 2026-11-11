#!/usr/bin/env python3
"""R5 HTTP content measurements; raw per-request records, no hidden retries."""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('base', help='HTTP server base URL')
parser.add_argument('output', type=Path, help='output JSONL')
parser.add_argument('--corpus-dir', type=Path, required=True, help='directory containing structured/math/code/prose _24k.txt files')
parser.add_argument('--held-prompts', type=Path, required=True, help='held-out prompts JSON array')
args = parser.parse_args()
base = args.base.rstrip('/')
out = args.output
short = ['The capital of France is', 'Write a Python function that returns the nth Fibonacci number.',
         'List three primary colors and explain why they matter in painting.',
         'Summarize the following passage in one sentence: ' + ''.join(
             'Paragraph %d: the river wound through the valley past terraced fields of barley, where farmers repaired stone walls each spring and traded seed in the market town. ' % i for i in range(1, 36)),
         'Write a Python class `LRUCache` with get and put methods, then show a ```python fenced example that uses it.',
         'Return only a JSON object describing a user with keys name, age, email, and a list field roles of strings. Use valid JSON with no commentary.',
         'Explain why the sky appears blue.']
held = json.loads(args.held_prompts.read_text())
sets = [(name, [short[0] + '\n\n' + (args.corpus_dir / (name + '_24k.txt')).read_text()]) for name in ('structured', 'math', 'code', 'prose')]
sets += [('short', short), ('held', held[:40])]
assert len(held) >= 40

def ask(prompt):
    body = json.dumps({'model': 'deepseek-v4.1-flash-pair', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 256, 'ignore_eos': True, 'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}).encode()
    req = urllib.request.Request(base + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    start = time.monotonic(); first = last = None; usage = {}; finish = None
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data:'): continue
            data = line[5:].strip()
            if data == '[DONE]': break
            obj = json.loads(data)
            if obj.get('usage'): usage = obj['usage']
            for choice in obj.get('choices') or []:
                finish = choice.get('finish_reason') or finish
                if (choice.get('delta') or {}).get('content'):
                    last = time.monotonic(); first = first or last
    n = usage.get('completion_tokens', 0); draft = usage.get('draft') or {}; accepted = draft.get('accepted')
    rounds = n - accepted if accepted is not None else None
    return {'prompt_tokens': usage.get('prompt_tokens'), 'tokens': n, 'accepted': accepted, 'rejected': draft.get('rejected'), 'verify_rounds_inferred': rounds, 'tokens_per_verify_inferred': n / rounds if rounds and rounds > 0 else None, 'ttft_s': first-start if first else None, 'decode_s': last-first if first and last else None, 'finish_reason': finish}

with out.open('w') as f:
    for rep in range(2):
        for name, prompts in (sets if rep == 0 else sets[::-1]):
            order = range(len(prompts)) if rep == 0 else reversed(range(len(prompts)))
            for i in order:
                try: result = ask(prompts[i])
                except Exception as e: result = {'error': repr(e)}
                rec = {'rep': rep, 'set': name, 'index': i, **result}
                f.write(json.dumps(rec) + '\n'); f.flush(); print(json.dumps(rec), flush=True)
