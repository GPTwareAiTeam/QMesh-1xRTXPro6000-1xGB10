"""Diagnostic client; run only against the isolated dev server.

Run once per server configuration: python adaptive_batch_gate.py URL WIDTH OUTPUT.
Compare output JSON files across configurations. Token differences characterize batch
numerics; they do not by themselves establish cross-user state corruption.
"""
import concurrent.futures
import json
import sys
import threading
import time
import urllib.error
import urllib.request

url, width, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if width not in (1, 2, 8):
    raise SystemExit('width must be 1, 2, or 8')
url = url.rstrip('/') + '/v1/chat/completions'

def request(prompt, max_tokens=64, barrier=None):
    body = json.dumps({'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': max_tokens, 'temperature': 0, 'seed': 42}).encode()
    req = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=7200) as response:
            status, result = response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        status, result = exc.code, json.load(exc)
    return {'status': status, 'result': result, 'seconds': time.monotonic() - started}

same = 'Write a detailed explanation of why the sky appears blue, including the physics.'
different = [f'Write a detailed explanation of the mathematical number {i}, including examples.' for i in range(width)]
report = {'width': width, 'same_prompt': same, 'different_prompts': different}
report['same_single'] = request(same)
report['different_single'] = [request(prompt) for prompt in different]
for name, prompts in [('same', [same] * width), ('different', different)]:
    barrier = threading.Barrier(width)
    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:
        report[name] = list(pool.map(lambda p: request(p, barrier=barrier), prompts))
    if any(x['status'] != 200 for x in report[name]):
        print(name, 'failed', file=sys.stderr)
with open(output, 'w') as f:
    json.dump(report, f, indent=2)
all_results = report['same'] + report['different'] + [report['same_single']] + report['different_single']
if any(x['status'] != 200 for x in all_results):
    raise SystemExit(1)
if any(len(x['result']['choices'][0]['token_ids']) < 16 for x in all_results):
    raise SystemExit('insufficient generated tokens for a meaningful invariance gate')
def tokens(result):
    return result['result']['choices'][0]['token_ids']
for name, actual, reference in [('same', report['same'], [report['same_single']] * width),
                                ('different', report['different'], report['different_single'])]:
    for row, (a, b) in enumerate(zip(actual, reference)):
        x, y = tokens(a), tokens(b)
        first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v),
                     min(len(x), len(y)) if len(x) != len(y) else None)
        print(f'{name} row={row} first_token_difference={first}')
