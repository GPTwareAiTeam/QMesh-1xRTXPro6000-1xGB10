"""B1 repeated chat greedy decoding; records output and requires uncached exact identity."""
import json
import sys
import urllib.error
import urllib.request

url, output = sys.argv[1].rstrip('/') + '/v1/chat/completions', sys.argv[2]
prompt = 'Write a detailed explanation of why the sky appears blue, including the physics.'
body = json.dumps({'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 64,
                   'temperature': 0, 'seed': 42}).encode()
results = []
for _ in range(4):
    req = urllib.request.Request(url, body, {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=7200) as response:
            results.append({'status': response.status, 'result': json.load(response)})
    except urllib.error.HTTPError as exc:
        results.append({'status': exc.code, 'result': json.load(exc)})
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
if any(r['status'] != 200 for r in results):
    raise SystemExit('request failed')
if any(r['result']['usage']['prompt_tokens_details']['cached_tokens'] != 0 for r in results):
    raise SystemExit('unexpected prompt-cache hit')
ids = [r['result']['choices'][0]['token_ids'] for r in results]
choices = [r['result']['choices'][0] for r in results]
if any('token_margins' in c and len(c['token_margins']) != len(c['token_ids']) for c in choices):
    raise SystemExit('token margins do not align with token IDs')
if any(len(x) < 16 for x in ids):
    raise SystemExit('insufficient output')
if any(x != ids[0] for x in ids[1:]):
    for i, seq in enumerate(ids[1:], 1):
        diff = next((j for j, (a, b) in enumerate(zip(ids[0], seq)) if a != b), None)
        if diff is None and len(ids[0]) != len(seq):
            diff = min(len(ids[0]), len(seq))
        if diff is not None and diff < len(ids[0]) and 'token_margins' in choices[0]:
            print(f'reference_margin={choices[0]["token_margins"][diff]}')
        print(f'repeat={i} first_diff={diff}')
    raise SystemExit('B1 repeatability failed')
print('PASS: four uncached B1 requests with identical token IDs')
