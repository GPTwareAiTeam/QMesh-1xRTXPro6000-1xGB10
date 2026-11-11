#!/usr/bin/env python3
"""Needles in a haystack, sudoingX's ladder shape: 8k / 32k / 64k / 128k tokens, three planted facts at 10 / 50 / 90 %
depth, one question per fact in the same request. The haystack is the release corpus (prose, code, math, structured,
multilingual 128k files, interleaved by paragraph), cut to size by a measured chars-per-token ratio and checked with the
server's reported prompt tokens. A fact counts when its unique answer string is in the reply. Greedy, 192 tokens."""
import json, sys, time, urllib.request
from pathlib import Path

base, output = sys.argv[1], sys.argv[2]
corpus = Path(__import__('os').environ.get('R5_CORPUS', 'corpus'))  # dir holding <type>_2k / _24k / _128k .txt
parts = [p for t in ('prose', 'code', 'math', 'structured', 'multilingual')
         for p in (corpus / f'{t}_128k.txt').read_text().split('\n\n') if p.strip()]
# interleave content types so every depth mixes them
step = len(parts) // 5
mixed = [parts[i + k * step] for i in range(step) for k in range(5)]
hay = '\n\n'.join(mixed)
FACTS = [('The vault code for the Aldersgate archive is 47-QX-9183.', 'What is the vault code for the Aldersgate archive?', '47-QX-9183'),
         ('The ferry captain on the Marrowind route is named Ilse Vantongeren.', 'Who is the ferry captain on the Marrowind route?', 'Vantongeren'),
         ('The rare blue orchid in greenhouse 12 blooms only in the month of Quintober.', 'In which month does the rare blue orchid in greenhouse 12 bloom?', 'Quintober')]
CPT = 3.3   # chars per token on this mix; corrected per rung from the first reply's prompt_tokens


def ask(text):
    body = json.dumps({'model': 'deepseek-v4.1-flash-pair', 'messages': [{'role': 'user', 'content': text}],
                       'temperature': 0, 'max_tokens': 192, 'stream': True,
                       'stream_options': {'include_usage': True}}).encode()
    req = urllib.request.Request(base + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    start = time.monotonic(); first = last = None; usage = {}; out = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            d = line[5:].strip()
            if d == '[DONE]':
                break
            obj = json.loads(d)
            if obj.get('usage'):
                usage = obj['usage']
            for c in obj.get('choices') or []:
                piece = (c.get('delta') or {}).get('content')
                if piece:
                    last = time.monotonic(); first = first or last; out.append(piece)
    return ''.join(out), usage, start, first, last


def build(tokens, cpt):
    n = int(tokens * cpt)
    body = hay[:n]
    for frac, (fact, _, _) in sorted(zip((0.9, 0.5, 0.1), FACTS), reverse=True):
        cut = body.rfind('\n\n', 0, int(len(body) * frac)) + 2
        body = body[:cut] + fact + '\n\n' + body[cut:]
    q = '\n'.join(f'{i + 1}. {q}' for i, (_, q, _) in enumerate(FACTS))
    return ('Read the following documents carefully. Afterwards answer the questions.\n\n' + body +
            '\n\nAnswer each question in one short line:\n' + q)


with open(output, 'w') as f:
    for target in (8192, 32768, 65536, 131072):
        cpt = CPT
        for attempt in range(2):
            text, usage, s, a, b = ask(build(target - 300, cpt))
            got = usage.get('prompt_tokens') or 0
            if abs(got - target) / target < 0.05 or attempt == 1:
                break
            cpt = cpt * target / max(got, 1)
        hits = [ans.lower() in text.lower() for _, _, ans in FACTS]
        dec = (usage.get('completion_tokens', 0) - 1) / (b - a) if a and b and b > a else None
        row = {'target': target, 'prompt_tokens': got, 'hits': hits, 'score': f'{sum(hits)}/3',
               'ttft_s': a - s if a else None, 'prefill_tok_s': got / (a - s) if a else None,
               'decode_tok_s': dec, 'reply': text}
        f.write(json.dumps(row) + '\n'); f.flush()
        print(json.dumps({k: v for k, v in row.items() if k != 'reply'}), flush=True)
