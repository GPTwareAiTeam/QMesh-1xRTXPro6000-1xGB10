"""Concurrent non-streaming serving wave with the engine's per-job timing.

Engine time_generate is time from its first decode forward to final emission. It
excludes prefill but includes shared scheduler work and request overlap; it is not
GPU kernel time and does not by itself prove simultaneous admission.
"""
import argparse
import json
import threading
import urllib.request

PROMPTS = {
    'A': 'Explain in three sentences why serving eight independent requests together keeps a GPU busier than serving the same eight one after another, and say what has to be true about the memory each request holds.',
    'B': 'Name four things that must be true for a distributed inference server to be deterministic, and say which of them is hardest to guarantee.',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('url')
    p.add_argument('spec')
    p.add_argument('--tokens', type=int, default=128)
    a = p.parse_args()
    names = a.spec.split(',')
    if not names or any(n not in PROMPTS for n in names):
        p.error('spec must be comma-separated A/B prompts')
    barrier = threading.Barrier(len(names))
    results = [None] * len(names)

    def run(i, name):
        try:
            body = json.dumps({'model': 'deepseek-v4.1-flash-pair', 'messages': [{'role': 'user', 'content': PROMPTS[name]}],
                               'max_tokens': a.tokens, 'temperature': 0, 'stream': False}).encode()
            req = urllib.request.Request(a.url.rstrip('/') + '/v1/chat/completions', data=body,
                                         headers={'Content-Type': 'application/json'})
            barrier.wait(timeout=30)
            with urllib.request.urlopen(req, timeout=900) as response:
                result = json.load(response)
            choice = result['choices'][0]
            count = result['usage']['completion_tokens']
            duration = choice['time_generate']
            ids = choice['token_ids']
            if not isinstance(duration, (int, float)) or duration <= 0 or count < 2 or len(ids) < 2:
                raise ValueError('missing engine timing or token data')
            results[i] = {'row': i, 'prompt': name, 'time_generate_s': duration,
                          'completion_tokens': count, 'finish': choice['finish_reason'], 'token_ids': ids}
        except Exception as exc:
            results[i] = {'row': i, 'error': repr(exc)}

    threads = [threading.Thread(target=run, args=(i, name)) for i, name in enumerate(names)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(json.dumps({'spec': names, 'requests': results}), flush=True)
    if any('error' in row for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
