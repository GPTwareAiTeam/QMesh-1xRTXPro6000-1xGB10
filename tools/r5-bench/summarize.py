#!/usr/bin/env python3
"""Summarize recorded HTTP benchmark JSONL without discarding errors or short completions."""
import argparse, collections, json, pathlib, statistics
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=pathlib.Path, help='directory containing benchmark JSONL files')
p = parser.parse_args().directory
for filename in ('content.jsonl','concurrency.jsonl','prefill.jsonl'):
    path=p/filename
    if not path.exists(): continue
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    print('\n'+filename)
    if filename=='content.jsonl':
        groups=collections.defaultdict(list)
        for r in rows: groups[(r['set'],r['rep'])].append(r)
        for key,rs in groups.items():
            good=[r for r in rs if r.get('decode_s') and r.get('tokens',0)>1]
            n=sum(r['tokens']-1 for r in good); t=sum(r['decode_s'] for r in good)
            dr=[r for r in good if r.get('verify_rounds_inferred')]
            rounds=sum(r['verify_rounds_inferred'] for r in dr)
            print(key, 'requests',len(rs),'errors',sum('error' in r for r in rs),'tok/s',round(n/t,2) if t else None,'tokens/round inferred',round(sum(r['tokens'] for r in dr)/rounds,3) if rounds else None,'mean tokens',round(statistics.mean(r['tokens'] for r in good),1) if good else None,'finish',dict(collections.Counter(r.get('finish_reason') for r in rs)))
    elif filename=='concurrency.jsonl':
        for r in rows: print(r['target'],r['users'],r['rep'],'total',r.get('total_decode_tok_s'),'user',r.get('mean_user_decode_tok_s'),'last_ttft',r.get('last_ttft_s'),'peak_p7_mib',r.get('peak_p7_driver_mib'),'errors',r.get('errors'),'prompt_tokens',[x.get('prompt_tokens') for x in r['requests']])
    else:
        for r in rows: print(r)
