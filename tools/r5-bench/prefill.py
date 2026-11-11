#!/usr/bin/env python3
"""Cold HTTP TTFT prefill estimate at requested context sizes; no prefix reuse."""
import json, sys, time, urllib.request
base,out=sys.argv[1:]
unit='The archive recorded the movements of ships, the price of grain, the weather at dawn and the names of the clerks on duty, one entry per line, in a careful hand that did not change for forty years. '
with open(out,'w') as f:
    for target in (24000,131000,1000000):
        prompt=f'Cold context {target}. Summarise the following archive.\n'+unit*(int(target * 940000 / 1010525)//40)
        body=json.dumps({'model':'deepseek-v4.1-flash-pair','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':16,'stream':True,'stream_options':{'include_usage':True}}).encode()
        req=urllib.request.Request(base+'/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
        start=time.monotonic(); first=None; usage={}
        try:
            with urllib.request.urlopen(req,timeout=3600) as r:
                for raw in r:
                    line=raw.decode().strip()
                    if not line.startswith('data:'): continue
                    d=line[5:].strip()
                    if d=='[DONE]': break
                    obj=json.loads(d)
                    if obj.get('usage'): usage=obj['usage']
                    for c in obj.get('choices') or []:
                        if (c.get('delta') or {}).get('content') and first is None: first=time.monotonic()
            row={'target':target,'prompt_tokens':usage.get('prompt_tokens'),'tokens':usage.get('completion_tokens'),'ttft_s':first-start if first else None}
            row['prompt_tokens_per_ttft_s']=row['prompt_tokens']/row['ttft_s'] if row['prompt_tokens'] and row['ttft_s'] else None
        except Exception as e: row={'target':target,'error':repr(e)}
        print(json.dumps(row),flush=True); f.write(json.dumps(row)+'\n'); f.flush()
