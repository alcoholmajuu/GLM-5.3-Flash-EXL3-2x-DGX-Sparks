"""MIT: one fixed 400-token request with bounded profiler, separate from timing."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--url',default='http://127.0.0.1:8000')
p.add_argument('--long-body',action='store_true',help='Profile a separate 4096-input/1024-output body continuation')
p.add_argument('--case',default='code_01',choices=['code_01','json_01','prose_ja_01'])
a=p.parse_args()
def control(action):
    request=urllib.request.Request(a.url+'/'+action,data=b'',method='POST')
    with urllib.request.urlopen(request,timeout=180) as response:
        return dict(action=action,status=response.status,body=response.read().decode())
events=[]
events.append(control('start_profile'))
try:
    if a.long_body:
        command=[sys.executable,str(Path(__file__).with_name('probe_body_acceptance.py')),
            '--url',a.url,'--output',str(a.output),'--cases',a.case,
            '--input-tokens','4096','--output-tokens','1024']
    else:
        command=[sys.executable,str(Path(__file__).with_name('run_stream.py')),
            '--url',a.url,'--output',str(a.output),'--prompt-id',a.case,
            '--run-kind','profile_only_not_throughput']
    subprocess.run(command,check=True)
finally:
    events.append(control('stop_profile'))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'profile_control.json').write_text(json.dumps(dict(events=events,
        scope='Profile-only request: timing is instrumented and excluded from normal throughput comparisons'),indent=2)+'\n')
