"""MIT: sequential fixed-panel benchmark, warmup separated from measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request
import yaml


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--label',required=True)
    p.add_argument('--launch-receipt',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists(), 'Never replace benchmark evidence'
    config=yaml.safe_load(a.config.read_text())
    assert config['concurrency']==1 and config['output_tokens']==400
    prompts_path=a.config.parent.parent/config['prompts']
    prompts=json.loads(prompts_path.read_text())
    launch=json.loads(a.launch_receipt.read_text())
    assert launch['executed'] and all(row['exit_code']==0 for row in launch['plans'])
    a.output.mkdir(parents=True)
    report=dict(complete=False,label=a.label,config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
                prompts_sha256=hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
                launch_receipt_sha256=hashlib.sha256(a.launch_receipt.read_bytes()).hexdigest(),
                launch=launch,runs=[],summary={})
    def save():
        pending=a.output/'summary.json.tmp'
        pending.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        pending.replace(a.output/'summary.json')
    save()
    deadline=time.monotonic()+1800
    while True:
        try:
            with urllib.request.urlopen(a.url+'/health',timeout=3) as response:
                assert response.status==200
            break
        except (OSError,AssertionError):
            if time.monotonic()>deadline:
                raise TimeoutError('API readiness deadline reached')
            time.sleep(2)
    for warmup,count in ((True,config['warmup_runs_per_prompt']),(False,config['measured_runs_per_prompt'])):
        for repeat in range(count):
            out=a.output/f'{"warmup" if warmup else "measured"}_{repeat:02d}'
            command=[sys.executable,str(Path(__file__).with_name('run_stream.py')),
                     '--output',str(out),'--config',str(a.config),'--url',a.url,
                     '--run-kind',a.label,'--metrics']
            if warmup:
                command.append('--warmup')
            subprocess.run(command,check=True)
            for prompt in prompts:
                result=json.loads((out/(prompt['id']+'.json')).read_text())
                assert result['passed'] and result['completion_tokens']==400
                assert result['config_sha256']==report['config_sha256']
                report['runs'].append(dict(prompt_id=prompt['id'],warmup=warmup,repeat=repeat,
                                           result=str((out/(prompt['id']+'.json')).relative_to(a.output)),
                                           tok_per_second=result['decode_tok_per_second'],
                                           characters_per_decode_second=result['characters_per_decode_second']))
            save()
    for prompt in prompts:
        rows=[r for r in report['runs'] if not r['warmup'] and r['prompt_id']==prompt['id']]
        assert len(rows)==config['measured_runs_per_prompt']
        report['summary'][prompt['id']]=dict(runs=len(rows),
            median_decode_tok_per_second=statistics.median(r['tok_per_second'] for r in rows),
            median_characters_per_decode_second={channel:statistics.median(r['characters_per_decode_second'][channel] for r in rows)
                                                 for channel in ('reasoning','answer')})
    report['complete']=True
    save()
    print(json.dumps(report['summary'],ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
