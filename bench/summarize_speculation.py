"""MIT: derive per-request speculation averages from preserved Prometheus bytes."""
import argparse
import json
from pathlib import Path
import re

def counters(text):
    out={}
    for line in text.splitlines():
        if not line.startswith('vllm:spec_decode_') or '_created' in line:
            continue
        metric,value=line.rsplit(' ',1)
        if '_total' not in metric:
            continue
        out[metric]=float(value)
    return out

def summarize(result):
    before=counters(result['metrics_before'])
    after=counters(result['metrics_after'])
    assert before.keys()==after.keys()
    delta={k:after[k]-before[k] for k in before}
    assert all(v>=0 for v in delta.values())
    def value(name):
        return sum(v for k,v in delta.items() if k.startswith('vllm:spec_decode_'+name+'_total{'))
    drafts=value('num_drafts')
    accepted=value('num_accepted_tokens')
    drafted=value('num_draft_tokens')
    positions={int(re.search(r'position="(\d+)"',k)[1]):v for k,v in delta.items()
               if k.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{')}
    pos=[positions[i] for i in sorted(positions)]
    assert not pos or sum(pos)==accepted
    return dict(raw_counter_deltas=delta,drafts=drafts,drafted=drafted,accepted=accepted,
        mean_accepted_draft_tokens=accepted/drafts if drafts else None,
        mean_acceptance_length_including_bonus=1+accepted/drafts if drafts else None,
        position_unconditional=[v/drafts if drafts else None for v in pos],
        position_conditional=[v/(drafts if i==0 else pos[i-1])
            if (drafts if i==0 else pos[i-1]) else None for i,v in enumerate(pos)],
        scope='Async aggregate metric deltas; not individual step observations. Final output may be length-clipped.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for path in sorted(a.directory.rglob('*.json')):
        result=json.loads(path.read_text())
        if not result.get('passed') or 'metrics_before' not in result:
            continue
        rows.append(dict(file=str(path),prompt=result['prompt']['id'],warmup=result['warmup'],
            tok_per_second=result['decode_tok_per_second'],**summarize(result)))
    assert not a.output.exists()
    a.output.write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    print(json.dumps(rows,indent=2))
