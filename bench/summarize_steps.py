"""MIT: summarize CPU scheduler observations without inventing emitted counts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
requests=defaultdict(list)
files=[]
for path in sorted(a.directory.rglob('scheduler-*.jsonl')):
    files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for line in path.read_text().splitlines():
        row=json.loads(line)
        assert 0<=row['accepted_draft_tokens']<=row['valid_draft_tokens']
        requests[row['request_id']].append(row)
rows=[]
for request_id,steps in requests.items():
    drafted=sum(s['valid_draft_tokens'] for s in steps)
    accepted=sum(s['accepted_draft_tokens'] for s in steps)
    k=max(s['valid_draft_tokens'] for s in steps)
    positions=[sum(s['accepted_draft_tokens']>i for s in steps) for i in range(k)]
    rows.append(dict(request_id=request_id,steps=len(steps),drafted=drafted,accepted=accepted,
        mean_accepted=accepted/len(steps),mean_accepted_plus_bonus=1+accepted/len(steps),
        acceptance_position_unconditional=[v/len(steps) for v in positions],
        acceptance_position_conditional=[v/(len(steps) if i==0 else positions[i-1])
            if (len(steps) if i==0 else positions[i-1]) else None for i,v in enumerate(positions)],
        histogram={str(i):sum(s['accepted_draft_tokens']==i for s in steps) for i in range(k+1)}))
assert rows,'No scheduler step observations found'
assert not a.output.exists()
a.output.write_text(json.dumps(dict(files=files,requests=rows,
    scope='CPU scheduler observations. Accepted+bonus is conventional, not exact emitted count at the final output limit.'),indent=2)+'\n')
print(json.dumps(dict(requests=len(rows),steps=sum(r['steps'] for r in rows))))
