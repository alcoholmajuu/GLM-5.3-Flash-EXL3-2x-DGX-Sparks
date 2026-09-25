"""MIT: preserve kernel attribution and interval overlap for bounded torch traces."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

def category(name):
    s=name.lower()
    if 'nccl' in s or 'allreduce' in s or 'all_gather' in s:
        return 'communication'
    if any(k in s for k in ['moe','expert','router','routing']):
        return 'moe_and_routing'
    if 'exl3' in s:
        return 'packed_linear_or_quant_unattributed_module'
    if any(k in s for k in ['flash','attention','kda','kpool','indexer','fused_recurrent','delta_rule','chunk_','rope','rotary']):
        return 'attention_and_position'
    if any(k in s for k in ['cutlass','gemm','gemv','splitkreduce']):
        return 'dense_matmul_unattributed_module'
    return 'other_gpu'

def union(intervals):
    end=None
    total=0
    for a,b in sorted(intervals):
        if end is None or a>end:
            total+=b-a
            end=b
        elif b>end:
            total+=b-end
            end=b
    return total

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=[]
for path in sorted(a.directory.rglob('*.pt.trace.json.gz')):
    with gzip.open(path,'rt') as f:
        trace=json.load(f)
    kernels=defaultdict(lambda:dict(count=0,sum_us=0))
    intervals=defaultdict(list)
    cpu=[]
    runtime=[]
    runtime_names=defaultdict(lambda:dict(count=0,sum_us=0))
    for event in trace['traceEvents']:
        if event.get('ph')!='X' or 'dur' not in event:
            continue
        interval=(event['ts'],event['ts']+event['dur'])
        cat=event.get('cat','')
        if cat=='kernel':
            name=event['name']
            kernels[name]['count']+=1
            kernels[name]['sum_us']+=event['dur']
            intervals[category(name)].append(interval)
        elif cat=='cpu_op':
            cpu.append(interval)
        elif cat=='cuda_runtime':
            runtime.append(interval)
            runtime_names[event['name']]['count']+=1
            runtime_names[event['name']]['sum_us']+=event['dur']
    all_gpu=[x for values in intervals.values() for x in values]
    span=(max(b for _,b in all_gpu)-min(x for x,_ in all_gpu)) if all_gpu else 0
    rows.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        kernel_categories={k:dict(summed_ms=sum(y-x for x,y in v)/1000,union_ms=union(v)/1000) for k,v in intervals.items()},
        all_gpu_union_ms=union(all_gpu)/1000,cpu_op_union_ms=union(cpu)/1000,
        gpu_kernel_span_ms=span/1000,gpu_kernel_gap_ms=(span-union(all_gpu))/1000,
        cuda_runtime_union_ms=union(runtime)/1000,cuda_runtime_calls=dict(runtime_names),
        kernels=dict(sorted(kernels.items(),key=lambda kv:-kv[1]['sum_us']))))
report=dict(rows=rows,scope='Bounded profiled iterations, separate from speed benchmark. Name-based kernel categories retain every name for audit. CPU and GPU overlap; times are not additive step-latency fractions.')
assert rows,'No GPU trace files found'
assert not a.output.exists()
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='kernels'} for r in rows],indent=2))
