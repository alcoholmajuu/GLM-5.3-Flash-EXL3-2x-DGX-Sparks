"""MIT: strict counter deltas for an isolated prefix-cache integration test."""
import math
import re


def counters(text):
    result={name:{} for name in ('queries','hits')}
    pattern=re.compile(r'^vllm:prefix_cache_(queries|hits)_total(\{.*\})?\s+(\S+)(?:\s+\S+)?$')
    for line in text.splitlines():
        match=pattern.fullmatch(line)
        if not match:
            continue
        name,labels,value=match.groups()
        labels=labels or ''
        value=float(value)
        if not math.isfinite(value) or value<0 or labels in result[name]:
            raise ValueError('Invalid or duplicate prefix cache counter')
        result[name][labels]=value
    if not result['queries'] or result['queries'].keys()!=result['hits'].keys():
        raise ValueError('Missing or mismatched prefix cache counters')
    return result


def delta(before,after):
    left,right=counters(before),counters(after)
    output={}
    for name in left:
        if left[name].keys()!=right[name].keys():
            raise ValueError('Prefix cache metric series changed')
        differences=[right[name][key]-value for key,value in left[name].items()]
        if any(value<0 for value in differences):
            raise ValueError('Prefix cache counter reset')
        output[name+'_tokens']=sum(differences)
    output['actual_hit']=output['hits_tokens']>0
    return output
