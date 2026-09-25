"""MIT: exact copy of original packed replicated MLA q_a/kv_a, indexer wq_b and MTP eh_proj.

Main layers come from the approved 4.05bpw shards (index hash pinned), layer 45 from
mtp.safetensors. No requantization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
assert not a.output.exists()
index_path = a.source/'model.safetensors.index.json'
assert hashlib.sha256(index_path.read_bytes()).hexdigest() == 'ee1f2dbea800dea0b4225c38193f7ef41180ed42dd51f80a0cee58daf44cf606'
with (a.source/'mtp.safetensors').open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == '1bd09bd4eff4adc568e4ac3e92efb436c2d3aa5f0dceab4c72005d796066822d'
index = json.loads(index_path.read_text())['weight_map']
rx = re.compile(r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_a_proj|kv_a_proj_with_mqa|indexer\.wq_b)|eh_proj)\.(?:trellis|suh|svh|mul1)')
selected = {k: v for k, v in index.items() if rx.fullmatch(k)}
payload = {}
for shard in sorted(set(selected.values())):
    with safe_open(str(a.source/shard), framework='pt', device='cpu') as f:
        for key in sorted(k for k, v in selected.items() if v == shard):
            payload[key] = f.get_tensor(key).clone()
with safe_open(str(a.source/'mtp.safetensors'), framework='pt', device='cpu') as f:
    for key in f.keys():
        if rx.fullmatch(key):
            assert key not in payload
            payload[key] = f.get_tensor(key).clone()
modules = sorted({k.rsplit('.', 1)[0] for k in payload})
for name in modules:
    assert all(name+'.'+s in payload for s in ('trellis', 'suh', 'svh', 'mul1')), name
a.output.parent.mkdir(parents=True, exist_ok=True)
save_file(payload, str(a.output), metadata={'source_revision': '2a30229e67012798ba9f0cd832bb78abf4c363d5',
                                          'operation': 'exact tensor copy; no requantization'})
a.output.chmod(0o644)
with a.output.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
receipt = dict(source_revision='2a30229e67012798ba9f0cd832bb78abf4c363d5', output_sha256=digest,
               tensor_count=len(payload), module_count=len(modules), modules=modules,
               tensor_bytes=sum(t.numel()*t.element_size() for t in payload.values()),
               operation='exact original tensor copy')
a.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'modules'}), flush=True)
