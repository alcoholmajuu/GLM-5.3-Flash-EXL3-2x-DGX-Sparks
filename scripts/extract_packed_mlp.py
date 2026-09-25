"""MIT: copy original approved non-routed MLP tensors, without requantization."""
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
p.add_argument('--head-only', action='store_true')
p.add_argument('--attention-only', action='store_true',help='Original o/q_b/qkv matrices for a separate decode trial')
a = p.parse_args()
assert not (a.head_only and a.attention_only)
index_path = a.source / 'model.safetensors.index.json'
assert hashlib.sha256(index_path.read_bytes()).hexdigest() == 'ee1f2dbea800dea0b4225c38193f7ef41180ed42dd51f80a0cee58daf44cf606'
assert not a.output.exists()
index = json.loads(index_path.read_text())['weight_map']
selected = {k: v for k, v in index.items() if re.fullmatch(
    r'model\.language_model\.layers\.\d+\.mlp\.(?:shared_experts\.)?(?:gate_proj|up_proj|down_proj)\.(?:trellis|suh|svh|mul1)', k)}
if a.head_only:
    selected = {k:v for k,v in index.items() if k in
                {'lm_head.'+s for s in ('trellis','suh','svh','mul1')}}
    assert len(selected) == 4
if a.attention_only:
    selected={k:v for k,v in index.items() if re.fullmatch(
        r'model\.language_model\.layers\.\d+\.self_attn\.(?:o_proj|q_b_proj|qkv_proj)\.(?:trellis|suh|svh|mul1)',k)}
assert selected
payload = {}
for shard in sorted(set(selected.values())):
    with safe_open(str(a.source / shard), framework='pt', device='cpu') as f:
        for key in sorted(k for k, v in selected.items() if v == shard):
            payload[key] = f.get_tensor(key).clone()
modules = sorted({k.rsplit('.', 1)[0] for k in payload})
for name in modules:
    assert all(name + '.' + s in payload for s in ('trellis', 'suh', 'svh', 'mul1'))
a.output.parent.mkdir(parents=True, exist_ok=True)
save_file(payload, str(a.output), metadata={'source_revision': '2a30229e67012798ba9f0cd832bb78abf4c363d5',
                                          'operation': 'exact tensor copy; no requantization'})
a.output.chmod(0o644)
with a.output.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
receipt = dict(source_revision='2a30229e67012798ba9f0cd832bb78abf4c363d5',
               input_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
               output_sha256=digest, tensor_count=len(payload), modules=modules,
               tensor_bytes=sum(t.numel() * t.element_size() for t in payload.values()),
               original_shards=sorted(set(selected.values())), operation='exact original tensor copy')
a.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('modules', 'original_shards')}), flush=True)
